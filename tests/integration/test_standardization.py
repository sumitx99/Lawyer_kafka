import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from pyconsumers.helpers.consumer_helper import create_consumer
from pyconsumers.helpers.outbox_helper import DocumentOutboxPublisher

import pytest
from sqlalchemy import text

from pyconsumers.consumers.rag_pipeline_consumers.standardization_consumer import StandardizationConsumer
from pyconsumers.core.rag_pipeline.execute_standardization import Standardization
from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedArtifact, ParsedBlock, ParserOutput
from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.parsing_repository import CONSUMER_NAME, ParsingRepository
from pyconsumers.helpers.parsing_repository import LeaseLost
from pyconsumers.helpers.postgres_helper import Database
from pyconsumers.helpers.secrets_control import get_settings
from pyconsumers.helpers.storage_helper import Storage
from pyconsumers.helpers.valkey_helper import Cache
from pyconsumers.core.rag_pipeline.connectors.unstructured.utils.vision import VisionEnricher
from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedAsset
from PIL import Image

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get('RAG_RUN_INTEGRATION') != '1',
                                                        reason='Requires local RAG services.')]


class FixtureParser:
    def __init__(self, content='Parsed text', started=None, release=None, fail=False):
        self.content, self.started, self.release, self.fail = content, started, release, fail
        self.calls = 0

    async def parse(self, source, output):
        self.calls += 1
        if self.started:
            self.started.set()
            await self.release.wait()
        if self.fail:
            raise RuntimeError('simulated parser failure')
        output.mkdir()
        (output / 'document.md').write_text(self.content)
        return ParserOutput(parser_version='fixture', blocks=[ParsedBlock(
            ref='#/texts/0', label='text', text=self.content, provenance=[{'page_no': 1}])],
            pages=[{'page_no': 1}], docling_document={'texts': [{'text': self.content}]})


@pytest.fixture
async def context():
    settings = get_settings().model_copy(update={'parse_max_attempts': 2})
    db, storage, cache = Database(settings), Storage(settings), Cache(settings)
    project, doc, event_id = uuid4(), uuid4(), uuid4()
    raw = b'fictional fixture bytes'
    key = f'projects/{project}/documents/{doc}/raw.pdf'
    event = RagEvent(event_id=event_id, event_type='doc.uploaded', project_id=project, doc_id=doc)
    await storage.put_bytes(key, raw, 'application/pdf')
    async with db.transaction() as session:
        await session.execute(text('''INSERT INTO documents
            (id,project_id,filename,media_type,byte_count,sha256,object_key)
            VALUES(:doc,:project,'fixture.pdf','application/pdf',:size,:sha,:key)'''),
            {'doc': doc, 'project': project, 'size': len(raw), 'sha': sha256(raw).hexdigest(), 'key': key})
    try:
        yield settings, db, storage, cache, event
    finally:
        async with db.transaction() as session:
            for table in ['processed_events', 'outbox_events', 'documents']:
                await session.execute(text(f'DELETE FROM {table} WHERE project_id=:project'), {'project': project})
        objects = await asyncio.to_thread(storage.client.list_objects_v2, Bucket=storage.bucket,
                                         Prefix=f'projects/{project}/')
        for item in objects.get('Contents', []):
            await storage.delete(item['Key'])
        keys = [key async for key in cache.client.scan_iter(match=f'rag:v1:{project}:*')]
        if keys:
            await cache.client.delete(*keys)
        await cache.close()
        await storage.close()
        await db.close()


async def test_parse_artifacts_atomic_completion_and_duplicate(context):
    settings, db, storage, cache, event = context
    parser = FixtureParser()
    service = Standardization(db, storage, cache, settings, parser=parser)
    assert await service.execute(event) == 'parsed'
    assert await service.execute(event) == 'duplicate'
    assert parser.calls == 1
    async with db.transaction() as session:
        row = (await session.execute(text('SELECT * FROM documents WHERE id=:id'), {'id': event.doc_id})).mappings().one()
        assert row['status'] == 'PARSED' and row['attempt_count'] == 1 and row['lease_owner'] is None
        artifact = ParsedArtifact.model_validate_json(await storage.get_bytes(row['parsed_object_key']))
        assert artifact.doc_id == event.doc_id and artifact.blocks[0].provenance[0]['page_no'] == 1
        assert await storage.get_bytes(artifact.markdown_key) == b'Parsed text'
        assert await session.scalar(text('SELECT count(*) FROM outbox_events WHERE aggregate_id=:id'), {'id': event.doc_id}) == 1
        payload = await session.scalar(text('SELECT payload FROM outbox_events WHERE aggregate_id=:id'), {'id': event.doc_id})
        parsed = RagEvent.model_validate(payload)
        assert parsed.event_type == 'doc.parsed' and parsed.correlation_id == event.correlation_id
        assert await session.scalar(text('SELECT count(*) FROM processed_events WHERE event_id=:id'), {'id': event.event_id}) == 1


async def test_retry_then_failed_and_durable_dead_letter_ack(context):
    settings, db, storage, cache, event = context
    service = Standardization(db, storage, cache, settings, parser=FixtureParser(fail=True))
    assert await service.execute(event) == 'retry'
    assert await service.execute(event) == 'failed'
    consumer, producer = AsyncMock(), SimpleNamespace(client=AsyncMock())
    record = SimpleNamespace(topic='doc.uploaded', partition=0, offset=5, key=b'fixture', value=event.model_dump_json().encode())
    worker = StandardizationConsumer(db, storage, cache, settings, service=service)
    await worker.process_record(record, consumer, producer)
    producer.client.send_and_wait.assert_awaited_once()
    payload = json.loads(producer.client.send_and_wait.call_args.args[1])
    assert payload['error_code'] == 'ParsingFailed'
    consumer.commit.assert_awaited_once()
    assert await service.execute(event) == 'duplicate'
    async with db.transaction() as session:
        assert await session.scalar(text('SELECT count(*) FROM outbox_events WHERE aggregate_id=:id'), {'id': event.doc_id}) == 0
        assert await session.scalar(text('SELECT status FROM documents WHERE id=:id'), {'id': event.doc_id}) == 'FAILED'


async def test_active_lease_and_stale_worker_cannot_overwrite(context):
    settings, db, storage, cache, event = context
    started, release = asyncio.Event(), asyncio.Event()
    first = Standardization(db, storage, cache, settings,
                            parser=FixtureParser('stale output', started=started, release=release))
    second = Standardization(db, storage, cache, settings, parser=FixtureParser('winning output'))
    task = asyncio.create_task(first.execute(event))
    try:
        await asyncio.wait_for(started.wait(), 5)
        assert await second.execute(event) == 'busy'
        async with db.transaction() as session:
            await session.execute(text('UPDATE documents SET lease_expires_at=:expiry WHERE id=:id'),
                                  {'expiry': datetime.now(timezone.utc)-timedelta(seconds=1), 'id': event.doc_id})
        assert await second.execute(event) == 'parsed'
    finally:
        release.set()
        assert await task == 'busy'
    async with db.transaction() as session:
        manifest = await session.scalar(text('SELECT parsed_object_key FROM documents WHERE id=:id'), {'id': event.doc_id})
        artifact = ParsedArtifact.model_validate_json(await storage.get_bytes(manifest))
        assert await storage.get_bytes(artifact.markdown_key) == b'winning output'
        assert await session.scalar(text('SELECT count(*) FROM outbox_events WHERE aggregate_id=:id'), {'id': event.doc_id}) == 1


async def test_integrity_failure_and_project_filter(context):
    settings, db, storage, cache, event = context
    parser = FixtureParser()
    service = Standardization(db, storage, cache, settings, parser=parser)
    other = event.model_copy(update={'project_id': uuid4()})
    assert await service.execute(other) == 'missing'
    async with db.transaction() as session:
        await session.execute(text("UPDATE documents SET sha256=:sha WHERE id=:id"), {'sha': '0'*64, 'id': event.doc_id})
    assert await service.execute(event) == 'failed'
    assert parser.calls == 0


async def test_heartbeat_ownership_and_crash_attempt_limit(context):
    settings, db, storage, cache, event = context
    repo = ParsingRepository(db, settings)
    state, row = await repo.claim(event, 'worker-a')
    assert state == 'claimed'
    async with db.transaction() as session:
        await session.execute(text('UPDATE documents SET lease_expires_at=:expiry WHERE id=:id'),
                              {'expiry': datetime.now(timezone.utc)+timedelta(seconds=5), 'id': event.doc_id})
    await repo.heartbeat(event, 'worker-a')
    async with db.transaction() as session:
        expiry = await session.scalar(text('SELECT lease_expires_at FROM documents WHERE id=:id'), {'id': event.doc_id})
        assert expiry > datetime.now(timezone.utc)+timedelta(seconds=90)
    with pytest.raises(LeaseLost):
        await repo.heartbeat(event, 'worker-b')
    async with db.transaction() as session:
        await session.execute(text('UPDATE documents SET attempt_count=:maximum,lease_expires_at=:expiry WHERE id=:id'),
                              {'maximum': settings.parse_max_attempts, 'expiry': datetime.now(timezone.utc)-timedelta(seconds=1), 'id': event.doc_id})
    state, _ = await repo.claim(event, 'worker-b')
    assert state == 'failed'


async def test_parsed_outbox_publishes_without_api(context):
    settings, db, storage, cache, event = context
    assert await Standardization(db, storage, cache, settings, parser=FixtureParser()).execute(event) == 'parsed'
    topic = 'infra.parsed.' + uuid4().hex
    settings = settings.model_copy(update={'kafka_topic_parsed': topic})
    publisher = DocumentOutboxPublisher(db, settings)
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    consumer = create_consumer(topic, 'parsed-test-' + uuid4().hex, settings)
    try:
        await admin.start()
        await admin.create_topics([NewTopic(topic, 1, 1)])
        await consumer.start()
        assert await publisher.dispatch_one(project_filter=event.project_id)
        record = await asyncio.wait_for(consumer.getone(), 20)
        parsed = RagEvent.model_validate_json(record.value)
        assert parsed.event_type == 'doc.parsed' and parsed.doc_id == event.doc_id
        async with db.transaction() as session:
            assert await session.scalar(text('SELECT status FROM outbox_events WHERE id=:id'), {'id': parsed.event_id}) == 'PUBLISHED'
    finally:
        await publisher.close()
        await consumer.stop()
        await admin.delete_topics([topic])
        await admin.close()


async def test_visual_enrichment_is_persisted_with_source_and_summary(context):
    settings, db, storage, cache, event = context
    class ImageParser(FixtureParser):
        async def parse(self, source, output):
            result = await super().parse(source, output)
            Image.new('RGB', (100, 100), 'white').save(output / 'picture.png')
            result.assets = [ParsedAsset(ref='#/pictures/0', path='picture.png', provenance=[{'page_no': 1}])]
            return result
    observation = dict(kind='diagram', title='Workflow', x_axis=None, y_axis=None, legend=[], visible_text=[],
                       numeric_values=[], observations=['An arrow connects intake and review.'],
                       interpretation='The workflow proceeds from intake to review.', limitations=[], status='interpreted')
    client = SimpleNamespace(describe_image=AsyncMock(return_value=json.dumps(observation)))
    vision = VisionEnricher(settings, cache, client=client)
    service = Standardization(db, storage, cache, settings, parser=ImageParser(), vision=vision)
    assert await service.execute(event) == 'parsed'
    async with db.transaction() as session:
        row = (await session.execute(text('SELECT parsed_object_key,parse_summary FROM documents WHERE id=:id'),
                                     {'id': event.doc_id})).mappings().one()
        assert row['parse_summary']['visual_description_count'] == 1
        assert row['parse_summary']['vision_model'] == settings.vision_model
        artifact = ParsedArtifact.model_validate_json(await storage.get_bytes(row['parsed_object_key']))
        assert artifact.assets[0].visual_description.observation.kind == 'diagram'
        assert artifact.blocks[-1].provenance == [{'page_no': 1}]
        assert artifact.blocks[-1].data['source_asset_ref'] == '#/pictures/0'
        assert await storage.get_bytes(artifact.assets[0].path)
        assert b'Model-derived visual interpretation' in await storage.get_bytes(artifact.markdown_key)
