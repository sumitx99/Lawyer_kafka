"""Crash recovery and failed-job handling across parsing, indexing, the outbox and Kafka."""
import asyncio
import os
from uuid import uuid4, uuid5

import pytest
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.structs import TopicPartition
from sqlalchemy import text

from conftest import FakeModel, counts, parsed_event
from pyconsumers.consumers.rag_pipeline_consumers.rag_markdown_consumer import IndexingConsumer
from pyconsumers.consumers.rag_pipeline_consumers.stall_sweeper import StallSweeper
from pyconsumers.core.rag_pipeline.execute_indexing import Indexing
from pyconsumers.core.rag_pipeline.execute_standardization import Standardization
from pyconsumers.core.rag_pipeline.indexing.indexer import IndexingRepository
from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedBlock, ParserOutput
from pyconsumers.helpers.consumer_helper import EventProducer, create_consumer
from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.outbox_helper import DocumentOutboxPublisher
from scripts.requeue import requeue

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get('RAG_RUN_INTEGRATION') != '1',
                                                        reason='Requires local RAG services.')]


class TextParser:
    async def parse(self, source, output):
        output.mkdir()
        (output / 'document.md').write_text('Recovered text')
        return ParserOutput(parser_version='fixture', pages=[{'page_no': 1}], docling_document={},
                            blocks=[ParsedBlock(ref='#/texts/0', label='text', text='Recovered text', provenance=[{'page_no': 1}])])


async def set_doc(db, doc, sql, **values):
    async with db.transaction() as session:
        await session.execute(text(f'UPDATE documents SET {sql} WHERE id=:doc'), {'doc': doc, **values})


async def outbox(db, doc):
    async with db.transaction() as session:
        return (await session.execute(text('SELECT id, event_type, status, payload FROM outbox_events '
                                           'WHERE aggregate_id=:doc ORDER BY created_at'), {'doc': doc})).mappings().all()


async def test_sweeper_recovers_crashed_indexing_worker_once(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    state, _ = await IndexingRepository(db, settings).claim(parsed_event(project, doc), 'crashed-worker')
    assert state == 'claimed'
    await set_doc(db, doc, "lease_expires_at=now()-interval '1 second'")  # the worker died; its lease ran out
    sweeper = StallSweeper(db, settings)
    assert (await sweeper.sweep_once(project))['recovered'] == 1
    assert (await sweeper.sweep_once(project))['recovered'] == 0
    state = await counts(db, doc)
    assert (state['status'], state['error_message']) == ('PARSED', 'LeaseExpired')
    [row] = await outbox(db, doc)
    assert row['event_type'] == 'doc.parsed' and row['id'] == uuid5(doc, 'sweep:lease:INDEXING:1')
    event = RagEvent.model_validate(row['payload'])
    assert await Indexing(db, storage, cache, settings, model=FakeModel()).execute(event) == 'indexed'
    assert (await counts(db, doc))['chunks'] == 3


async def test_sweeper_recovers_crashed_parsing_worker_without_the_api(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    await set_doc(db, doc, """status='PARSING', attempt_count=1, parsed_object_key=NULL, lease_owner='crashed',
        lease_expires_at=now()-interval '1 second'""")
    assert (await StallSweeper(db, settings).sweep_once(project))['recovered'] == 1
    [row] = await outbox(db, doc)
    assert (row['event_type'], (await counts(db, doc))['status']) == ('doc.uploaded', 'PENDING')
    # The worker's own outbox publisher delivers parse-stage recovery events (API not needed).
    topic = 'reliability.uploaded.' + uuid4().hex
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    publisher = DocumentOutboxPublisher(db, settings.model_copy(update={'kafka_topic_uploaded': topic}))
    consumer = create_consumer(topic, 'reliability-' + uuid4().hex, settings)
    try:
        await admin.start()
        await admin.create_topics([NewTopic(topic, 1, 1)])
        await consumer.start()
        assert await publisher.dispatch_one(project_filter=project)
        delivered = RagEvent.model_validate_json((await asyncio.wait_for(consumer.getone(), 20)).value)
    finally:
        await publisher.close()
        await consumer.stop()
        await admin.delete_topics([topic])
        await admin.close()
    assert delivered.event_id == row['id']
    service = Standardization(db, storage, cache, settings, parser=TextParser(), vision=None)
    service.vision.enrich = lambda result, output, project_id: asyncio.sleep(0, result)
    assert await service.execute(delivered) == 'parsed'
    assert (await counts(db, doc))['status'] == 'PARSED'


async def test_sweeper_leaves_live_workers_alone_and_fails_exhausted_documents(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    await set_doc(db, doc, "status='INDEXING', lease_owner='live', lease_expires_at=now()+interval '1 minute'")
    swept = await StallSweeper(db, settings).sweep_once(project)
    assert {key: swept[key] for key in ('recovered', 'failed', 'requeued_stuck')} == {'recovered': 0, 'failed': 0, 'requeued_stuck': 0}
    assert (await counts(db, doc))['status'] == 'INDEXING'
    await set_doc(db, doc, "lease_expires_at=now()-interval '1 second', index_attempt_count=:max",
                  max=settings.index_max_attempts)
    assert (await StallSweeper(db, settings).sweep_once(project))['failed'] == 1
    state = await counts(db, doc)
    assert (state['status'], state['error_message']) == ('FAILED', 'IndexAttemptLimitExceeded')
    assert await outbox(db, doc) == []


async def test_sweeper_requeues_documents_stuck_without_an_event_once(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    sweeper = StallSweeper(db, settings.model_copy(update={'sweep_stuck_seconds': 0}))
    assert (await sweeper.sweep_once(project))['requeued_stuck'] == 1  # PARSED, but its event was lost
    assert (await sweeper.sweep_once(project))['requeued_stuck'] == 0  # a pending event now exists
    async with db.transaction() as session:
        await session.execute(text("UPDATE outbox_events SET status='PUBLISHED' WHERE aggregate_id=:doc"), {'doc': doc})
    assert (await sweeper.sweep_once(project))['requeued_stuck'] == 0  # same state: same recovery id, not re-sent
    [row] = await outbox(db, doc)
    assert row['event_type'] == 'doc.parsed'


async def test_requeue_failed_restarts_from_the_stage_that_failed(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    await set_doc(db, doc, "status='FAILED', index_attempt_count=3, error_message='EmbeddingFailure'")
    assert await requeue(project, [doc], 'failed', dry_run=True, db=db)
    assert (await counts(db, doc))['status'] == 'FAILED'  # dry run changes nothing
    await requeue(project, [doc], 'failed', db=db)
    state = await counts(db, doc)
    assert (state['status'], state['index_attempt_count']) == ('PARSED', 0)
    [row] = await outbox(db, doc)
    assert row['event_type'] == 'doc.parsed'
    assert await Indexing(db, storage, cache, settings, model=FakeModel()).execute(RagEvent.model_validate(row['payload'])) == 'indexed'

    await set_doc(db, doc, "status='FAILED', parsed_object_key=NULL, attempt_count=3, error_message='ParseFailure'")
    await requeue(project, [doc], 'failed', db=db)
    assert (await counts(db, doc))['status'] == 'PENDING'
    assert (await outbox(db, doc))[-1]['event_type'] == 'doc.uploaded'
    with pytest.raises(ValueError):
        await requeue(project, [doc], 'failed', db=db)  # no longer FAILED


class Hang:
    """A service that never finishes, so the consumer is killed mid-document."""
    def __init__(self):
        self.started = asyncio.Event()

    async def execute(self, event):
        self.started.set()
        await asyncio.Event().wait()


async def test_kafka_redelivers_when_a_worker_dies_before_commit(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    topic, group = 'reliability.parsed.' + uuid4().hex, 'reliability-indexing-' + uuid4().hex
    settings = settings.model_copy(update={'kafka_topic_parsed': topic, 'indexing_group_id': group})
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    producer = EventProducer(settings)
    hang = Hang()
    first = second = None
    try:
        await admin.start()
        await admin.create_topics([NewTopic(topic, 1, 1), NewTopic(topic + '.dlq', 1, 1)])
        await producer.start()
        await producer.send(parsed_event(project, doc))
        first = asyncio.create_task(IndexingConsumer(db, storage, cache, settings, service=hang).run())
        await asyncio.wait_for(hang.started.wait(), 60)
        first.cancel()  # worker dies mid-document: nothing was committed
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(IndexingConsumer(
            db, storage, cache, settings, service=Indexing(db, storage, cache, settings, model=FakeModel())).run())
        for _ in range(120):
            if (await counts(db, doc))['status'] == 'INDEXED':
                break
            await asyncio.sleep(0.5)
        state = await counts(db, doc)
        assert (state['status'], state['chunks'], state['index_version']) == ('INDEXED', 3, 1)
        offsets = await admin.list_consumer_group_offsets(group)
        assert offsets[TopicPartition(topic, 0)].offset == 1  # committed exactly once, after the durable write
    finally:
        for task in (first, second):
            if task and not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        await producer.close()
        await admin.delete_topics([topic, topic + '.dlq'])
        await admin.close()
