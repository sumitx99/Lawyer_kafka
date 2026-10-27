"""Folder pipeline consumer side: fair queue, transitions, retries, fencing, sweeper, Kafka wake-up."""
import asyncio
from collections import Counter
from contextlib import suppress
from hashlib import sha256
import json
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from sqlalchemy import text

from pyconsumers.consumers.rag_pipeline_consumers.stage_worker import StageWorker
from pyconsumers.consumers.rag_pipeline_consumers.stall_sweeper import StallSweeper
from pyconsumers.core.rag_pipeline.userallocation.UserAllocation import LeaseLost, StageError, StageResult, UserAllocation
from pyconsumers.helpers.consumer_helper import EventProducer, create_consumer
from pyconsumers.helpers.postgres_helper import Database
from pyconsumers.helpers.secrets_control import get_settings
from pyconsumers.helpers.valkey_helper import Cache

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get('RAG_RUN_INTEGRATION') != '1',
                                                        reason='Requires local RAG services.')]


@pytest.fixture
async def env():
    settings = get_settings().model_copy(update={'stage_retry_base_seconds': 0})
    db, cache = Database(settings), Cache(settings)
    made = []

    async def folder(user, files=1, file_type='pdf', first='standardization'):
        thread = uuid4()
        async with db.transaction() as session:
            await session.execute(text("INSERT INTO unstructured_thread(id,name,created_by) VALUES(:id,'test',:user)"),
                                  {'id': thread, 'user': user})
            docs = []
            for n in range(files):
                doc = uuid4()
                await session.execute(text(f"""INSERT INTO unstructured_data(id,thread_id,file_name,file_type,category,
                    byte_count,sha256,s3_path,upload_status,{first}_status,{first}_updated_at,created_at)
                    VALUES(:doc,:thread,:name,:type,:category,10,:sha,:path,'COMPLETED','WAITING',now(),
                    now() + (:n * interval '1 millisecond'))"""),
                    {'doc': doc, 'thread': thread, 'name': f'f{n}.{file_type}', 'type': file_type,
                     'category': 'csv' if file_type == 'csv' else 'document', 'sha': sha256(doc.bytes).hexdigest(),
                     'path': f'THREAD_{thread}/{file_type}/doc_{doc}/f{n}', 'n': n})
                docs.append(doc)
        made.append(thread)
        return thread, docs

    try:
        yield SimpleNamespace(settings=settings, db=db, cache=cache, folder=folder)
    finally:
        async with db.transaction() as session:
            await session.execute(text('DELETE FROM outbox_events WHERE aggregate_id = ANY(:t)'), {'t': made})
            await session.execute(text('DELETE FROM unstructured_thread WHERE id = ANY(:t)'), {'t': made})
        await cache.close()
        await db.close()


def queue(env, stage='standardization', **limits):
    return UserAllocation(env.db, env.cache, env.settings.model_copy(update={f'{stage}_{k}': v for k, v in limits.items()}), stage)


async def stage_of(env, doc, *stages):
    async with env.db.transaction() as session:
        row = (await session.execute(text('SELECT * FROM unstructured_data WHERE id=:doc'), {'doc': doc})).mappings().one()
    return tuple(row[f'{s}_status'] for s in stages) if len(stages) > 1 else row[f'{stages[0]}_status']


async def messages(env, thread):
    async with env.db.transaction() as session:
        return Counter(await session.scalars(text('SELECT event_type FROM outbox_events WHERE aggregate_id=:t'), {'t': thread}))


async def test_fair_queue_serves_every_user_and_caps_the_stage(env):
    big, big_docs = await env.folder('user-a', files=5)  # older and larger
    small, small_docs = await env.folder('user-b', files=1)
    q = queue(env, max_workers=2, max_per_user=1)
    first = await q.get_next_document('w1')
    second = await q.get_next_document('w2')
    assert first.doc_id == big_docs[0] and first.row['created_by'] == 'user-a'
    assert second.doc_id == small_docs[0]  # user-a is at its cap, so user-b is not stuck behind 4 more files
    assert await q.get_next_document('w3') is None  # stage cap of 2 running
    await q.complete(first, StageResult())
    assert (await q.get_next_document('w4')).doc_id == big_docs[1]  # the freed slot goes to the next file


async def test_concurrent_claims_never_exceed_limits_or_double_claim(env):
    for user in ('u1', 'u2', 'u3', 'u4'):
        await env.folder(user, files=5)
    q = queue(env, max_workers=5, max_per_user=2)
    claims = [c for c in await asyncio.gather(*(q.get_next_document(f'w{n}') for n in range(15))) if c]
    assert len(claims) == 5 and len({c.doc_id for c in claims}) == 5
    assert max(Counter(c.row['created_by'] for c in claims).values()) <= 2


async def test_full_folder_flow_runs_metaclassifier_exactly_once_per_file(env):
    thread, (normal, text_only, image_only) = await env.folder('user-a', files=3)
    calls = Counter()

    async def standardization(claim):
        calls['standardization'] += 1
        nxt = {text_only: {'image': 'INAPPLICABLE'}, image_only: {'markdown': 'INAPPLICABLE'}}.get(claim.doc_id, {})
        return StageResult(updates={'s3_path_tomarkdown': f'md/{claim.doc_id}', 'page_count': 2}, summary={'pages': 2}, next=nxt)

    async def other(claim):
        calls[claim.stage] += 1
        await asyncio.sleep(0.05)  # markdown and image of one file finish at nearly the same time
        return StageResult(summary={'ok': True})

    stages = ['standardization', 'markdown', 'image', 'metaclassifier']
    workers = {s: StageWorker(env.db, env.cache, env.settings, s, standardization if s == 'standardization' else other,
                              producer=SimpleNamespace(client=None)) for s in stages}
    while True:  # drain every stage, markdown and image concurrently, until nothing is left
        progressed = await workers['standardization'].process_next()
        progressed |= any(await asyncio.gather(workers['markdown'].process_next(), workers['image'].process_next()))
        progressed |= await workers['metaclassifier'].process_next()
        if not progressed:
            break
    assert calls == {'standardization': 3, 'markdown': 2, 'image': 2, 'metaclassifier': 3}
    assert await stage_of(env, normal, *stages) == ('COMPLETED', 'COMPLETED', 'COMPLETED', 'COMPLETED')
    assert await stage_of(env, text_only, *stages) == ('COMPLETED', 'COMPLETED', 'INAPPLICABLE', 'COMPLETED')
    assert await stage_of(env, image_only, *stages) == ('COMPLETED', 'INAPPLICABLE', 'COMPLETED', 'COMPLETED')
    async with env.db.transaction() as session:
        status = await session.scalar(text('SELECT status FROM unstructured_thread WHERE id=:t'), {'t': thread})
        summary, path = (await session.execute(text('SELECT processing_summary, s3_path_tomarkdown FROM unstructured_data WHERE id=:d'),
                                               {'d': normal})).one()
    assert status == 'COMPLETED' and path == f'md/{normal}'
    assert summary == {'standardization': {'pages': 2}, 'markdown': {'ok': True}, 'image': {'ok': True}, 'metaclassifier': {'ok': True}}
    sent = await messages(env, thread)
    # A stage gets a message only when it becomes WAITING: no markdown for the image-only file, no image for the text-only one.
    assert (sent['thread.markdown'], sent['thread.image'], sent['thread.metaclassifier']) == (2, 2, 3)


async def test_retries_back_off_then_fail_with_a_dead_letter(env):
    thread, [doc] = await env.folder('user-a')
    calls = []

    async def broken(claim):
        calls.append(claim.row['standardization_attempts'])
        raise RuntimeError('Docling crashed')

    producer = SimpleNamespace(client=SimpleNamespace(send_and_wait=None))
    sent = []

    async def send(topic, value, **kwargs):
        sent.append((topic, json.loads(value)))
    producer.client.send_and_wait = send
    worker = StageWorker(env.db, env.cache, env.settings, 'standardization', broken, producer=producer)
    for _ in range(3):
        assert await worker.process_next() is True
    assert await worker.process_next() is False
    assert calls == [1, 2, 3] and await stage_of(env, doc, 'standardization') == 'FAILED'
    [(topic, record)] = sent
    assert topic == 'STNDZ_TOPIC_DLQ' and (record['doc_id'], record['error'], record['attempts']) == (str(doc), 'RuntimeError', 3)
    async with env.db.transaction() as session:
        assert await session.scalar(text('SELECT status FROM unstructured_thread WHERE id=:t'), {'t': thread}) == 'FAILED'

    # Backoff: with a real base delay, a failed attempt is not picked up again immediately.
    _, [later] = await env.folder('user-b')
    slow = StageWorker(env.db, env.cache, env.settings.model_copy(update={'stage_retry_base_seconds': 60}),
                       'standardization', broken, producer=producer)
    assert await slow.process_next() is True and await slow.process_next() is False
    assert await stage_of(env, later, 'standardization') == 'WAITING'

    # A terminal error fails at the first attempt.
    _, [bad] = await env.folder('user-c')

    async def unreadable(claim):
        raise StageError('Encrypted PDF', terminal=True)
    await StageWorker(env.db, env.cache, env.settings, 'standardization', unreadable, producer=producer).process_next()
    assert await stage_of(env, bad, 'standardization') == 'FAILED'


async def test_a_worker_that_lost_its_lease_cannot_write(env):
    thread, [doc] = await env.folder('user-a')
    q = queue(env)
    stale = await q.get_next_document('stale-worker')
    async with env.db.transaction() as session:
        await session.execute(text("UPDATE unstructured_data SET standardization_lease_expires_at=now()-interval '1 second' WHERE id=:d"), {'d': doc})
    fresh = await q.get_next_document('fresh-worker')  # an expired lease no longer counts as running... but status is still RUNNING
    assert fresh is None  # the sweeper must reset it first; nobody can steal a RUNNING row directly
    for call in (q.complete(stale, StageResult()), q.heartbeat(stale), q.fail(stale, 'x')):
        with pytest.raises(LeaseLost):
            await call


async def test_sweeper_recovers_dead_stages_and_requeues_stuck_ones_once(env):
    thread, [doc] = await env.folder('user-a')
    q = queue(env)
    await q.get_next_document('dead-worker')
    async with env.db.transaction() as session:
        await session.execute(text("UPDATE unstructured_data SET standardization_lease_expires_at=now()-interval '1 second' WHERE id=:d"), {'d': doc})
    sweeper = StallSweeper(env.db, env.settings)
    counts = await sweeper.sweep_folders(thread_filter=thread)
    assert counts['stage_recovered'] == 1
    assert await stage_of(env, doc, 'standardization') == 'WAITING'
    assert (await messages(env, thread))['thread.standardization'] == 1
    # The recovered file is claimable again; once attempts are used up a dead worker means FAILED 'stalled'.
    for _ in range(2):
        await q.get_next_document('dead-worker')
        async with env.db.transaction() as session:
            await session.execute(text("UPDATE unstructured_data SET standardization_lease_expires_at=now()-interval '1 second' WHERE id=:d"), {'d': doc})
        await sweeper.sweep_folders(thread_filter=thread)
    async with env.db.transaction() as session:
        status, error, attempts = (await session.execute(text(
            'SELECT standardization_status, standardization_error, standardization_attempts FROM unstructured_data WHERE id=:d'),
            {'d': doc})).one()
    assert (status, error, attempts) == ('FAILED', 'stalled', 3)

    # WAITING too long with no recent message: one message per stuck period, not one per sweep.
    stuck_thread, [stuck] = await env.folder('user-b', first='markdown')
    async with env.db.transaction() as session:
        await session.execute(text("UPDATE unstructured_data SET markdown_updated_at=now()-interval '1 hour' WHERE id=:d"), {'d': stuck})
    assert (await sweeper.sweep_folders(thread_filter=stuck_thread))['stage_requeued'] == 1
    async with env.db.transaction() as session:
        await session.execute(text("UPDATE outbox_events SET status='PUBLISHED' WHERE aggregate_id=:t"), {'t': stuck_thread})
    assert (await sweeper.sweep_folders(thread_filter=stuck_thread))['stage_requeued'] == 0
    assert (await messages(env, stuck_thread))['thread.markdown'] == 1


async def test_kafka_message_wakes_the_worker_and_bad_messages_are_dead_lettered(env):
    thread, [doc] = await env.folder('user-a')
    topic = 'test.stndz.' + uuid4().hex
    settings = env.settings.model_copy(update={'kafka_topic_standardization': topic, 'stage_poll_seconds': 600})
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    producer = EventProducer(settings)
    handled = asyncio.Event()

    async def handler(claim):
        handled.set()
        return StageResult(next={'markdown': 'INAPPLICABLE', 'image': 'INAPPLICABLE'})

    worker = StageWorker(env.db, env.cache, settings, 'standardization', handler, group_id='test-' + uuid4().hex)
    worker.wake.clear()  # prove the Kafka message, not the start-up drain, does the waking
    dlq = create_consumer(topic + '_DLQ', 'test-dlq-' + uuid4().hex, settings)
    task = None
    try:
        await admin.start()
        await admin.create_topics([NewTopic(topic, 1, 1), NewTopic(topic + '_DLQ', 1, 1)])
        await producer.start()
        await dlq.start()
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(3)  # group join
        assert not handled.is_set()
        await producer.client.send_and_wait(topic, b'{"not": "a folder message"}')
        await producer.send_thread('thread.standardization', thread)
        await asyncio.wait_for(handled.wait(), 30)
        record = await asyncio.wait_for(dlq.getone(), 20)
        assert json.loads(record.value)['error_code'] == 'InvalidMessage'
        for _ in range(40):
            if await stage_of(env, doc, 'standardization') == 'COMPLETED':
                break
            await asyncio.sleep(0.25)
        assert await stage_of(env, doc, 'standardization', 'metaclassifier') == ('COMPLETED', 'WAITING')
    finally:
        if task:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await producer.close()
        await dlq.stop()
        await admin.delete_topics([topic, topic + '_DLQ'])
        await admin.close()


async def test_worker_outbox_publishes_next_stage_messages_without_the_api(env):
    from pyconsumers.helpers.outbox_helper import DocumentOutboxPublisher
    thread, [doc] = await env.folder('user-a')
    q = queue(env)
    await q.complete(await q.get_next_document('w1'), StageResult(next={'image': 'INAPPLICABLE'}))
    topic = 'test.markdown.' + uuid4().hex
    settings = env.settings.model_copy(update={'kafka_topic_markdown': topic})
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    publisher, consumer = DocumentOutboxPublisher(env.db, settings), create_consumer(topic, 'test-' + uuid4().hex, settings)
    try:
        await admin.start()
        await admin.create_topics([NewTopic(topic, 1, 1)])
        await consumer.start()
        assert await publisher.dispatch_one(project_filter=thread)
        record = await asyncio.wait_for(consumer.getone(), 20)
    finally:
        await publisher.close()
        await consumer.stop()
        await admin.delete_topics([topic])
        await admin.close()
    assert json.loads(record.value) == {'unstructured_thread_id': str(thread)} and record.key == str(thread).encode()
    async with env.db.transaction() as session:
        assert await session.scalar(text("SELECT status FROM outbox_events WHERE aggregate_id=:t AND event_type='thread.markdown'"),
                                    {'t': thread}) == 'PUBLISHED'


# ---- Phase D: real stage handlers against Postgres + SeaweedFS (fake parser, vision and embedder) ----

class FixtureParser:
    """Docling stand-in: the fictional lease text plus three pictures (a chart, a logo, a table)."""
    async def parse(self, source, output):
        from io import BytesIO
        from PIL import Image
        from conftest import BLOCKS
        from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedAsset, ParserOutput
        (output / 'assets').mkdir(parents=True)
        (output / 'document.md').write_text('# Lease agreement')
        assets = []
        for n, colour in enumerate(('red', 'blue', 'green')):
            buffer = BytesIO()
            Image.new('RGB', (16, 16), colour).save(buffer, format='PNG')
            (output / f'assets/picture-{n}.png').write_bytes(buffer.getvalue())
            assets.append(ParsedAsset(ref=f'#/pictures/{n}', path=f'assets/picture-{n}.png', provenance=[{'page_no': n + 1}]))
        return ParserOutput(parser_version='fixture', pages=[{'page_no': 1}, {'page_no': 2}, {'page_no': 3}],
                            blocks=BLOCKS, assets=assets, docling_document={})


class FixtureVision:
    def __init__(self, kinds):
        self.kinds, self.calls = kinds, 0

    async def describe(self, data, context, scope):
        from pyconsumers.core.rag_pipeline.schemas.visual import VisualDescription
        kind = self.kinds[self.calls % len(self.kinds)]
        self.calls += 1
        return VisualDescription(model='fixture-vision', prompt_version='visual-v2', image_sha256=sha256(data).hexdigest(),
                                 observation={'kind': kind, 'title': None, 'x_axis': None, 'y_axis': None, 'legend': [],
                                              'visible_text': [], 'numeric_values': [], 'observations': [f'A {kind}.'],
                                              'interpretation': f'Shows a {kind}.', 'limitations': [], 'status': 'interpreted'})


class FixtureLLM:
    """Metaclassifier model stand-in: cites whichever passage prints each fact, and invents one amount."""
    def __init__(self):
        self.calls = 0

    async def complete_json(self, model, system, prompt, schema, context_tokens, timeout):
        import re
        self.calls += 1
        passages = dict(re.findall(r'^\[(\d+)\] \(.*?\)\n((?:(?!^\[\d+\] ).*\n?)*)', prompt, re.M))
        cite = lambda needle: [int(n) for n, body in passages.items() if needle in body][:1]
        empty = {'value': '', 'evidence': []}
        return json.dumps({
            'summary': 'A lease.', 'document_type': 'lease_deed', 'title': {'value': 'Lease agreement', 'evidence': cite('Lease agreement')},
            'parties': [], 'dates': [],
            'amounts': [{'label': 'rent', 'as_printed': '25000 rupees', 'evidence': cite('25000')},
                        {'label': 'deposit', 'as_printed': 'Rs. 99,999', 'evidence': cite('25000')}],
            'jurisdiction': empty, 'governing_law': empty, 'property': empty, 'court': empty, 'case_number': empty,
            'judges': [], 'citation': empty, 'statutes_cited': []})


@pytest.fixture
async def handlers(env):
    from conftest import FakeModel
    from pyconsumers.core.rag_pipeline.execute_indexing import MarkdownRag
    from pyconsumers.core.rag_pipeline.execute_metaclassifier import MetaClassifier
    from pyconsumers.core.rag_pipeline.execute_standardization import FolderStandardization
    from pyconsumers.core.rag_pipeline.multimodal_processors.image_rag import ImageRag
    from pyconsumers.helpers.storage_helper import Storage
    storage, model, vision = Storage(env.settings), FakeModel(), FixtureVision(['chart', 'logo', 'table'])
    written = []

    async def upload(doc, data):
        async with env.db.transaction() as session:
            path = await session.scalar(text('UPDATE unstructured_data SET sha256=:sha WHERE id=:d RETURNING s3_path'),
                                        {'d': doc, 'sha': sha256(data).hexdigest()})
        await storage.put_bytes(path, data)
        written.append(path.rsplit('/', 1)[0])

    built = {'standardization': FolderStandardization(storage, env.settings, parser=FixtureParser()),
             'markdown': MarkdownRag(env.db, storage, env.settings, model),
             'image': ImageRag(env.db, storage, env.cache, env.settings, model, vision=vision),
             'metaclassifier': MetaClassifier(env.db, env.settings, model, client=FixtureLLM())}
    workers = {stage: StageWorker(env.db, env.cache, env.settings, stage, handler, producer=SimpleNamespace(client=None))
               for stage, handler in built.items()}
    try:
        yield SimpleNamespace(workers=workers, built=built, upload=upload, vision=vision, model=model)
    finally:
        for prefix in written:  # the raw file plus everything standardization wrote under parsed/
            for name in ['f0', 'parsed/document.md', 'parsed/document.json'] + [f'parsed/assets/picture-{n}.png' for n in range(3)]:
                with suppress(Exception):
                    await storage.delete(f'{prefix}/{name}')
        await storage.close()


async def rows(env, sql, doc):
    async with env.db.transaction() as session:
        return (await session.execute(text(sql), {'d': doc})).mappings().all()


async def drain(workers):
    while any([await workers[stage].process_next() for stage in ('standardization', 'markdown', 'image', 'metaclassifier')]):
        pass


async def test_real_handlers_index_text_and_images_and_rerun_without_duplicates(env, handlers):
    thread, [doc] = await env.folder('user-a')
    await handlers.upload(doc, b'%PDF-1.7 fictional lease')
    await drain(handlers.workers)
    assert await stage_of(env, doc, 'standardization', 'markdown', 'image', 'metaclassifier') == (
        'COMPLETED', 'COMPLETED', 'COMPLETED', 'COMPLETED')
    [file] = await rows(env, 'SELECT * FROM unstructured_data WHERE id=:d', doc)
    assert (file['page_count'], file['language']) == (3, 'en')
    assert file['processing_summary']['image'].items() >= {'images': 3, 'logos_dropped': 1, 'stored': 2}.items()

    nodes = await rows(env, 'SELECT id, chunk_index, total_chunks, node_type, ts_config::text AS config, content FROM text_nodes '
                            'WHERE doc_id=:d ORDER BY chunk_index', doc)
    assert [n['chunk_index'] for n in nodes] == list(range(len(nodes))) and {n['total_chunks'] for n in nodes} == {len(nodes)}
    assert {n['node_type'] for n in nodes} == {'text', 'table'} and {n['config'] for n in nodes} == {'english'}
    links = await rows(env, 'SELECT source_node_id, target_node_id, relationship FROM node_relationships WHERE doc_id=:d', doc)
    assert len(links) == 2 * (len(nodes) - 1)
    assert {(l['source_node_id'], l['target_node_id']) for l in links if l['relationship'] == 'NEXT'} == {
        (a['id'], b['id']) for a, b in zip(nodes, nodes[1:])}
    async with env.db.transaction() as session:  # the generated tsv makes the text searchable with English stemming
        hits = await session.scalar(text("SELECT count(*) FROM text_nodes WHERE doc_id=:d AND tsv @@ to_tsquery('english','terminating')"),
                                    {'d': doc})
    assert hits >= 1  # "terminating" finds "terminate" and "Termination" (stemmed); a plain word match would not

    images = await rows(env, 'SELECT image_index, image_type, page_no, s3_path, metadata FROM image_rag_vector_table '
                             'WHERE doc_id=:d ORDER BY image_index', doc)
    assert [(i['image_index'], i['image_type'], i['page_no']) for i in images] == [(0, 'graph', 1), (2, 'table', 3)]  # logo dropped
    assert all(i['metadata']['model_derived'] and i['s3_path'].endswith(f'picture-{i["image_index"]}.png') for i in images)

    [meta] = await rows(env, 'SELECT * FROM unstructured_metadata WHERE doc_id=:d', doc)
    assert (meta['title'], meta['document_type'], meta['language']) == ('Lease agreement', 'lease_deed', 'en')
    assert meta['details']['amounts'] == [{'label': 'rent', 'as_printed': '25000 rupees'}]  # the invented deposit is dropped
    assert meta['details']['dropped_unsupported'] == ['amounts[1]'] and meta['title_embedding'] is not None
    assert meta['evidence']['title'][0]['kind'] == 'text'

    # Retrigger markdown and image: the same rows are replaced, never duplicated; metaclassifier is not re-queued.
    async with env.db.transaction() as session:
        await session.execute(text("""UPDATE unstructured_data SET markdown_status='WAITING', image_status='WAITING',
            markdown_attempts=0, image_attempts=0 WHERE id=:d"""), {'d': doc})
    await drain(handlers.workers)
    again = await rows(env, 'SELECT id FROM text_nodes WHERE doc_id=:d ORDER BY chunk_index', doc)
    assert [n['id'] for n in again] == [n['id'] for n in nodes]
    assert len(await rows(env, 'SELECT 1 FROM node_relationships WHERE doc_id=:d', doc)) == len(links)
    assert len(await rows(env, 'SELECT 1 FROM image_rag_vector_table WHERE doc_id=:d', doc)) == 2
    assert (await messages(env, thread))['thread.metaclassifier'] == 1
    async with env.db.transaction() as session:  # retrigger the metaclassifier itself: one row, replaced
        await session.execute(text("UPDATE unstructured_data SET metaclassifier_status='WAITING', metaclassifier_attempts=0 WHERE id=:d"),
                              {'d': doc})
    await drain(handlers.workers)
    assert len(await rows(env, 'SELECT 1 FROM unstructured_metadata WHERE doc_id=:d', doc)) == 1


async def test_image_upload_skips_markdown_and_a_stale_handler_writes_nothing(env, handlers):
    from io import BytesIO
    from PIL import Image
    thread, [photo] = await env.folder('user-a', file_type='png')
    buffer = BytesIO()
    Image.new('RGB', (16, 16), 'white').save(buffer, format='PNG')
    await handlers.upload(photo, buffer.getvalue())
    handlers.vision.kinds = ['photo']
    await drain(handlers.workers)
    assert await stage_of(env, photo, 'markdown', 'image', 'metaclassifier') == ('INAPPLICABLE', 'COMPLETED', 'COMPLETED')
    [stored] = await rows(env, 'SELECT image_type, s3_path FROM image_rag_vector_table WHERE doc_id=:d', photo)
    assert stored['image_type'] == 'image' and stored['s3_path'].endswith('/f0')  # the original upload, no copy

    # A markdown handler whose lease was taken over cannot write its nodes.
    _, [doc] = await env.folder('user-b')
    await handlers.upload(doc, b'%PDF-1.7 fictional lease')
    await handlers.workers['standardization'].process_next()
    claim = await queue(env, 'markdown').get_next_document('slow-worker')
    async with env.db.transaction() as session:
        await session.execute(text("UPDATE unstructured_data SET markdown_lease_expires_at=now()-interval '1 second' WHERE id=:d"),
                              {'d': doc})
    with pytest.raises(LeaseLost):
        await handlers.built['markdown'](claim)
    assert await rows(env, 'SELECT 1 FROM text_nodes WHERE doc_id=:d', doc) == []


async def test_metaclassifier_skips_empty_files_and_a_stale_one_writes_nothing(env, handlers):
    _, [empty] = await env.folder('user-a', first='metaclassifier')  # no text nodes, no images
    assert await handlers.workers['metaclassifier'].process_next()
    assert await stage_of(env, empty, 'metaclassifier') == 'INAPPLICABLE'

    _, [doc] = await env.folder('user-b')
    await handlers.upload(doc, b'%PDF-1.7 fictional lease')
    for stage in ('standardization', 'markdown', 'image'):
        await handlers.workers[stage].process_next()
    claim = await queue(env, 'metaclassifier').get_next_document('slow-worker')
    async with env.db.transaction() as session:
        await session.execute(text("UPDATE unstructured_data SET metaclassifier_lease_expires_at=now()-interval '1 second' WHERE id=:d"),
                              {'d': doc})
    with pytest.raises(LeaseLost):
        await handlers.built['metaclassifier'](claim)
    assert await rows(env, 'SELECT 1 FROM unstructured_metadata WHERE doc_id=:d', doc) == []


async def test_csv_worker_completes_folder_and_writes_parquet(env):
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from pyconsumers.core.rag_pipeline.execute_parquet import CsvParquet
    from pyconsumers.helpers.storage_helper import Storage

    thread, [doc] = await env.folder('csv-user', file_type='csv', first='parquet')
    storage = Storage(env.settings)
    raw = b'Party;Amount;Code\nAsha Verma;15000;001\nRavi Kumar;25000;002\n'
    async with env.db.transaction() as session:
        path = await session.scalar(text('UPDATE unstructured_data SET sha256=:sha, byte_count=:size WHERE id=:d RETURNING s3_path'),
                                    {'d': doc, 'sha': sha256(raw).hexdigest(), 'size': len(raw)})
    key = path.rsplit('/', 1)[0] + '/parquet/data.parquet'
    try:
        await storage.put_bytes(path, raw)
        worker = StageWorker(env.db, env.cache, env.settings, 'parquet', CsvParquet(storage, env.settings),
                             producer=SimpleNamespace(client=None))
        assert await worker.process_next()
        assert await stage_of(env, doc, 'parquet') == 'COMPLETED'
        [file] = await rows(env, 'SELECT processing_summary FROM unstructured_data WHERE id=:d', doc)
        assert file['processing_summary']['parquet']['parquet_key'] == key
        table = parquet.read_table(pa.BufferReader(await storage.get_bytes(key)))
        assert table.num_rows == 2 and table['Code'].to_pylist() == ['001', '002']
        async with env.db.transaction() as session:
            assert await session.scalar(text('SELECT status FROM unstructured_thread WHERE id=:t'), {'t': thread}) == 'COMPLETED'
    finally:
        try:
            await storage.delete(key)
            await storage.delete(path)
        finally:
            await storage.close()
