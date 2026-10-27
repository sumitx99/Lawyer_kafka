import os

import pytest
from sqlalchemy import text

from pyconsumers.core.rag_pipeline.execute_indexing import Indexing
from conftest import FakeModel, counts, parsed_event

pytestmark = [pytest.mark.integration, pytest.mark.skipif(os.environ.get('RAG_RUN_INTEGRATION') != '1',
                                                        reason='Requires local RAG services.')]


async def test_index_persists_cited_chunks_and_is_idempotent(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    service = Indexing(db, storage, cache, settings, model=FakeModel())
    event = parsed_event(project, doc)
    assert await service.execute(event) == 'indexed'
    state = await counts(db, doc)
    assert (state['status'], state['sections'], state['chunks'], state['ready'], state['index_version']) == ('INDEXED', 2, 3, 3, 1)
    summary = state['index_summary']
    assert (summary['table_count'], summary['table_row_count'], summary['table_chunk_count']) == (1, 2, 1)
    async with db.transaction() as session:
        table = (await session.execute(text('''SELECT ordinal, chunk_type, table_id, heading_path, page_start, content,
            source_metadata FROM chunks WHERE doc_id=:doc AND search_vector @@ plainto_tsquery('english','security deposit')'''),
            {'doc': doc})).mappings().one()
    assert (table['chunk_type'], table['table_id'], table['heading_path'], table['page_start']) == (
        'table', '#/tables/0', 'Lease agreement > Termination', 2)
    assert table['ordinal'] == 2 and '| Item | Amount |' in table['content']
    assert table['source_metadata']['row_pages'] == {'1': 2, '2': 2}
    async with db.transaction() as session:
        rent = (await session.execute(text('''SELECT heading_path, page_start, page_end, source_metadata FROM chunks
            WHERE doc_id=:doc AND search_vector @@ plainto_tsquery('english','rent month')'''), {'doc': doc})).mappings().one()
    assert rent['heading_path'] == 'Lease agreement > Rent' and (rent['page_start'], rent['page_end']) == (1, 1)
    assert rent['source_metadata']['block_refs'] == ['#/texts/2']

    # The same event, and a later duplicate event, never add rows.
    assert await service.execute(event) == 'duplicate'
    assert await service.execute(parsed_event(project, doc)) == 'duplicate'
    assert (await counts(db, doc))['chunks'] == 3


async def test_crashed_worker_lease_is_recovered_without_duplicates(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    service = Indexing(db, storage, cache, settings, model=FakeModel())
    assert await service.execute(parsed_event(project, doc)) == 'indexed'
    async with db.transaction() as session:
        await session.execute(text('''UPDATE documents SET status='INDEXING', lease_owner='crashed-worker',
            lease_expires_at=now()-interval '1 minute', index_attempt_count=0 WHERE id=:doc'''), {'doc': doc})
    assert await service.execute(parsed_event(project, doc)) == 'indexed'
    state = await counts(db, doc)
    assert (state['status'], state['chunks'], state['index_version']) == ('INDEXED', 3, 2)


async def test_embedding_failure_retries_then_fails_terminally(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    service = Indexing(db, storage, cache, settings, model=FakeModel(fail=True))
    assert await service.execute(parsed_event(project, doc)) == 'retry'
    assert (await counts(db, doc))['status'] == 'PARSED'
    assert await service.execute(parsed_event(project, doc)) == 'failed'
    state = await counts(db, doc)
    assert (state['status'], state['chunks'], state['error_message']) == ('FAILED', 0, 'EmbeddingFailure')


async def test_mismatched_manifest_is_terminal(parsed_doc):
    settings, db, storage, cache, project, doc = parsed_doc
    async with db.transaction() as session:
        await session.execute(text("UPDATE documents SET sha256=:sha WHERE id=:doc"), {'doc': doc, 'sha': '0' * 64})
    assert await Indexing(db, storage, cache, settings, model=FakeModel()).execute(parsed_event(project, doc)) == 'failed'
    state = await counts(db, doc)
    assert (state['status'], state['chunks'], state['error_message']) == ('FAILED', 0, 'ValueError')
