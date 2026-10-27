"""Shared integration fixtures: a PARSED fictional lease with its manifest and raw file in storage."""
import asyncio
from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import text

from pyconsumers.core.rag_pipeline.schemas.parsed import ParsedArtifact, ParsedBlock
from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.postgres_helper import Database
from pyconsumers.helpers.secrets_control import get_settings
from pyconsumers.helpers.storage_helper import Storage
from pyconsumers.helpers.valkey_helper import Cache


class FakeModel:
    """Deterministic stand-in for BGE-M3: whitespace token counts and unit vectors."""
    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def count_tokens(self, text):
        return len(text.split())

    async def embed_async(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError('simulated embedding failure')
        return [[1.0] + [0.0] * 1023 for _ in texts]


BLOCKS = [ParsedBlock(ref='#/texts/0', label='title', text='Lease agreement'),
          ParsedBlock(ref='#/texts/1', label='section_header', level=1, text='Rent', provenance=[{'page_no': 1}]),
          ParsedBlock(ref='#/texts/2', label='text', text='The tenant pays rent of 25000 rupees on the fifth day of each month.',
                      provenance=[{'page_no': 1, 'bbox': {'l': 1, 't': 2, 'r': 3, 'b': 0}}]),
          ParsedBlock(ref='#/texts/3', label='section_header', level=1, text='Termination'),
          ParsedBlock(ref='#/texts/4', label='text', text='Either party may terminate with two months written notice.',
                      provenance=[{'page_no': 2}]),
          ParsedBlock(ref='#/tables/0', label='table', text='| Item | Amount |', provenance=[{'page_no': 2}],
                      data={'grid': [[{'text': 'Item', 'column_header': True}, {'text': 'Amount', 'column_header': True}],
                                     [{'text': 'Security deposit'}, {'text': '75000'}],
                                     [{'text': 'Maintenance'}, {'text': '2000'}]]})]


@pytest.fixture
async def parsed_doc():
    settings = get_settings().model_copy(update={'index_max_attempts': 2, 'embedding_max_attempts': 1})
    db, storage, cache = Database(settings), Storage(settings), Cache(settings)
    project, doc = uuid4(), uuid4()
    raw = b'fictional lease bytes'
    digest = sha256(raw).hexdigest()
    manifest_key = f'projects/{project}/documents/{doc}/parsed/fixture/document.json'
    artifact = ParsedArtifact(parser_version='fixture', blocks=BLOCKS, pages=[{'page_no': 1}, {'page_no': 2}],
                              docling_document={}, doc_id=doc, project_id=project, source_sha256=digest,
                              source_filename='lease.pdf', markdown_key=manifest_key[:-4] + 'md', manifest_key=manifest_key)
    await storage.put_bytes(manifest_key, artifact.model_dump_json().encode(), 'application/json')
    await storage.put_bytes(f'projects/{project}/documents/{doc}/raw.pdf', raw, 'application/pdf')
    async with db.transaction() as session:
        await session.execute(text('''INSERT INTO documents (id,project_id,filename,media_type,byte_count,sha256,object_key,
            parsed_object_key,status) VALUES (:doc,:project,'lease.pdf','application/pdf',:size,:sha,:key,:manifest,'PARSED')'''),
            {'doc': doc, 'project': project, 'size': len(raw), 'sha': digest,
             'key': f'projects/{project}/documents/{doc}/raw.pdf', 'manifest': manifest_key})
    try:
        yield settings, db, storage, cache, project, doc
    finally:
        async with db.transaction() as session:
            for table in ['processed_events', 'outbox_events', 'documents']:
                await session.execute(text(f'DELETE FROM {table} WHERE project_id=:project'), {'project': project})
        objects = await asyncio.to_thread(storage.client.list_objects_v2, Bucket=storage.bucket, Prefix=f'projects/{project}/')
        for item in objects.get('Contents', []):
            await storage.delete(item['Key'])
        await cache.close()
        await storage.close()
        await db.close()


def parsed_event(project, doc):
    return RagEvent(event_type='doc.parsed', project_id=project, doc_id=doc)


async def counts(db, doc):
    async with db.transaction() as session:
        row = (await session.execute(text('''SELECT (SELECT count(*) FROM sections WHERE doc_id=:doc) AS sections,
            (SELECT count(*) FROM chunks WHERE doc_id=:doc) AS chunks,
            (SELECT count(*) FROM chunks WHERE doc_id=:doc AND embedding_status='READY' AND embedding IS NOT NULL) AS ready,
            d.status, d.index_version, d.index_attempt_count, d.index_summary, d.error_message
            FROM documents d WHERE d.id=:doc'''), {'doc': doc})).mappings().one()
    return dict(row)
