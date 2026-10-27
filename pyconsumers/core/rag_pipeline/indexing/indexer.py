"""Idempotently persist sections, chunks, vectors and search text.

The document row is the lock: a worker claims it (PARSED -> INDEXING) with a lease, and the
final write replaces the document's sections and chunks in one transaction that only commits
while that lease is still held. Re-running after a crash or a duplicate event therefore
replaces rows instead of duplicating them. Full-text search_vector is a generated column.
"""
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid5

from sqlalchemy import text

from ....helpers.parsing_repository import LeaseLost
from ..userallocation.UserAllocation import lock_claim

CONSUMER_NAME = 'indexing-v1'


class IndexingRepository:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings

    async def claim(self, event, owner):
        params = {'doc': event.doc_id, 'project': event.project_id, 'event': event.event_id, 'consumer': CONSUMER_NAME}
        async with self.db.transaction() as session:
            row = (await session.execute(text('SELECT * FROM documents WHERE id=:doc AND project_id=:project FOR UPDATE'),
                                         params)).mappings().first()
            if row is None:
                return 'missing', None
            if await session.scalar(text('SELECT 1 FROM processed_events WHERE event_id=:event AND consumer_name=:consumer'), params):
                return 'duplicate', dict(row)
            if row['status'] == 'INDEXED':
                await self._processed(session, event)
                return 'duplicate', dict(row)
            if row['status'] == 'FAILED':
                return 'failed', dict(row)
            now = datetime.now(timezone.utc)
            if row['status'] not in {'PARSED', 'INDEXING'} or (row['lease_expires_at'] and row['lease_expires_at'] > now):
                return 'busy', dict(row)
            if row['index_attempt_count'] >= self.settings.index_max_attempts:
                await session.execute(text("""UPDATE documents SET status='FAILED', lease_owner=NULL, lease_expires_at=NULL,
                    error_message='IndexAttemptLimitExceeded' WHERE id=:doc AND project_id=:project"""), params)
                return 'failed', dict(row)
            params |= {'owner': owner, 'expires': now + timedelta(seconds=self.settings.index_lease_seconds)}
            row = (await session.execute(text("""UPDATE documents SET status='INDEXING',
                index_attempt_count=index_attempt_count+1, lease_owner=:owner, lease_expires_at=:expires,
                error_message=NULL WHERE id=:doc AND project_id=:project RETURNING *"""), params)).mappings().one()
            return 'claimed', dict(row)

    async def heartbeat(self, event, owner):
        now = datetime.now(timezone.utc)
        async with self.db.transaction() as session:
            value = await session.scalar(text("""UPDATE documents SET lease_expires_at=:expires
                WHERE id=:doc AND project_id=:project AND lease_owner=:owner AND status='INDEXING'
                AND lease_expires_at>:now RETURNING id"""),
                {'doc': event.doc_id, 'project': event.project_id, 'owner': owner, 'now': now,
                 'expires': now + timedelta(seconds=self.settings.index_lease_seconds)})
            if value is None:
                raise LeaseLost()

    async def complete(self, event, owner, sections, chunks, vectors, summary):
        doc, project = event.doc_id, event.project_id
        base = {'doc': doc, 'project': project}
        async with self.db.transaction() as session:
            held = await session.scalar(text("""SELECT id FROM documents WHERE id=:doc AND project_id=:project
                AND lease_owner=:owner AND status='INDEXING' AND lease_expires_at>now() FOR UPDATE"""), base | {'owner': owner})
            if held is None:
                raise LeaseLost()
            await session.execute(text('DELETE FROM sections WHERE doc_id=:doc AND project_id=:project'), base)
            section_ids = {}
            if sections:
                rows = []
                for section in sections:
                    section_ids[section.ordinal] = uuid5(doc, f'{summary["chunker_version"]}:section:{section.ordinal}')
                    pages = section.pages
                    rows.append(base | {'id': section_ids[section.ordinal], 'ordinal': section.ordinal,
                                        'heading': json.dumps(section.heading_path), 'content': section.content,
                                        'start': pages[0] if pages else None, 'end': pages[-1] if pages else None})
                await session.execute(text("""INSERT INTO sections (id,project_id,doc_id,ordinal,heading_path,content,page_start,page_end)
                    VALUES (:id,:project,:doc,:ordinal,CAST(:heading AS jsonb),:content,:start,:end)"""), rows)
            if chunks:
                rows = [base | {'id': uuid5(doc, f'{summary["chunker_version"]}:chunk:{chunk.ordinal}'),
                                'section': section_ids[chunk.section_ordinal], 'ordinal': chunk.ordinal,
                                'heading': ' > '.join(chunk.heading_path), 'content': chunk.content,
                                'tokens': chunk.token_count, 'start': chunk.page_start, 'end': chunk.page_end,
                                'type': chunk.chunk_type, 'table': chunk.table_id,
                                'meta': json.dumps(chunk.source_metadata), 'model': self.settings.embedding_model,
                                'embedding': '[' + ','.join(repr(float(v)) for v in vector) + ']'}
                        for chunk, vector in zip(chunks, vectors, strict=True)]
                await session.execute(text("""INSERT INTO chunks (id,project_id,doc_id,section_id,ordinal,chunk_type,heading_path,
                    content,token_count,page_start,page_end,table_id,source_metadata,embedding,embedding_status,embedding_model)
                    VALUES (:id,:project,:doc,:section,:ordinal,:type,:heading,:content,:tokens,:start,:end,:table,
                    CAST(:meta AS jsonb),CAST(:embedding AS vector),'READY',:model)"""), rows)
            await session.execute(text("""UPDATE documents SET status='INDEXED', index_version=index_version+1,
                index_summary=CAST(:summary AS jsonb), lease_owner=NULL, lease_expires_at=NULL, error_message=NULL
                WHERE id=:doc AND project_id=:project"""), base | {'summary': json.dumps(summary)})
            await self._processed(session, event)

    async def failure(self, event, owner, code, terminal=False):
        async with self.db.transaction() as session:
            row = (await session.execute(text("""UPDATE documents SET status=CASE
                WHEN index_attempt_count>=:maximum OR :terminal THEN 'FAILED' ELSE 'PARSED' END,
                lease_owner=NULL, lease_expires_at=NULL, error_message=:error
                WHERE id=:doc AND project_id=:project AND lease_owner=:owner AND status='INDEXING'
                AND lease_expires_at>now() RETURNING status"""),
                {'doc': event.doc_id, 'project': event.project_id, 'owner': owner,
                 'maximum': self.settings.index_max_attempts, 'terminal': terminal, 'error': code})).first()
            if row is None:
                raise LeaseLost()
            return 'retry' if row.status == 'PARSED' else 'failed'

    async def acknowledge_failed(self, event):
        async with self.db.transaction() as session:
            state = await session.scalar(text('SELECT status FROM documents WHERE id=:doc AND project_id=:project FOR UPDATE'),
                                         {'doc': event.doc_id, 'project': event.project_id})
            if state == 'FAILED':
                await self._processed(session, event)

    @staticmethod
    async def _processed(session, event):
        await session.execute(text('''INSERT INTO processed_events(event_id,consumer_name,project_id)
            VALUES(:event,:consumer,:project) ON CONFLICT DO NOTHING'''),
            {'event': event.event_id, 'consumer': CONSUMER_NAME, 'project': event.project_id})


# ---- Folder pipeline (design doc): text_nodes + node_relationships --------------------------

async def write_text_nodes(db, claim, chunks, vectors, ts_config, embedding_model, version):
    """Delete-then-insert the file's chunks and PREVIOUS/NEXT links, fenced by the stage lease."""
    doc, thread = claim.doc_id, claim.thread_id
    ids = [uuid5(doc, f'{version}:node:{chunk.ordinal}') for chunk in chunks]
    async with db.transaction() as session:
        await lock_claim(session, claim)
        await session.execute(text('DELETE FROM text_nodes WHERE doc_id=:doc'), {'doc': doc})  # links cascade
        if not chunks:
            return 0
        await session.execute(text("""INSERT INTO text_nodes (id,doc_id,thread_id,chunk_index,total_chunks,node_type,
            heading_path,content,token_count,page_start,page_end,table_id,metadata,ts_config,embedding,embedding_model)
            VALUES (:id,:doc,:thread,:index,:total,:type,:heading,:content,:tokens,:start,:end,:table,
            CAST(:meta AS jsonb),CAST(:config AS regconfig),CAST(:embedding AS vector),:model)"""),
            [{'id': node, 'doc': doc, 'thread': thread, 'index': chunk.ordinal, 'total': len(chunks),
              'type': chunk.chunk_type, 'heading': ' > '.join(chunk.heading_path), 'content': chunk.content,
              'tokens': chunk.token_count, 'start': chunk.page_start, 'end': chunk.page_end, 'table': chunk.table_id,
              'meta': json.dumps(chunk.source_metadata), 'config': ts_config, 'model': embedding_model,
              'embedding': '[' + ','.join(repr(float(v)) for v in vector) + ']'}
             for node, chunk, vector in zip(ids, chunks, vectors, strict=True)])
        links = [{'id': uuid5(source, kind), 'doc': doc, 'source': source, 'target': target, 'kind': kind}
                 for before, after in zip(ids, ids[1:])
                 for source, target, kind in ((before, after, 'NEXT'), (after, before, 'PREVIOUS'))]
        if links:
            await session.execute(text("""INSERT INTO node_relationships (id,doc_id,source_node_id,target_node_id,relationship)
                VALUES (:id,:doc,:source,:target,:kind)"""), links)
    return len(chunks)
