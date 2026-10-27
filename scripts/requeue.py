"""Re-queue documents through the durable outbox; the worker (make worker) processes them.

Run from the backend root:
  python -m scripts.requeue --reindex --project-id UUID (--all | --doc-id UUID ...)
      INDEXED -> PARSED: rebuild sections, chunks and vectors with the current chunkers.
      Queued at most once per chunker version.
  python -m scripts.requeue --failed --project-id UUID (--all | --doc-id UUID ...)
      FAILED -> restart from the stage that failed: re-index when a parsed artifact exists,
      otherwise re-parse. Attempt counters are reset. Fix the cause first.
Add --dry-run to list what would be queued without changing anything.
"""
import argparse
import asyncio
from uuid import UUID, uuid5

from sqlalchemy import text

from pyconsumers.core.rag_pipeline.chunking.table_chunker import TABLE_CHUNKER_VERSION
from pyconsumers.core.rag_pipeline.chunking.text_chunker import CHUNKER_VERSION
from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.postgres_helper import Database

VERSION = f'{CHUNKER_VERSION}+{TABLE_CHUNKER_VERSION}'


def plan(row, mode):
    """Return (new status, event type, counter to reset, event key) for one document."""
    if mode == 'reindex':
        return 'PARSED', 'doc.parsed', 'index_attempt_count', f'reindex:{VERSION}'
    key = f'requeue-failed:{row["error_message"]}:{row["updated_at"].isoformat()}'
    if row['parsed_object_key']:
        return 'PARSED', 'doc.parsed', 'index_attempt_count', key
    return 'PENDING', 'doc.uploaded', 'attempt_count', key


async def requeue(project_id, doc_ids, mode, every=False, dry_run=False, db=None):
    required = 'INDEXED' if mode == 'reindex' else 'FAILED'
    owned, db = db is None, db or Database()
    queued = []
    try:
        async with db.transaction() as session:
            if every:
                doc_ids = list(await session.scalars(text(
                    'SELECT id FROM documents WHERE project_id=:project AND status=:status ORDER BY created_at'),
                    {'project': project_id, 'status': required}))
            for doc_id in doc_ids:
                row = (await session.execute(text('SELECT * FROM documents WHERE id=:doc AND project_id=:project FOR UPDATE'),
                                             {'doc': doc_id, 'project': project_id})).mappings().first()
                if row is None or row['status'] != required:
                    raise ValueError(f'{doc_id}: must exist in the project and have status {required}.')
                status, event_type, counter, key = plan(row, mode)
                event = RagEvent(event_id=uuid5(doc_id, key), event_type=event_type, project_id=project_id, doc_id=doc_id)
                if await session.scalar(text('SELECT 1 FROM outbox_events WHERE id=:id'), {'id': event.event_id}):
                    print(f'{doc_id}: already queued ({key}).')
                    continue
                queued.append((doc_id, row['filename'], status, event_type))
                if dry_run:
                    continue
                await session.execute(text(f"""UPDATE documents SET status=:status, {counter}=0, lease_owner=NULL,
                    lease_expires_at=NULL, error_message=NULL WHERE id=:doc AND project_id=:project"""),
                    {'status': status, 'doc': doc_id, 'project': project_id})
                await session.execute(text("""INSERT INTO outbox_events(id,project_id,event_type,aggregate_id,payload)
                    VALUES(:id,:project,:type,:doc,CAST(:payload AS jsonb))"""),
                    {'id': event.event_id, 'project': project_id, 'type': event_type, 'doc': doc_id,
                     'payload': event.model_dump_json()})
        for doc_id, filename, status, event_type in queued:
            print(f'{"would queue" if dry_run else "queued"}: {filename} ({doc_id}) -> {status} via {event_type}')
        print(f'{len(queued)} document(s) {"would be " if dry_run else ""}queued. Start the worker (make worker) to process them.')
        return queued
    finally:
        if owned:
            await db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--reindex', action='store_const', const='reindex', dest='mode')
    mode.add_argument('--failed', action='store_const', const='failed', dest='mode')
    parser.add_argument('--project-id', type=UUID, required=True)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument('--doc-id', type=UUID, action='append')
    which.add_argument('--all', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    asyncio.run(requeue(args.project_id, args.doc_id or [], args.mode, every=args.all, dry_run=args.dry_run))
