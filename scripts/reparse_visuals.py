"""Requeue selected PARSED documents for visual-v1 enrichment.

Run from the backend root: python -m scripts.reparse_visuals --project-id UUID --doc-id UUID
Indexed documents are not changed.
"""
import argparse
import asyncio
from uuid import UUID, uuid5

from sqlalchemy import text

from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.postgres_helper import Database


async def requeue(project_id, doc_ids):
    db = Database()
    try:
        async with db.transaction() as session:
            for doc_id in doc_ids:
                event_id = uuid5(doc_id, 'visual-v1-reparse')
                existing = await session.scalar(text('SELECT 1 FROM outbox_events WHERE id=:id'), {'id': event_id})
                if existing:
                    print(f'{doc_id}: visual reparse already queued or completed.')
                    continue
                row = (await session.execute(text('SELECT status FROM documents WHERE id=:doc AND project_id=:project FOR UPDATE'),
                                             {'doc': doc_id, 'project': project_id})).first()
                if row is None or row.status != 'PARSED':
                    raise ValueError('Every selected document must exist in the project and have status PARSED.')
                event = RagEvent(event_id=event_id, event_type='doc.uploaded', project_id=project_id, doc_id=doc_id)
                await session.execute(text("""UPDATE documents SET status='PENDING',attempt_count=0,
                    lease_owner=NULL,lease_expires_at=NULL,error_message=NULL,
                    parsed_object_key=NULL,parse_summary='{}'::jsonb WHERE id=:doc AND project_id=:project"""),
                    {'doc': doc_id, 'project': project_id})
                await session.execute(text("""INSERT INTO outbox_events(id,project_id,event_type,aggregate_id,payload)
                    VALUES(:id,:project,'doc.uploaded',:doc,CAST(:payload AS jsonb))"""),
                    {'id': event.event_id, 'project': project_id, 'doc': doc_id, 'payload': event.model_dump_json()})
        print('Selected documents queued through the durable outbox. The API dispatcher and parsing worker must be running.')
    finally:
        await db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-id', type=UUID, required=True)
    parser.add_argument('--doc-id', type=UUID, action='append', required=True)
    args = parser.parse_args()
    asyncio.run(requeue(args.project_id, args.doc_id))
