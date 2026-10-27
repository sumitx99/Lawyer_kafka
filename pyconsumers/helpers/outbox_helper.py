"""Dispatch document-pipeline outbox rows (doc.uploaded, doc.parsed) without depending on the API process.

The API dispatches the same table; per-row leases with SKIP LOCKED keep the two from double-sending.
"""
import asyncio
from contextlib import suppress
from datetime import datetime, timedelta, timezone
import json
import logging
from uuid import uuid4

from sqlalchemy import text

from .consumer_helper import EventProducer
from .events import THREAD_EVENTS, RagEvent, ThreadMessage

logger = logging.getLogger(__name__)


EVENT_TYPES = ('doc.uploaded', 'doc.parsed', *THREAD_EVENTS)


class DocumentOutboxPublisher:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings
        self.owner = str(uuid4())
        self.producer = None

    async def close(self):
        if self.producer:
            producer, self.producer = self.producer, None
            with suppress(Exception):
                await asyncio.wait_for(producer.close(), 5)

    async def dispatch_one(self, project_filter=None):
        async with self.db.transaction() as session:
            project_clause = ' AND project_id=:project' if project_filter is not None else ''
            row = (await session.execute(text("""SELECT * FROM outbox_events WHERE event_type = ANY(:types)
                AND status='PENDING' AND available_at<=now()
                AND (lease_expires_at IS NULL OR lease_expires_at<=now())""" + project_clause +
                ' ORDER BY created_at,id LIMIT 1 FOR UPDATE SKIP LOCKED'),
                {'types': list(EVENT_TYPES), **({'project': project_filter} if project_filter is not None else {})})).mappings().first()
            if row is None:
                return False
            row = dict(row)
            await session.execute(text('''UPDATE outbox_events SET lease_owner=:owner,
                lease_expires_at=now()+interval '60 seconds', attempt_count=attempt_count+1 WHERE id=:id'''),
                {'owner': self.owner, 'id': row['id']})
        status, error = 'PUBLISHED', None
        try:
            if row['event_type'] in THREAD_EVENTS:
                if ThreadMessage.model_validate(row['payload']).unstructured_thread_id != row['aggregate_id']:
                    raise ValueError('Outbox folder identity mismatch.')
                event = None
            else:
                event = RagEvent.model_validate(row['payload'])
                if (event.event_id, event.event_type, event.project_id, event.doc_id) != (
                        row['id'], row['event_type'], row['project_id'], row['aggregate_id']) or event.event_type not in EVENT_TYPES:
                    raise ValueError('Outbox identity mismatch.')
        except ValueError:
            status, error = 'FAILED', 'Invalid event envelope.'
        else:
            try:
                async with asyncio.timeout(20):
                    if self.producer is None:
                        self.producer = EventProducer(self.settings)
                        await self.producer.start()
                    if event is None:
                        await self.producer.send_thread(row['event_type'], row['aggregate_id'])
                    else:
                        await self.producer.send(event)
            except Exception as failure:
                status, error = 'PENDING', type(failure).__name__
                await self.close()
        now = datetime.now(timezone.utc)
        async with self.db.transaction() as session:
            await session.execute(text('''UPDATE outbox_events SET status=:status,error_message=:error,
                published_at=:published, available_at=:available, lease_owner=NULL, lease_expires_at=NULL
                WHERE id=:id AND status='PENDING' AND lease_owner=:owner'''),
                {'status': status, 'error': error, 'published': now if status == 'PUBLISHED' else None,
                 'available': now + timedelta(seconds=min(300, 2 ** min(row['attempt_count'] + 1, 8))),
                 'id': row['id'], 'owner': self.owner})
        return True

    async def run(self):
        try:
            while True:
                try:
                    if await self.dispatch_one():
                        continue
                except Exception as error:
                    logger.warning('Document outbox will retry (%s)', type(error).__name__)
                await asyncio.sleep(1)
        finally:
            await self.close()
