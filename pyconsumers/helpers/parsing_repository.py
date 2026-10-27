"""SQL persistence for parsing; uses the API-owned schema without importing its repo."""
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid5

from sqlalchemy import text

from .events import RagEvent

CONSUMER_NAME = 'standardization-v1'


class LeaseLost(Exception):
    pass


class ParsingRepository:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings

    async def claim(self, event, owner):
        params = {'doc': event.doc_id, 'project': event.project_id, 'event': event.event_id,
                  'consumer': CONSUMER_NAME}
        async with self.db.transaction() as session:
            row = (await session.execute(text('SELECT * FROM documents WHERE id=:doc AND project_id=:project FOR UPDATE'), params)).mappings().first()
            if row is None:
                return 'missing', None
            done = await session.scalar(text('SELECT 1 FROM processed_events WHERE event_id=:event AND consumer_name=:consumer'), params)
            if done:
                return 'duplicate', dict(row)
            if row['status'] in {'PARSED', 'INDEXING', 'INDEXED'}:
                await self._processed(session, event)
                return 'duplicate', dict(row)
            if row['status'] == 'FAILED':
                return 'failed', dict(row)
            now = datetime.now(timezone.utc)
            if row['lease_expires_at'] and row['lease_expires_at'] > now:
                return 'busy', dict(row)
            if row['attempt_count'] >= self.settings.parse_max_attempts:
                await session.execute(text("""UPDATE documents SET status='FAILED', lease_owner=NULL,
                    lease_expires_at=NULL,error_message='ParseAttemptLimitExceeded'
                    WHERE id=:doc AND project_id=:project"""), params)
                return 'failed', dict(row)
            params |= {'owner': owner, 'expires': now + timedelta(seconds=self.settings.parse_lease_seconds)}
            row = (await session.execute(text("""UPDATE documents SET status='PARSING', attempt_count=attempt_count+1,
                lease_owner=:owner, lease_expires_at=:expires, error_message=NULL
                WHERE id=:doc AND project_id=:project RETURNING *"""), params)).mappings().one()
            return 'claimed', dict(row)

    async def heartbeat(self, event, owner):
        now = datetime.now(timezone.utc)
        async with self.db.transaction() as session:
            value = await session.scalar(text("""UPDATE documents SET lease_expires_at=:expires
                WHERE id=:doc AND project_id=:project AND lease_owner=:owner AND status='PARSING'
                AND lease_expires_at>:now RETURNING id"""),
                {'doc': event.doc_id, 'project': event.project_id, 'owner': owner, 'now': now,
                 'expires': now + timedelta(seconds=self.settings.parse_lease_seconds)})
            if value is None:
                raise LeaseLost()

    async def complete(self, event, owner, manifest_key, summary):
        parsed = RagEvent(event_id=uuid5(event.event_id, 'doc.parsed:v1'),
                          correlation_id=event.correlation_id, event_type='doc.parsed',
                          project_id=event.project_id, doc_id=event.doc_id)
        async with self.db.transaction() as session:
            value = await session.scalar(text("""UPDATE documents SET status='PARSED', parsed_object_key=:manifest,
                parse_summary=CAST(:summary AS jsonb), lease_owner=NULL, lease_expires_at=NULL, error_message=NULL
                WHERE id=:doc AND project_id=:project AND lease_owner=:owner AND status='PARSING'
                AND lease_expires_at>now() RETURNING id"""),
                {'doc': event.doc_id, 'project': event.project_id, 'owner': owner,
                 'manifest': manifest_key, 'summary': json.dumps(summary)})
            if value is None:
                raise LeaseLost()
            await session.execute(text("""INSERT INTO outbox_events
                (id,project_id,event_type,aggregate_id,payload) VALUES (:id,:project,'doc.parsed',:doc,CAST(:payload AS jsonb))
                ON CONFLICT (id) DO NOTHING"""),
                {'id': parsed.event_id, 'project': parsed.project_id, 'doc': parsed.doc_id,
                 'payload': parsed.model_dump_json()})
            await self._processed(session, event)

    async def failure(self, event, owner, code, terminal=False):
        async with self.db.transaction() as session:
            row = (await session.execute(text("""UPDATE documents SET status=CASE
                WHEN attempt_count>=:maximum OR :terminal THEN 'FAILED' ELSE 'PENDING' END,
                lease_owner=NULL, lease_expires_at=NULL, error_message=:error
                WHERE id=:doc AND project_id=:project AND lease_owner=:owner AND status='PARSING'
                AND lease_expires_at>now() RETURNING status"""),
                {'doc': event.doc_id, 'project': event.project_id, 'owner': owner,
                 'maximum': self.settings.parse_max_attempts, 'terminal': terminal, 'error': code})).first()
            if row is None:
                raise LeaseLost()
            return row.status.lower()

    async def acknowledge_failed(self, event):
        async with self.db.transaction() as session:
            state = await session.scalar(text("SELECT status FROM documents WHERE id=:doc AND project_id=:project FOR UPDATE"),
                                         {'doc': event.doc_id, 'project': event.project_id})
            if state == 'FAILED':
                await self._processed(session, event)

    @staticmethod
    async def _processed(session, event):
        await session.execute(text('''INSERT INTO processed_events(event_id,consumer_name,project_id)
            VALUES(:event,:consumer,:project) ON CONFLICT DO NOTHING'''),
            {'event': event.event_id, 'consumer': CONSUMER_NAME, 'project': event.project_id})
