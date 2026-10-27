"""Recover documents left behind by crashed workers or lost events.

Every sweep, in one transaction per batch with SKIP LOCKED (safe with several workers):
  1. PARSING/INDEXING rows whose lease expired (the worker died) go back to PENDING/PARSED with
     a recovery event in the outbox, or to FAILED once their attempt limit is used up.
  2. PENDING/PARSED rows idle longer than SWEEP_STUCK_SECONDS with no pending outbox event get
     one recovery event (the original was dead-lettered, expired from Kafka, or never sent).
Recovery event ids are derived from the document and its attempt counters, so repeated sweeps
queue each recovery once, and duplicates that do reach a consumer are idempotent there.

Folder pipeline (design doc), per stage on unstructured_data:
  3. RUNNING with an expired lease (the worker died) goes back to WAITING with error 'stalled' and a
     folder message; the attempt was counted at claim, so after STAGE_MAX_ATTEMPTS it becomes FAILED.
  4. WAITING longer than the stage's stuck threshold gets one folder message per threshold period.
"""
import asyncio
import logging
from uuid import uuid5

from sqlalchemy import text

from ...core.rag_pipeline.userallocation.UserAllocation import OUTBOX_INSERT, queue_message, refresh_thread_status
from ...core.rag_pipeline.userallocation.UserAllocation import STAGES as FOLDER_STAGES
from ...helpers.events import RagEvent

logger = logging.getLogger(__name__)

STAGES = {
    'PARSING': ('PENDING', 'doc.uploaded', 'attempt_count', 'parse_max_attempts', 'ParseAttemptLimitExceeded'),
    'INDEXING': ('PARSED', 'doc.parsed', 'index_attempt_count', 'index_max_attempts', 'IndexAttemptLimitExceeded'),
}
NEXT_EVENT = {'PENDING': 'doc.uploaded', 'PARSED': 'doc.parsed'}


class StallSweeper:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings

    async def sweep_once(self, project_filter=None):
        counts = {'recovered': 0, 'failed': 0, 'requeued_stuck': 0}
        scope = ' AND project_id=:project' if project_filter is not None else ''
        params = {'batch': self.settings.sweep_batch_size, 'project': project_filter,
                  'stuck': self.settings.sweep_stuck_seconds}
        async with self.db.transaction() as session:
            expired = (await session.execute(text(f"""SELECT * FROM documents
                WHERE status IN ('PARSING','INDEXING') AND lease_expires_at < now(){scope}
                ORDER BY lease_expires_at LIMIT :batch FOR UPDATE SKIP LOCKED"""), params)).mappings().all()
            for row in expired:
                back_to, event_type, counter, limit, code = STAGES[row['status']]
                if row[counter] >= getattr(self.settings, limit):
                    await session.execute(text("""UPDATE documents SET status='FAILED', lease_owner=NULL,
                        lease_expires_at=NULL, error_message=:code WHERE id=:doc"""), {'doc': row['id'], 'code': code})
                    counts['failed'] += 1
                    continue
                await session.execute(text("""UPDATE documents SET status=:status, lease_owner=NULL,
                    lease_expires_at=NULL, error_message='LeaseExpired' WHERE id=:doc"""),
                    {'doc': row['id'], 'status': back_to})
                await self._enqueue(session, row, event_type, f'sweep:lease:{row["status"]}:{row[counter]}')
                counts['recovered'] += 1
            stuck = (await session.execute(text(f"""SELECT * FROM documents d
                WHERE status IN ('PENDING','PARSED') AND updated_at < now() - (:stuck * interval '1 second'){scope}
                AND NOT EXISTS (SELECT 1 FROM outbox_events o WHERE o.aggregate_id=d.id AND o.status='PENDING')
                ORDER BY updated_at LIMIT :batch FOR UPDATE SKIP LOCKED"""), params)).mappings().all()
            for row in stuck:
                key = f'sweep:stuck:{row["status"]}:{row["attempt_count"]}:{row["index_attempt_count"]}'
                if await self._enqueue(session, row, NEXT_EVENT[row['status']], key):
                    counts['requeued_stuck'] += 1
        counts |= await self.sweep_folders(thread_filter=project_filter)
        if any(counts.values()):
            logger.info('Stall sweep: %s', counts)
        return counts

    async def sweep_folders(self, thread_filter=None):
        counts = {'stage_recovered': 0, 'stage_failed': 0, 'stage_requeued': 0}
        scope = ' AND d.thread_id=:thread' if thread_filter is not None else ''
        for stage in FOLDER_STAGES:
            stuck = (self.settings.standardization_stuck_seconds if stage == 'standardization'
                     else self.settings.stage_stuck_seconds)
            params = {'batch': self.settings.sweep_batch_size, 'thread': thread_filter, 'stuck': stuck,
                      'max': self.settings.stage_max_attempts, 'type': f'thread.{stage}'}
            async with self.db.transaction() as session:
                dead = (await session.execute(text(f"""SELECT d.id, d.thread_id, d.{stage}_attempts AS attempts
                    FROM unstructured_data d WHERE d.{stage}_status='RUNNING' AND d.{stage}_lease_expires_at < now(){scope}
                    ORDER BY d.{stage}_lease_expires_at LIMIT :batch FOR UPDATE SKIP LOCKED"""), params)).mappings().all()
                threads, recovered = set(), set()
                for row in dead:
                    final = row['attempts'] >= self.settings.stage_max_attempts
                    await session.execute(text(f"""UPDATE unstructured_data SET {stage}_status=:status, {stage}_owner=NULL,
                        {stage}_lease_expires_at=NULL, {stage}_error='stalled', {stage}_updated_at=now() WHERE id=:doc"""),
                        {'status': 'FAILED' if final else 'WAITING', 'doc': row['id']})
                    counts['stage_failed' if final else 'stage_recovered'] += 1
                    threads.add(row['thread_id'])
                    if not final:
                        recovered.add(row['thread_id'])
                waiting = (await session.execute(text(f"""SELECT DISTINCT d.thread_id FROM unstructured_data d
                    WHERE d.{stage}_status='WAITING' AND d.{stage}_updated_at < now() - (:stuck * interval '1 second'){scope}
                    AND NOT EXISTS (SELECT 1 FROM outbox_events o WHERE o.aggregate_id=d.thread_id AND o.event_type=:type
                        AND (o.status='PENDING' OR o.created_at > now() - (:stuck * interval '1 second')))
                    LIMIT :batch"""), params)).scalars().all()
                counts['stage_requeued'] += len(set(waiting) - recovered)
                wake = recovered | set(waiting)
                if wake:
                    await session.execute(OUTBOX_INSERT, [queue_message(stage, thread) for thread in sorted(wake)])
                for thread in threads:
                    await refresh_thread_status(session, thread)
        return counts

    @staticmethod
    async def _enqueue(session, row, event_type, key):
        event = RagEvent(event_id=uuid5(row['id'], key), event_type=event_type,
                         project_id=row['project_id'], doc_id=row['id'])
        result = await session.execute(text("""INSERT INTO outbox_events(id,project_id,event_type,aggregate_id,payload)
            VALUES(:id,:project,:type,:doc,CAST(:payload AS jsonb)) ON CONFLICT (id) DO NOTHING"""),
            {'id': event.event_id, 'project': event.project_id, 'type': event_type, 'doc': event.doc_id,
             'payload': event.model_dump_json()})
        return result.rowcount == 1

    async def run(self):
        while True:
            try:
                await self.sweep_once()
            except Exception as error:
                logger.warning('Stall sweep will retry (%s)', type(error).__name__)
            await asyncio.sleep(self.settings.sweep_interval_seconds)
