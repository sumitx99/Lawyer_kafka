"""Postgres-backed fair queue for one pipeline stage (design doc: UserAllocation).

get_next_document: one transaction takes the oldest WAITING file of this stage whose user is under
max_per_user while the stage is under max_workers, and sets it RUNNING with an owner and a lease.
A per-stage advisory lock makes the limits exact across workers; FOR UPDATE SKIP LOCKED keeps two
workers from taking the same file. Running counts are read from the table, so finishing a file frees
its slot automatically and a crash leaves nothing in memory to rebuild.

finish: only the current owner (fencing) can record the result. In the same transaction it starts the
next stages, queues their folder messages in the outbox and refreshes the folder status:
  standardization -> markdown and image (WAITING, or INAPPLICABLE when the handler says so)
  markdown + image -> metaclassifier, set WAITING exactly once, when both are COMPLETED/INAPPLICABLE
A failed attempt goes back to WAITING with an exponential backoff; after STAGE_MAX_ATTEMPTS, or for a
terminal error, the stage is FAILED and the caller sends a dead letter.
"""
import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4

from sqlalchemy import text

from ....helpers.events import ThreadMessage

STAGES = ('standardization', 'markdown', 'image', 'metaclassifier', 'parquet')
DONE = {'COMPLETED', 'INAPPLICABLE'}
# Columns a stage handler may set on unstructured_data besides its own status columns.
RESULT_COLUMNS = {'s3_path_tomarkdown', 's3_path_images', 'page_count', 'language'}


class LeaseLost(Exception):
    """This worker no longer owns the file: another worker or the sweeper took it over."""


class StageError(Exception):
    """Raised by a stage handler. terminal=True fails the stage now instead of retrying."""
    def __init__(self, message, terminal=False):
        super().__init__(message)
        self.terminal = terminal


@dataclass
class StageResult:
    status: str = 'COMPLETED'  # or INAPPLICABLE
    updates: dict = field(default_factory=dict)  # RESULT_COLUMNS only
    summary: dict = field(default_factory=dict)  # merged into processing_summary[stage]
    next: dict = field(default_factory=dict)  # standardization only: {'markdown'|'image': 'WAITING'|'INAPPLICABLE'}


@dataclass
class Claim:
    stage: str
    owner: str
    row: dict

    @property
    def doc_id(self):
        return self.row['id']

    @property
    def thread_id(self):
        return self.row['thread_id']


def stages_for(file_type):
    return ('parquet',) if file_type == 'csv' else ('standardization', 'markdown', 'image', 'metaclassifier')


def file_state(row):
    if row['upload_status'] == 'RUNNING':
        return 'uploading'
    if row['upload_status'] == 'FAILED':
        return 'failed'
    statuses = [row[f'{stage}_status'] for stage in stages_for(row['file_type'])]
    if 'FAILED' in statuses:
        return 'failed'
    return 'done' if all(status in DONE for status in statuses) else 'processing'


def derive_thread_status(rows):
    """Mirrors the API's rule (services/threads/unstructured/thread_service.py)."""
    states = {file_state(row) for row in rows}
    if not states or 'uploading' in states:
        return 'UPLOADING'
    if 'processing' in states:
        return 'PROCESSING'
    if states == {'done'}:
        return 'COMPLETED'
    return 'FAILED' if states == {'failed'} else 'PARTIAL'


def transitions(stage, row, result):
    """Status changes for later stages once `stage` finished with `result` (row is before this finish)."""
    changes = {}
    if stage == 'standardization':
        for later in ('markdown', 'image'):
            status = result.next.get(later, 'WAITING')
            if status not in {'WAITING', 'INAPPLICABLE'}:
                raise ValueError(f'Invalid next status for {later}: {status}')
            changes[later] = status
    if stage in {'standardization', 'markdown', 'image'}:
        after = {name: changes.get(name, row[f'{name}_status']) for name in ('markdown', 'image')}
        if stage != 'standardization':
            after[stage] = result.status
        if all(after[name] in DONE for name in after) and row['metaclassifier_status'] is None:
            changes['metaclassifier'] = 'WAITING'
    return changes


def retry_delay(settings, attempts):
    return min(600, settings.stage_retry_base_seconds * 2 ** max(0, attempts - 1))


def queue_message(stage, thread_id):
    return {'id': uuid4(), 'project': thread_id, 'type': f'thread.{stage}', 'thread': thread_id,
            'payload': ThreadMessage(unstructured_thread_id=thread_id).model_dump_json()}


OUTBOX_INSERT = text("""INSERT INTO outbox_events(id,project_id,event_type,aggregate_id,payload)
    VALUES(:id,:project,:type,:thread,CAST(:payload AS jsonb))""")


async def refresh_thread_status(session, thread_id):
    rows = (await session.execute(text('SELECT * FROM unstructured_data WHERE thread_id=:thread'),
                                  {'thread': thread_id})).mappings().all()
    status = derive_thread_status(rows)
    await session.execute(text('UPDATE unstructured_thread SET status=:status WHERE id=:thread AND status<>:status'),
                          {'thread': thread_id, 'status': status})
    return status


async def lock_claim(session, claim):
    """Fence: lock the file row and prove this worker still owns the stage with a live lease.

    Handlers call it inside the transaction that writes their output (text_nodes, images), so a worker
    whose lease was taken over can never overwrite the new owner's rows.
    """
    s = claim.stage
    row = (await session.execute(text(f"""SELECT * FROM unstructured_data WHERE id = :doc
        AND {s}_owner = :owner AND {s}_status = 'RUNNING' AND {s}_lease_expires_at > now() FOR UPDATE"""),
        {'doc': claim.doc_id, 'owner': claim.owner})).mappings().first()
    if row is None:
        raise LeaseLost()
    return row


class UserAllocation:
    def __init__(self, db, cache, settings, stage):
        if stage not in STAGES:
            raise ValueError(f'Unknown stage {stage}')
        self.db, self.cache, self.settings, self.stage = db, cache, settings, stage
        self.max_workers = getattr(settings, f'{stage}_max_workers')
        self.max_per_user = getattr(settings, f'{stage}_max_per_user')

    def column(self, name, stage=None):
        return f'{stage or self.stage}_{name}'

    async def get_next_document(self, owner):
        s = self.stage
        async with self.db.transaction() as session:
            await session.execute(text('SELECT pg_advisory_xact_lock(hashtext(:key))'), {'key': f'userallocation:{s}'})
            row = (await session.execute(text(f"""
                WITH running AS (
                    SELECT t.created_by, count(*) AS n FROM unstructured_data d
                    JOIN unstructured_thread t ON t.id = d.thread_id
                    WHERE d.{s}_status = 'RUNNING' AND d.{s}_lease_expires_at > now()
                    GROUP BY t.created_by),
                candidate AS (
                    SELECT d.id FROM unstructured_data d
                    JOIN unstructured_thread t ON t.id = d.thread_id
                    LEFT JOIN running r ON r.created_by = t.created_by
                    WHERE d.{s}_status = 'WAITING' AND d.upload_status = 'COMPLETED'
                      AND coalesce(d.{s}_updated_at, '-infinity') <= now()
                      AND coalesce(r.n, 0) < :per_user
                      AND (SELECT coalesce(sum(n), 0) FROM running) < :workers
                    ORDER BY d.{s}_updated_at NULLS FIRST, d.created_at, d.id
                    LIMIT 1 FOR UPDATE OF d SKIP LOCKED)
                UPDATE unstructured_data d SET {s}_status = 'RUNNING', {s}_owner = :owner,
                    {s}_lease_expires_at = now() + (:lease * interval '1 second'),
                    {s}_attempts = {s}_attempts + 1, {s}_updated_at = now()
                FROM candidate, unstructured_thread t
                WHERE d.id = candidate.id AND t.id = d.thread_id
                RETURNING d.*, t.created_by, t.jurisdiction, t.tags"""),
                {'per_user': self.max_per_user, 'workers': self.max_workers, 'owner': owner,
                 'lease': self.settings.stage_lease_seconds})).mappings().first()
            if row is None:
                return None
            await refresh_thread_status(session, row['thread_id'])
        claim = Claim(s, owner, dict(row))
        await self.notify(claim, 'RUNNING')
        return claim

    async def heartbeat(self, claim):
        s = self.stage
        async with self.db.transaction() as session:
            value = await session.scalar(text(f"""UPDATE unstructured_data SET
                {s}_lease_expires_at = now() + (:lease * interval '1 second')
                WHERE id = :doc AND {s}_owner = :owner AND {s}_status = 'RUNNING' AND {s}_lease_expires_at > now()
                RETURNING id"""), {'doc': claim.doc_id, 'owner': claim.owner, 'lease': self.settings.stage_lease_seconds})
        if value is None:
            raise LeaseLost()

    async def _locked(self, session, claim):
        return await lock_claim(session, claim)

    async def complete(self, claim, result: StageResult):
        s = self.stage
        if result.status not in DONE:
            raise ValueError('A stage result must be COMPLETED or INAPPLICABLE.')
        unknown = set(result.updates) - RESULT_COLUMNS
        if unknown:
            raise ValueError(f'Stage handlers may not set {sorted(unknown)}.')
        async with self.db.transaction() as session:
            row = await self._locked(session, claim)
            changes = transitions(s, row, result)
            values = {f'{s}_status': result.status, f'{s}_owner': None, f'{s}_lease_expires_at': None,
                      f'{s}_error': None, **result.updates}
            for later, status in changes.items():
                values |= {f'{later}_status': status, f'{later}_attempts': 0, f'{later}_error': None,
                           f'{later}_owner': None, f'{later}_lease_expires_at': None}
            assignments = ', '.join([f'{name} = :{name}' for name in values] +
                                    [f'{s}_updated_at = now()'] +
                                    [f'{later}_updated_at = now()' for later in changes] +
                                    ['processing_summary = processing_summary || CAST(:summary AS jsonb)'])
            await session.execute(text(f'UPDATE unstructured_data SET {assignments} WHERE id = :doc'),
                                  {**values, 'doc': claim.doc_id, 'summary': json.dumps({s: result.summary})})
            wake = sorted(later for later, status in changes.items() if status == 'WAITING')
            if wake:
                await session.execute(OUTBOX_INSERT, [queue_message(later, claim.thread_id) for later in wake])
            await refresh_thread_status(session, claim.thread_id)
        await self.notify(claim, result.status)
        return wake

    async def fail(self, claim, error: str, terminal=False):
        """Record a failed attempt. Returns 'retry' (back to WAITING after a backoff) or 'failed'."""
        s = self.stage
        async with self.db.transaction() as session:
            row = await self._locked(session, claim)
            attempts = row[f'{s}_attempts']
            final = terminal or attempts >= self.settings.stage_max_attempts
            delay = 0 if final else retry_delay(self.settings, attempts)
            await session.execute(text(f"""UPDATE unstructured_data SET {s}_status = :status, {s}_owner = NULL,
                {s}_lease_expires_at = NULL, {s}_error = :error,
                {s}_updated_at = now() + (:delay * interval '1 second') WHERE id = :doc"""),
                {'status': 'FAILED' if final else 'WAITING', 'error': error[:2000], 'delay': delay, 'doc': claim.doc_id})
            await refresh_thread_status(session, claim.thread_id)
        state = 'failed' if final else 'retry'
        await self.notify(claim, 'FAILED' if final else 'WAITING')
        return state

    async def notify(self, claim, status):
        """Live progress (design doc): progress:doc:{id} for 1 hour, plus a folder pub/sub ping. Best effort."""
        with suppress(Exception):
            async with asyncio.timeout(1):
                payload = json.dumps({'doc_id': str(claim.doc_id), 'thread_id': str(claim.thread_id), 'stage': self.stage,
                                      'status': status, 'at': datetime.now(timezone.utc).isoformat()})
                await self.cache.client.set(f'progress:doc:{claim.doc_id}', payload, ex=3600)
                await self.cache.client.publish(f'progress:thread:{claim.thread_id}', 'changed')
