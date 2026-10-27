import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from pyconsumers.consumers.rag_pipeline_consumers.stage_worker import StageWorker
from pyconsumers.core.rag_pipeline.userallocation.UserAllocation import (Claim, LeaseLost, StageError, StageResult,
                                                                         derive_thread_status, retry_delay, transitions)
from pyconsumers.helpers.secrets_control import Settings


def settings(**changes):
    return Settings(_env_file=None, postgres_password='t', valkey_password='t', storage_access_key='t',
                    storage_secret_key='t', **changes)


def row(**values):
    base = {'markdown_status': None, 'image_status': None, 'metaclassifier_status': None}
    return base | values


@pytest.mark.parametrize('stage,before,result,expected', [
    ('standardization', row(), StageResult(), {'markdown': 'WAITING', 'image': 'WAITING'}),
    ('standardization', row(), StageResult(next={'image': 'INAPPLICABLE'}), {'markdown': 'WAITING', 'image': 'INAPPLICABLE'}),
    ('standardization', row(), StageResult(next={'markdown': 'INAPPLICABLE', 'image': 'INAPPLICABLE'}),
     {'markdown': 'INAPPLICABLE', 'image': 'INAPPLICABLE', 'metaclassifier': 'WAITING'}),
    ('markdown', row(markdown_status='RUNNING', image_status='RUNNING'), StageResult(), {}),
    ('markdown', row(markdown_status='RUNNING', image_status='INAPPLICABLE'), StageResult(), {'metaclassifier': 'WAITING'}),
    ('image', row(markdown_status='COMPLETED', image_status='RUNNING'), StageResult(status='INAPPLICABLE'),
     {'metaclassifier': 'WAITING'}),
    ('image', row(markdown_status='COMPLETED', image_status='RUNNING', metaclassifier_status='WAITING'), StageResult(), {}),
    ('metaclassifier', row(), StageResult(), {}), ('parquet', row(), StageResult(), {}),
])
def test_stage_transitions(stage, before, result, expected):
    assert transitions(stage, before, result) == expected


def test_transitions_reject_invalid_next_status():
    with pytest.raises(ValueError):
        transitions('standardization', row(), StageResult(next={'markdown': 'COMPLETED'}))


def doc(file_type='pdf', upload='COMPLETED', **stages):
    values = {'file_type': file_type, 'upload_status': upload}
    return values | {f'{s}_status': stages.get(s) for s in ('standardization', 'markdown', 'image', 'metaclassifier', 'parquet')}


def test_folder_status_matches_the_api_rule():
    done = dict(standardization='COMPLETED', markdown='COMPLETED', image='INAPPLICABLE', metaclassifier='COMPLETED')
    assert derive_thread_status([doc(**done), doc('csv', parquet='COMPLETED')]) == 'COMPLETED'
    assert derive_thread_status([doc(**done), doc(standardization='RUNNING')]) == 'PROCESSING'
    assert derive_thread_status([doc(**done), doc(**(done | {'image': 'FAILED'}))]) == 'PARTIAL'
    assert derive_thread_status([doc(upload='RUNNING')]) == 'UPLOADING'


def test_retry_backoff_grows_and_is_capped():
    s = settings(stage_retry_base_seconds=30)
    assert [retry_delay(s, n) for n in (1, 2, 3, 10)] == [30, 60, 120, 600]


class FakeQueue:
    def __init__(self, fail_state='retry'):
        self.claim = Claim('markdown', 'owner', {'id': uuid4(), 'thread_id': uuid4(), 'markdown_attempts': 3})
        self.fail_state, self.completed, self.failed = fail_state, [], []

    async def get_next_document(self, owner):
        claim, self.claim = self.claim, None
        return claim

    async def heartbeat(self, claim):
        pass

    async def complete(self, claim, result):
        self.completed.append(result)

    async def fail(self, claim, message, terminal=False):
        self.failed.append((message, terminal))
        if self.fail_state == 'lost':
            raise LeaseLost()
        return self.fail_state


def worker(handler, queue):
    w = StageWorker(None, None, settings(), 'markdown', handler, producer=SimpleNamespace(client=AsyncMock()))
    w.queue = queue
    return w


@pytest.mark.parametrize('error,terminal', [(StageError('bad layout', terminal=True), True), (ValueError('bad'), True),
                                            (RuntimeError('timeout'), False), (StageError('retry me'), False)])
async def test_worker_records_failures_with_the_right_kind(error, terminal):
    async def handler(claim):
        raise error
    queue = FakeQueue()
    w = worker(handler, queue)
    assert await w.process_next() is True and await w.process_next() is False
    assert queue.failed[0][1] is terminal and queue.completed == []
    w.producer.client.send_and_wait.assert_not_awaited()  # 'retry' sends no dead letter


async def test_worker_dead_letters_only_final_failures_and_ignores_lost_leases():
    async def handler(claim):
        raise StageError('Docling crashed')
    queue = FakeQueue(fail_state='failed')
    w = worker(handler, queue)
    await w.process_next()
    topic, payload = w.producer.client.send_and_wait.await_args.args
    assert topic == 'RAG_MARKDOWN_TOPIC_DLQ' and b'"error": "Docling crashed"' in payload and b'"attempts": 3' in payload
    lost = worker(handler, FakeQueue(fail_state='lost'))
    await lost.process_next()  # LeaseLost while recording the failure: nothing is sent
    lost.producer.client.send_and_wait.assert_not_awaited()


async def test_worker_completes_and_stops_when_the_queue_is_empty():
    async def handler(claim):
        await asyncio.sleep(0)
        return StageResult(summary={'chunks': 4})
    queue = FakeQueue()
    w = worker(handler, queue)
    assert await w.process_next() is True and await w.process_next() is False
    assert queue.completed[0].summary == {'chunks': 4}
