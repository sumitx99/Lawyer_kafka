from types import SimpleNamespace
from unittest.mock import AsyncMock
import json
from uuid import uuid4
import pytest

from pyconsumers.consumers.rag_pipeline_consumers.standardization_consumer import StandardizationConsumer
from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.secrets_control import Settings


def settings():
    return Settings(_env_file=None, postgres_password='test', valkey_password='test',
                    storage_access_key='test', storage_secret_key='test')


async def test_invalid_event_is_dead_lettered_before_commit():
    service = AsyncMock()
    worker = StandardizationConsumer(None, None, None, settings(), service=service)
    record = SimpleNamespace(topic='doc.uploaded', partition=2, offset=7, value=b'invalid-json', key=None)
    calls = []
    async def send(*args, **kwargs):
        calls.append('dlq')
    async def commit(*args):
        calls.append('commit')
    consumer = SimpleNamespace(commit=commit)
    producer = SimpleNamespace(client=SimpleNamespace(send_and_wait=send))
    await worker.process_record(record, consumer, producer)
    assert calls == ['dlq', 'commit']
    service.execute.assert_not_awaited()


async def test_success_commits_after_durable_service_result():
    service = AsyncMock()
    service.execute.return_value = 'parsed'
    worker = StandardizationConsumer(None, None, None, settings(), service=service)
    event = RagEvent(event_type='doc.uploaded', project_id=uuid4(), doc_id=uuid4())
    record = SimpleNamespace(topic='doc.uploaded', partition=0, offset=10,
                             value=event.model_dump_json().encode(), key=b'doc')
    consumer, producer = AsyncMock(), SimpleNamespace(client=AsyncMock())
    await worker.process_record(record, consumer, producer)
    consumer.commit.assert_awaited_once()
    producer.client.send_and_wait.assert_not_awaited()


async def test_failed_dlq_delivery_does_not_commit():
    worker = StandardizationConsumer(None, None, None, settings(), service=AsyncMock())
    record = SimpleNamespace(topic='doc.uploaded', partition=0, offset=1, value=b'bad-json', key=None)
    consumer, producer = AsyncMock(), SimpleNamespace(client=AsyncMock())
    producer.client.send_and_wait.side_effect = ConnectionError('broker unavailable')
    with pytest.raises(ConnectionError):
        await worker.process_record(record, consumer, producer)
    consumer.commit.assert_not_awaited()
