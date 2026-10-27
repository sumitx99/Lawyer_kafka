import asyncio
import os
from uuid import uuid4

from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.structs import TopicPartition
import pytest

from pyconsumers.helpers.consumer_helper import EventProducer, commit_record, create_consumer
from pyconsumers.helpers.events import RagEvent
from pyconsumers.helpers.postgres_helper import Database
from pyconsumers.helpers.secrets_control import get_settings
from pyconsumers.helpers.storage_helper import Storage
from pyconsumers.helpers.valkey_helper import Cache

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get('RAG_RUN_INTEGRATION') != '1',
                                 reason='Set RAG_RUN_INTEGRATION=1 for real service tests.')]


async def test_database_connectivity():
    db = Database()
    try:
        assert await db.ping()
    finally:
        await db.close()


async def test_storage_roundtrip():
    storage = Storage()
    key = f'integration/{uuid4()}/sample.pdf'
    content = b'%PDF-1.7 integration fixture'
    try:
        assert await storage.ping()
        await storage.put_bytes(key, content, 'application/pdf')
        assert await storage.get_bytes(key) == content
    finally:
        await storage.delete(key)
        await storage.close()


async def test_cache_ttl_and_lock_ownership():
    cache = Cache()
    project = uuid4()
    key = cache.key(project, 'cache', uuid4())
    lock = cache.key(project, 'lock', uuid4())
    assert cache.key(project, 'retrieval', 'same') != cache.key(uuid4(), 'retrieval', 'same')
    try:
        await cache.set_json(key, {'status': 'PENDING'}, ttl=30)
        assert await cache.get_json(key) == {'status': 'PENDING'}
        assert 0 < await cache.client.ttl(key) <= 30
        token = await cache.acquire_lock(lock)
        assert token
        assert await cache.acquire_lock(lock) is None
        assert not await cache.release_lock(lock, 'wrong-owner')
        assert await cache.client.get(lock) == token
        assert await cache.release_lock(lock, token)
        assert await cache.client.get(lock) is None
    finally:
        await cache.client.delete(key, lock)
        await cache.close()


async def test_kafka_event_roundtrip_and_manual_commit():
    settings = get_settings()
    topic = f'infra.helper.{uuid4().hex}'
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    producer = EventProducer(settings)
    consumer = create_consumer(topic, f'infra-helper-{uuid4().hex}', settings)
    created = False
    try:
        await admin.start()
        await admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1,
                                           topic_configs={'retention.ms': '60000'})])
        created = True
        producer.topics['doc.uploaded'] = topic
        await producer.start()
        await consumer.start()
        assert consumer._enable_auto_commit is False
        event = RagEvent(event_type='doc.uploaded', project_id=uuid4(), doc_id=uuid4())
        await producer.send(event)
        record = await asyncio.wait_for(consumer.getone(), timeout=30)
        assert RagEvent.model_validate_json(record.value) == event
        assert record.key == f'{event.project_id}:{event.doc_id}'.encode()
        await commit_record(consumer, record)
        assert await consumer.committed(TopicPartition(topic, record.partition)) == record.offset + 1
    finally:
        await consumer.stop()
        await producer.close()
        if created:
            await admin.delete_topics([topic])
        await admin.close()
