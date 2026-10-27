"""Kafka transport primitives and the shared at-least-once consumer loop (retry, DLQ, manual commit)."""
import asyncio
from hashlib import sha256
import json
import logging
from uuid import NAMESPACE_URL, uuid5

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.structs import OffsetAndMetadata, TopicPartition

from .events import THREAD_EVENTS, RagEvent
from .secrets_control import Settings, get_settings


class EventProducer:
    def __init__(self, settings: Settings | None = None):
        settings = settings or get_settings()
        self.topics = {'doc.uploaded': settings.kafka_topic_uploaded,
                       'doc.parsed': settings.kafka_topic_parsed,
                       'query.requested': settings.kafka_topic_query}
        self.thread_topics = {event_type: getattr(settings, attribute) for event_type, attribute in THREAD_EVENTS.items()}
        self.client = AIOKafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers,
                                      enable_idempotence=True, acks='all', request_timeout_ms=15000)

    async def start(self):
        await self.client.start()

    async def send(self, event: RagEvent):
        identity = event.doc_id or event.query_id
        # Same document/query key preserves ordering within a partition.
        key = f'{event.project_id}:{identity}'.encode()
        return await self.client.send_and_wait(self.topics[event.event_type],
                                              event.model_dump_json().encode(), key=key)

    async def send_thread(self, event_type, thread_id):
        """One folder message {"unstructured_thread_id": X}, keyed by folder."""
        payload = json.dumps({'unstructured_thread_id': str(thread_id)}).encode()
        return await self.client.send_and_wait(self.thread_topics[event_type], payload, key=str(thread_id).encode())

    async def close(self):
        await self.client.stop()


def create_consumer(topic, group_id, settings: Settings | None = None, *, max_poll_interval_ms=900000):
    settings = settings or get_settings()
    return AIOKafkaConsumer(topic, bootstrap_servers=settings.kafka_bootstrap_servers,
                           group_id=group_id, enable_auto_commit=False,
                           auto_offset_reset='earliest', max_poll_records=1,
                           max_poll_interval_ms=max_poll_interval_ms, request_timeout_ms=30000)


async def commit_record(consumer, record):
    """Call only after durable effects. Process records sequentially per partition."""
    partition = TopicPartition(record.topic, record.partition)
    await consumer.commit({partition: OffsetAndMetadata(record.offset + 1, '')})


logger = logging.getLogger(__name__)


class PipelineConsumer:
    """Sequential per-partition processing: commit only after a durable result or an acknowledged DLQ write.

    Subclasses set event_type, done_states, failure_code, topic, group_id and max_poll_ms; the service's
    execute(event) returns a state, and service.repo.acknowledge_failed(event) records terminal failures.
    """
    event_type = ''
    done_states = frozenset()
    failure_code = 'ProcessingFailed'

    def __init__(self, settings, service):
        self.settings, self.service = settings, service

    @property
    def topic(self):
        raise NotImplementedError

    @property
    def group_id(self):
        raise NotImplementedError

    @property
    def max_poll_ms(self):
        return 900000

    async def process_record(self, record, consumer, producer):
        try:
            event = RagEvent.model_validate_json(record.value)
            if event.event_type != self.event_type:
                raise ValueError('Unexpected event type.')
        except ValueError:
            await self.dead_letter(record, producer, 'InvalidEvent', None)
            await commit_record(consumer, record)
            return
        retries = 0
        while True:
            try:
                state = await self.service.execute(event)
                if state in self.done_states:
                    await commit_record(consumer, record)
                    return
                if state in {'failed', 'missing'}:
                    await self.dead_letter(record, producer, self.failure_code if state == 'failed' else 'DocumentNotFound', event)
                    if state == 'failed':
                        await self.service.repo.acknowledge_failed(event)
                    await commit_record(consumer, record)
                    return
            except Exception as error:
                # A commit/DB/DLQ error must leave the record unacknowledged.
                logger.warning('%s will retry (%s)', type(self).__name__, type(error).__name__)
            retries += 1
            await asyncio.sleep(min(30, 2 ** min(retries, 5)))

    async def dead_letter(self, record, producer, error, event):
        identifier = str(uuid5(NAMESPACE_URL, f'{record.topic}:{record.partition}:{record.offset}'))
        payload = {'schema_version': 1, 'event_id': identifier, 'error_code': error,
                   'source': {'topic': record.topic, 'partition': record.partition, 'offset': record.offset},
                   'original_event': event.model_dump(mode='json') if event else None,
                   'payload_sha256': sha256(record.value or b'').hexdigest()}
        async with asyncio.timeout(20):
            await producer.client.send_and_wait(self.topic + '.dlq', json.dumps(payload).encode(), key=record.key)

    async def run(self):
        consumer = create_consumer(self.topic, self.group_id, self.settings, max_poll_interval_ms=self.max_poll_ms)
        producer = EventProducer(self.settings)
        try:
            await producer.start()
            await consumer.start()
            logger.info('%s listening on %s', type(self).__name__, self.topic)
            async for record in consumer:
                await self.process_record(record, consumer, producer)
        finally:
            await consumer.stop()
            await producer.close()
