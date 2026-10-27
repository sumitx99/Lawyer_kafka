"""One folder-pipeline stage: a Kafka listener that wakes a drainer that works the Postgres queue.

Kafka only says "this folder may have work". The listener validates {"unstructured_thread_id": X},
commits at once and wakes the drainer, so polling never stalls behind long documents. The drainer
claims the next file through UserAllocation (fair per user, capped per stage), runs the stage handler
while heartbeating the lease, and records the result; it repeats until nothing is claimable. It also
wakes every STAGE_POLL_SECONDS and once at start, so retries whose backoff has passed and work whose
message was lost are still picked up. A stage that ends FAILED gets a record on <topic>_DLQ.
"""
import asyncio
from contextlib import suppress
import json
import logging
from uuid import uuid4

from ...core.rag_pipeline.userallocation.UserAllocation import LeaseLost, StageError, UserAllocation
from ...helpers.consumer_helper import EventProducer, create_consumer
from ...helpers.events import THREAD_EVENTS, ThreadMessage

logger = logging.getLogger(__name__)


class StageWorker:
    def __init__(self, db, cache, settings, stage, handler, producer=None, group_id=None):
        self.settings, self.stage, self.handler = settings, stage, handler
        self.queue = UserAllocation(db, cache, settings, stage)
        self.topic = getattr(settings, THREAD_EVENTS[f'thread.{stage}'])
        self.group_id = group_id or f'lexaior-stage-{stage}-v1'  # never shared with the old doc.* consumers
        self.producer = producer
        self.wake = asyncio.Event()
        self.wake.set()  # drain once at start

    async def run(self):
        consumer = create_consumer(self.topic, self.group_id, self.settings, max_poll_interval_ms=300000)
        owned = self.producer is None
        self.producer = self.producer or EventProducer(self.settings)
        try:
            if owned:
                await self.producer.start()
            await consumer.start()
            logger.info('%s worker listening on %s', self.stage, self.topic)
            await asyncio.gather(self.listen(consumer), self.drain())
        finally:
            await consumer.stop()
            if owned:
                await self.producer.close()

    async def listen(self, consumer):
        async for record in consumer:
            try:
                ThreadMessage.model_validate_json(record.value)
            except ValueError:
                await self.dead_letter({'error_code': 'InvalidMessage', 'source': {
                    'topic': record.topic, 'partition': record.partition, 'offset': record.offset}})
            await consumer.commit()
            self.wake.set()

    async def drain(self):
        while True:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.wake.wait(), self.settings.stage_poll_seconds)
            self.wake.clear()
            try:
                while await self.process_next():
                    pass
            except Exception as error:
                logger.warning('%s drain will retry (%s)', self.stage, type(error).__name__)

    async def process_next(self):
        """Claim, process and record one file. Returns False when nothing is claimable."""
        claim = await self.queue.get_next_document(f'{self.stage}-{uuid4()}')
        if claim is None:
            return False
        work = asyncio.create_task(self.handler(claim))
        heartbeat = asyncio.create_task(self.heartbeat(claim))
        try:
            done, _ = await asyncio.wait({work, heartbeat}, timeout=self.settings.stage_timeout_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat.result()
                raise LeaseLost()
            if not done:
                raise StageError(f'{self.stage} exceeded {self.settings.stage_timeout_seconds} seconds.')
            await self.queue.complete(claim, work.result())
        except LeaseLost:
            logger.warning('%s lost the lease on %s; another worker owns it now', self.stage, claim.doc_id)
        except Exception as error:
            terminal = getattr(error, 'terminal', isinstance(error, ValueError))
            message = str(error) if isinstance(error, StageError) else type(error).__name__
            with suppress(LeaseLost):
                if await self.queue.fail(claim, message, terminal=terminal) == 'failed':
                    await self.dead_letter({'error_code': 'StageFailed', 'stage': self.stage, 'error': message,
                                            'unstructured_thread_id': str(claim.thread_id), 'doc_id': str(claim.doc_id),
                                            'attempts': claim.row[f'{self.stage}_attempts']})
        finally:
            for task in (work, heartbeat):
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        return True

    async def heartbeat(self, claim):
        while True:
            await asyncio.sleep(self.settings.stage_lease_seconds / 3)
            await self.queue.heartbeat(claim)

    async def dead_letter(self, record):
        """Best effort: the database already says FAILED, which is what the UI and alerts read."""
        try:
            async with asyncio.timeout(20):
                await self.producer.client.send_and_wait(self.topic + '_DLQ', json.dumps({'schema_version': 1, **record}).encode())
        except Exception as error:
            logger.warning('%s dead letter not sent (%s)', self.stage, type(error).__name__)
