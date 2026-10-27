"""Run: python -m pyconsumers.start.

Starts the per-document parsing and indexing consumers, one folder-pipeline StageWorker per stage
that has a handler, the outbox publisher and the stall sweeper.
SIGTERM/SIGINT cancel everything so consumers leave their group cleanly; work interrupted
mid-document is recovered by Kafka redelivery (offsets are only committed after durable
results) or, failing that, by the sweeper once its lease expires.
"""
import asyncio
from contextlib import suppress
import logging
import signal

from .consumers.rag_pipeline_consumers.rag_markdown_consumer import IndexingConsumer
from .consumers.rag_pipeline_consumers.stage_worker import StageWorker
from .consumers.rag_pipeline_consumers.stall_sweeper import StallSweeper
from .consumers.rag_pipeline_consumers.standardization_consumer import StandardizationConsumer
from .core.rag_pipeline.execute_indexing import Indexing, MarkdownRag
from .core.rag_pipeline.execute_parquet import CsvParquet
from .core.rag_pipeline.execute_metaclassifier import MetaClassifier
from .core.rag_pipeline.execute_standardization import FolderStandardization
from .core.rag_pipeline.multimodal_processors.image_rag import ImageRag
from .helpers.embedding_helper import BGEM3
from .helpers.outbox_helper import DocumentOutboxPublisher
from .helpers.postgres_helper import Database
from .helpers.secrets_control import get_settings
from .helpers.storage_helper import Storage
from .helpers.valkey_helper import Cache

logger = logging.getLogger(__name__)

# Folder pipeline handlers by stage: build(db, storage, cache, settings, model) -> async (claim) -> StageResult.
# Documents use standardization; CSV files use only parquet.
STAGE_HANDLERS = {
    'parquet': lambda db, storage, cache, settings, model: CsvParquet(storage, settings),
    'standardization': lambda db, storage, cache, settings, model: FolderStandardization(storage, settings),
    'markdown': lambda db, storage, cache, settings, model: MarkdownRag(db, storage, settings, model),
    'image': lambda db, storage, cache, settings, model: ImageRag(db, storage, cache, settings, model),
    'metaclassifier': lambda db, storage, cache, settings, model: MetaClassifier(db, settings, model),
}


async def keep_running(worker):
    while True:
        try:
            await worker.run()
        except Exception as error:
            logger.warning('%s connection will retry (%s)', type(worker).__name__, type(error).__name__)
            await asyncio.sleep(5)


async def main():
    settings = get_settings()
    db, storage, cache = Database(settings), Storage(settings), Cache(settings)
    model = BGEM3(settings)  # one copy of the 2.3 GB embedding model, shared by every stage
    publisher = DocumentOutboxPublisher(db, settings)
    tasks = [asyncio.create_task(publisher.run()),
             asyncio.create_task(StallSweeper(db, settings).run()),
             asyncio.create_task(keep_running(StandardizationConsumer(db, storage, cache, settings))),
             asyncio.create_task(keep_running(IndexingConsumer(db, storage, cache, settings,
                                                               service=Indexing(db, storage, cache, settings, model=model))))]
    tasks += [asyncio.create_task(keep_running(StageWorker(db, cache, settings, stage, build(db, storage, cache, settings, model))))
              for stage, build in STAGE_HANDLERS.items()]
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(name, stop.set)
    try:
        await stop.wait()
        logger.info('Shutdown requested; stopping workers.')
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await task
        await publisher.close()
        await cache.close()
        await storage.close()
        await db.close()
        logger.info('Worker stopped.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    asyncio.run(main())
