"""Batch embedding orchestration with tracked failures and retries.

A batch is retried a bounded number of times; a batch that still fails, or returns a vector
with the wrong dimension or non-finite values, fails the whole document so no partially
embedded index is ever marked ready.
"""
import asyncio
import logging
import math

from ....helpers.embedding_helper import EmbeddingUnavailable

logger = logging.getLogger(__name__)


class EmbeddingFailure(Exception):
    pass


class Embedder:
    def __init__(self, settings, model):
        self.settings, self.model = settings, model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors, size = [], self.settings.embedding_batch_size
        for start in range(0, len(texts), size):
            vectors.extend(await self._batch(texts[start:start + size], start // size))
        return vectors

    async def _batch(self, texts, number):
        for attempt in range(1, self.settings.embedding_max_attempts + 1):
            try:
                vectors = await self.model.embed_async(texts)
                self._validate(texts, vectors)
                return vectors
            except EmbeddingUnavailable:
                raise
            except Exception as error:
                if attempt == self.settings.embedding_max_attempts:
                    raise EmbeddingFailure(f'Embedding batch {number} failed after {attempt} attempts '
                                           f'({type(error).__name__}).') from None
                logger.warning('Embedding batch %s attempt %s failed (%s); retrying', number, attempt,
                               type(error).__name__)
                await asyncio.sleep(min(10, 2 ** attempt))

    def _validate(self, texts, vectors):
        if len(vectors) != len(texts):
            raise ValueError('Embedding count does not match input count.')
        for vector in vectors:
            if len(vector) != self.settings.embedding_dimension or not all(math.isfinite(v) for v in vector):
                raise ValueError('Embedding has the wrong dimension or non-finite values.')
