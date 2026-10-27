"""Orchestrate chunking, embedding and indexing under a fenced lease."""
import asyncio
from contextlib import suppress
from uuid import uuid4

from ...helpers.embedding_helper import BGEM3
from ...helpers.parsing_repository import LeaseLost
from .chunking.table_chunker import TABLE_CHUNKER_VERSION, chunk_tables
from .chunking.text_chunker import CHUNKER_VERSION, build_sections, chunk_sections
from .indexing.embedder import Embedder
from .indexing.indexer import IndexingRepository, write_text_nodes
from .schemas.parsed import ParsedArtifact, ParserOutput
from .userallocation.UserAllocation import StageError, StageResult


class Indexing:
    def __init__(self, db, storage, cache, settings, model=None):
        self.repo = IndexingRepository(db, settings)
        self.storage, self.cache, self.settings = storage, cache, settings
        self.model = model or BGEM3(settings)
        self.embedder = Embedder(settings, self.model)

    async def notify(self, event):
        with suppress(Exception):
            async with asyncio.timeout(1):
                await self.cache.client.publish(self.cache.key(event.project_id, 'document-progress', event.doc_id), 'changed')

    async def execute(self, event):
        owner = str(uuid4())
        state, row = await self.repo.claim(event, owner)
        if state != 'claimed':
            return state
        await self.notify(event)
        work = asyncio.create_task(self._index(row))
        heartbeat = asyncio.create_task(self._heartbeat(event, owner))
        try:
            done, _ = await asyncio.wait({work, heartbeat}, timeout=self.settings.index_timeout_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise TimeoutError()
            if heartbeat in done:
                heartbeat.result()
                raise LeaseLost()
            sections, chunks, vectors, summary = work.result()
            await self.repo.complete(event, owner, sections, chunks, vectors, summary)
            await self.notify(event)
            return 'indexed'
        except LeaseLost:
            return 'busy'
        except Exception as error:
            # Bad or mismatched manifests (ValueError, incl. pydantic ValidationError) are terminal;
            # a missing local model, timeouts and embedding errors are retried.
            state = await self.repo.failure(event, owner, type(error).__name__, terminal=isinstance(error, ValueError))
            await self.notify(event)
            return state
        finally:
            for task in (work, heartbeat):
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task

    async def _heartbeat(self, event, owner):
        while True:
            await asyncio.sleep(self.settings.index_lease_seconds / 3)
            try:
                await self.repo.heartbeat(event, owner)
            except Exception:
                raise LeaseLost() from None

    async def _index(self, row):
        if not row['parsed_object_key']:
            raise ValueError('Document has no parsed artifact.')
        artifact = ParsedArtifact.model_validate_json(await self.storage.get_bytes(row['parsed_object_key']))
        if (artifact.doc_id, artifact.project_id, artifact.source_sha256) != (row['id'], row['project_id'], row['sha256']):
            raise ValueError('Parsed artifact does not belong to this document version.')
        sections, skipped, tables = build_sections(artifact.blocks, table_captions(artifact))
        budget = (self.model.count_tokens, self.settings.chunk_target_tokens, self.settings.chunk_max_tokens)
        chunks = await asyncio.to_thread(chunk_sections, sections, *budget)
        chunks += await asyncio.to_thread(chunk_tables, tables, *budget)
        chunks.sort(key=lambda chunk: (chunk.section_ordinal, chunk.position, chunk.ordinal))
        for ordinal, chunk in enumerate(chunks):
            chunk.ordinal = ordinal
        vectors = await self.embedder.embed([chunk.embedding_text for chunk in chunks])
        table_chunks = [chunk for chunk in chunks if chunk.chunk_type == 'table']
        summary = {'chunker_version': f'{CHUNKER_VERSION}+{TABLE_CHUNKER_VERSION}', 'embedding_model': self.settings.embedding_model,
                   'embedding_model_revision': self.settings.embedding_model_revision,
                   'embedding_dimension': self.settings.embedding_dimension,
                   'parsed_object_key': row['parsed_object_key'], 'section_count': len(sections),
                   'chunk_count': len(chunks), 'token_count': sum(chunk.token_count for chunk in chunks),
                   'paged': any(chunk.page_start is not None for chunk in chunks),
                   'skipped_blocks': skipped, 'table_count': len(tables),
                   'table_row_count': sum(len(table.rows) for table in tables),
                   'table_chunk_count': len(table_chunks),
                   'multi_page_table_count': sum(len({part.page for part in table.parts}) > 1 for table in tables)}
        return sections, chunks, vectors, summary


def table_captions(artifact):
    """Map table refs to caption text using Docling's caption references."""
    texts = {block.ref: block.text for block in artifact.blocks}
    captions = {}
    for table in artifact.docling_document.get('tables', []) or []:
        refs = [item.get('$ref') for item in table.get('captions', []) or [] if isinstance(item, dict)]
        text = ' '.join(texts[ref].strip() for ref in refs if ref in texts and texts[ref].strip())
        if text and table.get('self_ref'):
            captions[table['self_ref']] = text
    return captions


# ---- Folder pipeline (design doc): Markdown RAG stage handler -------------------------------

def text_search_config(texts):
    """Postgres text-search config for a document: english when its letters are mostly Latin, else simple.

    Postgres ships stemmers for some European languages but none for Hindi and many others; 'simple'
    still matches exact words. BGE-M3 vectors cover meaning across languages either way.
    """
    letters = [ch for item in texts for ch in item if ch.isalpha()]
    latin = sum(ch.isascii() for ch in letters)
    return ('english', 'en') if letters and latin / len(letters) >= 0.9 else ('simple', None)


class MarkdownRag:
    """RAG_MARKDOWN stage: manifest -> heading/table chunks -> BGE-M3 -> text_nodes + node_relationships."""

    def __init__(self, db, storage, settings, model):
        self.db, self.storage, self.settings, self.model = db, storage, settings, model
        self.embedder = Embedder(settings, model)

    async def __call__(self, claim):
        manifest_key = (claim.row['processing_summary'].get('standardization') or {}).get('manifest_key')
        if not manifest_key:
            raise StageError('No parsed manifest; retrigger standardization.', terminal=True)
        artifact = ParserOutput.model_validate_json(await self.storage.get_bytes(manifest_key))
        sections, skipped, tables = build_sections(artifact.blocks, table_captions(artifact))
        budget = (self.model.count_tokens, self.settings.chunk_target_tokens, self.settings.chunk_max_tokens)
        chunks = await asyncio.to_thread(chunk_sections, sections, *budget)
        chunks += await asyncio.to_thread(chunk_tables, tables, *budget)
        chunks.sort(key=lambda chunk: (chunk.section_ordinal, chunk.position, chunk.ordinal))
        for ordinal, chunk in enumerate(chunks):
            chunk.ordinal = ordinal
        config, language = text_search_config([chunk.content for chunk in chunks])
        vectors = await self.embedder.embed([chunk.embedding_text for chunk in chunks])
        version = f'{CHUNKER_VERSION}+{TABLE_CHUNKER_VERSION}'
        written = await write_text_nodes(self.db, claim, chunks, vectors, config, self.settings.embedding_model, version)
        summary = {'chunker_version': version, 'embedding_model': self.settings.embedding_model,
                   'node_count': written, 'table_count': len(tables),
                   'table_node_count': sum(chunk.chunk_type == 'table' for chunk in chunks),
                   'token_count': sum(chunk.token_count for chunk in chunks), 'ts_config': config,
                   'skipped_blocks': skipped}
        return StageResult(status='COMPLETED' if written else 'INAPPLICABLE', updates={'language': language}, summary=summary)
