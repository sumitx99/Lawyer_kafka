"""Chunk record shared by the text and table chunkers."""
from dataclasses import dataclass


@dataclass
class IndexChunk:
    ordinal: int
    section_ordinal: int
    heading_path: list[str]
    content: str
    embedding_text: str
    token_count: int
    page_start: int | None
    page_end: int | None
    source_metadata: dict
    position: int = 0  # index of the first source block, for document-order sorting
    chunk_type: str = 'text'
    table_id: str | None = None
