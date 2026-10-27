"""Versioned parser output contract, shared with later ingestion stages."""
from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field
from .visual import VisualDescription


class ParsedBlock(BaseModel):
    ref: str
    label: str
    text: str = ''
    level: int | None = None
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    data: dict[str, Any] | None = None


class ParsedAsset(BaseModel):
    ref: str
    path: str
    media_type: str = 'image/png'
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    ocr_text: str = ''
    visual_description: VisualDescription | None = None


class ParserOutput(BaseModel):
    schema_version: Literal[1] = 1
    parser: str = 'docling'
    parser_version: str
    pages: list[dict[str, Any]] = Field(default_factory=list)
    blocks: list[ParsedBlock]
    assets: list[ParsedAsset] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    docling_document: dict[str, Any]


class ParsedArtifact(ParserOutput):
    doc_id: UUID
    project_id: UUID
    source_sha256: str
    source_filename: str
    markdown_key: str
    manifest_key: str
