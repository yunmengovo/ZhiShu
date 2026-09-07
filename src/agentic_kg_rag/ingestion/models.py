from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic_kg_rag.knowledge_base.models import DEFAULT_KNOWLEDGE_BASE_ID

BlockKind = Literal["title", "heading", "paragraph", "list", "table"]


class SourceBlock(BaseModel):
    block_id: str
    kind: BlockKind
    text: str
    order: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    heading_level: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    order: int = Field(ge=0)
    block_ids: list[str]
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedDocument(BaseModel):
    document_id: str
    knowledge_base_id: str = DEFAULT_KNOWLEDGE_BASE_ID
    filename: str
    file_type: str
    content_type: str | None = None
    checksum_sha256: str
    title: str
    status: Literal["ready", "failed"] = "ready"
    created_at: datetime
    raw_path: str
    processed_path: str
    blocks: list[SourceBlock]
    chunks: list[DocumentChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    document_id: str
    knowledge_base_id: str = DEFAULT_KNOWLEDGE_BASE_ID
    filename: str
    file_type: str
    title: str
    status: Literal["ready", "failed"]
    created_at: datetime
    block_count: int
    chunk_count: int
    duplicate: bool = False

    @classmethod
    def from_document(
        cls, document: ProcessedDocument, *, duplicate: bool = False
    ) -> DocumentSummary:
        return cls(
            document_id=document.document_id,
            knowledge_base_id=document.knowledge_base_id,
            filename=document.filename,
            file_type=document.file_type,
            title=document.title,
            status=document.status,
            created_at=document.created_at,
            block_count=len(document.blocks),
            chunk_count=len(document.chunks),
            duplicate=duplicate,
        )


class BatchDocumentResult(BaseModel):
    filename: str
    success: bool
    document: DocumentSummary | None = None
    error: str = ""


class BatchIngestResponse(BaseModel):
    knowledge_base_id: str
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    results: list[BatchDocumentResult]
