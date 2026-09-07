from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexResult(BaseModel):
    document_id: str
    filename: str
    collection: str
    indexed_chunks: int = Field(ge=0)


class IndexAllResult(BaseModel):
    collection: str
    indexed_documents: int = Field(ge=0)
    indexed_chunks: int = Field(ge=0)
    documents: list[IndexResult]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] = Field(default_factory=list)
    knowledge_base_id: str | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] = Field(default_factory=list)
    knowledge_base_id: str | None = None


class Evidence(BaseModel):
    rank: int = Field(ge=1)
    score: float
    source_type: Literal["vector", "web"] = "vector"
    document_id: str
    chunk_id: str
    filename: str
    title: str
    text: str
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    evidence: list[Evidence]


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    evidence: list[Evidence]


class RagStatus(BaseModel):
    mode: str
    collection: str
    collection_ready: bool
    indexed_chunks: int = Field(ge=0)
    chat_configured: bool
    embedding_configured: bool


class DocumentIndexStatus(BaseModel):
    document_id: str
    indexed_chunks: int = Field(ge=0)
    indexed: bool
