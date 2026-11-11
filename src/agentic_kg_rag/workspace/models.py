from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentAsset(BaseModel):
    document_id: str
    knowledge_base_id: str
    filename: str
    file_type: str
    title: str
    status: str
    created_at: datetime
    block_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    indexed_chunks: int = Field(ge=0)
    vector_status: str
    entity_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
    graph_status: str
    build_status: str = ""
    build_message: str = ""
    build_error: str = ""


class DocumentDeleteResult(BaseModel):
    document_id: str
    filename: str
    deleted_vectors: int = Field(ge=0)
    graph_deleted: bool
    files_deleted: bool
    warnings: list[str] = Field(default_factory=list)


class DocumentRebuildRequest(BaseModel):
    reparse: bool = True
    rebuild_vector: bool = True
    rebuild_graph: bool = True


class DocumentRebuildResult(BaseModel):
    document_id: str
    filename: str
    reparsed: bool
    indexed_chunks: int = Field(ge=0)
    entity_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)


class KnowledgeBaseDeleteResult(BaseModel):
    knowledge_base_id: str
    deleted_documents: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
