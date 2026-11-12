from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1, max_length=200)
    entity_type: str = Field(alias="type", min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    chunk_ids: list[str] = Field(default_factory=list)


class ExtractedRelationship(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    relation_type: str = Field(alias="type", min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    chunk_id: str = Field(min_length=1, max_length=100)
    evidence_text: str = ""


class ExtractionPayload(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class GraphExtractionResult(BaseModel):
    extraction_mode: str = "semantic"
    warnings: list[str] = Field(default_factory=list)
    failed_batches: int = 0
    document_id: str
    filename: str
    batches: int = Field(ge=0)
    entity_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)


class GraphRelationEvidence(BaseModel):
    rank: int = Field(ge=1)
    source: str
    source_type: str
    relation: str
    target: str
    target_type: str
    description: str = ""
    document_id: str
    chunk_id: str


class GraphSearchResponse(BaseModel):
    query: str
    relations: list[GraphRelationEvidence]


class GraphStatus(BaseModel):
    mode: str
    backend: str = "sqlite"
    connected: bool = True
    message: str = ""
    entity_count: int = Field(ge=0)
    relationship_count: int = Field(ge=0)
    document_count: int = Field(ge=0)


class GraphSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] = Field(default_factory=list)


class GraphNode(BaseModel):
    node_id: str
    label: str
    entity_type: str


class GraphEdge(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    description: str = ""
    document_id: str
    chunk_id: str


class GraphSubgraphRequest(BaseModel):
    query: str = Field(default="", max_length=4000)
    document_ids: list[str] = Field(default_factory=list)
    layer: Literal["all", "semantic", "structure"] = "all"
    limit: int = Field(default=80, ge=1, le=500)


class GraphSubgraphResponse(BaseModel):
    query: str = ""
    nodes: list[GraphNode]
    edges: list[GraphEdge]
