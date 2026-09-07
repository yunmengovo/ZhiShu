from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic_kg_rag.knowledge_graph.models import GraphRelationEvidence
from agentic_kg_rag.memory.models import MemorySnapshot
from agentic_kg_rag.rag.models import Evidence


class AgentAskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[str] = Field(default_factory=list)
    knowledge_base_id: str | None = None
    max_retries: int | None = Field(default=None, ge=0, le=3)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)


class AgentStep(BaseModel):
    node: str
    action: str
    detail: str


class AgentAskResponse(BaseModel):
    question: str
    answer: str
    model: str
    route: Literal["retrieve", "direct"]
    final_query: str
    retrieval_attempts: int = Field(ge=0)
    evidence: list[Evidence]
    graph_relations: list[GraphRelationEvidence]
    steps: list[AgentStep]
    session_id: str | None = None
    memory: MemorySnapshot | None = None
    citations_valid: bool = True
    citation_warnings: list[str] = Field(default_factory=list)
    used_web_search: bool = False
