from __future__ import annotations

from typing import Literal, TypedDict

from agentic_kg_rag.agent.models import AgentStep
from agentic_kg_rag.knowledge_graph.models import GraphRelationEvidence
from agentic_kg_rag.rag.models import Evidence


class AgentState(TypedDict, total=False):
    question: str
    query: str
    conversation_context: str
    document_ids: list[str]
    top_k: int
    max_retries: int
    retrieval_attempts: int
    route: Literal["retrieve", "direct"]
    grade: Literal["sufficient", "rewrite", "web", "stop"]
    evidence: list[Evidence]
    graph_relations: list[GraphRelationEvidence]
    answer: str
    citations_valid: bool
    citation_warnings: list[str]
    used_web_search: bool
    steps: list[AgentStep]
