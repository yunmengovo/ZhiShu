from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.exceptions import ModelError
from langchain_core.messages import AIMessage

from agentic_kg_rag.agent.router import get_agent_service
from agentic_kg_rag.agent.service import AgentService
from agentic_kg_rag.api import app
from agentic_kg_rag.knowledge_graph.models import (
    GraphRelationEvidence,
    GraphSearchResponse,
)
from agentic_kg_rag.memory.service import ConversationMemoryService
from agentic_kg_rag.memory.store import ConversationStore
from agentic_kg_rag.rag.models import Evidence, SearchResponse


class FakeChatModel:
    def __init__(self, *, rewrite: bool = False) -> None:
        self.rewrite = rewrite

    def invoke(self, messages: list[Any]) -> AIMessage:
        system = str(messages[0].content)
        user = str(messages[-1].content)
        if "路由器" in system:
            return AIMessage(content="DIRECT" if user == "你好" else "RETRIEVE")
        if "会话上下文改写器" in system:
            return AIMessage(content="LangGraph 有什么作用？")
        if "相关性评估器" in system:
            return AIMessage(content="RELEVANT")
        if "查询改写器" in system:
            return AIMessage(content="LangGraph memory persistence")
        if "简洁回应" in system:
            return AIMessage(content="你好，可以询问我知识库中的内容。")
        raise AssertionError(f"unexpected prompt: {system}")


class FakeRagService:
    def __init__(self, *, empty_first: bool = False, empty_always: bool = False) -> None:
        self.empty_first = empty_first
        self.empty_always = empty_always
        self.queries: list[str] = []
        self.graph_contexts: list[str] = []
        self.conversation_contexts: list[str] = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: list[str],
    ) -> SearchResponse:
        self.queries.append(query)
        if self.empty_always or (self.empty_first and len(self.queries) == 1):
            return SearchResponse(query=query, evidence=[])
        return SearchResponse(
            query=query,
            evidence=[
                Evidence(
                    rank=1,
                    score=0.9,
                    document_id="doc-1",
                    chunk_id="chunk-1",
                    filename="agentic-rag.mdx",
                    title="Agentic RAG",
                    text="Agents decide when to retrieve.",
                    section_title="Overview",
                )
            ],
        )

    def generate_answer(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        graph_context: str = "",
        conversation_context: str = "",
    ) -> str:
        assert question
        assert evidence or graph_context
        self.graph_contexts.append(graph_context)
        self.conversation_contexts.append(conversation_context)
        return "Agent 会决定何时检索。[1]"


class FakeGraphService:
    default_top_k = 5

    def search(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: list[str],
        vector_evidence: list[Evidence],
    ) -> GraphSearchResponse:
        assert query
        assert top_k == self.default_top_k
        assert document_ids == ["doc-1"]
        assert isinstance(vector_evidence, list)
        return GraphSearchResponse(
            query=query,
            relations=[
                GraphRelationEvidence(
                    rank=1,
                    source="Agent",
                    source_type="Concept",
                    relation="USES",
                    target="Retriever",
                    target_type="Component",
                    document_id="doc-1",
                    chunk_id="chunk-1",
                )
            ],
        )

    @staticmethod
    def format_relations(relations: list[GraphRelationEvidence]) -> str:
        return "\n".join(
            f"[G{item.rank}] {item.source} --{item.relation}--> {item.target}" for item in relations
        )


def make_agent(
    *,
    empty_first: bool = False,
    empty_always: bool = False,
    with_graph: bool = False,
    memory_service: ConversationMemoryService | None = None,
) -> tuple[AgentService, FakeRagService]:
    rag = FakeRagService(empty_first=empty_first, empty_always=empty_always)
    agent = AgentService(
        rag_service=rag,  # type: ignore[arg-type]
        chat_model=FakeChatModel(),
        model_name="fake-chat",
        default_top_k=5,
        default_max_retries=2,
        graph_service=FakeGraphService() if with_graph else None,  # type: ignore[arg-type]
        memory_service=memory_service,
    )
    return agent, rag


def test_agent_retrieves_grades_and_answers() -> None:
    agent, rag = make_agent()

    result = agent.ask("什么是 Agentic RAG？", document_ids=["doc-1"])

    assert result.route == "retrieve"
    assert result.retrieval_attempts == 1
    assert result.answer.endswith("[1]")
    assert rag.queries == ["什么是 Agentic RAG？"]
    assert [step.node for step in result.steps] == [
        "route_question",
        "retrieve",
        "grade_evidence",
        "generate_answer",
        "verify_citations",
    ]


def test_agent_combines_vector_and_graph_evidence() -> None:
    agent, rag = make_agent(with_graph=True)

    result = agent.ask("Agent 如何使用检索器？", document_ids=["doc-1"])

    assert len(result.evidence) == 1
    assert len(result.graph_relations) == 1
    assert result.graph_relations[0].relation == "USES"
    assert rag.graph_contexts == ["[G1] Agent --USES--> Retriever"]
    assert any("1 条图谱关系" in step.detail for step in result.steps)
    assert result.steps[-1].node == "verify_citations"


def test_agent_can_answer_with_graph_evidence_only() -> None:
    agent, rag = make_agent(empty_always=True, with_graph=True)

    result = agent.ask("Agent 如何使用检索器？", document_ids=["doc-1"])

    assert result.evidence == []
    assert len(result.graph_relations) == 1
    assert rag.graph_contexts == ["[G1] Agent --USES--> Retriever"]
    assert [step.node for step in result.steps] == [
        "route_question",
        "retrieve",
        "grade_evidence",
        "generate_answer",
        "verify_citations",
    ]


def test_agent_rewrites_and_retries_empty_retrieval() -> None:
    agent, rag = make_agent(empty_first=True)

    result = agent.ask("记忆怎么保存？", max_retries=1)

    assert result.retrieval_attempts == 2
    assert result.final_query == "LangGraph memory persistence"
    assert rag.queries == ["记忆怎么保存？", "LangGraph memory persistence"]
    assert any(step.node == "rewrite_query" for step in result.steps)


def test_agent_routes_greeting_without_retrieval() -> None:
    agent, rag = make_agent()

    result = agent.ask("你好")

    assert result.route == "direct"
    assert result.retrieval_attempts == 0
    assert result.evidence == []
    assert rag.queries == []


def test_agent_uses_persistent_history_to_contextualize_follow_up(tmp_path: Path) -> None:
    memory = ConversationMemoryService(
        store=ConversationStore(tmp_path / "memory.db"),
        chat_model=FakeChatModel(),
        max_tokens=10_000,
        keep_recent_messages=6,
        summary_max_tokens=500,
    )
    agent, rag = make_agent(memory_service=memory)

    first = agent.ask("LangGraph 是什么？", document_ids=["doc-1"])
    second = agent.ask(
        "它有什么作用？",
        document_ids=["doc-1"],
        session_id=first.session_id,
    )

    assert first.session_id
    assert second.session_id == first.session_id
    assert second.final_query == "LangGraph 有什么作用？"
    assert second.memory is not None
    assert second.memory.message_count == 4
    assert "LangGraph 是什么" in rag.conversation_contexts[-1]
    assert second.steps[0].node == "contextualize_question"


def test_agent_api_dependency_override() -> None:
    agent, _ = make_agent()
    app.dependency_overrides[get_agent_service] = lambda: agent
    try:
        response = TestClient(app).post(
            "/agent/ask",
            json={"question": "什么是 Agentic RAG？", "top_k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["steps"][0]["node"] == "route_question"


def test_agent_api_reports_upstream_model_failure_as_503() -> None:
    class UnavailableAgent:
        def ask(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise ModelError("temporary provider failure")

    app.dependency_overrides[get_agent_service] = lambda: UnavailableAgent()
    try:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/agent/ask",
            json={"question": "什么是 Agentic RAG？"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "上游模型服务暂时不可用，请稍后重试。"}


def test_clean_rewrite_removes_model_explanation() -> None:
    verbose = (
        "结合历史对话，‘它’指 LangSmith tracing。\n"
        "因此，改写后的独立问题是：\n\nLangSmith tracing 如何指定项目？"
    )

    assert AgentService._clean_rewrite(verbose) == "LangSmith tracing 如何指定项目？"
