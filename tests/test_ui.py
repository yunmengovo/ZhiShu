from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from agentic_kg_rag.ui import (
    AGENTIC_MODE,
    ask_question,
    build_ui,
    create_new_conversation,
    extract_selected_graph,
    index_selected,
    initialize_conversations,
    refresh_workspace,
    render_agent_trace,
    render_evidence,
    render_graph_relations,
    upload_documents,
)


class FakeBackend:
    def __init__(self) -> None:
        self.documents = [
            {
                "document_id": "doc-langgraph",
                "filename": "agentic-rag.mdx",
                "file_type": "mdx",
                "title": "Build a custom RAG agent with LangGraph",
                "status": "ready",
                "created_at": "2026-09-01T00:00:00Z",
                "block_count": 10,
                "chunk_count": 4,
                "duplicate": False,
            }
        ]
        self.index_calls: list[str] = []
        self.conversations = [
            {
                "session_id": "session-test",
                "title": "新会话",
                "created_at": "2026-09-01T00:00:00Z",
                "updated_at": "2026-09-01T00:00:00Z",
                "message_count": 0,
                "token_count": 0,
                "compressed_messages": 0,
                "summary_present": False,
                "summary": "",
                "messages": [],
                "max_tokens": 220000,
                "context_window_tokens": 256000,
            }
        ]

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def rag_status(self) -> dict[str, Any]:
        return {
            "mode": "local",
            "collection": "document_chunks",
            "collection_ready": True,
            "indexed_chunks": 4,
            "chat_configured": True,
            "embedding_configured": True,
        }

    def graph_status(self) -> dict[str, Any]:
        return {
            "mode": "local",
            "entity_count": 2,
            "relationship_count": 1,
            "document_count": 1,
        }

    def integration_status(self) -> dict[str, bool]:
        return {
            "mineru_configured": False,
            "tavily_configured": True,
            "langsmith_enabled": True,
        }

    def list_knowledge_bases(self) -> list[dict[str, Any]]:
        return [
            {
                "knowledge_base_id": "kb-default",
                "name": "默认知识空间",
                "description": "",
                "document_count": len(self.documents),
            }
        ]

    def list_documents(
        self,
        knowledge_base_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self.documents

    def list_assets(
        self,
        knowledge_base_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                **document,
                "knowledge_base_id": knowledge_base_id or "kb-default",
                "indexed_chunks": document["chunk_count"],
                "vector_status": "ready",
                "entity_count": 2,
                "relationship_count": 1,
                "graph_status": "ready",
            }
            for document in self.documents
        ]

    def upload_documents_batch(
        self,
        paths: list[str],
        *,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        assert knowledge_base_id == "kb-default"
        assert [Path(path).name for path in paths] == ["notes.md"]
        document = {
            "document_id": "doc-notes",
            "knowledge_base_id": knowledge_base_id,
            "filename": "notes.md",
            "file_type": "md",
            "title": "Notes",
            "status": "ready",
            "created_at": "2026-09-01T00:00:00Z",
            "block_count": 2,
            "chunk_count": 2,
            "duplicate": False,
        }
        return {"results": [{"filename": "notes.md", "success": True, "document": document}]}

    def graph_subgraph(
        self,
        *,
        query: str = "",
        document_ids: list[str] | None = None,
        limit: int = 80,
    ) -> dict[str, Any]:
        return {
            "query": query,
            "nodes": [
                {"node_id": "n1", "label": "LangChain", "entity_type": "Framework"},
                {"node_id": "n2", "label": "LangGraph", "entity_type": "Framework"},
            ],
            "edges": [
                {
                    "edge_id": "e1",
                    "source_id": "n1",
                    "target_id": "n2",
                    "relation": "USES",
                    "description": "",
                    "document_id": "doc-langgraph",
                    "chunk_id": "chunk-1",
                }
            ],
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        return []

    def index_document(self, document_id: str) -> dict[str, Any]:
        self.index_calls.append(document_id)
        filename = "notes.md" if document_id == "doc-notes" else "agentic-rag.mdx"
        chunks = 2 if document_id == "doc-notes" else 4
        return {
            "document_id": document_id,
            "filename": filename,
            "collection": "document_chunks",
            "indexed_chunks": chunks,
        }

    def extract_graph(self, document_id: str) -> dict[str, Any]:
        return {
            "document_id": document_id,
            "filename": "agentic-rag.mdx",
            "batches": 1,
            "entity_count": 2,
            "relationship_count": 1,
        }

    def ask(
        self,
        question: str,
        *,
        top_k: int,
        document_ids: list[str],
        knowledge_base_id: str | None = None,
    ) -> dict[str, Any]:
        assert question == "什么是 Agentic RAG？"
        assert top_k == 5
        assert document_ids == ["doc-langgraph"]
        return {
            "question": question,
            "answer": "Agent 可以决定何时进行检索。[1]",
            "model": "fake-chat",
            "evidence": [
                {
                    "rank": 1,
                    "score": 0.8123,
                    "document_id": "doc-langgraph",
                    "chunk_id": "chunk-1",
                    "filename": "agentic-rag.mdx",
                    "title": "Build a custom RAG agent with LangGraph",
                    "text": "An agent decides when to retrieve evidence.",
                    "section_title": "Overview",
                    "page_start": None,
                    "page_end": None,
                    "metadata": {},
                }
            ],
        }

    def agent_ask(
        self,
        question: str,
        *,
        top_k: int,
        document_ids: list[str],
        session_id: str | None = None,
        knowledge_base_id: str | None = None,
    ) -> dict[str, Any]:
        result = self.ask(
            question,
            top_k=top_k,
            document_ids=document_ids,
            knowledge_base_id=knowledge_base_id,
        )
        return {
            **result,
            "route": "retrieve",
            "final_query": question,
            "retrieval_attempts": 1,
            "steps": [
                {
                    "node": "route_question",
                    "action": "retrieve",
                    "detail": "需要查询知识库",
                },
                {
                    "node": "retrieve",
                    "action": "search",
                    "detail": "命中 1 条证据",
                },
            ],
            "graph_relations": [
                {
                    "rank": 1,
                    "source": "LangChain",
                    "source_type": "Framework",
                    "relation": "BUILT_ON",
                    "target": "LangGraph",
                    "target_type": "Framework",
                    "description": "Agents use LangGraph.",
                    "document_id": "doc-langgraph",
                    "chunk_id": "chunk-1",
                }
            ],
            "session_id": session_id or "session-test",
            "memory": {
                "session_id": session_id or "session-test",
                "message_count": 2,
                "token_count": 42,
                "compressed_messages": 0,
                "summary_present": False,
                "compressed_this_turn": False,
                "max_tokens": 220000,
                "context_window_tokens": 256000,
            },
        }

    def create_conversation(self, title: str = "新会话") -> dict[str, Any]:
        conversation = {
            **self.conversations[0],
            "session_id": f"session-{len(self.conversations) + 1}",
            "title": title,
        }
        self.conversations.insert(0, conversation)
        return conversation

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.conversations

    def get_conversation(self, session_id: str) -> dict[str, Any]:
        return next(item for item in self.conversations if item["session_id"] == session_id)

    def delete_conversation(self, session_id: str) -> None:
        self.conversations = [
            item for item in self.conversations if item["session_id"] != session_id
        ]


def test_refresh_workspace_and_document_choices() -> None:
    status, base_update, qa_base_update, rows, index_update, scope_update, graph, jobs = (
        refresh_workspace(FakeBackend())
    )

    assert "1 份文档" in status
    assert "1 个空间" in status
    assert base_update["value"] == "kb-default"
    assert qa_base_update["value"] == "kb-default"
    assert rows[0][-1] == "doc-langgraph"
    assert index_update["choices"][0][1] == "doc-langgraph"
    assert scope_update["choices"] == index_update["choices"]
    assert "LangChain" in graph
    assert "暂无后台任务" in jobs


def test_upload_and_index_selected(tmp_path: Path) -> None:
    backend = FakeBackend()
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Notes", encoding="utf-8")

    operation, *_ = upload_documents(
        [str(file_path)],
        True,
        False,
        False,
        "kb-default",
        backend,
    )
    index_operation, *_ = index_selected(["doc-langgraph"], "kb-default", backend)
    graph_operation, *_ = extract_selected_graph(
        ["doc-langgraph"],
        "kb-default",
        backend,
    )

    assert "已索引 2 个分块" in operation
    assert "agentic-rag.mdx" in index_operation
    assert "2 个实体" in graph_operation
    assert backend.index_calls == ["doc-notes", "doc-langgraph"]


def test_ask_question_updates_chat_and_evidence() -> None:
    history, cleared, evidence, conversation_update, memory = ask_question(
        "什么是 Agentic RAG？",
        [],
        ["doc-langgraph"],
        5,
        AGENTIC_MODE,
        "session-test",
        "kb-default",
        FakeBackend(),
    )

    assert history[0] == {"role": "user", "content": "什么是 Agentic RAG？"}
    assert history[1]["content"].endswith("[1]")
    assert cleared == ""
    assert "相似度 `0.8123`" in evidence
    assert "Overview" in evidence
    assert "route_question" in evidence
    assert "BUILT_ON" in evidence
    assert conversation_update["value"] == "session-test"
    assert "42" in memory
    assert "220,000" in memory
    assert "256,000" in memory
    assert "36,000" in memory


def test_initialize_and_create_conversation() -> None:
    backend = FakeBackend()
    selector, history, memory = initialize_conversations(backend)
    created, created_history, *_ = create_new_conversation(backend)

    assert selector["value"] == "session-test"
    assert history == []
    assert "256,000" in memory
    assert created["value"] == "session-2"
    assert created_history == []


def test_render_empty_evidence_and_build_ui() -> None:
    assert "没有检索到" in render_evidence([])
    assert "基础固定链路" in render_agent_trace({}, "基础 RAG")
    assert "没有命中" in render_graph_relations([])
    demo = build_ui(FakeBackend())
    assert isinstance(demo, gr.Blocks)
    assert len(demo.get_config_file()["dependencies"]) >= 6
