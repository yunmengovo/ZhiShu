from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agentic_kg_rag.api import app
from agentic_kg_rag.memory.router import get_memory_service
from agentic_kg_rag.memory.service import ConversationMemoryService
from agentic_kg_rag.memory.store import ConversationStore


class FakeSummaryModel:
    def invoke(self, messages: list[Any]) -> AIMessage:
        assert "记忆压缩器" in str(messages[0].content)
        return AIMessage(content="用户正在讨论 LangGraph，并关注其会话记忆能力。")


def make_memory(tmp_path: Any) -> ConversationMemoryService:
    return ConversationMemoryService(
        store=ConversationStore(tmp_path / "memory.db"),
        chat_model=FakeSummaryModel(),
        max_tokens=30,
        keep_recent_messages=2,
        summary_max_tokens=20,
    )


def test_memory_persists_and_compresses_old_messages(tmp_path: Any) -> None:
    memory = make_memory(tmp_path)
    conversation = memory.create()

    memory.append_turn(
        conversation.session_id, "请介绍 LangGraph 的状态图和节点", "状态图包含多个节点。"
    )
    snapshot = memory.append_turn(
        conversation.session_id,
        "再详细解释它如何保存很长的会话历史",
        "可以持久化消息，并在达到上下文阈值后摘要旧消息。",
    )
    detail = memory.get(conversation.session_id)

    assert snapshot.compressed_this_turn is True
    assert snapshot.compressed_messages == 2
    assert detail.summary_present is True
    assert detail.message_count == 2
    assert detail.title == "请介绍 LangGraph 的状态图和节点"
    assert "LangGraph" in detail.summary
    assert "较早对话摘要" in memory.context(conversation.session_id)


def test_custom_conversation_title_is_preserved(tmp_path: Any) -> None:
    memory = make_memory(tmp_path)
    conversation = memory.create("我的研究笔记")

    memory.append_turn(conversation.session_id, "请解释 RAG", "RAG 会先检索再生成。")

    assert memory.get(conversation.session_id).title == "我的研究笔记"


def test_memory_api_create_list_get_delete(tmp_path: Any) -> None:
    memory = make_memory(tmp_path)
    app.dependency_overrides[get_memory_service] = lambda: memory
    client = TestClient(app)
    try:
        created = client.post("/memory/conversations", json={"title": "测试会话"})
        session_id = created.json()["session_id"]
        listed = client.get("/memory/conversations")
        loaded = client.get(f"/memory/conversations/{session_id}")
        deleted = client.delete(f"/memory/conversations/{session_id}")
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 200
    assert listed.json()[0]["title"] == "测试会话"
    assert loaded.json()["max_tokens"] == 30
    assert deleted.status_code == 204
