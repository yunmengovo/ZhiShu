from __future__ import annotations

from typing import Any

from langchain_core.exceptions import ModelError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from agentic_kg_rag.memory.models import (
    ConversationDetail,
    ConversationInfo,
    MemorySnapshot,
)
from agentic_kg_rag.memory.store import ConversationStore


class ConversationMemoryService:
    def __init__(
        self,
        *,
        store: ConversationStore,
        chat_model: Any,
        max_tokens: int,
        keep_recent_messages: int,
        summary_max_tokens: int,
        context_window_tokens: int | None = None,
    ) -> None:
        self.store = store
        self.chat_model = chat_model
        self.max_tokens = max_tokens
        self.context_window_tokens = context_window_tokens or max_tokens
        self.keep_recent_messages = max(2, keep_recent_messages)
        self.summary_max_tokens = summary_max_tokens

    def create(self, title: str = "新会话") -> ConversationDetail:
        return self._with_limit(self.store.create(title))

    def ensure(self, session_id: str | None) -> ConversationDetail:
        detail = self.store.get(session_id) if session_id else self.store.create()
        return self._with_limit(detail)

    def list(self) -> list[ConversationInfo]:
        return self.store.list()

    def get(self, session_id: str) -> ConversationDetail:
        return self._with_limit(self.store.get(session_id))

    def delete(self, session_id: str) -> None:
        self.store.delete(session_id)

    def context(self, session_id: str) -> str:
        conversation = self.store.get(session_id)
        sections: list[str] = []
        if conversation.summary:
            sections.append(f"较早对话摘要：\n{conversation.summary}")
        if conversation.messages:
            transcript = "\n".join(
                f"{'用户' if item.role == 'user' else '助手'}：{item.content}"
                for item in conversation.messages
            )
            sections.append(f"近期对话：\n{transcript}")
        return "\n\n".join(sections)

    def append_turn(self, session_id: str, question: str, answer: str) -> MemorySnapshot:
        self.store.append_turn(
            session_id,
            question=question,
            answer=answer,
            question_tokens=self._count_text(question),
            answer_tokens=self._count_text(answer),
        )
        compression_warning = ""
        try:
            compressed = self._compress_if_needed(session_id)
        except ModelError:
            compressed = False
            compression_warning = "达到压缩阈值，但摘要模型本轮不可用；消息已完整保留。"
        detail = self.store.get(session_id)
        return MemorySnapshot(
            session_id=session_id,
            message_count=detail.message_count,
            token_count=detail.token_count,
            compressed_messages=detail.compressed_messages,
            summary_present=detail.summary_present,
            compressed_this_turn=compressed,
            max_tokens=self.max_tokens,
            context_window_tokens=self.context_window_tokens,
            compression_warning=compression_warning,
        )

    def snapshot(self, session_id: str) -> MemorySnapshot:
        detail = self.store.get(session_id)
        return MemorySnapshot(
            session_id=session_id,
            message_count=detail.message_count,
            token_count=detail.token_count,
            compressed_messages=detail.compressed_messages,
            summary_present=detail.summary_present,
            max_tokens=self.max_tokens,
            context_window_tokens=self.context_window_tokens,
        )

    def _compress_if_needed(self, session_id: str) -> bool:
        conversation = self.store.get(session_id)
        if conversation.token_count <= self.max_tokens:
            return False
        old_messages = conversation.messages[: -self.keep_recent_messages]
        if not old_messages:
            return False
        transcript = "\n".join(
            f"{'用户' if item.role == 'user' else '助手'}：{item.content}" for item in old_messages
        )
        response = self.chat_model.invoke(
            [
                SystemMessage(
                    content=(
                        "你是会话记忆压缩器。把较早对话更新为简洁、可继续使用的滚动摘要。"
                        "保留用户目标、已确认事实、偏好、文档范围、专有名词、关键结论和未解决问题；"
                        "删除寒暄、重复表达和冗余细节。不得加入对话中不存在的信息。只输出摘要正文。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"已有摘要：\n{conversation.summary or '无'}\n\n"
                        f"需要并入摘要的较早消息：\n{transcript}\n\n"
                        f"摘要目标不超过约 {self.summary_max_tokens} tokens。"
                    )
                ),
            ]
        )
        summary = response.content if isinstance(response.content, str) else str(response.content)
        summary = self._truncate_summary(summary.strip())
        self.store.replace_summary(
            session_id,
            summary=summary,
            summary_token_count=self._count_text(summary),
            removed_message_ids=[item.id for item in old_messages],
        )
        return True

    def _truncate_summary(self, summary: str) -> str:
        while summary and self._count_text(summary) > self.summary_max_tokens:
            summary = summary[: max(1, int(len(summary) * 0.85))].rstrip()
        return summary

    def _with_limit(self, detail: ConversationDetail) -> ConversationDetail:
        detail.max_tokens = self.max_tokens
        detail.context_window_tokens = self.context_window_tokens
        return detail

    @staticmethod
    def _count_text(text: str) -> int:
        return count_tokens_approximately([HumanMessage(content=text)])
