from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from agentic_kg_rag.memory.models import (
    ConversationDetail,
    ConversationInfo,
    ConversationMessage,
)


class ConversationNotFoundError(RuntimeError):
    status_code = 404


class ConversationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    summary_token_count INTEGER NOT NULL DEFAULT 0,
                    compressed_messages INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL
                        REFERENCES conversations(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
                    ON conversation_messages(session_id, id);
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "summary_token_count" not in columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN summary_token_count "
                    "INTEGER NOT NULL DEFAULT 0"
                )

    def create(self, title: str = "新会话", session_id: str | None = None) -> ConversationDetail:
        now = self._now()
        resolved_id = session_id or f"session-{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversations
                    (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (resolved_id, title.strip() or "新会话", now, now),
            )
        return self.get(resolved_id)

    def append_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        question_tokens: int,
        answer_tokens: int,
    ) -> None:
        now = self._now()
        with self._connect() as connection:
            conversation = connection.execute(
                "SELECT title FROM conversations WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not conversation:
                raise ConversationNotFoundError(f"会话不存在：{session_id}")
            connection.executemany(
                """
                INSERT INTO conversation_messages
                    (session_id, role, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (session_id, "user", question, question_tokens, now),
                    (session_id, "assistant", answer, answer_tokens, now),
                ],
            )
            if conversation["title"] == "新会话":
                connection.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE session_id = ?",
                    (self._title(question), now, session_id),
                )
            else:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )

    def replace_summary(
        self,
        session_id: str,
        *,
        summary: str,
        summary_token_count: int,
        removed_message_ids: list[int],
    ) -> None:
        if not removed_message_ids:
            return
        now = self._now()
        placeholders = ",".join("?" for _ in removed_message_ids)
        with self._connect() as connection:
            connection.execute(
                f"DELETE FROM conversation_messages WHERE session_id = ? "
                f"AND id IN ({placeholders})",
                [session_id, *removed_message_ids],
            )
            connection.execute(
                """
                UPDATE conversations
                SET summary = ?,
                    summary_token_count = ?,
                    compressed_messages = compressed_messages + ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (summary, summary_token_count, len(removed_message_ids), now, session_id),
            )

    def list(self) -> list[ConversationInfo]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*,
                       count(m.id) AS message_count,
                       coalesce(sum(m.token_count), 0) AS message_tokens
                FROM conversations c
                LEFT JOIN conversation_messages m ON m.session_id = c.session_id
                GROUP BY c.session_id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [self._to_info(row) for row in rows]

    def get(self, session_id: str) -> ConversationDetail:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.*,
                       count(m.id) AS message_count,
                       coalesce(sum(m.token_count), 0) AS message_tokens
                FROM conversations c
                LEFT JOIN conversation_messages m ON m.session_id = c.session_id
                WHERE c.session_id = ?
                GROUP BY c.session_id
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise ConversationNotFoundError(f"会话不存在：{session_id}")
            message_rows = connection.execute(
                """
                SELECT id, role, content, token_count, created_at
                FROM conversation_messages
                WHERE session_id = ? ORDER BY id
                """,
                (session_id,),
            ).fetchall()
        info = self._to_info(row)
        return ConversationDetail(
            **info.model_dump(),
            summary=row["summary"],
            messages=[ConversationMessage(**dict(message)) for message in message_rows],
        )

    def delete(self, session_id: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE session_id = ?",
                (session_id,),
            )
        if cursor.rowcount == 0:
            raise ConversationNotFoundError(f"会话不存在：{session_id}")

    def _to_info(self, row: sqlite3.Row) -> ConversationInfo:
        return ConversationInfo(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
            token_count=int(row["message_tokens"]) + int(row["summary_token_count"]),
            compressed_messages=row["compressed_messages"],
            summary_present=bool(row["summary"]),
        )

    @staticmethod
    def _title(question: str) -> str:
        clean = " ".join(question.split())
        return clean[:42] + ("…" if len(clean) > 42 else "")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
