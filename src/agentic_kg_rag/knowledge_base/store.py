from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from agentic_kg_rag.knowledge_base.models import (
    DEFAULT_KNOWLEDGE_BASE_ID,
    KnowledgeBase,
)


class KnowledgeBaseError(ValueError):
    status_code = 400


class KnowledgeBaseNotFoundError(KnowledgeBaseError):
    status_code = 404


class KnowledgeBaseConflictError(KnowledgeBaseError):
    status_code = 409


class KnowledgeBaseStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    knowledge_base_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_bases
                    (knowledge_base_id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_KNOWLEDGE_BASE_ID,
                    "默认知识空间",
                    "未指定知识空间的文档会保存在这里。",
                    now,
                    now,
                ),
            )

    def create(self, name: str, description: str = "") -> KnowledgeBase:
        clean_name = self._clean_name(name)
        now = datetime.now(UTC)
        knowledge_base_id = f"kb-{uuid.uuid4().hex[:16]}"
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO knowledge_bases VALUES (?, ?, ?, ?, ?)",
                    (
                        knowledge_base_id,
                        clean_name,
                        description.strip(),
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise KnowledgeBaseConflictError(f"知识空间名称已存在：{clean_name}") from error
        return self.get(knowledge_base_id)

    def get(self, knowledge_base_id: str) -> KnowledgeBase:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_bases WHERE knowledge_base_id = ?",
                (knowledge_base_id,),
            ).fetchone()
        if row is None:
            raise KnowledgeBaseNotFoundError(f"知识空间不存在：{knowledge_base_id}")
        return self._to_model(row)

    def list(self) -> list[KnowledgeBase]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM knowledge_bases ORDER BY created_at ASC"
            ).fetchall()
        return [self._to_model(row) for row in rows]

    def update(
        self,
        knowledge_base_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> KnowledgeBase:
        current = self.get(knowledge_base_id)
        next_name = self._clean_name(name) if name is not None else current.name
        next_description = description.strip() if description is not None else current.description
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE knowledge_bases
                    SET name = ?, description = ?, updated_at = ?
                    WHERE knowledge_base_id = ?
                    """,
                    (next_name, next_description, now, knowledge_base_id),
                )
        except sqlite3.IntegrityError as error:
            raise KnowledgeBaseConflictError(f"知识空间名称已存在：{next_name}") from error
        return self.get(knowledge_base_id)

    def delete(self, knowledge_base_id: str) -> None:
        if knowledge_base_id == DEFAULT_KNOWLEDGE_BASE_ID:
            raise KnowledgeBaseConflictError("默认知识空间不能删除，可以删除其中的文档")
        self.get(knowledge_base_id)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM knowledge_bases WHERE knowledge_base_id = ?",
                (knowledge_base_id,),
            )

    @staticmethod
    def _clean_name(value: str) -> str:
        clean = " ".join(value.split())
        if not clean:
            raise KnowledgeBaseError("知识空间名称不能为空")
        if len(clean) > 80:
            raise KnowledgeBaseError("知识空间名称不能超过 80 个字符")
        return clean

    @staticmethod
    def _to_model(row: sqlite3.Row) -> KnowledgeBase:
        return KnowledgeBase(
            knowledge_base_id=row["knowledge_base_id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
