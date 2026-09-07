from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Protocol

from agentic_kg_rag.knowledge_graph.models import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphRelationEvidence,
    GraphStatus,
)


class GraphStoreError(RuntimeError):
    pass


class GraphStore(Protocol):
    def replace_document(
        self,
        *,
        document_id: str,
        filename: str,
        extraction_hash: str,
        batches: int,
        entities: list[ExtractedEntity],
        relationships: list[ExtractedRelationship],
    ) -> tuple[int, int]: ...

    def search_relations(
        self,
        text: str,
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[GraphRelationEvidence]: ...

    def list_relations(
        self,
        *,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[GraphRelationEvidence]: ...

    def delete_document(self, document_id: str) -> None: ...

    def status(self, *, mode: str) -> GraphStatus: ...

    def has_current_extraction(self, document_id: str, extraction_hash: str) -> bool: ...

    def document_counts(self, document_id: str) -> tuple[int, int, int]: ...

    def close(self) -> None: ...


class LocalGraphStore:
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
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS mentions (
                    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    PRIMARY KEY (entity_id, document_id, chunk_id)
                );
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    document_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    UNIQUE (source_id, target_id, relation_type, document_id, chunk_id)
                );
                CREATE TABLE IF NOT EXISTS graph_documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    extraction_hash TEXT NOT NULL,
                    batches INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_mentions_document ON mentions(document_id);
                CREATE INDEX IF NOT EXISTS idx_relationships_document
                    ON relationships(document_id);
                """
            )

    def replace_document(
        self,
        *,
        document_id: str,
        filename: str,
        extraction_hash: str,
        batches: int,
        entities: list[ExtractedEntity],
        relationships: list[ExtractedRelationship],
    ) -> tuple[int, int]:
        with self._connect() as connection:
            connection.execute("DELETE FROM relationships WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM mentions WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM graph_documents WHERE document_id = ?", (document_id,))

            entity_by_name: dict[str, ExtractedEntity] = {}
            for entity in entities:
                normalized = self.normalize_name(entity.name)
                if not normalized:
                    continue
                previous = entity_by_name.get(normalized)
                if previous is None or len(entity.description) > len(previous.description):
                    entity_by_name[normalized] = entity
            for relation in relationships:
                for name in (relation.source, relation.target):
                    normalized = self.normalize_name(name)
                    entity_by_name.setdefault(
                        normalized,
                        ExtractedEntity(name=name, type="Concept", description=""),
                    )

            entity_ids: dict[str, str] = {}
            for normalized, entity in entity_by_name.items():
                entity_id = self.entity_id(normalized)
                entity_ids[normalized] = entity_id
                connection.execute(
                    """
                    INSERT INTO entities (id, name, normalized_name, entity_type, description)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name = excluded.name,
                        entity_type = CASE
                            WHEN entities.entity_type = 'Concept' THEN excluded.entity_type
                            ELSE entities.entity_type
                        END,
                        description = CASE
                            WHEN length(excluded.description) > length(entities.description)
                            THEN excluded.description ELSE entities.description
                        END
                    """,
                    (
                        entity_id,
                        entity.name.strip(),
                        normalized,
                        entity.entity_type.strip(),
                        entity.description.strip(),
                    ),
                )
                for chunk_id in entity.chunk_ids:
                    connection.execute(
                        "INSERT OR IGNORE INTO mentions VALUES (?, ?, ?)",
                        (entity_id, document_id, chunk_id),
                    )

            inserted_relationships = 0
            for relation in relationships:
                source_id = entity_ids[self.normalize_name(relation.source)]
                target_id = entity_ids[self.normalize_name(relation.target)]
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO relationships
                        (source_id, target_id, relation_type, description, document_id, chunk_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_id,
                        target_id,
                        self.normalize_relation(relation.relation_type),
                        relation.description.strip(),
                        document_id,
                        relation.chunk_id,
                    ),
                )
                inserted_relationships += cursor.rowcount
                for entity_id in (source_id, target_id):
                    connection.execute(
                        "INSERT OR IGNORE INTO mentions VALUES (?, ?, ?)",
                        (entity_id, document_id, relation.chunk_id),
                    )

            connection.execute(
                "INSERT INTO graph_documents VALUES (?, ?, ?, ?)",
                (document_id, filename, extraction_hash, batches),
            )
            connection.execute(
                "DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM mentions) "
                "AND id NOT IN (SELECT source_id FROM relationships) "
                "AND id NOT IN (SELECT target_id FROM relationships)"
            )
            entity_count = connection.execute(
                "SELECT COUNT(DISTINCT entity_id) FROM mentions WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            return int(entity_count), inserted_relationships

    def search_relations(
        self,
        text: str,
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[GraphRelationEvidence]:
        normalized_text = self.normalize_name(text)
        tokens = self._tokens(normalized_text)
        with self._connect() as connection:
            entities = connection.execute(
                "SELECT id, normalized_name, description FROM entities"
            ).fetchall()
            scored_ids: dict[str, float] = {}
            for entity in entities:
                name = entity["normalized_name"]
                description = self.normalize_name(entity["description"])
                score = 0.0
                if name and name in normalized_text:
                    score += 10.0 + min(len(name), 40) / 40
                for token in tokens:
                    if len(token) >= 2 and token in name:
                        score += 2.0
                    elif len(token) >= 3 and token in description:
                        score += 0.25
                if score > 0:
                    scored_ids[entity["id"]] = score
            if not scored_ids:
                return []

            candidates = sorted(scored_ids, key=scored_ids.get, reverse=True)[:20]
            placeholders = ",".join("?" for _ in candidates)
            parameters: list[object] = [*candidates, *candidates]
            document_clause = ""
            if document_ids:
                document_placeholders = ",".join("?" for _ in document_ids)
                document_clause = f" AND r.document_id IN ({document_placeholders})"
                parameters.extend(document_ids)
            rows = connection.execute(
                f"""
                SELECT r.*, source.name AS source, source.entity_type AS source_type,
                       target.name AS target, target.entity_type AS target_type
                FROM relationships r
                JOIN entities source ON source.id = r.source_id
                JOIN entities target ON target.id = r.target_id
                WHERE (r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders}))
                {document_clause}
                """,
                parameters,
            ).fetchall()
        ranked = sorted(
            rows,
            key=lambda row: max(
                scored_ids.get(row["source_id"], 0),
                scored_ids.get(row["target_id"], 0),
            ),
            reverse=True,
        )[:top_k]
        return [
            GraphRelationEvidence(
                rank=index,
                source=row["source"],
                source_type=row["source_type"],
                relation=row["relation_type"],
                target=row["target"],
                target_type=row["target_type"],
                description=row["description"],
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
            )
            for index, row in enumerate(ranked, 1)
        ]

    def status(self, *, mode: str = "local") -> GraphStatus:
        with self._connect() as connection:
            entity_count = connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            relationship_count = connection.execute(
                "SELECT COUNT(*) FROM relationships"
            ).fetchone()[0]
            document_count = connection.execute("SELECT COUNT(*) FROM graph_documents").fetchone()[
                0
            ]
        return GraphStatus(
            mode=mode,
            backend="sqlite",
            connected=True,
            message=str(self.path),
            entity_count=entity_count,
            relationship_count=relationship_count,
            document_count=document_count,
        )

    def list_relations(
        self,
        *,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[GraphRelationEvidence]:
        parameters: list[object] = []
        document_clause = ""
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            document_clause = f"WHERE r.document_id IN ({placeholders})"
            parameters.extend(document_ids)
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT r.*, source.name AS source, source.entity_type AS source_type,
                       target.name AS target, target.entity_type AS target_type
                FROM relationships r
                JOIN entities source ON source.id = r.source_id
                JOIN entities target ON target.id = r.target_id
                {document_clause}
                ORDER BY r.id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
        return [self._relation_from_row(row, rank) for rank, row in enumerate(rows, 1)]

    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM relationships WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM mentions WHERE document_id = ?", (document_id,))
            connection.execute("DELETE FROM graph_documents WHERE document_id = ?", (document_id,))
            connection.execute(
                "DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM mentions) "
                "AND id NOT IN (SELECT source_id FROM relationships) "
                "AND id NOT IN (SELECT target_id FROM relationships)"
            )

    def close(self) -> None:
        return None

    def has_current_extraction(self, document_id: str, extraction_hash: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT extraction_hash FROM graph_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return bool(row and row["extraction_hash"] == extraction_hash)

    def document_counts(self, document_id: str) -> tuple[int, int, int]:
        with self._connect() as connection:
            entity_count = connection.execute(
                "SELECT COUNT(DISTINCT entity_id) FROM mentions WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            relationship_count = connection.execute(
                "SELECT COUNT(*) FROM relationships WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            row = connection.execute(
                "SELECT batches FROM graph_documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return int(entity_count), int(relationship_count), int(row["batches"] if row else 0)

    @staticmethod
    def _relation_from_row(row: sqlite3.Row, rank: int) -> GraphRelationEvidence:
        return GraphRelationEvidence(
            rank=rank,
            source=row["source"],
            source_type=row["source_type"],
            relation=row["relation_type"],
            target=row["target"],
            target_type=row["target_type"],
            description=row["description"],
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
        )

    @staticmethod
    def normalize_name(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return re.sub(r"\s+", " ", value).strip(" \t\n.,;:，。；：()（）[]【】")

    @staticmethod
    def normalize_relation(value: str) -> str:
        return re.sub(r"\s+", "_", value.strip()).upper()[:100]

    @staticmethod
    def entity_id(normalized_name: str) -> str:
        digest = hashlib.sha256(normalized_name.encode()).hexdigest()[:20]
        return f"entity-{digest}"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_.-]+|[\u4e00-\u9fff]{2,}", text)
        }
