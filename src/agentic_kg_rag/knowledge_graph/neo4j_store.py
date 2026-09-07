from __future__ import annotations

import hashlib
import threading
from typing import Any

from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import DriverError, Neo4jError

from agentic_kg_rag.knowledge_graph.models import (
    ExtractedEntity,
    ExtractedRelationship,
    GraphRelationEvidence,
    GraphStatus,
)
from agentic_kg_rag.knowledge_graph.store import GraphStoreError, LocalGraphStore


class Neo4jGraphStore:
    def __init__(
        self,
        *,
        uri: str,
        username: str,
        password: str,
        database: str,
    ) -> None:
        self.uri = uri
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._schema_ready = False
        self._schema_lock = threading.Lock()

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
        self._ensure_schema()
        entity_rows = self._entity_rows(entities, relationships)
        relationship_rows = self._relationship_rows(relationships)
        mention_rows = self._mention_rows(document_id, entities, relationship_rows)
        chunk_rows = sorted({row["chunk_id"] for row in mention_rows})
        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(
                    self._replace_document_tx,
                    document_id,
                    filename,
                    extraction_hash,
                    batches,
                    entity_rows,
                    relationship_rows,
                    mention_rows,
                    chunk_rows,
                )
        except (Neo4jError, DriverError) as error:
            code = getattr(error, "code", None) or type(error).__name__
            raise GraphStoreError(f"Neo4j 写入失败：{code}") from error
        return len({row["entity_id"] for row in mention_rows}), len(relationship_rows)

    @staticmethod
    def _replace_document_tx(
        tx: Any,
        document_id: str,
        filename: str,
        extraction_hash: str,
        batches: int,
        entities: list[dict[str, str]],
        relationships: list[dict[str, str]],
        mentions: list[dict[str, str]],
        chunk_ids: list[str],
    ) -> None:
        tx.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.document_id = $document_id DELETE r",
            document_id=document_id,
        ).consume()
        tx.run(
            "MATCH (c:Chunk {document_id: $document_id}) DETACH DELETE c",
            document_id=document_id,
        ).consume()
        tx.run(
            "MATCH (d:Document {document_id: $document_id}) DETACH DELETE d",
            document_id=document_id,
        ).consume()
        tx.run(
            """
            MERGE (d:Document {document_id: $document_id})
            SET d.filename = $filename,
                d.extraction_hash = $extraction_hash,
                d.batches = $batches
            """,
            document_id=document_id,
            filename=filename,
            extraction_hash=extraction_hash,
            batches=batches,
        ).consume()
        tx.run(
            """
            UNWIND $entities AS item
            MERGE (e:Entity {id: item.id})
            SET e.name = item.name,
                e.normalized_name = item.normalized_name,
                e.entity_type = CASE
                    WHEN coalesce(e.entity_type, 'Concept') = 'Concept' THEN item.entity_type
                    ELSE e.entity_type
                END,
                e.description = CASE
                    WHEN size(item.description) > size(coalesce(e.description, ''))
                    THEN item.description ELSE coalesce(e.description, '')
                END
            """,
            entities=entities,
        ).consume()
        tx.run(
            """
            MATCH (d:Document {document_id: $document_id})
            UNWIND $chunk_ids AS chunk_id
            MERGE (c:Chunk {id: chunk_id})
            SET c.document_id = $document_id
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            document_id=document_id,
            chunk_ids=chunk_ids,
        ).consume()
        tx.run(
            """
            UNWIND $mentions AS item
            MATCH (e:Entity {id: item.entity_id})
            MATCH (c:Chunk {id: item.chunk_id})
            MERGE (e)-[:MENTIONED_IN]->(c)
            """,
            mentions=mentions,
        ).consume()
        tx.run(
            """
            UNWIND $relationships AS item
            MATCH (source:Entity {id: item.source_id})
            MATCH (target:Entity {id: item.target_id})
            MERGE (source)-[r:RELATES_TO {key: item.key}]->(target)
            SET r.relation_type = item.relation_type,
                r.description = item.description,
                r.document_id = $document_id,
                r.chunk_id = item.chunk_id
            """,
            document_id=document_id,
            relationships=relationships,
        ).consume()
        tx.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)-[:MENTIONED_IN]->()
              AND NOT (e)-[:RELATES_TO]->()
              AND NOT ()-[:RELATES_TO]->(e)
            DELETE e
            """
        ).consume()

    def search_relations(
        self,
        text: str,
        *,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[GraphRelationEvidence]:
        self._ensure_schema()
        normalized_text = LocalGraphStore.normalize_name(text)
        tokens = LocalGraphStore._tokens(normalized_text)
        try:
            records, _, _ = self.driver.execute_query(
                "MATCH (e:Entity) RETURN e.id AS id, e.normalized_name AS normalized_name, "
                "e.description AS description",
                database_=self.database,
                routing_=RoutingControl.READ,
            )
            scored_ids: dict[str, float] = {}
            for record in records:
                name = record["normalized_name"] or ""
                description = LocalGraphStore.normalize_name(record["description"] or "")
                score = 0.0
                if name and name in normalized_text:
                    score += 10.0 + min(len(name), 40) / 40
                for token in tokens:
                    if len(token) >= 2 and token in name:
                        score += 2.0
                    elif len(token) >= 3 and token in description:
                        score += 0.25
                if score > 0:
                    scored_ids[record["id"]] = score
            if not scored_ids:
                return []
            candidate_ids = sorted(scored_ids, key=scored_ids.get, reverse=True)[:20]
            relation_records, _, _ = self.driver.execute_query(
                """
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                WHERE (source.id IN $candidate_ids OR target.id IN $candidate_ids)
                  AND (size($document_ids) = 0 OR r.document_id IN $document_ids)
                RETURN source.id AS source_id, source.name AS source,
                       source.entity_type AS source_type,
                       r.relation_type AS relation_type, r.description AS description,
                       target.id AS target_id, target.name AS target,
                       target.entity_type AS target_type,
                       r.document_id AS document_id, r.chunk_id AS chunk_id
                """,
                candidate_ids=candidate_ids,
                document_ids=document_ids or [],
                database_=self.database,
                routing_=RoutingControl.READ,
            )
        except (Neo4jError, DriverError) as error:
            code = getattr(error, "code", None) or type(error).__name__
            raise GraphStoreError(f"Neo4j 查询失败：{code}") from error

        ranked = sorted(
            relation_records,
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
                description=row["description"] or "",
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
            )
            for index, row in enumerate(ranked, 1)
        ]

    def status(self, *, mode: str = "neo4j") -> GraphStatus:
        try:
            self._ensure_schema()
            records, _, _ = self.driver.execute_query(
                """
                MATCH (e:Entity)
                WITH count(e) AS entity_count
                OPTIONAL MATCH ()-[r:RELATES_TO]->()
                WITH entity_count, count(r) AS relationship_count
                OPTIONAL MATCH (d:Document)
                RETURN entity_count, relationship_count, count(d) AS document_count
                """,
                database_=self.database,
                routing_=RoutingControl.READ,
            )
            row = records[0]
            return GraphStatus(
                mode=mode,
                backend="neo4j",
                connected=True,
                message=f"{self.uri} / {self.database}",
                entity_count=row["entity_count"],
                relationship_count=row["relationship_count"],
                document_count=row["document_count"],
            )
        except (Neo4jError, DriverError, GraphStoreError) as error:
            return GraphStatus(
                mode=mode,
                backend="neo4j",
                connected=False,
                message=f"Neo4j 未连接：{type(error).__name__}",
                entity_count=0,
                relationship_count=0,
                document_count=0,
            )

    def has_current_extraction(self, document_id: str, extraction_hash: str) -> bool:
        self._ensure_schema()
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (d:Document {document_id: $document_id})
                RETURN d.extraction_hash AS extraction_hash
                """,
                document_id=document_id,
                database_=self.database,
                routing_=RoutingControl.READ,
            )
        except (Neo4jError, DriverError) as error:
            code = getattr(error, "code", None) or type(error).__name__
            raise GraphStoreError(f"Neo4j 查询失败：{code}") from error
        return bool(records and records[0]["extraction_hash"] == extraction_hash)

    def document_counts(self, document_id: str) -> tuple[int, int, int]:
        self._ensure_schema()
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (d:Document {document_id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)<-[:MENTIONED_IN]-(e:Entity)
                WITH d, count(DISTINCT e) AS entity_count
                OPTIONAL MATCH ()-[r:RELATES_TO {document_id: $document_id}]->()
                RETURN entity_count, count(r) AS relationship_count, d.batches AS batches
                """,
                document_id=document_id,
                database_=self.database,
                routing_=RoutingControl.READ,
            )
        except (Neo4jError, DriverError) as error:
            code = getattr(error, "code", None) or type(error).__name__
            raise GraphStoreError(f"Neo4j 查询失败：{code}") from error
        if not records:
            return 0, 0, 0
        row = records[0]
        return int(row["entity_count"]), int(row["relationship_count"]), int(row["batches"])

    def list_relations(
        self,
        *,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[GraphRelationEvidence]:
        self._ensure_schema()
        try:
            records, _, _ = self.driver.execute_query(
                """
                MATCH (source:Entity)-[r:RELATES_TO]->(target:Entity)
                WHERE size($document_ids) = 0 OR r.document_id IN $document_ids
                RETURN source.name AS source, source.entity_type AS source_type,
                       r.relation_type AS relation_type, r.description AS description,
                       target.name AS target, target.entity_type AS target_type,
                       r.document_id AS document_id, r.chunk_id AS chunk_id
                LIMIT $limit
                """,
                document_ids=document_ids or [],
                limit=limit,
                database_=self.database,
                routing_=RoutingControl.READ,
            )
        except (Neo4jError, DriverError) as error:
            code = getattr(error, "code", None) or type(error).__name__
            raise GraphStoreError(f"Neo4j 查询失败：{code}") from error
        return [
            GraphRelationEvidence(
                rank=rank,
                source=row["source"],
                source_type=row["source_type"],
                relation=row["relation_type"],
                target=row["target"],
                target_type=row["target_type"],
                description=row["description"] or "",
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
            )
            for rank, row in enumerate(records, 1)
        ]

    def delete_document(self, document_id: str) -> None:
        self._ensure_schema()
        try:
            with self.driver.session(database=self.database) as session:
                session.execute_write(self._delete_document_tx, document_id)
        except (Neo4jError, DriverError) as error:
            code = getattr(error, "code", None) or type(error).__name__
            raise GraphStoreError(f"Neo4j 删除失败：{code}") from error

    @staticmethod
    def _delete_document_tx(tx: Any, document_id: str) -> None:
        tx.run(
            "MATCH ()-[r:RELATES_TO]->() WHERE r.document_id = $document_id DELETE r",
            document_id=document_id,
        ).consume()
        tx.run(
            "MATCH (c:Chunk {document_id: $document_id}) DETACH DELETE c",
            document_id=document_id,
        ).consume()
        tx.run(
            "MATCH (d:Document {document_id: $document_id}) DETACH DELETE d",
            document_id=document_id,
        ).consume()
        tx.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)-[:MENTIONED_IN]->()
              AND NOT (e)-[:RELATES_TO]->()
              AND NOT ()-[:RELATES_TO]->(e)
            DELETE e
            """
        ).consume()

    def close(self) -> None:
        self.driver.close()

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            try:
                self.driver.verify_connectivity()
                for query in (
                    (
                        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
                        "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
                    ),
                    (
                        "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
                        "FOR (d:Document) REQUIRE d.document_id IS UNIQUE"
                    ),
                    (
                        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                        "FOR (c:Chunk) REQUIRE c.id IS UNIQUE"
                    ),
                ):
                    self.driver.execute_query(query, database_=self.database)
            except (Neo4jError, DriverError) as error:
                code = getattr(error, "code", None) or type(error).__name__
                raise GraphStoreError(f"无法连接或初始化 Neo4j：{code}") from error
            self._schema_ready = True

    @staticmethod
    def _entity_rows(
        entities: list[ExtractedEntity],
        relationships: list[ExtractedRelationship],
    ) -> list[dict[str, str]]:
        entity_by_name: dict[str, ExtractedEntity] = {}
        for entity in entities:
            normalized = LocalGraphStore.normalize_name(entity.name)
            if not normalized:
                continue
            previous = entity_by_name.get(normalized)
            if previous is None or len(entity.description) > len(previous.description):
                entity_by_name[normalized] = entity
        for relation in relationships:
            for name in (relation.source, relation.target):
                normalized = LocalGraphStore.normalize_name(name)
                if normalized:
                    entity_by_name.setdefault(
                        normalized,
                        ExtractedEntity(name=name, type="Concept", description=""),
                    )
        return [
            {
                "id": LocalGraphStore.entity_id(normalized),
                "name": entity.name.strip(),
                "normalized_name": normalized,
                "entity_type": entity.entity_type.strip(),
                "description": entity.description.strip(),
            }
            for normalized, entity in entity_by_name.items()
        ]

    @staticmethod
    def _relationship_rows(
        relationships: list[ExtractedRelationship],
    ) -> list[dict[str, str]]:
        rows: dict[str, dict[str, str]] = {}
        for relation in relationships:
            source_name = LocalGraphStore.normalize_name(relation.source)
            target_name = LocalGraphStore.normalize_name(relation.target)
            relation_type = LocalGraphStore.normalize_relation(relation.relation_type)
            if not source_name or not target_name:
                continue
            raw_key = f"{source_name}|{target_name}|{relation_type}|{relation.chunk_id}"
            key = hashlib.sha256(raw_key.encode()).hexdigest()[:24]
            rows[key] = {
                "key": key,
                "source_id": LocalGraphStore.entity_id(source_name),
                "target_id": LocalGraphStore.entity_id(target_name),
                "relation_type": relation_type,
                "description": relation.description.strip(),
                "chunk_id": relation.chunk_id,
            }
        return list(rows.values())

    @staticmethod
    def _mention_rows(
        document_id: str,
        entities: list[ExtractedEntity],
        relationships: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        mentions: dict[tuple[str, str], dict[str, str]] = {}
        for entity in entities:
            normalized = LocalGraphStore.normalize_name(entity.name)
            if not normalized:
                continue
            entity_id = LocalGraphStore.entity_id(normalized)
            for chunk_id in entity.chunk_ids:
                mentions[(entity_id, chunk_id)] = {
                    "entity_id": entity_id,
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                }
        for relation in relationships:
            for entity_id in (relation["source_id"], relation["target_id"]):
                mentions[(entity_id, relation["chunk_id"])] = {
                    "entity_id": entity_id,
                    "chunk_id": relation["chunk_id"],
                    "document_id": document_id,
                }
        return list(mentions.values())
