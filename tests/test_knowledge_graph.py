from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from neo4j.exceptions import ServiceUnavailable

from agentic_kg_rag.api import app
from agentic_kg_rag.ingestion.service import DocumentIngestionService
from agentic_kg_rag.knowledge_graph.neo4j_store import Neo4jGraphStore
from agentic_kg_rag.knowledge_graph.router import get_knowledge_graph_service
from agentic_kg_rag.knowledge_graph.service import KnowledgeGraphService
from agentic_kg_rag.knowledge_graph.store import LocalGraphStore


class FakeExtractionChat:
    def invoke(self, messages: list[Any]) -> AIMessage:
        user_text = str(messages[-1].content)
        chunk_id = re.search(r'<chunk id="([^"]+)">', user_text).group(1)  # type: ignore[union-attr]
        return AIMessage(
            content=json.dumps(
                {
                    "entities": [
                        {
                            "name": "LangGraph",
                            "type": "Framework",
                            "description": "A graph runtime for agents.",
                            "chunk_ids": [chunk_id],
                        },
                        {
                            "name": "LangChain",
                            "type": "Framework",
                            "description": "An agent framework.",
                            "chunk_ids": [chunk_id],
                        },
                    ],
                    "relationships": [
                        {
                            "source": "LangChain",
                            "target": "LangGraph",
                            "type": "BUILT_ON",
                            "description": "LangChain agents are built on LangGraph.",
                            "chunk_id": chunk_id,
                        }
                    ],
                }
            )
        )


def make_graph_service(tmp_path: Path) -> tuple[DocumentIngestionService, KnowledgeGraphService]:
    ingestion = DocumentIngestionService(
        data_dir=tmp_path / "data",
        chunk_size=200,
        chunk_overlap=20,
        max_upload_bytes=1024 * 1024,
    )
    service = KnowledgeGraphService(
        ingestion_service=ingestion,
        store=LocalGraphStore(tmp_path / "graph.db"),
        chat_model=FakeExtractionChat(),
        model_name="fake-chat",
        mode="local",
        batch_chars=1000,
        default_top_k=5,
    )
    return ingestion, service


def add_document(ingestion: DocumentIngestionService) -> str:
    summary = ingestion.ingest_bytes(
        filename="agents.md",
        content=(
            b"# Agents\n\nLangChain agents are built on LangGraph. "
            b"LangGraph provides graph orchestration."
        ),
    )
    return summary.document_id


def test_extract_store_search_and_idempotency(tmp_path: Path) -> None:
    ingestion, service = make_graph_service(tmp_path)
    document_id = add_document(ingestion)

    first = service.extract_document(document_id)
    second = service.extract_document(document_id)
    search = service.search("LangChain 和 LangGraph 有什么关系？")
    status = service.status()

    assert first.entity_count == 2
    assert first.relationship_count == 1
    assert second == first
    assert search.relations[0].relation == "BUILT_ON"
    assert search.relations[0].chunk_id.startswith("chunk-")
    assert status.entity_count == 2
    assert status.relationship_count == 1
    assert status.document_count == 1


def test_graph_api_dependency_override(tmp_path: Path) -> None:
    ingestion, service = make_graph_service(tmp_path)
    document_id = add_document(ingestion)
    app.dependency_overrides[get_knowledge_graph_service] = lambda: service
    try:
        extraction = TestClient(app).post(f"/graph/extract/{document_id}")
        search = TestClient(app).post(
            "/graph/search",
            json={"query": "LangGraph", "top_k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert extraction.status_code == 200
    assert search.status_code == 200
    assert search.json()["relations"][0]["source"] == "LangChain"


def test_neo4j_status_reports_disconnected_without_crashing(monkeypatch: Any) -> None:
    class OfflineDriver:
        def verify_connectivity(self) -> None:
            raise ServiceUnavailable("offline")

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        "agentic_kg_rag.knowledge_graph.neo4j_store.GraphDatabase.driver",
        lambda *args, **kwargs: OfflineDriver(),
    )
    store = Neo4jGraphStore(
        uri="bolt://127.0.0.1:7687",
        username="neo4j",
        password="password",
        database="neo4j",
    )

    status = store.status()

    assert status.backend == "neo4j"
    assert status.connected is False
    assert status.entity_count == 0


def test_neo4j_maps_extracted_entities_and_relationships() -> None:
    entities = [
        {
            "name": "LangGraph",
            "type": "Framework",
            "description": "Graph runtime",
            "chunk_ids": ["chunk-1"],
        }
    ]
    relationships = [
        {
            "source": "LangChain",
            "target": "LangGraph",
            "type": "built on",
            "description": "Agent runtime",
            "chunk_id": "chunk-1",
        }
    ]
    from agentic_kg_rag.knowledge_graph.models import (
        ExtractedEntity,
        ExtractedRelationship,
    )

    entity_rows = Neo4jGraphStore._entity_rows(
        [ExtractedEntity.model_validate(item) for item in entities],
        [ExtractedRelationship.model_validate(item) for item in relationships],
    )
    relation_rows = Neo4jGraphStore._relationship_rows(
        [ExtractedRelationship.model_validate(item) for item in relationships]
    )

    assert {item["name"] for item in entity_rows} == {"LangChain", "LangGraph"}
    assert relation_rows[0]["relation_type"] == "BUILT_ON"
    assert relation_rows[0]["chunk_id"] == "chunk-1"
