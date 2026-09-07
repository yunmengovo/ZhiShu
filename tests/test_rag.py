from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

from agentic_kg_rag.api import app
from agentic_kg_rag.ingestion.service import DocumentIngestionService
from agentic_kg_rag.rag.router import get_rag_service
from agentic_kg_rag.rag.service import RagService


class FakeEmbeddings:
    terms = ("南湖", "王蕾", "备份")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [float(text.count(term)) for term in self.terms]
        return vector if any(vector) else [0.01, 0.01, 0.01]


class FakeChatModel:
    def invoke(self, messages: list[object]) -> AIMessage:
        assert "王蕾负责技术工作" in str(messages[-1].content)
        return AIMessage(content="Atlas 的技术负责人是王蕾。[1]")


def make_rag_service(tmp_path: Path) -> tuple[DocumentIngestionService, RagService]:
    ingestion = DocumentIngestionService(
        data_dir=tmp_path,
        chunk_size=120,
        chunk_overlap=20,
        max_upload_bytes=1024 * 1024,
    )
    rag = RagService(
        ingestion_service=ingestion,
        qdrant=QdrantClient(":memory:"),
        embeddings=FakeEmbeddings(),
        chat_model=FakeChatModel(),
        collection_name="test_chunks",
        default_top_k=3,
        model_name="fake-chat",
        mode="memory",
        chat_configured=True,
        embedding_configured=True,
    )
    return ingestion, rag


def add_atlas_document(ingestion: DocumentIngestionService) -> str:
    summary = ingestion.ingest_bytes(
        filename="atlas.md",
        content=(
            "# Atlas 使用说明\n\nAtlas 部署在南湖示范工厂。\n\n## 负责人\n\n王蕾负责技术工作。\n"
        ).encode(),
        content_type="text/markdown",
    )
    return summary.document_id


def test_index_search_and_grounded_answer(tmp_path: Path) -> None:
    ingestion, rag = make_rag_service(tmp_path)
    document_id = add_atlas_document(ingestion)

    indexed = rag.index_document(document_id)
    result = rag.search("谁是王蕾？", top_k=1)
    answer = rag.ask("谁负责 Atlas 的技术工作？", top_k=1)
    status = rag.status()

    assert indexed.indexed_chunks >= 1
    assert result.evidence[0].document_id == document_id
    assert "王蕾" in result.evidence[0].text
    assert answer.answer == "Atlas 的技术负责人是王蕾。[1]"
    assert answer.evidence[0].filename == "atlas.md"
    assert status.collection_ready is True
    assert status.indexed_chunks == indexed.indexed_chunks


def test_document_filter_and_empty_unindexed_search(tmp_path: Path) -> None:
    ingestion, rag = make_rag_service(tmp_path)
    document_id = add_atlas_document(ingestion)

    assert rag.search("南湖").evidence == []
    rag.index_document(document_id)
    assert rag.search("南湖", document_ids=["doc-does-not-exist"]).evidence == []


def test_rag_api_with_dependency_override(tmp_path: Path) -> None:
    ingestion, rag = make_rag_service(tmp_path)
    document_id = add_atlas_document(ingestion)
    app.dependency_overrides[get_rag_service] = lambda: rag
    try:
        index_response = TestClient(app).post(f"/rag/index/{document_id}")
        ask_response = TestClient(app).post(
            "/rag/ask",
            json={"question": "谁负责 Atlas 的技术工作？", "top_k": 1},
        )
    finally:
        app.dependency_overrides.clear()

    assert index_response.status_code == 200
    assert ask_response.status_code == 200
    payload = ask_response.json()
    assert payload["answer"].endswith("[1]")
    assert payload["evidence"][0]["chunk_id"]
