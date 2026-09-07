from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pytest

from agentic_kg_rag.config import Settings
from agentic_kg_rag.ingestion.service import DocumentIngestionService
from agentic_kg_rag.integrations.mineru import MinerUError, MinerUService
from agentic_kg_rag.jobs.manager import JobManager
from agentic_kg_rag.jobs.store import JobStore
from agentic_kg_rag.knowledge_base.models import DEFAULT_KNOWLEDGE_BASE_ID
from agentic_kg_rag.knowledge_base.scope import resolve_document_scope
from agentic_kg_rag.knowledge_base.store import (
    KnowledgeBaseConflictError,
    KnowledgeBaseStore,
)
from agentic_kg_rag.knowledge_graph.models import GraphRelationEvidence
from agentic_kg_rag.rag.models import Evidence
from agentic_kg_rag.retrieval.fusion import fuse_local_evidence, merge_web_evidence
from agentic_kg_rag.runtime_config import RUNTIME


def make_ingestion(path: Path) -> DocumentIngestionService:
    return DocumentIngestionService(
        data_dir=path,
        chunk_size=200,
        chunk_overlap=20,
        max_upload_bytes=1024 * 1024,
    )


def test_knowledge_base_catalog_and_document_isolation(tmp_path: Path) -> None:
    catalog = KnowledgeBaseStore(tmp_path / "catalog.db")
    ingestion = make_ingestion(tmp_path / "data")
    project = catalog.create("项目资料", "一组隔离的资料")
    same_content = b"# Atlas\n\nA reusable document."

    default_document = ingestion.ingest_bytes(filename="atlas.md", content=same_content)
    project_document = ingestion.ingest_bytes(
        filename="atlas.md",
        content=same_content,
        knowledge_base_id=project.knowledge_base_id,
    )

    assert default_document.document_id != project_document.document_id
    assert [item.document_id for item in ingestion.list_documents(project.knowledge_base_id)] == [
        project_document.document_id
    ]
    assert resolve_document_scope(
        document_ids=[],
        knowledge_base_id=project.knowledge_base_id,
        ingestion=ingestion,
        knowledge_bases=catalog,
    ) == [project_document.document_id]
    with pytest.raises(ValueError, match="不属于"):
        resolve_document_scope(
            document_ids=[default_document.document_id],
            knowledge_base_id=project.knowledge_base_id,
            ingestion=ingestion,
            knowledge_bases=catalog,
        )


def test_knowledge_base_names_are_unique_and_default_cannot_be_deleted(tmp_path: Path) -> None:
    catalog = KnowledgeBaseStore(tmp_path / "catalog.db")
    catalog.create("产品手册")

    with pytest.raises(KnowledgeBaseConflictError):
        catalog.create("  产品手册  ")
    with pytest.raises(KnowledgeBaseConflictError, match="默认知识空间不能删除"):
        catalog.delete(DEFAULT_KNOWLEDGE_BASE_ID)


def test_runtime_policy_is_code_owned_not_environment_owned() -> None:
    settings = Settings(_env_file=None, max_upload_mb=999, vector_top_k=49)  # type: ignore[call-arg]

    assert settings.max_upload_mb == RUNTIME.ingestion.max_upload_mb
    assert settings.vector_top_k == RUNTIME.retrieval.vector_top_k


def test_mineru_markdown_archive_and_reprocessing(tmp_path: Path) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("result/full.md", "# 解析结果\n\n复杂 PDF 中的表格已经转换。")

    markdown = MinerUService._markdown_from_zip(output.getvalue())
    ingestion = make_ingestion(tmp_path / "data")
    summary = ingestion.ingest_extracted_markdown(
        filename="complex.pdf",
        content=b"a pretend pdf payload",
        markdown=markdown,
    )
    reprocessed = ingestion.reprocess(summary.document_id)

    assert "表格已经转换" in ingestion.load(summary.document_id).chunks[0].text
    assert reprocessed.document_id == summary.document_id
    with pytest.raises(MinerUError):
        MinerUService._markdown_from_zip(b"not a zip")


def test_reciprocal_rank_fusion_boosts_graph_backed_chunk_and_merges_web() -> None:
    local = [
        Evidence(
            rank=1,
            score=0.9,
            document_id="doc-1",
            chunk_id="chunk-1",
            filename="a.md",
            title="A",
            text="first",
        ),
        Evidence(
            rank=2,
            score=0.8,
            document_id="doc-1",
            chunk_id="chunk-2",
            filename="a.md",
            title="A",
            text="second",
        ),
    ]
    graph = [
        GraphRelationEvidence(
            rank=1,
            source="B",
            source_type="Concept",
            relation="USES",
            target="C",
            target_type="Concept",
            document_id="doc-1",
            chunk_id="chunk-2",
        )
    ]
    fused = fuse_local_evidence(local, graph, limit=2, rrf_constant=60)
    web = Evidence(
        rank=1,
        score=0.7,
        source_type="web",
        document_id="web-1",
        chunk_id="web-1",
        filename="example.com",
        title="Latest",
        text="fresh information",
        metadata={"url": "https://example.com"},
    )

    assert fused[0].chunk_id == "chunk-2"
    assert fused[0].metadata["graph_boosted"] is True
    merged = merge_web_evidence(fused, [web], limit=3)
    assert [item.rank for item in merged] == [1, 2, 3]
    assert merged[-1].source_type == "web"


def test_persistent_background_job_lifecycle(tmp_path: Path) -> None:
    manager = JobManager(
        JobStore(tmp_path / "jobs.db", retained_jobs=20),
        workers=1,
    )

    def work(progress):  # type: ignore[no-untyped-def]
        progress(60, "处理中")
        return {"answer": 42}

    submitted = manager.submit("demo", "target-1", work)
    deadline = time.monotonic() + 3
    current = manager.store.get(submitted.job_id)
    while current.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
        time.sleep(0.01)
        current = manager.store.get(submitted.job_id)
    manager.shutdown()

    assert current.status == "succeeded"
    assert current.progress == 100
    assert current.result == {"answer": 42}
