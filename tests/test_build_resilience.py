from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage
from openai import BadRequestError, RateLimitError
from test_knowledge_graph import FakeExtractionChat, add_document, make_graph_service
from test_rag import add_atlas_document, make_rag_service

from agentic_kg_rag.build_cache import BuildCache
from agentic_kg_rag.embeddings import SafeEmbeddings, split_utf8
from agentic_kg_rag.ingestion.chunking import build_chunks
from agentic_kg_rag.ingestion.models import SourceBlock
from agentic_kg_rag.jobs.manager import JobConflictError, JobManager
from agentic_kg_rag.jobs.store import JobStore
from agentic_kg_rag.knowledge_graph.service import KnowledgeGraphExtractionError
from agentic_kg_rag.resilience import ModelCallError, call_model
from agentic_kg_rag.ui import BackendClient, BackendError
from agentic_kg_rag.workspace.service import DocumentBuildError, WorkspaceService


def api_error(status: int, *, retry_after: str | None = None):
    response = httpx.Response(
        status,
        request=httpx.Request("POST", "https://example.invalid"),
        headers={"retry-after": retry_after} if retry_after else {},
    )
    cls = RateLimitError if status == 429 else BadRequestError
    return cls("synthetic provider rejection", response=response, body={"code": "synthetic"})


@pytest.mark.parametrize("budget", [4, 32, 480])
def test_unicode_segments_are_lossless_and_bounded(budget: int) -> None:
    text = "中文来源。分段保留！🙂\nEnglish words and numbers 2026. " * 40
    parts = split_utf8(text, budget)
    assert "".join(parts) == text
    assert all(0 < len(x.encode()) <= budget for x in parts)


class SizedEmbeddings:
    def __init__(self) -> None:
        self.requests: list[list[str]] = []
        self.fail_on = 0

    def embed_documents(self, texts):
        self.requests.append(texts)
        if len(self.requests) == self.fail_on:
            raise api_error(400)
        assert all(len(t.encode()) <= 480 for t in texts)
        return [[1.0, float(len(t)), 2.0] for t in texts]


def test_length_safe_pooling_and_cache(tmp_path: Path) -> None:
    backend = SizedEmbeddings()
    embeddings = SafeEmbeddings(
        backend,
        model_name="BAAI/bge-large-zh-v1.5",
        endpoint="https://example.invalid/v1",
        cache=BuildCache(tmp_path / "c.db"),
    )
    text = "中文正文，超过模型的输入长度。" * 100
    result = embeddings.embed_documents([text, text])
    assert result[0] == result[1]
    assert math.isclose(sum(x * x for x in result[0]), 1)
    calls = len(backend.requests)
    assert embeddings.embed_query(text) == result[0]
    assert len(backend.requests) == calls
    assert sum(map(len, backend.requests)) == len(set(split_utf8(text, 480)))


def test_failed_embedding_batch_reuses_successful_segments(tmp_path: Path) -> None:
    backend = SizedEmbeddings()
    backend.fail_on = 2
    embeddings = SafeEmbeddings(
        backend, model_name="bge", endpoint="test", cache=BuildCache(tmp_path / "c.db")
    )
    embeddings.batch_size = 1
    text = "一" * 160 + "二" * 160 + "三" * 160
    with pytest.raises(ModelCallError):
        embeddings.embed_documents([text])
    first_request = backend.requests[0]
    assert len(embeddings.embed_documents([text])) == 1
    assert backend.requests.count(first_request) == 1


def test_retry_after_and_jitter_respected() -> None:
    calls = []
    delays = []

    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise api_error(429, retry_after="2")
        return "ok"

    assert (
        call_model(
            operation, label="测试", slots=threading.BoundedSemaphore(1), sleep=delays.append
        )
        == "ok"
    )
    assert len(calls) == 3
    assert all(2 <= d <= 2.2 for d in delays)


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_permanent_provider_errors_not_retried(status: int) -> None:
    calls = []

    def operation():
        calls.append(1)
        raise api_error(status)

    with pytest.raises(ModelCallError):
        call_model(
            operation,
            label="测试",
            slots=threading.BoundedSemaphore(1),
            sleep=lambda _: pytest.fail("must not retry permanent failures"),
        )
    assert len(calls) == 1


def test_long_retry_after_is_deferred_without_early_retry() -> None:
    with pytest.raises(ModelCallError, match="限流"):
        call_model(
            lambda: (_ for _ in ()).throw(api_error(429, retry_after="120")),
            label="测试",
            slots=threading.BoundedSemaphore(1),
            sleep=lambda _: pytest.fail("must not wait or retry early"),
        )


def test_timeouts_retry_but_are_bounded() -> None:
    calls = []

    def operation():
        calls.append(1)
        raise httpx.ReadTimeout("synthetic timeout")

    with pytest.raises(ModelCallError, match="3次"):
        call_model(
            operation, label="测试", slots=threading.BoundedSemaphore(1), sleep=lambda _: None
        )
    assert len(calls) == 3


def test_chunk_provenance_only_references_overlapping_blocks() -> None:
    blocks = [
        SourceBlock(block_id=f"block-{i}", kind="paragraph", text=word * 80, order=i)
        for i, word in enumerate(["甲", "乙", "丙"])
    ]
    chunks = build_chunks("doc-test", blocks, chunk_size=80, chunk_overlap=0)
    assert len(chunks) == 3
    assert [c.block_ids for c in chunks] == [["block-0"], ["block-1"], ["block-2"]]
    text = "\n".join(b.text for b in blocks)
    for c in chunks:
        assert text[c.metadata["segment_start"] : c.metadata["segment_end"]] == c.text


def test_failed_reindex_keeps_previous_vectors(tmp_path: Path) -> None:
    ingestion, rag = make_rag_service(tmp_path)
    doc = add_atlas_document(ingestion)
    before = rag.index_document(doc)

    class Broken:
        def embed_documents(self, texts):
            raise ModelCallError("synthetic offline")

    rag.embeddings = Broken()
    with pytest.raises(ModelCallError):
        rag.index_document(doc)
    assert rag.document_index_status(doc).indexed_chunks == before.indexed_chunks
    rag.qdrant.close()


def test_graph_invalid_source_is_rejected_without_writes(tmp_path: Path) -> None:
    ingestion, graph = make_graph_service(tmp_path)
    doc = add_document(ingestion)

    class Fabricated:
        def invoke(self, messages):
            return AIMessage(
                content='{"entities":[],"relationships":[{"source":"X",'
                '"target":"Y","type":"USES","chunk_id":"invented"}]}'
            )

    graph.chat_model = Fabricated()
    with pytest.raises(KnowledgeGraphExtractionError):
        graph.extract_document(doc)
    assert graph.status().relationship_count == 0
    graph.close()


def test_graph_checkpoint_resumes_after_midway_failure(tmp_path: Path) -> None:
    ingestion, graph = make_graph_service(tmp_path)
    doc = add_document(ingestion)
    loaded = ingestion.load(doc)
    loaded.chunks.append(
        loaded.chunks[0].model_copy(update={"chunk_id": "chunk-second", "order": 1})
    )
    Path(loaded.processed_path).write_text(loaded.model_dump_json())
    graph.batch_chars = len(loaded.chunks[0].text)
    graph.cache = BuildCache(tmp_path / "cache.db")

    class Flaky(FakeExtractionChat):
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                raise ModelCallError("synthetic timeout")
            return super().invoke(messages)

    model = Flaky()
    graph.chat_model = model
    with pytest.raises(ModelCallError):
        graph.extract_document(doc)
    assert graph.status().relationship_count == 0
    graph.extract_document(doc)
    assert model.calls == 3  # first successful batch was not generated again
    graph.close()


def test_independent_build_stage_survives_other_stage_failure() -> None:
    called = []

    class BrokenRag:
        def index_document(self, document_id):
            raise ModelCallError("嵌入暂不可用")

    class WorkingGraph:
        def extract_document(self, document_id, **kwargs):
            called.append(document_id)
            return SimpleNamespace(entity_count=3, relationship_count=2)

    workspace = WorkspaceService(
        ingestion=SimpleNamespace(load=lambda _: SimpleNamespace(filename="test.md")),
        rag=BrokenRag(),
        graph=WorkingGraph(),
        knowledge_bases=None,
    )
    with pytest.raises(DocumentBuildError) as error:
        workspace.rebuild_document(
            "doc-test", reparse=False, rebuild_vector=True, rebuild_graph=True
        )
    assert called == ["doc-test"]
    assert error.value.result["entity_count"] == 3
    assert error.value.result["stage_errors"] == {"vector": "嵌入暂不可用"}


def test_duplicate_job_reused_and_conflicting_job_blocked(tmp_path: Path) -> None:
    manager = JobManager(JobStore(tmp_path / "jobs.db", retained_jobs=20), workers=1)
    release = threading.Event()

    def work(progress):
        release.wait(3)
        return {"ok": True}

    try:
        first = manager.submit("build", "doc-test", work, request_key="vector")
        second = manager.submit("build", "doc-test", work, request_key="vector")
        assert first.job_id == second.job_id
        with pytest.raises(JobConflictError):
            manager.submit("build", "doc-test", work, request_key="graph")
    finally:
        release.set()
        manager.shutdown()
    deadline = time.monotonic() + 3
    while manager.store.get(first.job_id).status not in {"failed", "succeeded"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert manager.store.get(first.job_id).status == "succeeded"


def test_frontend_timeout_is_not_reported_as_connection_failure(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("synthetic timeout")

    monkeypatch.setattr(httpx.Client, "request", timeout)
    with pytest.raises(BackendError, match="请求超时"):
        BackendClient("http://localhost:8000").health()


def test_same_dimension_different_embedding_profile_never_mixes_results(tmp_path: Path) -> None:
    ingestion, rag = make_rag_service(tmp_path)
    doc = add_atlas_document(ingestion)
    rag.embedding_profile = "encoder-v1"
    rag.index_document(doc)
    assert rag.search("王蕾").evidence
    rag.embedding_profile = "encoder-v2"
    assert rag.search("王蕾").evidence == []
    assert rag.document_index_status(doc).indexed_chunks == 0
    rag.index_document(doc)
    assert rag.search("王蕾").evidence
    rag.qdrant.close()


def test_streamed_graph_is_buffered_before_return() -> None:
    from langchain_core.messages import AIMessageChunk

    from agentic_kg_rag.resilience import ReliableChatModel

    class Streamed:
        def stream(self, messages):
            yield AIMessageChunk(content='{"entities":')
            yield AIMessageChunk(content='[],"relationships":[]}')

    response = ReliableChatModel(Streamed(), stream=True).invoke([])
    assert response.content == '{"entities":[],"relationships":[]}'


def test_quota_exhaustion_does_not_retry() -> None:
    error = api_error(429)
    error.body = {"code": "insufficient_quota"}
    with pytest.raises(ModelCallError):
        call_model(
            lambda: (_ for _ in ()).throw(error),
            label="测试",
            slots=threading.BoundedSemaphore(1),
            sleep=lambda _: pytest.fail("quota exhaustion must not retry"),
        )


def test_http_service_error_retries_with_retry_after_milliseconds() -> None:
    calls, waits = [], []

    def operation():
        calls.append(1)
        if len(calls) == 1:
            response = httpx.Response(
                503,
                headers={"retry-after-ms": "1500"},
                request=httpx.Request("GET", "https://example.invalid"),
            )
            response.raise_for_status()
        return "recovered"

    assert (
        call_model(operation, label="测试", slots=threading.BoundedSemaphore(1), sleep=waits.append)
        == "recovered"
    )
    assert len(calls) == 2 and 1.5 <= waits[0] <= 1.7


def test_local_qdrant_concurrent_writes_and_reads_are_serialized(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from qdrant_client import models

    from agentic_kg_rag.stores.qdrant import create_qdrant_client

    q = create_qdrant_client(mode="local", path=tmp_path / "vectors", url="", api_key="")
    q.create_collection(
        "test", vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE)
    )

    def write(i):
        q.upsert(
            "test",
            points=[models.PointStruct(id=i * 40 + j, vector=[1.0, 0.5, 0.3]) for j in range(40)],
        )
        q.count("test", exact=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    assert q.count("test", exact=True).count == 480
    q.close()


def test_queue_wait_does_not_consume_retry_budget(monkeypatch) -> None:
    import agentic_kg_rag.resilience as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    class DelayedSlot:
        def acquire(self, timeout):
            clock[0] += 250
            return True

        def release(self):
            pass

    calls = []

    def work():
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("synthetic timeout")
        return "success"

    assert call_model(work, label="测试", slots=DelayedSlot(), sleep=lambda _: None) == "success"
    assert len(calls) == 2
