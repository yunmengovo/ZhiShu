"""Typed, code-owned runtime policies.

These values describe how ZhiShu behaves rather than where services are
deployed or which credentials are used.  They intentionally live in code so
that ``.env`` remains a small deployment/secret contract.
"""

from pydantic import BaseModel, ConfigDict, Field


class RuntimeSection(BaseModel):
    model_config = ConfigDict(frozen=True)


class IngestionRuntimeConfig(RuntimeSection):
    max_upload_mb: int = Field(default=30, ge=1, le=500)
    chunk_size: int = Field(default=800, ge=200, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)
    batch_upload_limit: int = Field(default=50, ge=1, le=200)


class RetrievalRuntimeConfig(RuntimeSection):
    vector_top_k: int = Field(default=8, ge=1, le=50)
    graph_top_k: int = Field(default=12, ge=1, le=100)
    web_top_k: int = Field(default=5, ge=1, le=10)
    fusion_pool_multiplier: int = Field(default=2, ge=1, le=5)
    rrf_constant: int = Field(default=60, ge=1, le=200)


class ModelRuntimeConfig(RuntimeSection):
    chat_timeout_seconds: float = Field(default=90, ge=5, le=300)
    graph_generation_seconds: float = Field(default=180, ge=30, le=600)
    embedding_timeout_seconds: float = Field(default=30, ge=5, le=120)
    max_attempts: int = Field(default=3, ge=1, le=5)
    retry_budget_seconds: float = Field(default=300, ge=10, le=600)
    retry_base_delay: float = Field(default=1, ge=0.1, le=10)
    max_retry_delay: float = Field(default=15, ge=1, le=60)
    chat_concurrency: int = Field(default=2, ge=1, le=8)
    embedding_concurrency: int = Field(default=2, ge=1, le=8)
    # Conservative UTF-8 byte budgets do not assume an OpenAI tokenizer for BGE.
    bge_input_bytes: int = Field(default=480, ge=64, le=500)
    embedding_input_bytes: int = Field(default=1800, ge=64, le=8000)
    embedding_batch_size: int = Field(default=8, ge=1, le=64)


class KnowledgeGraphRuntimeConfig(RuntimeSection):
    extraction_batch_chars: int = Field(default=2400, ge=1000, le=100_000)
    visualization_max_relations: int = Field(default=120, ge=10, le=500)


class AgentRuntimeConfig(RuntimeSection):
    max_retries: int = Field(default=1, ge=0, le=3)
    enable_citation_repair: bool = True
    web_time_sensitive_terms: tuple[str, ...] = (
        "最新",
        "现在",
        "当前",
        "今天",
        "近期",
        "实时",
        "价格",
        "版本",
        "发布",
        "news",
        "latest",
        "current",
        "today",
        "recent",
        "price",
        "version",
    )


class MemoryRuntimeConfig(RuntimeSection):
    model_context_window: int = Field(default=256_000, ge=8192)
    max_tokens: int = Field(default=220_000, ge=4096)
    keep_recent_messages: int = Field(default=12, ge=2, le=100)
    summary_max_tokens: int = Field(default=4000, ge=256, le=32_000)


class JobRuntimeConfig(RuntimeSection):
    max_pending: int = Field(default=100, ge=1, le=1000)
    workers: int = Field(default=4, ge=1, le=16)
    retained_jobs: int = Field(default=500, ge=20, le=5000)


class MinerURuntimeConfig(RuntimeSection):
    model_version: str = "vlm"
    poll_interval_seconds: float = Field(default=3.0, ge=0.5, le=30)
    timeout_seconds: int = Field(default=600, ge=30, le=3600)


class RuntimeConfig(RuntimeSection):
    models: ModelRuntimeConfig = ModelRuntimeConfig()
    ingestion: IngestionRuntimeConfig = IngestionRuntimeConfig()
    retrieval: RetrievalRuntimeConfig = RetrievalRuntimeConfig()
    knowledge_graph: KnowledgeGraphRuntimeConfig = KnowledgeGraphRuntimeConfig()
    agent: AgentRuntimeConfig = AgentRuntimeConfig()
    memory: MemoryRuntimeConfig = MemoryRuntimeConfig()
    jobs: JobRuntimeConfig = JobRuntimeConfig()
    mineru: MinerURuntimeConfig = MinerURuntimeConfig()


RUNTIME = RuntimeConfig()
