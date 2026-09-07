from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_kg_rag.runtime_config import RUNTIME, RuntimeConfig

load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_host: str = "127.0.0.1"
    api_port: int = 8000
    ui_port: int = 7860
    ui_api_base_url: str = "http://127.0.0.1:8000"
    data_dir: Path = Path("data")

    chat_model: str = ""
    chat_base_url: str = ""
    chat_api_key: str = Field(default="", repr=False)

    embedding_model: str = "text-embedding-3-large"
    embedding_base_url: str = ""
    embedding_api_key: str = Field(default="", repr=False)
    embedding_check_ctx_length: bool = False

    qdrant_mode: str = "local"
    qdrant_path: Path = Path("data/qdrant")
    qdrant_collection: str = "document_chunks"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = Field(default="", repr=False)

    graph_mode: str = "local"
    graph_path: Path = Path("data/graph/knowledge.db")

    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = Field(default="", repr=False)
    neo4j_database: str = "neo4j"

    memory_path: Path = Path("data/memory/conversations.db")
    catalog_path: Path = Path("data/catalog.db")
    jobs_path: Path = Path("data/jobs.db")

    tavily_api_key: str = Field(default="", repr=False)
    langsmith_tracing: bool = False
    langsmith_api_key: str = Field(default="", repr=False)
    langsmith_project: str = "agentic-kg-rag"
    mineru_token: str = Field(default="", repr=False)
    mineru_base_url: str = "https://mineru.net"

    @property
    def runtime(self) -> RuntimeConfig:
        return RUNTIME

    @property
    def max_upload_mb(self) -> int:
        return RUNTIME.ingestion.max_upload_mb

    @property
    def chunk_size(self) -> int:
        return RUNTIME.ingestion.chunk_size

    @property
    def chunk_overlap(self) -> int:
        return RUNTIME.ingestion.chunk_overlap

    @property
    def vector_top_k(self) -> int:
        return RUNTIME.retrieval.vector_top_k

    @property
    def graph_top_k(self) -> int:
        return RUNTIME.retrieval.graph_top_k

    @property
    def kg_extraction_batch_chars(self) -> int:
        return RUNTIME.knowledge_graph.extraction_batch_chars

    @property
    def max_agent_retries(self) -> int:
        return RUNTIME.agent.max_retries

    @property
    def model_context_window(self) -> int:
        return RUNTIME.memory.model_context_window

    @property
    def memory_max_tokens(self) -> int:
        return RUNTIME.memory.max_tokens

    @property
    def memory_keep_recent_messages(self) -> int:
        return RUNTIME.memory.keep_recent_messages

    @property
    def memory_summary_max_tokens(self) -> int:
        return RUNTIME.memory.summary_max_tokens

    @property
    def enable_web_search(self) -> bool:
        return bool(self.tavily_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
