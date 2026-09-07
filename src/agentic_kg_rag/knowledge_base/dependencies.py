from functools import lru_cache

from agentic_kg_rag.config import get_settings
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseStore


@lru_cache
def get_knowledge_base_store() -> KnowledgeBaseStore:
    return KnowledgeBaseStore(get_settings().catalog_path)
