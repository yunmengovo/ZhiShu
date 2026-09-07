from __future__ import annotations

from agentic_kg_rag.ingestion.service import DocumentIngestionService
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseStore


def resolve_document_scope(
    *,
    document_ids: list[str],
    knowledge_base_id: str | None,
    ingestion: DocumentIngestionService,
    knowledge_bases: KnowledgeBaseStore,
) -> list[str]:
    if not knowledge_base_id:
        return document_ids
    knowledge_bases.get(knowledge_base_id)
    allowed = {
        item.document_id
        for item in ingestion.list_documents(knowledge_base_id=knowledge_base_id)
    }
    if not document_ids:
        return sorted(allowed)
    outside = sorted(set(document_ids) - allowed)
    if outside:
        raise ValueError(
            "以下文档不属于所选知识空间：" + ", ".join(outside)
        )
    return document_ids
