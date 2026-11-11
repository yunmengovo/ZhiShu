from __future__ import annotations

from collections.abc import Callable

from agentic_kg_rag.ingestion.service import DocumentIngestionService
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseStore
from agentic_kg_rag.knowledge_graph.service import KnowledgeGraphService
from agentic_kg_rag.rag.service import RagService
from agentic_kg_rag.resilience import ModelCallError
from agentic_kg_rag.workspace.models import (
    DocumentAsset,
    DocumentDeleteResult,
    DocumentRebuildResult,
    KnowledgeBaseDeleteResult,
)


class DocumentBuildError(RuntimeError):
    def __init__(self, message: str, result: dict) -> None:
        super().__init__(message)
        self.result = result


class WorkspaceService:
    def __init__(
        self,
        *,
        ingestion: DocumentIngestionService,
        rag: RagService,
        graph: KnowledgeGraphService,
        knowledge_bases: KnowledgeBaseStore,
    ) -> None:
        self.ingestion = ingestion
        self.rag = rag
        self.graph = graph
        self.knowledge_bases = knowledge_bases

    def list_assets(self, knowledge_base_id: str | None = None) -> list[DocumentAsset]:
        if knowledge_base_id:
            self.knowledge_bases.get(knowledge_base_id)
        assets: list[DocumentAsset] = []
        for document in self.ingestion.list_documents(knowledge_base_id=knowledge_base_id):
            index = self.rag.document_index_status(document.document_id)
            try:
                entities, relationships, batches = self.graph.store.document_counts(
                    document.document_id
                )
                graph_status = "ready" if batches > 0 else "pending"
            except Exception:  # noqa: BLE001 - status must degrade without breaking the library
                entities, relationships, graph_status = 0, 0, "unavailable"
            assets.append(
                DocumentAsset(
                    **document.model_dump(),
                    indexed_chunks=index.indexed_chunks,
                    vector_status="ready" if index.indexed else "pending",
                    entity_count=entities,
                    relationship_count=relationships,
                    graph_status=graph_status,
                )
            )
        return assets

    def delete_document(self, document_id: str) -> DocumentDeleteResult:
        document = self.ingestion.load(document_id)
        warnings: list[str] = []
        deleted_vectors = 0
        graph_deleted = False
        try:
            deleted_vectors = self.rag.delete_document(document_id)
        except Exception as error:  # noqa: BLE001 - continue cross-store cleanup
            warnings.append(f"向量清理失败：{error}")
        try:
            self.graph.delete_document(document_id)
            graph_deleted = True
        except Exception as error:  # noqa: BLE001 - continue cross-store cleanup
            warnings.append(f"图谱清理失败：{error}")
        self.ingestion.delete(document_id)
        return DocumentDeleteResult(
            document_id=document_id,
            filename=document.filename,
            deleted_vectors=deleted_vectors,
            graph_deleted=graph_deleted,
            files_deleted=True,
            warnings=warnings,
        )

    def rebuild_document(
        self,
        document_id: str,
        *,
        reparse: bool,
        rebuild_vector: bool,
        rebuild_graph: bool,
        progress: Callable[[int, str], None] | None = None,
    ) -> DocumentRebuildResult:
        document = self.ingestion.reprocess(document_id) if reparse else None
        loaded = self.ingestion.load(document_id)
        indexed_chunks = 0
        entity_count = 0
        relationship_count = 0
        failures = {}
        if rebuild_vector:
            if progress:
                progress(15, "正在构建向量索引（成功片段会缓存）")
            try:
                indexed_chunks = self.rag.index_document(document_id).indexed_chunks
            except Exception as error:  # noqa: BLE001 - preserve independent stage results
                failures["vector"] = (
                    str(error)
                    if isinstance(error, ModelCallError)
                    else f"向量构建失败（{type(error).__name__}）"
                )
        if rebuild_graph:
            if progress:
                progress(45, "正在构建知识图谱")
            try:
                extraction = self.graph.extract_document(
                    document_id,
                    progress=(lambda value, message: progress(45 + value // 2, message))
                    if progress
                    else None,
                )
                entity_count = extraction.entity_count
                relationship_count = extraction.relationship_count
            except Exception as error:  # noqa: BLE001 - independent stage boundary
                failures["graph"] = (
                    str(error)
                    if isinstance(error, ModelCallError)
                    else f"图谱构建失败（{type(error).__name__}）"
                )
        result = DocumentRebuildResult(
            document_id=document_id,
            filename=(document.filename if document else loaded.filename),
            reparsed=reparse,
            indexed_chunks=indexed_chunks,
            entity_count=entity_count,
            relationship_count=relationship_count,
        )
        if failures:
            partial = result.model_dump(mode="json")
            partial["stage_errors"] = failures
            raise DocumentBuildError(
                "；".join(failures.values()) + "；已成功的阶段保留，可重新提交", partial
            )
        return result

    def delete_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseDeleteResult:
        self.knowledge_bases.get(knowledge_base_id)
        warnings: list[str] = []
        deleted = 0
        for document in self.ingestion.list_documents(knowledge_base_id=knowledge_base_id):
            result = self.delete_document(document.document_id)
            deleted += 1
            warnings.extend(result.warnings)
        self.knowledge_bases.delete(knowledge_base_id)
        return KnowledgeBaseDeleteResult(
            knowledge_base_id=knowledge_base_id,
            deleted_documents=deleted,
            warnings=warnings,
        )
