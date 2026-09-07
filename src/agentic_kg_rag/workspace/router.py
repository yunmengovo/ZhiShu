from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.ingestion.service import IngestionError
from agentic_kg_rag.knowledge_base.dependencies import get_knowledge_base_store
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseError
from agentic_kg_rag.knowledge_graph.router import get_knowledge_graph_service
from agentic_kg_rag.knowledge_graph.service import KnowledgeGraphError
from agentic_kg_rag.rag.router import get_rag_service
from agentic_kg_rag.rag.service import RagError
from agentic_kg_rag.workspace.models import (
    DocumentAsset,
    DocumentDeleteResult,
    DocumentRebuildRequest,
    DocumentRebuildResult,
    KnowledgeBaseDeleteResult,
)
from agentic_kg_rag.workspace.service import WorkspaceService

router = APIRouter(tags=["workspace"])


@lru_cache
def get_workspace_service() -> WorkspaceService:
    return WorkspaceService(
        ingestion=get_ingestion_service(),
        rag=get_rag_service(),
        graph=get_knowledge_graph_service(),
        knowledge_bases=get_knowledge_base_store(),
    )


@router.get("/workspace/assets", response_model=list[DocumentAsset])
def list_assets(
    service: Annotated[WorkspaceService, Depends(get_workspace_service)],
    knowledge_base_id: str | None = None,
) -> list[DocumentAsset]:
    try:
        return service.list_assets(knowledge_base_id)
    except KnowledgeBaseError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResult)
async def delete_document(
    document_id: str,
    service: Annotated[WorkspaceService, Depends(get_workspace_service)],
) -> DocumentDeleteResult:
    try:
        return await run_in_threadpool(service.delete_document, document_id)
    except (IngestionError, RagError, KnowledgeGraphError) as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/documents/{document_id}/rebuild", response_model=DocumentRebuildResult)
async def rebuild_document(
    document_id: str,
    request: DocumentRebuildRequest,
    service: Annotated[WorkspaceService, Depends(get_workspace_service)],
) -> DocumentRebuildResult:
    try:
        return await run_in_threadpool(
            service.rebuild_document,
            document_id,
            reparse=request.reparse,
            rebuild_vector=request.rebuild_vector,
            rebuild_graph=request.rebuild_graph,
        )
    except (IngestionError, RagError, KnowledgeGraphError) as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.delete(
    "/knowledge-bases/{knowledge_base_id}",
    response_model=KnowledgeBaseDeleteResult,
)
async def delete_knowledge_base(
    knowledge_base_id: str,
    service: Annotated[WorkspaceService, Depends(get_workspace_service)],
) -> KnowledgeBaseDeleteResult:
    try:
        return await run_in_threadpool(service.delete_knowledge_base, knowledge_base_id)
    except KnowledgeBaseError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
