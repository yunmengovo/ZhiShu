from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.ingestion.service import DocumentIngestionService
from agentic_kg_rag.knowledge_base.dependencies import get_knowledge_base_store
from agentic_kg_rag.knowledge_base.models import (
    CreateKnowledgeBaseRequest,
    KnowledgeBase,
    UpdateKnowledgeBaseRequest,
)
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseError, KnowledgeBaseStore

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


def _with_document_count(
    knowledge_base: KnowledgeBase,
    ingestion: DocumentIngestionService,
) -> KnowledgeBase:
    knowledge_base.document_count = len(
        ingestion.list_documents(knowledge_base_id=knowledge_base.knowledge_base_id)
    )
    return knowledge_base


@router.post("", response_model=KnowledgeBase, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    request: CreateKnowledgeBaseRequest,
    store: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
    ingestion: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> KnowledgeBase:
    try:
        return _with_document_count(store.create(request.name, request.description), ingestion)
    except KnowledgeBaseError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.get("", response_model=list[KnowledgeBase])
def list_knowledge_bases(
    store: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
    ingestion: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> list[KnowledgeBase]:
    return [_with_document_count(item, ingestion) for item in store.list()]


@router.get("/{knowledge_base_id}", response_model=KnowledgeBase)
def get_knowledge_base(
    knowledge_base_id: str,
    store: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
    ingestion: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> KnowledgeBase:
    try:
        return _with_document_count(store.get(knowledge_base_id), ingestion)
    except KnowledgeBaseError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.patch("/{knowledge_base_id}", response_model=KnowledgeBase)
def update_knowledge_base(
    knowledge_base_id: str,
    request: UpdateKnowledgeBaseRequest,
    store: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
    ingestion: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> KnowledgeBase:
    try:
        updated = store.update(
            knowledge_base_id,
            name=request.name,
            description=request.description,
        )
        return _with_document_count(updated, ingestion)
    except KnowledgeBaseError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
