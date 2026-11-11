from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from starlette.concurrency import run_in_threadpool

from agentic_kg_rag.build_cache import BuildCache
from agentic_kg_rag.clients import normalize_openai_base_url
from agentic_kg_rag.config import get_settings
from agentic_kg_rag.embeddings import SafeEmbeddings
from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.ingestion.service import IngestionError
from agentic_kg_rag.knowledge_base.dependencies import get_knowledge_base_store
from agentic_kg_rag.knowledge_base.scope import resolve_document_scope
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseError, KnowledgeBaseStore
from agentic_kg_rag.rag.models import (
    AskRequest,
    AskResponse,
    IndexAllResult,
    IndexResult,
    RagStatus,
    SearchRequest,
    SearchResponse,
)
from agentic_kg_rag.rag.service import RagError, RagService
from agentic_kg_rag.resilience import ReliableChatModel
from agentic_kg_rag.runtime_config import RUNTIME
from agentic_kg_rag.stores.qdrant import create_qdrant_client

router = APIRouter(prefix="/rag", tags=["rag"])


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        base_url=normalize_openai_base_url(settings.embedding_base_url),
        api_key=settings.embedding_api_key or None,
        check_embedding_ctx_length=False,
        request_timeout=RUNTIME.models.embedding_timeout_seconds,
        max_retries=0,
    )
    embeddings = SafeEmbeddings(
        embeddings,
        model_name=settings.embedding_model,
        endpoint=normalize_openai_base_url(settings.embedding_base_url) or "",
        cache=BuildCache(settings.data_dir / "cache" / "embeddings.db"),
    )
    chat_model = ReliableChatModel(
        ChatOpenAI(
            model=settings.chat_model,
            base_url=normalize_openai_base_url(settings.chat_base_url),
            api_key=settings.chat_api_key or None,
            temperature=0,
            max_retries=0,
            timeout=RUNTIME.models.chat_timeout_seconds,
            max_tokens=4096,
        )
    )
    qdrant = create_qdrant_client(
        mode=settings.qdrant_mode,
        path=settings.qdrant_path,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    return RagService(
        ingestion_service=get_ingestion_service(),
        qdrant=qdrant,
        embeddings=embeddings,
        chat_model=chat_model,
        collection_name=settings.qdrant_collection,
        default_top_k=settings.vector_top_k,
        model_name=settings.chat_model,
        mode=settings.qdrant_mode,
        chat_configured=all([settings.chat_model, settings.chat_base_url, settings.chat_api_key]),
        embedding_configured=all(
            [
                settings.embedding_model,
                settings.embedding_base_url,
                settings.embedding_api_key,
            ]
        ),
    )


@router.get("/status", response_model=RagStatus)
def rag_status(service: Annotated[RagService, Depends(get_rag_service)]) -> RagStatus:
    try:
        return service.status()
    except RagError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/index/{document_id}", response_model=IndexResult)
async def index_document(
    document_id: str,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> IndexResult:
    try:
        return await run_in_threadpool(service.index_document, document_id)
    except (RagError, IngestionError) as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/index-all", response_model=IndexAllResult)
async def index_all(
    service: Annotated[RagService, Depends(get_rag_service)],
) -> IndexAllResult:
    try:
        return await run_in_threadpool(service.index_all)
    except (RagError, IngestionError) as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
    knowledge_bases: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
) -> SearchResponse:
    try:
        document_ids = resolve_document_scope(
            document_ids=request.document_ids,
            knowledge_base_id=request.knowledge_base_id,
            ingestion=service.ingestion_service,
            knowledge_bases=knowledge_bases,
        )
        return await run_in_threadpool(
            service.search,
            request.query,
            top_k=request.top_k,
            document_ids=document_ids,
        )
    except (RagError, KnowledgeBaseError, ValueError) as error:
        status_code = getattr(error, "status_code", 400)
        raise HTTPException(status_code=status_code, detail=str(error)) from error


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
    knowledge_bases: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
) -> AskResponse:
    try:
        document_ids = resolve_document_scope(
            document_ids=request.document_ids,
            knowledge_base_id=request.knowledge_base_id,
            ingestion=service.ingestion_service,
            knowledge_bases=knowledge_bases,
        )
        return await run_in_threadpool(
            service.ask,
            request.question,
            top_k=request.top_k,
            document_ids=document_ids,
        )
    except (RagError, KnowledgeBaseError, ValueError) as error:
        status_code = getattr(error, "status_code", 400)
        raise HTTPException(status_code=status_code, detail=str(error)) from error
