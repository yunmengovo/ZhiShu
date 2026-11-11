from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from agentic_kg_rag.build_cache import BuildCache
from agentic_kg_rag.clients import normalize_openai_base_url
from agentic_kg_rag.config import get_settings
from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.ingestion.service import IngestionError
from agentic_kg_rag.knowledge_graph.models import (
    GraphExtractionResult,
    GraphSearchRequest,
    GraphSearchResponse,
    GraphStatus,
    GraphSubgraphRequest,
    GraphSubgraphResponse,
)
from agentic_kg_rag.knowledge_graph.neo4j_store import Neo4jGraphStore
from agentic_kg_rag.knowledge_graph.service import KnowledgeGraphError, KnowledgeGraphService
from agentic_kg_rag.knowledge_graph.store import LocalGraphStore
from agentic_kg_rag.rag.router import get_rag_service
from agentic_kg_rag.resilience import ReliableChatModel

router = APIRouter(prefix="/graph", tags=["knowledge-graph"])


@lru_cache
def get_knowledge_graph_service() -> KnowledgeGraphService:
    settings = get_settings()
    mode = settings.graph_mode.strip().lower()
    if mode == "local":
        store = LocalGraphStore(settings.graph_path)
    elif mode == "neo4j":
        if not settings.neo4j_password:
            raise KnowledgeGraphError("GRAPH_MODE=neo4j 时必须配置 NEO4J_PASSWORD")
        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
    else:
        raise KnowledgeGraphError("GRAPH_MODE 仅支持 local 或 neo4j")
    rag_service = get_rag_service()
    return KnowledgeGraphService(
        ingestion_service=get_ingestion_service(),
        store=store,
        chat_model=ReliableChatModel(rag_service.chat_model.model, stream=True),
        model_name=settings.chat_model,
        mode=mode,
        batch_chars=settings.kg_extraction_batch_chars,
        default_top_k=settings.graph_top_k,
        cache=BuildCache(settings.data_dir / "cache" / "graph_batches.db"),
        cache_namespace=normalize_openai_base_url(settings.chat_base_url) or "",
    )


@router.get("/status", response_model=GraphStatus)
def graph_status(
    service: Annotated[KnowledgeGraphService, Depends(get_knowledge_graph_service)],
) -> GraphStatus:
    return service.status()


@router.post("/extract/{document_id}", response_model=GraphExtractionResult)
async def extract_graph(
    document_id: str,
    service: Annotated[KnowledgeGraphService, Depends(get_knowledge_graph_service)],
) -> GraphExtractionResult:
    try:
        return await run_in_threadpool(service.extract_document, document_id)
    except (KnowledgeGraphError, IngestionError) as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/search", response_model=GraphSearchResponse)
async def search_graph(
    request: GraphSearchRequest,
    service: Annotated[KnowledgeGraphService, Depends(get_knowledge_graph_service)],
) -> GraphSearchResponse:
    try:
        return await run_in_threadpool(
            service.search,
            request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except KnowledgeGraphError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/subgraph", response_model=GraphSubgraphResponse)
async def graph_subgraph(
    request: GraphSubgraphRequest,
    service: Annotated[KnowledgeGraphService, Depends(get_knowledge_graph_service)],
) -> GraphSubgraphResponse:
    try:
        return await run_in_threadpool(
            service.subgraph,
            query=request.query,
            document_ids=request.document_ids,
            limit=request.limit,
        )
    except KnowledgeGraphError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
