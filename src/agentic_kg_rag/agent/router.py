from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from agentic_kg_rag.agent.models import AgentAskRequest, AgentAskResponse
from agentic_kg_rag.agent.service import AgentService
from agentic_kg_rag.config import get_settings
from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.knowledge_base.dependencies import get_knowledge_base_store
from agentic_kg_rag.knowledge_base.scope import resolve_document_scope
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseError, KnowledgeBaseStore
from agentic_kg_rag.knowledge_graph.router import get_knowledge_graph_service
from agentic_kg_rag.memory.router import get_memory_service
from agentic_kg_rag.memory.store import ConversationNotFoundError
from agentic_kg_rag.rag.router import get_rag_service
from agentic_kg_rag.rag.service import RagError
from agentic_kg_rag.retrieval.web import WebSearchService

router = APIRouter(prefix="/agent", tags=["agent"])


@lru_cache
def get_agent_service() -> AgentService:
    settings = get_settings()
    rag_service = get_rag_service()
    return AgentService(
        rag_service=rag_service,
        chat_model=rag_service.chat_model,
        model_name=settings.chat_model,
        default_top_k=settings.vector_top_k,
        default_max_retries=settings.max_agent_retries,
        graph_service=get_knowledge_graph_service(),
        memory_service=get_memory_service(),
        web_search_service=(
            WebSearchService(settings.tavily_api_key) if settings.tavily_api_key else None
        ),
    )


@router.post("/ask", response_model=AgentAskResponse)
async def agent_ask(
    request: AgentAskRequest,
    service: Annotated[AgentService, Depends(get_agent_service)],
    knowledge_bases: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
) -> AgentAskResponse:
    try:
        document_ids = resolve_document_scope(
            document_ids=request.document_ids,
            knowledge_base_id=request.knowledge_base_id,
            ingestion=get_ingestion_service(),
            knowledge_bases=knowledge_bases,
        )
        return await run_in_threadpool(
            service.ask,
            request.question,
            top_k=request.top_k,
            document_ids=document_ids,
            max_retries=request.max_retries,
            session_id=request.session_id,
        )
    except (RagError, ConversationNotFoundError, KnowledgeBaseError, ValueError) as error:
        status_code = getattr(error, "status_code", 400)
        raise HTTPException(status_code=status_code, detail=str(error)) from error
