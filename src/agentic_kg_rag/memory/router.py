from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from agentic_kg_rag.config import get_settings
from agentic_kg_rag.memory.models import (
    ConversationDetail,
    ConversationInfo,
    CreateConversationRequest,
)
from agentic_kg_rag.memory.service import ConversationMemoryService
from agentic_kg_rag.memory.store import ConversationNotFoundError, ConversationStore
from agentic_kg_rag.rag.router import get_rag_service

router = APIRouter(prefix="/memory", tags=["memory"])


@lru_cache
def get_memory_service() -> ConversationMemoryService:
    settings = get_settings()
    return ConversationMemoryService(
        store=ConversationStore(settings.memory_path),
        chat_model=get_rag_service().chat_model,
        max_tokens=settings.memory_max_tokens,
        keep_recent_messages=settings.memory_keep_recent_messages,
        summary_max_tokens=settings.memory_summary_max_tokens,
        context_window_tokens=settings.model_context_window,
    )


@router.post("/conversations", response_model=ConversationDetail)
def create_conversation(
    request: CreateConversationRequest,
    service: Annotated[ConversationMemoryService, Depends(get_memory_service)],
) -> ConversationDetail:
    return service.create(request.title)


@router.get("/conversations", response_model=list[ConversationInfo])
def list_conversations(
    service: Annotated[ConversationMemoryService, Depends(get_memory_service)],
) -> list[ConversationInfo]:
    return service.list()


@router.get("/conversations/{session_id}", response_model=ConversationDetail)
def get_conversation(
    session_id: str,
    service: Annotated[ConversationMemoryService, Depends(get_memory_service)],
) -> ConversationDetail:
    try:
        return service.get(session_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.delete("/conversations/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    session_id: str,
    service: Annotated[ConversationMemoryService, Depends(get_memory_service)],
) -> Response:
    try:
        service.delete(session_id)
    except ConversationNotFoundError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    return Response(status_code=status.HTTP_204_NO_CONTENT)
