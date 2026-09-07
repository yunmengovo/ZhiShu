from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from agentic_kg_rag.config import get_settings
from agentic_kg_rag.ingestion.models import DocumentSummary
from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.ingestion.service import DocumentIngestionService, IngestionError
from agentic_kg_rag.integrations.mineru import MinerUError, MinerUService
from agentic_kg_rag.knowledge_base.dependencies import get_knowledge_base_store
from agentic_kg_rag.knowledge_base.models import DEFAULT_KNOWLEDGE_BASE_ID
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseError, KnowledgeBaseStore

router = APIRouter(prefix="/integrations", tags=["integrations"])


@lru_cache
def get_mineru_service() -> MinerUService:
    settings = get_settings()
    return MinerUService(token=settings.mineru_token, base_url=settings.mineru_base_url)


@router.get("/status")
def integration_status(
    mineru: Annotated[MinerUService, Depends(get_mineru_service)],
) -> dict[str, bool]:
    settings = get_settings()
    return {
        "mineru_configured": mineru.configured,
        "tavily_configured": bool(settings.tavily_api_key),
        "langsmith_enabled": bool(settings.langsmith_tracing and settings.langsmith_api_key),
    }


@router.post("/mineru/ingest", response_model=DocumentSummary)
async def mineru_ingest(
    file: Annotated[UploadFile, File()],
    ingestion: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
    mineru: Annotated[MinerUService, Depends(get_mineru_service)],
    knowledge_bases: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
    knowledge_base_id: Annotated[str, Form()] = DEFAULT_KNOWLEDGE_BASE_ID,
) -> DocumentSummary:
    filename = file.filename or ""
    try:
        knowledge_bases.get(knowledge_base_id)
        content = await file.read(ingestion.max_upload_bytes + 1)
        markdown = await run_in_threadpool(
            mineru.extract_markdown,
            filename=filename,
            content=content,
        )
        return await run_in_threadpool(
            ingestion.ingest_extracted_markdown,
            filename=filename,
            content=content,
            markdown=markdown,
            content_type=file.content_type,
            knowledge_base_id=knowledge_base_id,
        )
    except (MinerUError, IngestionError, KnowledgeBaseError) as error:
        status_code = getattr(error, "status_code", 400)
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    finally:
        await file.close()
