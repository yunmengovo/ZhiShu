from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from agentic_kg_rag.config import get_settings
from agentic_kg_rag.ingestion.models import (
    BatchDocumentResult,
    BatchIngestResponse,
    DocumentSummary,
    ProcessedDocument,
)
from agentic_kg_rag.ingestion.parsers import SUPPORTED_EXTENSIONS
from agentic_kg_rag.ingestion.service import DocumentIngestionService, IngestionError
from agentic_kg_rag.knowledge_base.dependencies import get_knowledge_base_store
from agentic_kg_rag.knowledge_base.models import DEFAULT_KNOWLEDGE_BASE_ID
from agentic_kg_rag.knowledge_base.store import KnowledgeBaseError, KnowledgeBaseStore
from agentic_kg_rag.runtime_config import RUNTIME

router = APIRouter(prefix="/documents", tags=["documents"])


@lru_cache
def get_ingestion_service() -> DocumentIngestionService:
    settings = get_settings()
    return DocumentIngestionService(
        data_dir=settings.data_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )


@router.get("/supported-formats")
def supported_formats() -> dict[str, list[str]]:
    return {"extensions": sorted(SUPPORTED_EXTENSIONS)}


@router.post("", response_model=DocumentSummary, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: Annotated[UploadFile, File()],
    service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
    knowledge_bases: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
    knowledge_base_id: Annotated[str, Form()] = DEFAULT_KNOWLEDGE_BASE_ID,
) -> DocumentSummary:
    filename = file.filename or ""
    try:
        content = await file.read(service.max_upload_bytes + 1)
        knowledge_bases.get(knowledge_base_id)
        return await run_in_threadpool(
            service.ingest_bytes,
            filename=filename,
            content=content,
            content_type=file.content_type,
            knowledge_base_id=knowledge_base_id,
        )
    except (IngestionError, KnowledgeBaseError) as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    finally:
        await file.close()


@router.post("/batch", response_model=BatchIngestResponse)
async def ingest_document_batch(
    files: Annotated[list[UploadFile], File()],
    service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
    knowledge_bases: Annotated[KnowledgeBaseStore, Depends(get_knowledge_base_store)],
    knowledge_base_id: Annotated[str, Form()] = DEFAULT_KNOWLEDGE_BASE_ID,
) -> BatchIngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")
    if len(files) > RUNTIME.ingestion.batch_upload_limit:
        raise HTTPException(
            status_code=400,
            detail=f"a batch can contain at most {RUNTIME.ingestion.batch_upload_limit} files",
        )
    try:
        knowledge_bases.get(knowledge_base_id)
    except KnowledgeBaseError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    results: list[BatchDocumentResult] = []
    for file in files:
        filename = file.filename or ""
        try:
            content = await file.read(service.max_upload_bytes + 1)
            document = await run_in_threadpool(
                service.ingest_bytes,
                filename=filename,
                content=content,
                content_type=file.content_type,
                knowledge_base_id=knowledge_base_id,
            )
            results.append(
                BatchDocumentResult(filename=filename, success=True, document=document)
            )
        except IngestionError as error:
            results.append(
                BatchDocumentResult(filename=filename, success=False, error=str(error))
            )
        finally:
            await file.close()
    succeeded = sum(item.success for item in results)
    return BatchIngestResponse(
        knowledge_base_id=knowledge_base_id,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
    knowledge_base_id: str | None = None,
) -> list[DocumentSummary]:
    return service.list_documents(knowledge_base_id=knowledge_base_id)


@router.get("/{document_id}", response_model=ProcessedDocument)
def get_document(
    document_id: str,
    service: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> ProcessedDocument:
    try:
        return service.load(document_id)
    except IngestionError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
