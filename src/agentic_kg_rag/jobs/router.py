from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from agentic_kg_rag.jobs.dependencies import get_job_manager
from agentic_kg_rag.jobs.manager import JobConflictError, JobManager, ProgressCallback
from agentic_kg_rag.jobs.models import JobInfo
from agentic_kg_rag.jobs.store import JobNotFoundError
from agentic_kg_rag.workspace.models import DocumentRebuildRequest
from agentic_kg_rag.workspace.router import get_workspace_service
from agentic_kg_rag.workspace.service import WorkspaceService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobInfo])
def list_jobs(
    manager: Annotated[JobManager, Depends(get_job_manager)],
    limit: int = 50,
) -> list[JobInfo]:
    return manager.store.list(limit=max(1, min(limit, 200)))


@router.get("/{job_id}", response_model=JobInfo)
def get_job(
    job_id: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
) -> JobInfo:
    try:
        return manager.store.get(job_id)
    except JobNotFoundError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error


@router.post("/documents/{document_id}/rebuild", response_model=JobInfo)
def submit_document_rebuild(
    document_id: str,
    request: DocumentRebuildRequest,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    workspace: Annotated[WorkspaceService, Depends(get_workspace_service)],
) -> JobInfo:
    workspace.ingestion.load(document_id)

    def run(progress: ProgressCallback) -> dict[str, object]:
        progress(10, "正在重新解析文档" if request.reparse else "正在准备文档")
        result = workspace.rebuild_document(
            document_id,
            reparse=request.reparse,
            rebuild_vector=request.rebuild_vector,
            rebuild_graph=request.rebuild_graph,
            progress=progress,
            graph_mode=request.graph_mode,
        )
        progress(95, "正在写入最终状态")
        return result.model_dump(mode="json")

    try:
        return manager.submit(
            "document_rebuild", document_id, run, request_key=request.model_dump_json()
        )
    except JobConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/knowledge-bases/{knowledge_base_id}/rebuild", response_model=JobInfo)
def submit_knowledge_base_rebuild(
    knowledge_base_id: str,
    manager: Annotated[JobManager, Depends(get_job_manager)],
    workspace: Annotated[WorkspaceService, Depends(get_workspace_service)],
) -> JobInfo:
    workspace.knowledge_bases.get(knowledge_base_id)

    def run(progress: ProgressCallback) -> dict[str, object]:
        documents = workspace.ingestion.list_documents(knowledge_base_id=knowledge_base_id)
        results: list[dict[str, object]] = []
        total = max(len(documents), 1)
        for index, document in enumerate(documents, 1):
            progress(
                5 + int(85 * (index - 1) / total),
                f"正在处理 {document.filename}（{index}/{len(documents)}）",
            )
            result = workspace.rebuild_document(
                document.document_id,
                reparse=False,
                rebuild_vector=True,
                rebuild_graph=True,
            )
            results.append(result.model_dump(mode="json"))
        return {"knowledge_base_id": knowledge_base_id, "documents": results}

    return manager.submit("knowledge_base_rebuild", knowledge_base_id, run)
