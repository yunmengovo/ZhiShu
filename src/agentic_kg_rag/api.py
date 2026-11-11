from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from langchain_core.exceptions import ModelError
from openai import OpenAIError
from starlette.requests import Request
from starlette.responses import JSONResponse

from agentic_kg_rag.agent.router import get_agent_service
from agentic_kg_rag.agent.router import router as agent_router
from agentic_kg_rag.config import get_settings
from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.ingestion.router import router as documents_router
from agentic_kg_rag.integrations.router import get_mineru_service
from agentic_kg_rag.integrations.router import router as integrations_router
from agentic_kg_rag.jobs.router import get_job_manager
from agentic_kg_rag.jobs.router import router as jobs_router
from agentic_kg_rag.knowledge_base.dependencies import get_knowledge_base_store
from agentic_kg_rag.knowledge_base.router import router as knowledge_base_router
from agentic_kg_rag.knowledge_graph.router import (
    get_knowledge_graph_service,
)
from agentic_kg_rag.knowledge_graph.router import (
    router as graph_router,
)
from agentic_kg_rag.memory.router import get_memory_service
from agentic_kg_rag.memory.router import router as memory_router
from agentic_kg_rag.rag.router import get_rag_service
from agentic_kg_rag.rag.router import router as rag_router
from agentic_kg_rag.resilience import ModelCallError
from agentic_kg_rag.workspace.router import get_workspace_service
from agentic_kg_rag.workspace.router import router as workspace_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Initialize once before accepting requests; concurrent lazy construction can
    # otherwise mistake a newly running job for a previous-process interruption.
    get_job_manager()
    yield
    get_agent_service.cache_clear()
    get_memory_service.cache_clear()
    get_workspace_service.cache_clear()
    get_knowledge_base_store.cache_clear()
    if get_job_manager.cache_info().currsize:
        get_job_manager().shutdown()
    get_job_manager.cache_clear()
    if get_knowledge_graph_service.cache_info().currsize:
        get_knowledge_graph_service().close()
    get_knowledge_graph_service.cache_clear()
    if get_rag_service.cache_info().currsize:
        get_rag_service().qdrant.close()
    get_rag_service.cache_clear()
    get_ingestion_service.cache_clear()
    get_mineru_service.cache_clear()


app = FastAPI(title="知枢 ZhiShu API", version="0.2.0", lifespan=lifespan)
app.include_router(documents_router)
app.include_router(rag_router)
app.include_router(agent_router)
app.include_router(graph_router)
app.include_router(memory_router)
app.include_router(knowledge_base_router)
app.include_router(workspace_router)
app.include_router(jobs_router)
app.include_router(integrations_router)


async def upstream_model_error_handler(
    request: Request,
    error: Exception,
) -> JSONResponse:
    del request, error
    return JSONResponse(
        status_code=503,
        content={"detail": "上游模型服务暂时不可用，请稍后重试。"},
    )


app.add_exception_handler(ModelError, upstream_model_error_handler)
app.add_exception_handler(OpenAIError, upstream_model_error_handler)


@app.exception_handler(ModelCallError)
async def model_call_error_handler(request: Request, error: ModelCallError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content={"detail": str(error)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    settings = get_settings()
    uvicorn.run(app, host=settings.app_host, port=settings.api_port)
