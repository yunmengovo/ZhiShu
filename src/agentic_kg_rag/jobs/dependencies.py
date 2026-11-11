from functools import lru_cache

from agentic_kg_rag.config import get_settings
from agentic_kg_rag.jobs.manager import JobManager
from agentic_kg_rag.jobs.store import JobStore
from agentic_kg_rag.runtime_config import RUNTIME


@lru_cache
def get_job_manager() -> JobManager:
    settings = get_settings()
    return JobManager(
        JobStore(settings.jobs_path, retained_jobs=RUNTIME.jobs.retained_jobs),
        workers=RUNTIME.jobs.workers,
    )
