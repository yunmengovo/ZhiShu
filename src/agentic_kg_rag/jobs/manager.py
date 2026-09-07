from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agentic_kg_rag.jobs.models import JobInfo
from agentic_kg_rag.jobs.store import JobStore

ProgressCallback = Callable[[int, str], None]
JobFunction = Callable[[ProgressCallback], dict[str, Any]]


class JobManager:
    def __init__(self, store: JobStore, *, workers: int) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="zhishu-job")

    def submit(self, job_type: str, target_id: str, function: JobFunction) -> JobInfo:
        job = self.store.create(job_type, target_id)
        self.executor.submit(self._run, job.job_id, function)
        return job

    def _run(self, job_id: str, function: JobFunction) -> None:
        self.store.update(job_id, status="running", progress=1, message="任务开始执行")

        def progress(value: int, message: str) -> None:
            self.store.update(job_id, progress=value, message=message)

        try:
            result = function(progress)
        except Exception as error:  # noqa: BLE001 - background job boundary
            self.store.update(
                job_id,
                status="failed",
                message="任务执行失败",
                error=f"{type(error).__name__}: {error}",
            )
            return
        self.store.update(
            job_id,
            status="succeeded",
            progress=100,
            message="任务执行完成",
            result=result,
            error="",
        )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
