from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

from agentic_kg_rag.jobs.models import JobInfo
from agentic_kg_rag.jobs.store import JobStore
from agentic_kg_rag.resilience import ModelCallError, notice
from agentic_kg_rag.runtime_config import RUNTIME

ProgressCallback = Callable[[int, str], None]
JobFunction = Callable[[ProgressCallback], dict[str, Any]]


class JobConflictError(RuntimeError):
    status_code = 409


class JobManager:
    def __init__(self, store: JobStore, *, workers: int) -> None:
        self.store = store
        self._lock = Lock()
        self._active: dict[str, tuple[str, str]] = {}
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="zhishu-job")

    def submit(
        self, job_type: str, target_id: str, function: JobFunction, *, request_key: str = ""
    ) -> JobInfo:
        identity = f"{job_type}:{request_key}"
        with self._lock:
            if target_id in self._active:
                active_identity, active_id = self._active[target_id]
                if active_identity == identity:
                    return self.store.get(active_id)
                raise JobConflictError("该文档已有构建任务，请等待当前任务完成后再操作")
            if len(self._active) >= RUNTIME.jobs.max_pending:
                raise JobConflictError("后台任务队列已满，请等待部分任务完成后重试")
            job = self.store.create(job_type, target_id)
            self._active[target_id] = (identity, job.job_id)
            try:
                self.executor.submit(self._run, job.job_id, function)
            except Exception:
                self._active.pop(target_id, None)
                self.store.update(job.job_id, status="failed", message="任务未能启动")
                raise
            return job

    def _run(self, job_id: str, function: JobFunction) -> None:
        self.store.update(job_id, status="running", progress=1, message="任务开始执行")

        def progress(value: int, message: str) -> None:
            self.store.update(job_id, progress=value, message=message)

        token = notice.set(lambda message: self.store.update(job_id, message=message))
        try:
            result = function(progress)
        except Exception as error:  # noqa: BLE001 - background job boundary
            self.store.update(
                job_id,
                status="failed",
                message="任务执行失败",
                error=(
                    str(error)
                    if isinstance(error, ModelCallError) or hasattr(error, "result")
                    else f"构建未完成（{type(error).__name__}），请重试或检查服务配置"
                ),
                result=getattr(error, "result", {}),
            )
        else:
            self.store.update(
                job_id,
                status="succeeded",
                progress=100,
                message=(
                    "结构图已就绪；部分语义关系待补全"
                    if result.get("graph_warnings")
                    else "任务执行完成"
                ),
                result=result,
                error="",
            )
        finally:
            notice.reset(token)
            target = self.store.get(job_id).target_id
            with self._lock:
                self._active.pop(target, None)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
