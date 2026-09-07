from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_kg_rag.jobs.models import JobInfo


class JobNotFoundError(LookupError):
    status_code = 404


class JobStore:
    def __init__(self, path: Path, *, retained_jobs: int) -> None:
        self.path = path
        self.retained_jobs = retained_jobs
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            now = datetime.now(UTC).isoformat()
            connection.execute(
                """
                UPDATE jobs SET status = 'failed', error = '服务重启导致任务中断',
                    message = '任务已中断', updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (now,),
            )

    def create(self, job_type: str, target_id: str) -> JobInfo:
        now = datetime.now(UTC).isoformat()
        job_id = f"job-{uuid.uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, 'queued', 0, ?, '{}', '', ?, ?)",
                (job_id, job_type, target_id, "等待执行", now, now),
            )
            self._prune(connection)
        return self.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> JobInfo:
        current = self.get(job_id)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs SET status = ?, progress = ?, message = ?, result_json = ?,
                    error = ?, updated_at = ? WHERE job_id = ?
                """,
                (
                    status or current.status,
                    current.progress if progress is None else max(0, min(100, progress)),
                    current.message if message is None else message,
                    json.dumps(current.result if result is None else result, ensure_ascii=False),
                    current.error if error is None else error,
                    datetime.now(UTC).isoformat(),
                    job_id,
                ),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> JobInfo:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(f"任务不存在：{job_id}")
        return self._to_model(row)

    def list(self, *, limit: int = 50) -> list[JobInfo]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_model(row) for row in rows]

    def _prune(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            DELETE FROM jobs WHERE job_id IN (
                SELECT job_id FROM jobs ORDER BY created_at DESC LIMIT -1 OFFSET ?
            ) AND status IN ('succeeded', 'failed')
            """,
            (self.retained_jobs,),
        )

    @staticmethod
    def _to_model(row: sqlite3.Row) -> JobInfo:
        return JobInfo(
            job_id=row["job_id"],
            job_type=row["job_type"],
            target_id=row["target_id"],
            status=row["status"],
            progress=row["progress"],
            message=row["message"],
            result=json.loads(row["result_json"]),
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
