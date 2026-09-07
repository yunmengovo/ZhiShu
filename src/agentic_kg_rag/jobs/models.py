from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobInfo(BaseModel):
    job_id: str
    job_type: str
    target_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    created_at: datetime
    updated_at: datetime
