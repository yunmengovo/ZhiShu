from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

DEFAULT_KNOWLEDGE_BASE_ID = "kb-default"


class KnowledgeBase(BaseModel):
    knowledge_base_id: str
    name: str
    description: str = ""
    created_at: datetime
    updated_at: datetime
    document_count: int = Field(default=0, ge=0)


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class UpdateKnowledgeBaseRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=500)
