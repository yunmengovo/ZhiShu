from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    id: int
    role: str
    content: str
    token_count: int = Field(ge=0)
    created_at: str


class ConversationInfo(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    compressed_messages: int = Field(ge=0)
    summary_present: bool


class ConversationDetail(ConversationInfo):
    summary: str = ""
    messages: list[ConversationMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=0, ge=0)
    context_window_tokens: int = Field(default=0, ge=0)


class CreateConversationRequest(BaseModel):
    title: str = Field(default="新会话", max_length=100)


class MemorySnapshot(BaseModel):
    session_id: str
    message_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    compressed_messages: int = Field(ge=0)
    summary_present: bool
    compressed_this_turn: bool = False
    max_tokens: int = Field(default=0, ge=0)
    context_window_tokens: int = Field(default=0, ge=0)
    compression_warning: str = ""
