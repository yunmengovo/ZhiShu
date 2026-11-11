"""Bounded model retries; SDK retries must be disabled by the caller."""

from __future__ import annotations

import contextvars
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import BoundedSemaphore
from typing import TypeVar

import httpx
from openai import APIConnectionError, APIStatusError

from agentic_kg_rag.runtime_config import RUNTIME

T = TypeVar("T")
notice: contextvars.ContextVar[Callable[[str], None] | None] = contextvars.ContextVar(
    "model_retry_notice", default=None
)
chat_slots = BoundedSemaphore(RUNTIME.models.chat_concurrency)
embedding_slots = BoundedSemaphore(RUNTIME.models.embedding_concurrency)


class ModelCallError(RuntimeError):
    status_code = 503


def error_status(error: Exception) -> int | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "status_code", None)
        if code is None and isinstance(current, httpx.HTTPStatusError):
            code = current.response.status_code
        if isinstance(code, int):
            return code
        current = current.__cause__
    return None


def retryable(error: Exception) -> bool:
    code = error_status(error)
    if code is not None:
        # Exhausted credits / billing require intervention, not more requests.
        body = str(getattr(error, "body", "")).lower()
        if any(x in body for x in ("insufficient_quota", "billing", "credit balance")):
            return False
        return code in {408, 409, 429, 500, 502, 503, 504}
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (APIConnectionError, httpx.TransportError, TimeoutError)):
            return True
        current = current.__cause__
    return False


def retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    milliseconds = response.headers.get("retry-after-ms")
    if milliseconds:
        try:
            return max(0.0, float(milliseconds) / 1000)
        except ValueError:
            pass
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            return max(0.0, (date - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def call_model(
    operation: Callable[[], T],
    *,
    label: str,
    slots: BoundedSemaphore,
    attempts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    policy = RUNTIME.models
    maximum = attempts or policy.max_attempts
    started = time.monotonic()
    for attempt in range(1, maximum + 1):
        remaining = policy.retry_budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise ModelCallError(f"{label}已达到重试时间预算")
        observer = notice.get()
        if observer:
            observer(f"正在等待{label}并发配额")
        queued_at = time.monotonic()
        if not slots.acquire(timeout=policy.queue_wait_seconds):
            raise ModelCallError(f"{label}排队等待超过上限，尚未发起本次请求")
        # Queue wait is not model execution/retry time.
        started += time.monotonic() - queued_at
        if observer:
            observer(f"已获得{label}并发配额，正在发送请求")
        try:
            return operation()
        except Exception as error:
            if not isinstance(error, (APIStatusError, APIConnectionError, httpx.HTTPError)) and (
                error_status(error) is None and not retryable(error)
            ):
                raise
            code = error_status(error)
            if attempt == maximum or not retryable(error):
                if code in {401, 403}:
                    message = "鉴权失败，请检查模型密钥与访问权限"
                elif code in {400, 413, 422}:
                    message = "参数或输入长度被拒绝，请检查模型和输入预算"
                else:
                    message = f"请求未成功（已尝试{attempt}次），请稍后重试"
                raise ModelCallError(f"{label}{message}") from error
            delay = min(policy.max_retry_delay, policy.retry_base_delay * 2 ** (attempt - 1))
            delay = random.uniform(delay / 2, delay)
            server_delay = retry_after(error)
            if server_delay is not None:
                # Never retry sooner than the provider requested.
                if server_delay > policy.max_retry_delay:
                    raise ModelCallError(f"{label}限流等待时间较长，请稍后重新提交") from error
                delay = max(delay, server_delay) + random.uniform(0, 0.2)
            if time.monotonic() - started + delay >= policy.retry_budget_seconds:
                raise ModelCallError(f"{label}已达到重试时间预算") from error
        finally:
            slots.release()
        observer = notice.get()
        if observer:
            observer(f"{label}暂时不可用，{delay:.1f}秒后重试（{attempt + 1}/{maximum}）")
        sleep(delay)
    raise AssertionError("unreachable")


class ReliableChatModel:
    def __init__(self, model: object, *, stream: bool = False) -> None:
        self.model = model
        self.stream = stream

    def invoke(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        def operation():  # type: ignore[no-untyped-def]
            if not self.stream:
                return self.model.invoke(*args, **kwargs)  # type: ignore[attr-defined]
            # Buffer privately: retries never duplicate already-published answers.
            started = last_notice = time.monotonic()
            combined = None
            received = 0
            for chunk in self.model.stream(*args, **kwargs):  # type: ignore[attr-defined]
                if time.monotonic() - started > RUNTIME.models.graph_generation_seconds:
                    raise TimeoutError("graph generation deadline exceeded")
                combined = chunk if combined is None else combined + chunk
                received += len(str(chunk.content))
                observer = notice.get()
                if observer and time.monotonic() - last_notice >= 5:
                    observer(f"模型已连接，正在推理或生成图谱（回答{received}字符）")
                    last_notice = time.monotonic()
            if combined is None:
                raise ValueError("model returned an empty stream")
            return combined

        return call_model(operation, label="聊天模型", slots=chat_slots)
