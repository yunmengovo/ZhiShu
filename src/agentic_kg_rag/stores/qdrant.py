from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from qdrant_client import QdrantClient


class SerializedLocalQdrant:
    """Local persistence shares a SQLite connection; serialize all access across threads."""

    def __init__(self, client: QdrantClient) -> None:
        self.client = client
        self.storage_lock = RLock()

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.client, name)
        if not callable(value):
            return value

        def locked(*args: Any, **kwargs: Any) -> Any:
            with self.storage_lock:
                return value(*args, **kwargs)

        return locked


def create_qdrant_client(
    *,
    mode: str,
    path: Path,
    url: str,
    api_key: str,
) -> QdrantClient | SerializedLocalQdrant:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "local":
        path.mkdir(parents=True, exist_ok=True)
        return SerializedLocalQdrant(
            QdrantClient(path=str(path), force_disable_check_same_thread=True)
        )
    if normalized_mode == "memory":
        return SerializedLocalQdrant(QdrantClient(":memory:", force_disable_check_same_thread=True))
    if normalized_mode == "server":
        return QdrantClient(url=url, api_key=api_key or None)
    raise ValueError("QDRANT_MODE must be one of: local, memory, server")
