from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient


def create_qdrant_client(
    *,
    mode: str,
    path: Path,
    url: str,
    api_key: str,
) -> QdrantClient:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "local":
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if normalized_mode == "memory":
        return QdrantClient(":memory:")
    if normalized_mode == "server":
        return QdrantClient(url=url, api_key=api_key or None)
    raise ValueError("QDRANT_MODE must be one of: local, memory, server")
