"""Content-addressed successful build steps; no API keys or raw input are stored."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class BuildCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS build_cache (key TEXT PRIMARY KEY, value TEXT)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def get(self, key: str) -> Any | None:
        with self._connect() as db:
            row = db.execute("SELECT value FROM build_cache WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (ValueError, TypeError):
            return None

    def put(self, key: str, value: Any) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO build_cache VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False, allow_nan=False)),
            )
