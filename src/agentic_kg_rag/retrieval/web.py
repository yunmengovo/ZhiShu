from __future__ import annotations

import hashlib
from urllib.parse import urlparse

from tavily import TavilyClient

from agentic_kg_rag.rag.models import Evidence


class WebSearchError(RuntimeError):
    pass


class WebSearchService:
    def __init__(self, api_key: str) -> None:
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, *, top_k: int) -> list[Evidence]:
        try:
            response = self.client.search(
                query=query,
                max_results=top_k,
                auto_parameters=True,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as error:
            raise WebSearchError(f"Tavily 搜索失败：{type(error).__name__}") from error
        evidence: list[Evidence] = []
        for rank, item in enumerate(response.get("results", []), 1):
            url = str(item.get("url", "")).strip()
            content = str(item.get("content", "")).strip()
            if not url or not content:
                continue
            digest = hashlib.sha256(url.encode()).hexdigest()[:20]
            hostname = urlparse(url).hostname or "web"
            evidence.append(
                Evidence(
                    rank=rank,
                    score=float(item.get("score", 0.0)),
                    source_type="web",
                    document_id=f"web-{digest}",
                    chunk_id=f"web-{digest}",
                    filename=hostname,
                    title=str(item.get("title", hostname)),
                    text=content,
                    metadata={"url": url},
                )
            )
        return evidence
