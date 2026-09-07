from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def normalize_openai_base_url(value: str) -> str | None:
    """Accept either an OpenAI base URL or a copied resource endpoint URL."""
    normalized = value.strip().rstrip("/")
    if not normalized:
        return None
    for suffix in ("/chat/completions", "/embeddings", "/responses"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    parsed = urlsplit(normalized)
    if parsed.scheme and parsed.netloc and not parsed.path:
        return urlunsplit((parsed.scheme, parsed.netloc, "/v1", parsed.query, parsed.fragment))
    return normalized
