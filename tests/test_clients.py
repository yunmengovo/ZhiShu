from agentic_kg_rag.clients import normalize_openai_base_url


def test_normalizes_copied_resource_urls() -> None:
    assert (
        normalize_openai_base_url("https://example.test/v1/embeddings") == "https://example.test/v1"
    )
    assert (
        normalize_openai_base_url("https://example.test/v1/chat/completions/")
        == "https://example.test/v1"
    )


def test_preserves_base_url() -> None:
    assert normalize_openai_base_url("https://example.test/v1") == "https://example.test/v1"
    assert normalize_openai_base_url("https://example.test") == "https://example.test/v1"
    assert normalize_openai_base_url("") is None
