"""Tests use deterministic substitutes and must never send traces to external services."""

import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
