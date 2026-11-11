"""Length-safe, resumable embeddings with no text truncation.

A long source chunk is embedded as bounded segments, then length-weighted and
L2-normalized into one vector. Original source text and citation IDs stay intact.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from agentic_kg_rag.build_cache import BuildCache
from agentic_kg_rag.resilience import call_model, embedding_slots
from agentic_kg_rag.runtime_config import RUNTIME


def split_utf8(text: str, budget: int) -> list[str]:
    if budget < 4:
        raise ValueError("UTF-8 budget must be at least 4")
    pieces: list[str] = []
    # Prefer sentence / paragraph endings, but split oversized sentences as well.
    current = ""
    for sentence in re.split(r"(?<=[。！？!?\n])", text):
        if len((current + sentence).encode()) <= budget:
            current += sentence
            continue
        if current:
            pieces.append(current)
            current = ""
        for char in sentence:
            if len((current + char).encode()) > budget:
                pieces.append(current)
                current = ""
            current += char
    if current:
        pieces.append(current)
    return pieces


class SafeEmbeddings:
    def __init__(self, model: Any, *, model_name: str, endpoint: str, cache: BuildCache) -> None:
        self.model = model
        self.cache = cache
        self.namespace = hashlib.sha256(
            f"embedding-v1:{endpoint}:{model_name}".encode()
        ).hexdigest()
        self.budget = (
            RUNTIME.models.bge_input_bytes
            if "bge" in model_name.lower()
            else RUNTIME.models.embedding_input_bytes
        )
        self.profile = hashlib.sha256(
            f"{self.namespace}:{self.budget}:weighted-pooling-v1:context-v1".encode()
        ).hexdigest()
        self.batch_size = RUNTIME.models.embedding_batch_size

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.namespace}:{text}".encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("embedding input must not be blank")
        groups = [split_utf8(text, self.budget) for text in texts]
        unique = list(dict.fromkeys(part for group in groups for part in group))
        vectors = {text: self.cache.get(self._key(text)) for text in unique}
        pending = [text for text in unique if vectors[text] is None]
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            result = call_model(
                lambda batch=batch: self.model.embed_documents(batch),
                label="嵌入模型",
                slots=embedding_slots,
            )
            if len(result) != len(batch):
                raise ValueError("embedding response count differs from input")
            for text, vector in zip(batch, result, strict=True):
                if not vector or not all(math.isfinite(x) for x in vector):
                    raise ValueError("embedding response contains an invalid vector")
                self.cache.put(self._key(text), vector)
                vectors[text] = vector
        pooled = []
        for group in groups:
            dimension = len(vectors[group[0]])
            if any(len(vectors[part]) != dimension for part in group):
                raise ValueError("embedding response dimensions are inconsistent")
            if len(group) == 1:
                pooled.append(vectors[group[0]])
                continue
            weights = [len(part.encode()) for part in group]
            vector = [
                sum(vectors[p][i] * w for p, w in zip(group, weights, strict=True)) / sum(weights)
                for i in range(dimension)
            ]
            norm = math.sqrt(sum(x * x for x in vector))
            if norm == 0:
                raise ValueError("pooled embedding has zero norm")
            pooled.append([x / norm for x in vector])
        return pooled

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
