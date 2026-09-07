from __future__ import annotations

from agentic_kg_rag.knowledge_graph.models import GraphRelationEvidence
from agentic_kg_rag.rag.models import Evidence


def fuse_local_evidence(
    vector_evidence: list[Evidence],
    graph_relations: list[GraphRelationEvidence],
    *,
    limit: int,
    rrf_constant: int,
) -> list[Evidence]:
    """Deduplicate chunks and apply reciprocal-rank graph boosting."""
    by_chunk: dict[str, Evidence] = {}
    fusion_scores: dict[str, float] = {}
    for rank, evidence in enumerate(vector_evidence, 1):
        key = evidence.chunk_id or f"text:{evidence.text.strip().casefold()}"
        if key not in by_chunk:
            by_chunk[key] = evidence.model_copy(deep=True)
            fusion_scores[key] = 1.0 / (rrf_constant + rank)
        else:
            fusion_scores[key] += 1.0 / (rrf_constant + rank)

    graph_ranks: dict[str, int] = {}
    for rank, relation in enumerate(graph_relations, 1):
        if relation.chunk_id:
            graph_ranks[relation.chunk_id] = min(
                graph_ranks.get(relation.chunk_id, rank),
                rank,
            )
    for key, rank in graph_ranks.items():
        if key in fusion_scores:
            fusion_scores[key] += 1.0 / (rrf_constant + rank)

    ranked = sorted(
        by_chunk.items(),
        key=lambda item: (fusion_scores[item[0]], item[1].score),
        reverse=True,
    )[:limit]
    result: list[Evidence] = []
    for rank, (key, evidence) in enumerate(ranked, 1):
        metadata = dict(evidence.metadata)
        metadata["vector_score"] = evidence.score
        metadata["fusion_score"] = fusion_scores[key]
        metadata["graph_boosted"] = key in graph_ranks
        result.append(
            evidence.model_copy(
                update={
                    "rank": rank,
                    "score": fusion_scores[key],
                    "metadata": metadata,
                }
            )
        )
    return result


def merge_web_evidence(
    local: list[Evidence],
    web: list[Evidence],
    *,
    limit: int,
) -> list[Evidence]:
    seen = {item.chunk_id for item in local}
    merged = [item.model_copy(deep=True) for item in local]
    for item in web:
        if item.chunk_id not in seen:
            seen.add(item.chunk_id)
            merged.append(item.model_copy(deep=True))
    return [item.model_copy(update={"rank": rank}) for rank, item in enumerate(merged[:limit], 1)]
