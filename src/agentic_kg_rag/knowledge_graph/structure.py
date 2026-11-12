"""Deterministic source structure, explicitly separate from inferred semantic facts."""

from __future__ import annotations

import re

from agentic_kg_rag.ingestion.models import ProcessedDocument
from agentic_kg_rag.knowledge_graph.models import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionPayload,
)

STRUCTURAL_PREFIX = "DOC_"


def build_structure(document: ProcessedDocument) -> ExtractionPayload:
    entities, relations = [], []
    root = f"{document.title[:120]} [文档:{document.document_id[-8:]}]"

    def entity(name, kind, cid):
        entities.append(ExtractedEntity(name=name, type=kind, chunk_ids=[cid]))

    for chunk in document.chunks:
        entity(root, "Document", chunk.chunk_id)
        section = f"{(chunk.section_title or '正文')[:100]} [章节:{document.document_id[-8:]}]"
        entity(section, "Section", chunk.chunk_id)
        relations.append(
            ExtractedRelationship(
                source=root,
                target=section,
                type="DOC_HAS_SECTION",
                description="文档结构：章节归属，非模型推断",
                chunk_id=chunk.chunk_id,
            )
        )
        for label, url in re.findall(r"(?<!!)\[([^\]\n]+)\]\(([^\s)]+)\)", chunk.text):
            if not label.strip() or len(url) > 160:
                continue
            target = f"{label.strip()[:90]} [{url[:90]}]"[:200]
            entity(target, "Reference", chunk.chunk_id)
            relations.append(
                ExtractedRelationship(
                    source=section,
                    target=target,
                    type="DOC_REFERENCES",
                    description=f"文档显式链接：{url}"[:1000],
                    chunk_id=chunk.chunk_id,
                )
            )
    return ExtractionPayload(entities=entities, relationships=relations)
