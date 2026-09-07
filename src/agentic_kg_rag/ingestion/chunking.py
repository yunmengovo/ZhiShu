from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from agentic_kg_rag.ingestion.models import DocumentChunk, SourceBlock


@dataclass(slots=True)
class _Segment:
    block_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    section_title: str | None = None

    def append(self, block: SourceBlock) -> None:
        self.block_ids.append(block.block_id)
        self.texts.append(block.text)
        if block.page_number is not None:
            self.pages.append(block.page_number)
        if block.section_title:
            self.section_title = block.section_title


def _group_blocks(blocks: list[SourceBlock]) -> list[_Segment]:
    segments: list[_Segment] = []
    current = _Segment()
    current_page: int | None = None
    current_section: str | None = None

    for block in blocks:
        page_changed = block.page_number is not None and block.page_number != current_page
        section_changed = bool(block.section_title and block.section_title != current_section)
        if current.block_ids and (
            page_changed or (block.kind in {"title", "heading"} and section_changed)
        ):
            segments.append(current)
            current = _Segment()
        current.append(block)
        if block.page_number is not None:
            current_page = block.page_number
        if block.section_title:
            current_section = block.section_title
    if current.block_ids:
        segments.append(current)
    return segments


def build_chunks(
    document_id: str,
    blocks: list[SourceBlock],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    chunks: list[DocumentChunk] = []

    for segment in _group_blocks(blocks):
        text = "\n".join(segment.texts)
        for piece in splitter.split_text(text):
            order = len(chunks)
            digest = hashlib.sha256(f"{document_id}:{order}:{piece}".encode()).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk-{digest}",
                    document_id=document_id,
                    text=piece,
                    order=order,
                    block_ids=segment.block_ids,
                    page_start=min(segment.pages) if segment.pages else None,
                    page_end=max(segment.pages) if segment.pages else None,
                    section_title=segment.section_title,
                )
            )
    return chunks
