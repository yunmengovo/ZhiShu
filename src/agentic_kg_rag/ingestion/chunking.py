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
        add_start_index=True,
        keep_separator="end",
        separators=["\n\n", "\n", "。", "！", "？", ". ", "；", "，", " ", ""],
    )
    chunks: list[DocumentChunk] = []

    for segment in _group_blocks(blocks):
        text = "\n".join(segment.texts)
        spans = []
        offset = 0
        for block_id, block_text in zip(segment.block_ids, segment.texts, strict=True):
            spans.append((block_id, offset, offset + len(block_text)))
            offset += len(block_text) + 1
        for split in splitter.create_documents([text]):
            piece = split.page_content
            start = split.metadata["start_index"]
            end = start + len(piece)
            source_ids = [bid for bid, left, right in spans if left < end and right > start]
            order = len(chunks)
            digest = hashlib.sha256(f"{document_id}:{order}:{piece}".encode()).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk-{digest}",
                    document_id=document_id,
                    text=piece,
                    order=order,
                    block_ids=source_ids,
                    metadata={
                        "segment_start": start,
                        "segment_end": end,
                        "content_sha256": hashlib.sha256(piece.encode()).hexdigest(),
                    },
                    page_start=min(segment.pages) if segment.pages else None,
                    page_end=max(segment.pages) if segment.pages else None,
                    section_title=segment.section_title,
                )
            )
    return chunks


def prose_chunks(blocks: list[SourceBlock], chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Mask fenced code using source offsets, keeping existing chunk IDs and provenance."""
    code_spans: dict[str, tuple[int, int]] = {}
    for segment in _group_blocks(blocks):
        offset = 0
        for bid, text in zip(segment.block_ids, segment.texts, strict=True):
            code_spans[bid] = (offset, offset + len(text))
            offset += len(text) + 1
    code_ids = {b.block_id for b in blocks if b.metadata.get("content_type") == "code"}
    result = []
    for chunk in chunks:
        code = code_ids.intersection(chunk.block_ids)
        if not code:
            result.append(chunk)
            continue
        start = chunk.metadata.get("segment_start")
        if start is None:
            # Older provenance cannot safely locate code: omit the mixed chunk conservatively.
            continue
        chars = list(chunk.text)
        for bid in code:
            left, right = code_spans[bid]
            for i in range(max(0, left - start), min(len(chars), right - start)):
                if chars[i] != "\n":
                    chars[i] = " "
        text = "".join(chars)
        if text.strip():
            result.append(chunk.model_copy(update={"text": text}))
    return result
