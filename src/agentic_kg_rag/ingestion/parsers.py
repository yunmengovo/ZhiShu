from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as WordDocument
from docx.document import Document as WordDocumentType
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from markdownify import markdownify

from agentic_kg_rag.ingestion.models import BlockKind, SourceBlock

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".mdx", ".txt", ".html", ".htm"}


class UnsupportedDocumentError(ValueError):
    pass


@dataclass(slots=True)
class ParsedContent:
    title: str
    blocks: list[SourceBlock]
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _make_block(
    blocks: list[SourceBlock],
    *,
    kind: BlockKind,
    text: str,
    page_number: int | None = None,
    section_title: str | None = None,
    heading_level: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    cleaned = _clean_text(text)
    if not cleaned:
        return
    order = len(blocks)
    blocks.append(
        SourceBlock(
            block_id=f"block-{order:05d}",
            kind=kind,
            text=cleaned,
            order=order,
            page_number=page_number,
            section_title=section_title,
            heading_level=heading_level,
            metadata=metadata or {},
        )
    )


def _prepare_markdown_text(text: str, fallback_title: str) -> tuple[str, str]:
    title = fallback_title
    if text.startswith("---\n"):
        closing = text.find("\n---", 4)
        if closing != -1:
            frontmatter = text[4:closing]
            title_match = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', frontmatter, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            text = text[closing + 4 :]

    text = re.sub(r"^\s*(?:import|export)\s+.+?;\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*:::\w*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*</?[A-Z][^>]*>\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[A-Z][\w.]*\b[^>]*/>", "", text)
    return text, title


def _parse_markdown_text(text: str, fallback_title: str) -> ParsedContent:
    text, fallback_title = _prepare_markdown_text(text, fallback_title)
    blocks: list[SourceBlock] = []
    section_title: str | None = None
    title = fallback_title
    paragraph_lines: list[str] = []
    code_lines: list[str] = []
    in_code_fence = False

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            _make_block(
                blocks,
                kind="paragraph",
                text=" ".join(paragraph_lines),
                section_title=section_title,
            )
            paragraph_lines = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            _make_block(
                blocks,
                kind="paragraph",
                text="\n".join(code_lines),
                section_title=section_title,
                metadata={"content_type": "code"},
            )
            code_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            if in_code_fence:
                flush_code()
                in_code_fence = False
            else:
                flush_paragraph()
                in_code_fence = True
            continue
        if in_code_fence:
            code_lines.append(raw_line.rstrip())
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        list_match = re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)(.+)$", line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            if not blocks and level == 1:
                title = heading
                kind: BlockKind = "title"
            else:
                kind = "heading"
            section_title = heading
            _make_block(
                blocks,
                kind=kind,
                text=heading,
                section_title=section_title,
                heading_level=level,
            )
        elif list_match:
            flush_paragraph()
            _make_block(
                blocks,
                kind="list",
                text=list_match.group(1),
                section_title=section_title,
            )
        elif not line:
            flush_paragraph()
        else:
            paragraph_lines.append(line)
    flush_code()
    flush_paragraph()
    return ParsedContent(title=title, blocks=blocks)


def _iter_word_blocks(document: WordDocumentType) -> Iterable[WordParagraph | WordTable]:
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield WordParagraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield WordTable(child, document)


def parse_docx(path: Path) -> ParsedContent:
    document = WordDocument(path)
    blocks: list[SourceBlock] = []
    title = path.stem
    section_title: str | None = None
    table_count = 0

    for item in _iter_word_blocks(document):
        if isinstance(item, WordParagraph):
            text = item.text.strip()
            if not text:
                continue
            style_name = item.style.name if item.style is not None else ""
            heading_match = re.match(r"Heading\s+(\d+)", style_name, re.IGNORECASE)
            font_sizes = [run.font.size.pt for run in item.runs if run.font.size is not None]
            is_visual_title = not blocks and font_sizes and max(font_sizes) >= 18
            if style_name.lower() == "title" or is_visual_title:
                title = text
                section_title = text
                kind: BlockKind = "title"
                level = 0
            elif heading_match:
                section_title = text
                kind = "heading"
                level = int(heading_match.group(1))
            elif style_name.lower().startswith("list"):
                kind = "list"
                level = None
            else:
                kind = "paragraph"
                level = None
            _make_block(
                blocks,
                kind=kind,
                text=text,
                section_title=section_title,
                heading_level=level,
                metadata={"style": style_name} if style_name else None,
            )
        else:
            rows: list[str] = []
            for row in item.rows:
                values = [_clean_text(cell.text) for cell in row.cells]
                rows.append(" | ".join(values))
            table_count += 1
            _make_block(
                blocks,
                kind="table",
                text="\n".join(rows),
                section_title=section_title,
                metadata={"table_index": table_count, "row_count": len(rows)},
            )

    return ParsedContent(
        title=title,
        blocks=blocks,
        metadata={"paragraph_count": len(document.paragraphs), "table_count": table_count},
    )


def parse_pdf(path: Path) -> ParsedContent:
    document = pymupdf.open(path)
    blocks: list[SourceBlock] = []
    metadata = document.metadata or {}
    title = (metadata.get("title") or "").strip() or path.stem
    page_count = document.page_count
    section_title: str | None = None

    for page_index, page in enumerate(document, start=1):
        page_data = page.get_text("dict", sort=True)
        for block in page_data["blocks"]:
            if block.get("type") != 0:
                continue
            lines = block.get("lines", [])
            spans = [span for line in lines for span in line.get("spans", [])]
            text = "\n".join(
                "".join(span.get("text", "") for span in line.get("spans", [])) for line in lines
            )
            max_font_size = max((float(span.get("size", 0)) for span in spans), default=0)
            if not blocks and max_font_size >= 18:
                kind: BlockKind = "title"
                section_title = _clean_text(text)
                title = section_title or title
                heading_level = 0
            elif max_font_size >= 14:
                kind = "heading"
                section_title = _clean_text(text)
                heading_level = 1
            else:
                kind = "paragraph"
                heading_level = None
            _make_block(
                blocks,
                kind=kind,
                text=text,
                page_number=page_index,
                section_title=section_title,
                heading_level=heading_level,
                metadata={
                    "bbox": [round(value, 2) for value in block.get("bbox", [])],
                    "max_font_size": round(max_font_size, 2),
                },
            )

    document.close()
    return ParsedContent(title=title, blocks=blocks, metadata={"page_count": page_count})


def parse_html(path: Path) -> ParsedContent:
    raw_html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")
    for node in soup(["script", "style", "noscript", "nav"]):
        node.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else path.stem
    body = soup.body or soup
    markdown = markdownify(str(body), heading_style="ATX")
    parsed = _parse_markdown_text(markdown, title)
    parsed.title = title or parsed.title
    return parsed


def parse_text(path: Path) -> ParsedContent:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[SourceBlock] = []
    for paragraph in re.split(r"\n\s*\n", text):
        _make_block(blocks, kind="paragraph", text=paragraph)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), path.stem)
    title = first_line[:120]
    return ParsedContent(title=title, blocks=blocks)


def parse_document(path: Path) -> ParsedContent:
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentError(
            f"Unsupported file type {extension!r}; supported: {supported}"
        )
    if extension == ".pdf":
        return parse_pdf(path)
    if extension == ".docx":
        return parse_docx(path)
    if extension in {".md", ".markdown", ".mdx"}:
        return _parse_markdown_text(path.read_text(encoding="utf-8", errors="replace"), path.stem)
    if extension in {".html", ".htm"}:
        return parse_html(path)
    return parse_text(path)
