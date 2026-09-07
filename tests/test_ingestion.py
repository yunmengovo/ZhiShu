from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from docx import Document
from fastapi.testclient import TestClient

from agentic_kg_rag.api import app
from agentic_kg_rag.ingestion.router import get_ingestion_service
from agentic_kg_rag.ingestion.service import (
    DocumentIngestionService,
    UnsupportedFileError,
)


def make_service(data_dir: Path) -> DocumentIngestionService:
    return DocumentIngestionService(
        data_dir=data_dir,
        chunk_size=120,
        chunk_overlap=20,
        max_upload_bytes=2 * 1024 * 1024,
    )


def test_markdown_ingestion_and_duplicate_detection(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    content = """# Atlas 使用说明

Atlas 部署在南湖示范工厂。

## 负责人

- 王蕾负责技术工作。
""".encode()

    first = service.ingest_bytes(filename="atlas.md", content=content, content_type="text/markdown")
    second = service.ingest_bytes(
        filename="atlas.md", content=content, content_type="text/markdown"
    )
    document = service.load(first.document_id)

    assert first.duplicate is False
    assert second.duplicate is True
    assert document.title == "Atlas 使用说明"
    assert any(block.kind == "heading" for block in document.blocks)
    assert any("南湖示范工厂" in chunk.text for chunk in document.chunks)
    assert len(service.list_documents()) == 1


def test_docx_parser_keeps_headings_and_tables(tmp_path: Path) -> None:
    word = Document()
    word.add_heading("设备维护手册", level=0)
    word.add_heading("责任人", level=1)
    word.add_paragraph("陈卓负责数据核对。")
    table = word.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "模块"
    table.cell(0, 1).text = "负责人"
    table.cell(1, 0).text = "ParseHub"
    table.cell(1, 1).text = "陈卓"
    buffer = BytesIO()
    word.save(buffer)

    service = make_service(tmp_path)
    summary = service.ingest_bytes(
        filename="manual.docx",
        content=buffer.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    document = service.load(summary.document_id)

    assert document.title == "设备维护手册"
    assert any(block.kind == "table" and "ParseHub" in block.text for block in document.blocks)
    assert any(chunk.section_title == "责任人" for chunk in document.chunks)


def test_pdf_parser_preserves_page_numbers(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Atlas service runs in the demonstration factory.")
    content = pdf.tobytes()
    pdf.close()

    service = make_service(tmp_path)
    summary = service.ingest_bytes(filename="manual.pdf", content=content)
    document = service.load(summary.document_id)

    assert document.metadata["page_count"] == 1
    assert document.blocks[0].page_number == 1
    assert document.chunks[0].page_start == 1
    assert document.chunks[0].page_end == 1


def test_html_ingestion_removes_script_content(tmp_path: Path) -> None:
    html = b"""
    <html><head><title>Operations Guide</title><script>ignore me</script></head>
    <body><h1>Backup</h1><p>Backups run at 22:30.</p></body></html>
    """
    service = make_service(tmp_path)
    summary = service.ingest_bytes(filename="guide.html", content=html, content_type="text/html")
    document = service.load(summary.document_id)

    assert document.title == "Operations Guide"
    assert any("22:30" in chunk.text for chunk in document.chunks)
    assert all("ignore me" not in block.text for block in document.blocks)


def test_mdx_ingestion_uses_frontmatter_and_removes_components(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    content = b"""---
title: Agentic RAG
description: Test page
---

import Demo from "/demo.mdx";

## Retrieval

Agents can decide when to retrieve evidence.

<Demo />

```python
# This is code, not a document heading
print("retrieve")
```
"""
    summary = service.ingest_bytes(filename="agentic-rag.mdx", content=content)
    document = service.load(summary.document_id)

    assert document.title == "Agentic RAG"
    assert any(block.section_title == "Retrieval" for block in document.blocks)
    assert all("import Demo" not in block.text for block in document.blocks)
    assert all("<Demo" not in block.text for block in document.blocks)
    assert any(block.metadata.get("content_type") == "code" for block in document.blocks)


def test_rejects_unsupported_file(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    with pytest.raises(UnsupportedFileError):
        service.ingest_bytes(filename="archive.zip", content=b"not-a-zip")


def test_upload_api_returns_summary(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    app.dependency_overrides[get_ingestion_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/documents",
            files={"file": ("note.txt", b"Atlas is ready.", "text/plain")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    payload = response.json()
    assert payload["file_type"] == "txt"
    assert payload["chunk_count"] == 1
