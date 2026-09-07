from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from agentic_kg_rag.ingestion.chunking import build_chunks
from agentic_kg_rag.ingestion.models import DocumentSummary, ProcessedDocument
from agentic_kg_rag.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    UnsupportedDocumentError,
    parse_document,
)
from agentic_kg_rag.knowledge_base.models import DEFAULT_KNOWLEDGE_BASE_ID

PARSER_VERSION = "1"


class IngestionError(ValueError):
    status_code = 400


class FileTooLargeError(IngestionError):
    status_code = 413


class UnsupportedFileError(IngestionError):
    status_code = 415


class DocumentNotFoundError(IngestionError):
    status_code = 404


class DocumentIngestionService:
    def __init__(
        self,
        *,
        data_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_bytes: int,
    ) -> None:
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_upload_bytes = max_upload_bytes
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def ingest_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None = None,
        knowledge_base_id: str = DEFAULT_KNOWLEDGE_BASE_ID,
        force: bool = False,
    ) -> DocumentSummary:
        knowledge_base_id = self._validate_knowledge_base_id(knowledge_base_id)
        safe_name = Path(filename).name.strip()
        if not safe_name:
            raise IngestionError("filename is required")
        extension = Path(safe_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(f"unsupported file type: {extension or '(none)'}")
        if not content:
            raise IngestionError("uploaded file is empty")
        if len(content) > self.max_upload_bytes:
            raise FileTooLargeError(
                f"file size exceeds limit of {self.max_upload_bytes // (1024 * 1024)} MB"
            )

        checksum = hashlib.sha256(content).hexdigest()
        document_id = self._document_id(knowledge_base_id, checksum)
        processed_path = self.processed_dir / f"{document_id}.json"
        processing_config = {
            "parser_version": PARSER_VERSION,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        if processed_path.exists() and not force:
            existing = self.load(document_id)
            if existing.metadata.get("processing") == processing_config:
                return DocumentSummary.from_document(existing, duplicate=True)

        document_raw_dir = self.raw_dir / document_id
        document_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = document_raw_dir / safe_name
        raw_path.write_bytes(content)

        try:
            parsed = parse_document(raw_path)
        except UnsupportedDocumentError as error:
            raise UnsupportedFileError(str(error)) from error
        except Exception as error:
            raise IngestionError(f"failed to parse {safe_name}: {error}") from error

        chunks = build_chunks(
            document_id,
            parsed.blocks,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        document = ProcessedDocument(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            filename=safe_name,
            file_type=extension.removeprefix("."),
            content_type=content_type,
            checksum_sha256=checksum,
            title=parsed.title,
            created_at=datetime.now(UTC),
            raw_path=str(raw_path),
            processed_path=str(processed_path),
            blocks=parsed.blocks,
            chunks=chunks,
            metadata={**parsed.metadata, "processing": processing_config},
        )
        temporary_path = processed_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(processed_path)
        return DocumentSummary.from_document(document)

    def load(self, document_id: str) -> ProcessedDocument:
        if not document_id.startswith("doc-") or "/" in document_id or ".." in document_id:
            raise DocumentNotFoundError("document not found")
        path = self.processed_dir / f"{document_id}.json"
        if not path.exists():
            raise DocumentNotFoundError(f"document not found: {document_id}")
        return ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def list_documents(self, knowledge_base_id: str | None = None) -> list[DocumentSummary]:
        documents: list[DocumentSummary] = []
        for path in sorted(self.processed_dir.glob("doc-*.json")):
            document = ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))
            if knowledge_base_id is None or document.knowledge_base_id == knowledge_base_id:
                documents.append(DocumentSummary.from_document(document))
        return sorted(documents, key=lambda item: item.created_at, reverse=True)

    def reprocess(self, document_id: str) -> DocumentSummary:
        document = self.load(document_id)
        raw_path = Path(document.raw_path)
        if not raw_path.exists():
            raise DocumentNotFoundError(f"raw document not found: {document_id}")
        processing = document.metadata.get("processing") or {}
        markdown_path = document.metadata.get("mineru_markdown_path")
        if processing.get("parser") == "mineru" and markdown_path:
            extracted_path = Path(str(markdown_path))
            if extracted_path.exists():
                return self.ingest_extracted_markdown(
                    filename=document.filename,
                    content=raw_path.read_bytes(),
                    markdown=extracted_path.read_text(encoding="utf-8"),
                    content_type=document.content_type,
                    knowledge_base_id=document.knowledge_base_id,
                )
        return self.ingest_bytes(
            filename=document.filename,
            content=raw_path.read_bytes(),
            content_type=document.content_type,
            knowledge_base_id=document.knowledge_base_id,
            force=True,
        )

    def ingest_extracted_markdown(
        self,
        *,
        filename: str,
        content: bytes,
        markdown: str,
        content_type: str | None = None,
        knowledge_base_id: str = DEFAULT_KNOWLEDGE_BASE_ID,
    ) -> DocumentSummary:
        safe_name = Path(filename).name.strip()
        knowledge_base_id = self._validate_knowledge_base_id(knowledge_base_id)
        extension = Path(safe_name).suffix.lower()
        if not safe_name or extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(f"unsupported file type: {extension or '(none)'}")
        if not content or not markdown.strip():
            raise IngestionError("MinerU extracted content is empty")
        if len(content) > self.max_upload_bytes:
            raise FileTooLargeError(
                f"file size exceeds limit of {self.max_upload_bytes // (1024 * 1024)} MB"
            )

        checksum = hashlib.sha256(content).hexdigest()
        document_id = self._document_id(knowledge_base_id, checksum)
        processed_path = self.processed_dir / f"{document_id}.json"
        document_raw_dir = self.raw_dir / document_id
        document_raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = document_raw_dir / safe_name
        raw_path.write_bytes(content)
        markdown_path = document_raw_dir / "mineru-full.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        parsed = parse_document(markdown_path)
        chunks = build_chunks(
            document_id,
            parsed.blocks,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        processing_config = {
            "parser_version": PARSER_VERSION,
            "parser": "mineru",
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        document = ProcessedDocument(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            filename=safe_name,
            file_type=extension.removeprefix("."),
            content_type=content_type,
            checksum_sha256=checksum,
            title=parsed.title,
            created_at=datetime.now(UTC),
            raw_path=str(raw_path),
            processed_path=str(processed_path),
            blocks=parsed.blocks,
            chunks=chunks,
            metadata={
                **parsed.metadata,
                "processing": processing_config,
                "mineru_markdown_path": str(markdown_path),
            },
        )
        temporary_path = processed_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(processed_path)
        return DocumentSummary.from_document(document)

    def delete(self, document_id: str) -> DocumentSummary:
        document = self.load(document_id)
        processed_path = Path(document.processed_path)
        if processed_path.exists():
            processed_path.unlink()
        raw_dir = self.raw_dir / document_id
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        return DocumentSummary.from_document(document)

    @staticmethod
    def _validate_knowledge_base_id(value: str) -> str:
        clean = value.strip()
        if not re.fullmatch(r"kb-[A-Za-z0-9-]{1,80}", clean):
            raise IngestionError("invalid knowledge_base_id")
        return clean

    @staticmethod
    def _document_id(knowledge_base_id: str, checksum: str) -> str:
        if knowledge_base_id == DEFAULT_KNOWLEDGE_BASE_ID:
            return f"doc-{checksum[:20]}"
        digest = hashlib.sha256(f"{knowledge_base_id}:{checksum}".encode()).hexdigest()
        return f"doc-{digest[:20]}"
