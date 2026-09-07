from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from agentic_kg_rag.config import get_settings
from agentic_kg_rag.ingestion.parsers import SUPPORTED_EXTENSIONS
from agentic_kg_rag.ingestion.service import DocumentIngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest every supported document in a directory.")
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/documents.json"),
        help="JSON manifest used for selective indexing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.directory.resolve()
    if not root.is_dir():
        raise SystemExit(f"directory does not exist: {root}")

    settings = get_settings()
    service = DocumentIngestionService(
        data_dir=settings.data_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_mb * 1024 * 1024,
    )
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    documents = []
    for path in paths:
        summary = service.ingest_bytes(filename=path.name, content=path.read_bytes())
        item = summary.model_dump(mode="json")
        item["source_path"] = str(path.relative_to(root))
        if summary.chunk_count == 0:
            print(f"skipped (no content): {item['source_path']}")
            continue
        documents.append(item)
        state = "duplicate" if summary.duplicate else "ingested"
        print(f"{state}: {item['source_path']} -> {summary.document_id}")

    manifest = {
        "source_root": str(root),
        "created_at": datetime.now(UTC).isoformat(),
        "document_count": len(documents),
        "documents": documents,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {args.manifest} ({len(documents)} documents)")


if __name__ == "__main__":
    main()
