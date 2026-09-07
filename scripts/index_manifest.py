from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_kg_rag.rag.router import get_rag_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index only documents listed in a manifest.")
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    document_ids = [item["document_id"] for item in manifest["documents"]]
    service = get_rag_service()
    total_chunks = 0
    for document_id in document_ids:
        result = service.index_document(document_id)
        total_chunks += result.indexed_chunks
        print(f"indexed: {result.filename} ({result.indexed_chunks} chunks)")
    print(f"complete: {len(document_ids)} documents, {total_chunks} chunks")


if __name__ == "__main__":
    main()
