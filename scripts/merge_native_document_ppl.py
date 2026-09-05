#!/usr/bin/env python
"""Validate and merge native DocPPL results without mixing scoring contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.native_document_results import aggregate_rows, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("gpst", "pushdown"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-documents", type=int)
    parser.add_argument("--expected-samples-per-sentence", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_documents is not None and args.expected_documents <= 0:
        parser.error("--expected-documents must be positive")
    if args.expected_samples_per_sentence is not None and args.expected_samples_per_sentence <= 0:
        parser.error("--expected-samples-per-sentence must be positive")
    root = args.directory
    document_dir = root / "documents" if (root / "documents").is_dir() else root
    documents = sorted(document_dir.glob("document_*.json"))
    paths = documents or sorted(root.glob("shard_*.json"))
    try:
        rows = []
        for path in paths:
            row = json.loads(path.read_text())
            if documents:
                doc_id = row.get("document_id")
                if type(doc_id) is not int or path.name != f"document_{doc_id:05d}.json":
                    raise ValueError(f"document filename/ID mismatch: {path}")
            rows.append(row)
        expected = set(range(args.expected_documents)) if args.expected_documents is not None else None
        output = aggregate_rows(args.model, rows, expected, args.expected_samples_per_sentence)
        if args.output:
            atomic_json(args.output, output)
        else:
            print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError) as error:
        parser.exit(1, f"invalid document results: {error}\n")


if __name__ == "__main__":
    main()
