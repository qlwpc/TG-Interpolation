#!/usr/bin/env python
"""Assemble the local BBC document-PPL data contract without duplicating arrays.

The historical serialized Tree/TG arrays remain useful provenance artifacts,
but GPST and Pushdown evaluation consumes only ``native_model_topk_300_v2``.
Local arrays are hard-linked, so the unified view costs metadata only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
BBC = REPO / "dataset" / "bbc-news"


def _files(source: Path) -> Iterable[Path]:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            yield path


def _link_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in _files(source):
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.samefile(path):
                raise FileExistsError(f"refusing to replace {target}")
            continue
        os.link(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BBC / "testppl")
    parser.add_argument("--native-source", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "tree300": BBC / "testppl_tree",
        "tg300": BBC / "testppl_tg",
        "aligned": BBC / "testppl_aligned",
        "native_model_topk_300_v2": args.native_source,
    }
    for name, source in sources.items():
        _link_tree(source, output / name)
    manifest = {
        "format_version": 1,
        "document_count": 4966,
        "sentence_count": 148836,
        "candidate_slots": 300,
        "evaluation": {
            "gpst": "native_model_topk_300_v2 (strict-binary merge orders)",
            "pushdown": "native_model_topk_300_v2 (unary-free n-ary spans)",
            "serialized_tree_tg": "provenance only; not parsed by document-PPL evaluators",
        },
        "entries": {name: str(source) for name, source in sources.items()},
        "storage": "hard links for local arrays; native data is copied from RTX3090 before assembly",
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "manifest.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
