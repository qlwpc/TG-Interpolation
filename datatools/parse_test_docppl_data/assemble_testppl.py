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

import numpy as np


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
    parser.add_argument("--tree-source", type=Path, default=BBC / "testppl_tree")
    parser.add_argument("--tg-source", type=Path, default=BBC / "testppl_tg")
    parser.add_argument("--aligned-source", type=Path, default=BBC / "testppl_aligned")
    args = parser.parse_args()
    output = args.output
    sources = {
        "tree300": args.tree_source,
        "tg300": args.tg_source,
        "aligned": args.aligned_source,
        "native_model_topk_300_v2": args.native_source,
    }
    native = json.loads((args.native_source / "manifest.json").read_text())
    aligned = json.loads((args.aligned_source / "manifest.json").read_text())
    if native.get("status") != "complete" or native["candidate_slots"] != 300:
        raise ValueError("native candidate corpus is incomplete or has incompatible slots")
    documents, sentences = native["document_count"], native["sentence_count"]
    if (aligned["contract"]["documents"], aligned["contract"]["sentences"]) != (documents, sentences):
        raise ValueError("aligned corpus counts differ from native candidates")
    for path, prefix in ((args.tree_source, "tree"), (args.tg_source, "tg")):
        docs = np.load(path / f"{prefix}_doc_index.npy", mmap_mode="r")
        lengths = np.load(path / f"{prefix}_sent_index.npy", mmap_mode="r")
        tokens = np.load(path / f"{prefix}_300.npy", mmap_mode="r")
        if (len(docs) != documents or int(docs.sum(dtype=np.uint64)) != sentences or
                len(lengths) != 300 * sentences or int(lengths.sum(dtype=np.uint64)) != len(tokens)):
            raise ValueError(f"serialized corpus counts do not cover its arrays: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        _link_tree(source, output / name)
    manifest = {
        "format_version": 1,
        "document_count": documents,
        "sentence_count": sentences,
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
