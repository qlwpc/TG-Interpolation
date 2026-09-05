"""Completion receipts for the three aligned token streams of a parsed shard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datatools.parse_pretrain_data.pipeline_io import sha256_file

FORMATS = ("terminal", "tree", "tg")
TOKENIZATION_PROTOCOL = "explicit-bos-eos-ptb-mapping-v1"


def receipt_path(root: Path, stem: str) -> Path:
    return root / "manifests" / f"{stem}.json"


def verify_receipt(root: Path, stem: str, tokenizer_sha256: str, source_sha256=None) -> dict:
    path = receipt_path(root, stem)
    if not path.is_file():
        raise ValueError(f"missing tokenization receipt: {path}; rebuild with --overwrite")
    record = json.loads(path.read_text())
    if record.get("schema_version") != 1 or record.get("protocol") != TOKENIZATION_PROTOCOL:
        raise ValueError(f"incompatible tokenization receipt: {path}")
    if record.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError(f"tokenizer fingerprint changed: {path}")
    if source_sha256 is not None and record.get("source_sha256") != source_sha256:
        raise ValueError(f"parsed source fingerprint changed: {path}")
    if not isinstance(record.get("documents"), int) or record["documents"] <= 0:
        raise ValueError(f"invalid document count: {path}")
    for fmt in FORMATS:
        output = root / fmt / f"{stem}.npy"
        info = record.get("formats", {}).get(fmt, {})
        if not output.is_file() or sha256_file(output) != info.get("sha256"):
            raise ValueError(f"missing or changed token output: {output}")
        array = np.load(output, mmap_mode="r", allow_pickle=False)
        if (array.ndim != 1 or array.size != info.get("tokens") or
                str(array.dtype) != record.get("dtype")):
            raise ValueError(f"invalid token output: {output}")
    return record
