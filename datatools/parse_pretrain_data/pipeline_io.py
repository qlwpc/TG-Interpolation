"""Small, dependency-light integrity helpers for the pretraining pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_index(path: Path) -> dict[str, list[int]]:
    """Preserve the released selection order; never coerce or deduplicate IDs."""
    raw = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: split index must be an object")
    for key, values in raw.items():
        if not isinstance(values, list) or any(type(i) is not int or i < 0 for i in values):
            raise ValueError(f"{path}: {key} must contain a list of nonnegative integers")
        if len(values) != len(set(values)):
            raise ValueError(f"{path}: duplicate document indices in {key}")
    return raw


def validate_split_indices(dev_path: Path, test_path: Path, order, *, released=False) -> dict:
    dev, test = load_index(dev_path), load_index(test_path)
    unknown = (set(dev) | set(test)) - set(order)
    if unknown:
        raise ValueError(f"unknown split shard keys: {sorted(unknown)}")
    for key in set(dev) | set(test):
        if set(dev.get(key, [])) & set(test.get(key, [])):
            raise ValueError(f"dev/test document indices overlap in {key}")
    result = {
        "protocol": "released-bbc-indices-v1" if released else "custom-indices",
        "selection_order": "shard-order, then JSON list order; train is source order",
        "train_only_shards": [key for key in order if not dev.get(key) and not test.get(key)],
    }
    pins = json.loads(Path(__file__).with_name("bbc_split_manifest.json").read_text())
    for split, path, index in (("dev", dev_path, dev), ("test", test_path, test)):
        record = {"path": str(path.resolve()), "sha256": sha256_file(path),
                  "shards": len(index), "documents": sum(map(len, index.values()))}
        if released and any(record[key] != pins[split][key] for key in ("sha256", "shards", "documents")):
            raise ValueError(f"{split} index differs from the released BBC split: {path}")
        result[split] = record
    return result
