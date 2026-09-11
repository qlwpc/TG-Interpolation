"""Directly audit frozen, existing arrays; never infer training from row rules."""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import time
import sys

import numpy as np
from tokenizers import Tokenizer

from datatools.reserved_clean.common import atomic_json, digest, emit, normalize, records, sha_file


def documents(array, chunk=8_000_000):
    if array.ndim != 1 or array.dtype != np.dtype("<u2") or not len(array) or array[0] != 50257:
        raise ValueError("expected a nonempty little-endian uint16 document stream")
    pending = 0
    for begin in range(0, len(array), chunk):
        starts = np.flatnonzero(array[begin:begin+chunk] == 50257) + begin
        for end in starts:
            if end > pending:
                part = array[pending:end]
                if part[-1] != 50256 or np.count_nonzero(part == 50256) != 1:
                    raise ValueError(f"invalid EOS at offset {pending}")
                yield pending, part
            pending = int(end)
    part = array[pending:]
    if part[-1] != 50256 or np.count_nonzero(part == 50256) != 1:
        raise ValueError(f"invalid final EOS at offset {pending}")
    yield pending, part


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    # During raw scanning the gzip member is still open. The independently
    # pinned source count lets us read exactly its complete reserved prefix.
    targets = {}
    count = 0
    for r in islice(records(args.reserved), 66628):
        blob = base64.b64decode(r["terminal_b64"])
        if hashlib.sha256(blob).hexdigest() != r["strict_sha256"]:
            raise ValueError("target fingerprint mismatch")
        targets.setdefault(r["strict_sha256"], blob)
        count += 1
    if count != 66628 or len(targets) != 66523:
        raise ValueError("incomplete reserved target prefix")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    inputs = json.loads(args.inputs.read_text())
    atomic_json(args.output / "inputs.json", {"arrays": inputs, "tokenizer_sha256": sha_file(args.tokenizer),
        "script_sha256": sha_file(__file__), "normalization": "NFKC casefold punctuation-token-spacing v1",
        "reserved_unique_targets": len(targets)})
    seen, seen_bodies, matches = {}, {}, defaultdict(list)
    sources = []
    started = time.monotonic()
    with gzip.open(args.output / "array_bodies.jsonl.gz", "wt", compresslevel=1, encoding="utf-8") as out:
        for spec in inputs:
            path = Path(spec["path"])
            before = path.stat()
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            pending = []
            ndoc = 0

            def flush():
                decoded = tokenizer.decode_batch([ids for _, _, ids in pending], skip_special_tokens=False)
                for (key, source, _), decoded_text in zip(pending, decoded):
                    body = normalize(decoded_text)
                    bodykey = digest(body)
                    if bodykey in seen_bodies:
                        if seen_bodies[bodykey] != body:
                            raise ValueError("body hash collision")
                    else:
                        seen_bodies[bodykey] = body
                        emit(out, {"source": source, "body_sha256": bodykey, "body": body,
                                   "strict_sha256": key})
                pending.clear()

            for d, (offset, part) in enumerate(documents(arr)):
                blob = part.tobytes()
                key = hashlib.sha256(blob).hexdigest()
                source = [spec["name"], d, offset]
                if key in targets:
                    if targets[key] != blob:
                        raise ValueError("target hash collision")
                    matches[key].append(source)
                if key in seen:
                    if seen[key] != blob:
                        raise ValueError("array content hash collision")
                else:
                    seen[key] = blob
                    pending.append((key, source, part[1:-1].tolist()))
                if len(pending) >= 256:
                    flush()
                ndoc = d+1
                if ndoc % 250000 == 0:
                    print(json.dumps({"array": spec["name"], "documents": ndoc, "unique": len(seen),
                        "seconds": round(time.monotonic()-started)}), file=sys.stderr, flush=True)
            flush()
            actual_hash = sha_file(path)
            after = path.stat()
            if actual_hash != spec["sha256"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"array changed or differs from frozen input: {path}")
            if "documents" in spec and ndoc != spec["documents"]:
                raise ValueError(f"document count mismatch: {path}")
            sources.append({**spec, "documents": ndoc, "tokens": len(arr), "hash_verified": True})
    with (args.output / "strict_matches.jsonl").open("w") as f:
        for key, locations in sorted(matches.items()):
            emit(f, {"strict_sha256": key, "sources": locations})
    atomic_json(args.output / "complete.json", {"complete": True, "claim_type": "computed", "sources": sources,
        "unique_normalized_bodies": len(seen_bodies), "matched_reserved_unique": len(matches),
        "seconds": round(time.monotonic()-started),
        "outputs": {p.name: sha_file(p) for p in args.output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reserved", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
