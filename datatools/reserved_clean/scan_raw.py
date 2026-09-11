"""Freeze reserved rows, scan every historical raw row, export unique bodies.

Run in a NEW output directory. Source files are read-only. A completion receipt
is written only after source hashes and the independent census counts agree.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

from datatools.reserved_clean.common import RESERVED, atomic_json, body_from_parse, digest, emit, sha_file
from diagnostics import audit_bbc_raw_duplicates as legacy


def worker_init(config):
    legacy.init_worker(config)


def encode(batch):
    blobs = legacy.encode_batch(batch)
    return [(blob, body_from_parse(raw.decode("utf-8"))) for raw, blob in zip(batch, blobs)]


def run(args):
    manifest = json.loads(args.manifest.read_text())
    census = json.loads(args.census.read_text())
    expected = {s["shard"]: s for s in census["sources"]}
    historical = [s for s in manifest["order"] if s not in RESERVED]
    if set(historical) != set(manifest["test"]) or len(historical) != 89:
        raise ValueError("unexpected historical shard set")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "inputs.json", {
        "manifest_sha256": sha_file(args.manifest), "census_sha256": sha_file(args.census),
        "script_sha256": sha_file(__file__), "common_sha256": sha_file(Path(__file__).with_name("common.py")),
        "legacy_script_sha256": sha_file(legacy.__file__),
        "parsed_dir": str(args.parsed_dir), "workers": args.workers,
        "source_sha256": {s: expected[s]["file_sha256"] for s in manifest["order"]},
    })
    # Raw cache retains bytes, so every cache hit is byte-verified. Strict
    # candidate hits are also verified against the complete terminal bytes.
    cache, reserved_groups, normalized_seen = {}, {}, {}
    strict_seen = set()
    started = time.monotonic()
    source_results = []
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=worker_init,
                             initargs=(manifest["tokenizer"],)) as pool, \
            gzip.open(args.output / "reserved.jsonl.gz", "wt", compresslevel=1, encoding="utf-8") as reserved_out, \
            gzip.open(args.output / "historical_bodies.jsonl.gz", "wt", compresslevel=1, encoding="utf-8") as references:
        for shard in [*RESERVED, *historical]:
            path = args.parsed_dir / (shard + ".txt")
            before = path.stat()
            is_reserved = shard in RESERVED
            filehash, row = hashlib.sha256(), 0
            with path.open("rb") as f:
                while True:
                    batch = []
                    for _ in range(args.batch_rows):
                        raw = f.readline()
                        if not raw:
                            break
                        if not raw.endswith(b"\n"):
                            raise ValueError(f"unterminated source row: {shard}/{row+len(batch)}")
                        filehash.update(raw)
                        batch.append(raw)
                    if not batch:
                        break
                    keys, fresh = [], {}
                    for raw in batch:
                        key = hashlib.sha256(raw).digest()
                        keys.append(key)
                        if key in cache:
                            if cache[key][0] != raw:
                                raise ValueError("raw hash collision")
                        elif key in fresh:
                            if fresh[key] != raw:
                                raise ValueError("raw hash collision within batch")
                        else:
                            fresh[key] = raw
                    items = list(fresh.items())
                    chunks = [[r for _, r in items[i:i+32]] for i in range(0, len(items), 32)]
                    encoded = (v for values in pool.map(encode, chunks) for v in values)
                    for (key, raw), (blob, body) in zip(items, encoded):
                        strict = hashlib.sha256(blob).hexdigest()
                        strict_seen.add(strict)
                        cache[key] = (raw, strict, blob, body)
                    for i, key in enumerate(keys):
                        raw, strict, blob, body = cache[key]
                        location = [shard, row+i]
                        if is_reserved:
                            g = reserved_groups.setdefault(strict, {"blob": blob, "sources": [],
                                "historical_copies": 0, "first_historical_source": None})
                            if g["blob"] != blob:
                                raise ValueError("terminal hash collision")
                            g["sources"].append(location)
                            emit(reserved_out, {"source_shard": shard, "source_row": row+i,
                                "raw_sha256": key.hex(), "parsed": raw.decode("utf-8"),
                                "strict_sha256": strict, "terminal_b64": base64.b64encode(blob).decode(),
                                "body": body, "body_sha256": digest(body)})
                        else:
                            if strict in reserved_groups:
                                g = reserved_groups[strict]
                                if g["blob"] != blob:
                                    raise ValueError("terminal hash collision")
                                g["historical_copies"] += 1
                                if g["first_historical_source"] is None:
                                    g["first_historical_source"] = location
                            bodykey = digest(body)
                            if bodykey in normalized_seen:
                                if normalized_seen[bodykey] != body:
                                    raise ValueError("normalized body hash collision")
                            else:
                                normalized_seen[bodykey] = body
                                emit(references, {"source": location, "body_sha256": bodykey, "body": body})
                    row += len(batch)
                    if row % (args.batch_rows*16) == 0:
                        print(json.dumps({"shard": shard, "rows": row, "unique_raw": len(cache),
                            "seconds": round(time.monotonic()-started)}), file=sys.stderr, flush=True)
            after = path.stat()
            result = {"shard": shard, "rows": row, "sha256": filehash.hexdigest()}
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"source changed: {path}")
            if row != expected[shard]["rows"] or filehash.hexdigest() != expected[shard]["file_sha256"]:
                raise ValueError(f"source does not match frozen census: {result}")
            source_results.append(result)
            print(json.dumps({"complete_shard": result, "seconds": round(time.monotonic()-started)}),
                  file=sys.stderr, flush=True)
            if is_reserved:
                reserved_out.flush()
            else:
                references.flush()
    with (args.output / "strict_groups.jsonl").open("w") as f:
        for key, g in reserved_groups.items():
            emit(f, {"strict_sha256": key, **{k: v for k, v in g.items() if k != "blob"}})
    eligible = sum(g["historical_copies"] == 0 for g in reserved_groups.values())
    if eligible != 16501 or len(reserved_groups) != 66523 or len(strict_seen) != 2104821:
        raise ValueError(f"independent census mismatch: {eligible}, {len(reserved_groups)}, {len(strict_seen)}")
    result = {"complete": True, "stage": "raw_exact", "claim_type": "computed", "sources": source_results,
        "reserved_records": sum(len(g["sources"]) for g in reserved_groups.values()),
        "reserved_unique_strict": len(reserved_groups), "strict_candidates": eligible,
        "all_unique_strict": len(strict_seen), "historical_unique_normalized": len(normalized_seen),
        "seconds": round(time.monotonic()-started),
        "outputs": {p.name: sha_file(p) for p in args.output.iterdir() if p.is_file()}}
    atomic_json(args.output / "complete.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--parsed-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--batch-rows", type=int, default=4096)
    run(p.parse_args())
