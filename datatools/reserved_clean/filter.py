"""Materialize exact candidates, then run complete shingle matching."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

from datatools.reserved_clean.common import atomic_json, digest, emit, records, sha_file


def validate_receipt(folder):
    receipt = json.loads((folder / "complete.json").read_text())
    if receipt.get("complete") is not True:
        raise ValueError(f"incomplete prerequisite: {folder}")
    for name, expected in receipt["outputs"].items():
        if sha_file(folder/name) != expected:
            raise ValueError(f"changed prerequisite: {folder/name}")
    return receipt


def prepare(raw, output):
    receipt = validate_receipt(raw)
    output.mkdir(parents=True, exist_ok=False)
    groups = {r["strict_sha256"]: r for r in records(raw/"strict_groups.jsonl")}
    seen, n = set(), 0
    with gzip.open(output/"candidates.jsonl.gz", "wt", compresslevel=1, encoding="utf-8") as f, \
            (output/"candidates.tsv").open("w") as tsv, \
            (output/"raw_exclusions.jsonl").open("w") as exclusions:
        for r in records(raw/"reserved.jsonl.gz"):
            key = r["strict_sha256"]
            g = groups[key]
            if g["historical_copies"]:
                emit(exclusions, {"source_shard": r["source_shard"], "source_row": r["source_row"],
                    "strict_sha256": key, "reason": "historical_pool_exact", "match": g["first_historical_source"]})
            elif key in seen:
                emit(exclusions, {"source_shard": r["source_shard"], "source_row": r["source_row"],
                    "strict_sha256": key, "reason": "reserved_exact_duplicate", "representative": g["sources"][0]})
            else:
                r.update(candidate_id=n, strict_sources=g["sources"])
                if digest(r["body"]) != r["body_sha256"] or "\t" in r["body"] or "\n" in r["body"]:
                    raise ValueError("invalid normalized candidate body")
                emit(f, r)
                tsv.write(f"{n}\t{r['body']}\n")
                seen.add(key)
                n += 1
    if n != receipt["strict_candidates"]:
        raise ValueError("candidate count differs from raw scan")
    atomic_json(output/"complete.json", {"complete": True, "candidates": n,
        "raw_receipt_sha256": sha_file(raw/"complete.json"),
        "outputs": {p.name: sha_file(p) for p in output.iterdir() if p.is_file()}})


def merge_external(output, workers):
    merged = None
    for worker in range(workers):
        with (output/f"external.{worker}.tsv").open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if merged is None:
            merged = rows
            continue
        if len(rows) != len(merged):
            raise ValueError("worker candidate coverage differs")
        for a, b in zip(merged, rows):
            if (a["candidate"], a["words"]) != (b["candidate"], b["words"]):
                raise ValueError("worker candidate identities differ")
            bits = int(a["covered_bitmap"] or "0", 2) | int(b["covered_bitmap"] or "0", 2)
            n = int(a["words"])
            if ((int(b["exact"]) and not int(a["exact"])) or
                    (int(b["near"]) and not int(a["exact"]) and not int(a["near"])) or
                    a["first_source"] == "-"):
                for key in ("first_source", "jaccard", "containment"):
                    a[key] = b[key]
            a["exact"] = str(int(bool(int(a["exact"]) or int(b["exact"]))))
            a["near"] = str(int(bool(int(a["near"]) or int(b["near"]))))
            a["covered_bitmap"] = format(bits, f"0{n}b") if n else ""
            a["covered_words"] = str(bits.bit_count())
            a["coverage"] = str(bits.bit_count()/n if n else 0)
    if not merged:
        raise ValueError("matcher returned no candidates")
    with (output/"external.tsv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(merged)
    with (output/"external_edges.tsv").open("w") as f:
        for worker in range(workers):
            with (output/f"external_edges.{worker}.tsv").open() as src:
                header = next(src)
                if worker == 0: f.write(header)
                for line in src: f.write(line)


def match(args):
    candidate_receipt = validate_receipt(args.candidates)
    reference_hashes = {str(path): sha_file(path) for path in args.references}
    if args.raw:
        receipt = validate_receipt(args.raw)
        reference_hashes[str(args.raw/"historical_bodies.jsonl.gz")] = receipt["outputs"]["historical_bodies.jsonl.gz"]
    if args.arrays:
        receipt = validate_receipt(args.arrays)
        reference_hashes[str(args.arrays/"array_bodies.jsonl.gz")] = receipt["outputs"]["array_bodies.jsonl.gz"]
    args.output.mkdir(parents=True, exist_ok=False)
    src = Path(__file__).with_name("match.cpp")
    binary = args.output/"match"
    subprocess.run(["g++", "-O3", "-std=c++17", "-Wall", "-Wextra", "-o", str(binary), str(src)], check=True)
    paths = list(args.references)
    if args.raw:
        paths.append(args.raw/"historical_bodies.jsonl.gz")
    if args.arrays:
        paths.append(args.arrays/"array_bodies.jsonl.gz")
    if not paths:
        raise ValueError("at least one audited reference corpus is required")
    tsv = args.candidates/"candidates.tsv"
    total, unique = 0, {}
    workers = args.workers
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    with ExitStack() as stack:
        processes = []
        for worker in range(workers):
            log = stack.enter_context((args.output/f"external.{worker}.log").open("w"))
            processes.append(subprocess.Popen([str(binary), str(tsv), str(args.output/f"external.{worker}.tsv"),
                str(args.output/f"external_edges.{worker}.tsv"), "external"], stdin=subprocess.PIPE, text=True, stderr=log))
        try:
            for path in paths:
                for r in records(path):
                    total += 1
                    key, body = r["body_sha256"], r["body"]
                    if digest(body) != key:
                        raise ValueError("reference body fingerprint mismatch")
                    if key in unique:
                        if unique[key] != body:
                            raise ValueError("reference hash collision")
                        continue
                    unique[key] = body
                    source = json.dumps(r["source"], ensure_ascii=False, separators=(",", ":"))
                    worker = ((len(unique)-1)//128) % workers
                    processes[worker].stdin.write(f"{source}\t{body}\n")
                    if len(unique) % 100000 == 0:
                        print(json.dumps({"matching_unique_references": len(unique), "records_read": total}),
                              file=sys.stderr, flush=True)
            for process in processes: process.stdin.close()
            statuses = [process.wait() for process in processes]
            if any(statuses):
                raise RuntimeError(f"external matcher failed: {statuses}")
        except BaseException:
            for process in processes:
                if process.poll() is None: process.kill()
            for process in processes: process.wait()
            raise
    if not unique:
        raise ValueError("empty reference corpus cannot certify clean candidates")
    merge_external(args.output, workers)
    with tsv.open() as f, (args.output/"internal.log").open("w") as log:
        subprocess.run([str(binary), str(tsv), str(args.output/"internal.tsv"),
            str(args.output/"internal_edges.tsv"), "internal"], stdin=f, stderr=log, check=True)
    for path in paths:
        if sha_file(path) != reference_hashes[str(path)]:
            raise ValueError(f"reference changed during matching: {path}")
    atomic_json(args.output/"complete.json", {"complete": True,
        "candidate_receipt_sha256": sha_file(args.candidates/"complete.json"),
        "candidate_count": candidate_receipt["candidates"],
        "prerequisite_receipts": {name: sha_file(folder/"complete.json") for name,folder in
            (("raw",args.raw),("arrays",args.arrays)) if folder is not None},
        "references": reference_hashes, "reference_records": total,
        "unique_references": len(unique), "matcher_source_sha256": sha_file(src),
        "orchestrator_sha256": sha_file(__file__), "workers": workers,
        "protocol": {"normalization": "NFKC casefold punctuation-token-spacing v1", "near_ngram": 5,
            "jaccard": 0.8, "containment": 0.8, "containment_min_words": 50, "containment_min_shared_grams": 20,
            "local_ngram": 13, "local_coverage": 0.3, "templates": "candidate DF >= max(100, ceil(N/100)); local only",
            "retrieval": "all shared ngrams, full string equality, no approximate retrieval"},
        "outputs": {p.name: sha_file(p) for p in args.output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    m = sub.add_parser("match")
    m.add_argument("--candidates", type=Path, required=True)
    m.add_argument("--raw", type=Path)
    m.add_argument("--arrays", type=Path)
    m.add_argument("--references", type=Path, nargs="*", default=[])
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    prepare(a.raw, a.output) if a.command == "prepare" else match(a)
