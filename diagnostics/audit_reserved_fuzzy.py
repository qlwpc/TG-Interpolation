"""Fresh final-test versus actual-train audit; no sampling or corpus transfers.

python -m diagnostics.audit_reserved_fuzzy --output artifacts/reserved_fuzzy_20260906
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from tokenizers import Tokenizer

from datatools.reserved_clean.common import atomic_json, normalize, records, sha_file
from datatools.reserved_clean.scan_arrays import documents


def merge(paths, targets):
    csv.field_size_limit(sys.maxsize)
    merged = None
    for path in paths:
        with path.open() as f:
            rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        if len(rows) != len(targets):
            raise ValueError("incomplete worker target coverage")
        for i, r in enumerate(rows):
            if int(r["test_doc"]) != i or int(r["words"]) != len(targets[i]["body"].split()):
                raise ValueError("worker target identity mismatch")
            for key in ("jaccard", "containment", "local_pair"):
                r[key] = float(r[key])
            r["exact"] = bool(int(r["exact"]))
        if merged is None:
            merged = rows
            continue
        for a, b in zip(merged, rows):
            a["exact"] |= b["exact"]
            a["bitmap"] = format(int(a["bitmap"] or "0", 2) | int(b["bitmap"] or "0", 2), f'0{a["words"]}b')
            for score, prefix in (("jaccard", "jac"), ("containment", "cont"), ("local_pair", "local")):
                if b[score] > a[score]:
                    for key in (score, prefix + "_source", prefix + "_body"):
                        a[key] = b[key]
    for r, target in zip(merged, targets):
        r["test_doc"] = int(r["test_doc"])
        r["words"] = int(r["words"])
        r["covered_words"] = r["bitmap"].count("1")
        r["coverage"] = r["covered_words"] / r["words"]
        r["candidate_id"] = target["candidate_id"]
        r["source"] = [target["source_shard"], target["source_row"]]
    return merged


def run(args):
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.dataset
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha_file(root / name) != expected:
            raise ValueError(f"published dataset changed: {name}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if sha_file(args.tokenizer) != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer differs from frozen version")
    targets = list(records(root / "test_selected.jsonl.gz"))
    arr = np.load(root / "terminal/test.npy", mmap_mode="r")
    target_hashes = {}
    offsets = []
    with (args.output / "targets.tsv").open("w") as f:
        for d, (offset, part) in enumerate(documents(arr)):
            body = normalize(tokenizer.decode(part[1:-1].tolist(), skip_special_tokens=False))
            if body != targets[d]["body"]:
                raise ValueError("final scoring array differs from selected body")
            blob = part.tobytes()
            key = hashlib.sha256(blob).digest()
            target_hashes.setdefault(key, []).append((d, blob))
            offsets.append(offset)
            f.write(f"{d}\t{body}\n")
    if len(offsets) != len(targets):
        raise ValueError("test document count mismatch")
    subset = json.loads((root / "docppl/test_1000/document_map.json").read_text())
    for r in subset:
        if targets[r["full_split_doc_id"]]["candidate_id"] != r["candidate_id"]:
            raise ValueError("DocPPL subset mapping mismatch")
    src = Path(__file__).with_suffix(".cpp")
    binary = args.output / "matcher"
    subprocess.run(["g++", "-O3", "-std=c++17", "-Wall", "-Wextra", str(src), "-o", str(binary)], check=True)
    atomic_json(args.output / "inputs.json", {
        "train": str(args.train), "expected_train_sha256": args.expected_train_sha256,
        "dataset_manifest_sha256": sha_file(root / "manifest.json"),
        "python_sha256": sha_file(__file__), "cpp_sha256": sha_file(src),
        "included_cpp_sha256": sha_file(src.parent.parent / "datatools/reserved_clean/match.cpp"),
        "numpy_version": np.__version__, "workers": args.workers,
        "protocol": "NFKC/casefold/punctuation tokens; exact full strings; exhaustive shared 5grams; eligible containment requires >=50 tokens both sides and >=20 shared distinct grams; union of all shared 13gram positions, NO template masking",
    })
    processes, logs = [], []
    before = args.train.stat()
    train = np.load(args.train, mmap_mode="r")
    seen, seen_bodies, pending, strict_matches = {}, {}, [], []
    ndoc = unique = 0
    try:
        for w in range(args.workers):
            log = (args.output / f"worker.{w}.log").open("w"); logs.append(log)
            processes.append(subprocess.Popen([str(binary), str(args.output / "targets.tsv"), str(args.output / f"worker.{w}.tsv")], stdin=subprocess.PIPE, text=True, stderr=log))

        def flush():
            nonlocal unique
            decoded = tokenizer.decode_batch([x[1] for x in pending], skip_special_tokens=False)
            for (source, _), text in zip(pending, decoded):
                body = normalize(text)
                key = hashlib.sha256(body.encode()).digest()
                if key in seen_bodies:
                    if seen_bodies[key] != body:
                        raise ValueError("normalized hash collision")
                    continue
                seen_bodies[key] = body
                processes[(unique // 128) % args.workers].stdin.write(f"{source}\t{body}\n")
                unique += 1
            pending.clear()

        for d, (offset, part) in enumerate(documents(train)):
            blob = part.tobytes(); key = hashlib.sha256(blob).digest()
            for target_d, expected_blob in target_hashes.get(key, []):
                if blob != expected_blob:
                    raise ValueError("strict hash collision")
                strict_matches.append({"test_doc": target_d, "train_doc": d, "offset": offset})
            if key not in seen:
                seen[key] = blob
                pending.append((f"{d}:{offset}", part[1:-1].tolist()))
            elif seen[key] != blob:
                raise ValueError("train hash collision")
            if len(pending) >= 256:
                flush()
            ndoc = d + 1
            if ndoc % 100000 == 0:
                progress = {"train_documents": ndoc, "unique_strict": len(seen), "unique_normalized": unique, "seconds": round(time.monotonic()-start)}
                print(json.dumps(progress), flush=True)
                atomic_json(args.output / "progress.json", progress)
        flush()
        for p in processes:
            p.stdin.close()
        if any(p.wait() != 0 for p in processes):
            raise RuntimeError("matcher worker failed")
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
            p.wait()
        for log in logs:
            log.close()
    actual_hash = sha_file(args.train)
    after = args.train.stat()
    if actual_hash != args.expected_train_sha256 or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("training array changed or differs from frozen scope")
    scan_data = {"train_documents": ndoc, "train_tokens": len(train),
        "unique_strict_train": len(seen), "unique_normalized_train": unique,
        "train_sha256": actual_hash, "strict_matches": strict_matches,
        "seconds": round(time.monotonic()-start)}
    atomic_json(args.output / "scan_complete.json", scan_data)
    finish(args, targets, scan_data)


def finish(args, targets, scan_data):
    """Allow result aggregation to resume without repeating corpus decoding."""
    merged = merge([args.output / f"worker.{w}.tsv" for w in range(args.workers)], targets)
    with (args.output / "per_test.jsonl").open("w") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    def stats(rows):
        return {"documents": len(rows), "normalized_exact": sum(r["exact"] for r in rows),
                "near_at_threshold": {str(t): sum(r["jaccard"] >= t or r["containment"] >= t for r in rows) for t in (0.8, 0.7, 0.6)},
                "unmasked_13gram_coverage_ge_30pct": sum(r["coverage"] >= .3 for r in rows),
                "any_shared_13gram": sum(r["coverage"] > 0 for r in rows),
                "max_jaccard": max(r["jaccard"] for r in rows),
                "max_eligible_containment": max(r["containment"] for r in rows),
                "max_unmasked_coverage": max(r["coverage"] for r in rows)}
    subset = json.loads((args.dataset / "docppl/test_1000/document_map.json").read_text())
    result = {"complete": True, "claim_type": "computed", **scan_data, "test": stats(merged),
              "test_docppl_1000": stats([merged[r["full_split_doc_id"]] for r in subset]),
              "limits": "Lexical overlap audit of this frozen BBC training array only; lower similarities and semantic relations are not excluded.",
              "outputs": {p.name: sha_file(p) for p in args.output.iterdir() if p.is_file()}}
    atomic_json(args.output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("dataset/bbc-news-reserved-clean-v1"))
    p.add_argument("--train", type=Path, default=Path("dataset/bbc-news/terminal/train.npy"))
    p.add_argument("--tokenizer", type=Path, default=Path("dataset/bbc-news/TG_GPT2_tokenizer.json"))
    p.add_argument("--expected-train-sha256", default="8f900f2acd1b19109dab522d74a0d1af35171838fbf547ee569a205d1649ceb1")
    p.add_argument("--workers", type=int, default=8, choices=range(1, 33))
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
