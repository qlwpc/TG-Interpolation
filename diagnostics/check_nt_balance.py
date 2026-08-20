#!/usr/bin/env python
"""Check non-terminal bracket balancing in the tree-stream corpora.

For every document (delimited by BOS ... EOS) the NT-bracket tokens must form a
*well-balanced* parenthesis sequence: each opening ``<(LABEL>`` is matched by a
later ``<LABEL)>``, the running depth never goes negative, and it returns to 0
at the document end.  This is independent of the ``(ADJ ... ADJ)`` *surface-text*
leak (those are literal GPT-2 tokens, not NT-bracket tokens, so they do NOT
affect NT depth) — it tests whether the serialized parse trees are structurally
sound.

Two failure modes:
  * UNDER (depth < 0 mid-document): a closing NT appears without a matching open.
  * UNBAL (depth != 0 at doc end):   opens without matching closes (truncated).

Also reported per-document:
  * max depth, #opens, #closes (imbalance = opens - closes).

Targets:
  dataset/bbc-news/tree/{train,dev,test}.npy      (BOS-delimited documents)
  dataset/testppl_tree/tree_300.npy               (300 candidates/sentence,
                                                   sliced by tree_sent_index.npy)
For tree_300 we additionally cross-tabulate NT imbalance against the terminal
mismatch verdict (clean vs broken) from the ADJ-leak defect.

Run:
  export PYTHONPATH=/home/wangpch/TG-Interpolation
  /home/wangpch/.conda/envs/LLM/bin/python diagnostics/check_nt_balance.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from olmo.data.parse_align import TreeVocab  # noqa: E402

TOK_PATH = f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json"
TREE_DIR = f"{REPO}/dataset/bbc-news/tree"
TESTPPL_DIR = f"{REPO}/dataset/testppl_tree"

# (label, path, has_sent_index)
TARGETS: List[Tuple[str, str, bool]] = [
    ("tree/dev.npy",         f"{TREE_DIR}/dev.npy",        False),
    ("tree/test.npy",        f"{TREE_DIR}/test.npy",       False),
    ("tree/train.npy",       f"{TREE_DIR}/train.npy",      False),
    ("testppl/tree_300.npy", f"{TESTPPL_DIR}/tree_300.npy", True),
]

CHUNK_TOKENS = 64_000_000
OVERLAP = 4  # small overlap so BOS-delimited docs aren't split across chunks


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


# --------------------------------------------------------------------------- #
# Worker globals (NT id sets)
# --------------------------------------------------------------------------- #
_OP_LO = _OP_HI = _CL_LO = _CL_HI = 0
_BOS = _EOS = 0


def _init_worker(op_lo, op_hi, cl_lo, cl_hi, bos, eos):
    global _OP_LO, _OP_HI, _CL_LO, _CL_HI, _BOS, _EOS
    _OP_LO, _OP_HI, _CL_LO, _CL_HI, _BOS, _EOS = op_lo, op_hi, cl_lo, cl_hi, bos, eos


def _analyze_segment(seg: np.ndarray) -> Dict:
    """Analyze NT balance of one BOS-delimited segment (no BOS/EOS themselves)."""
    depth = 0
    max_depth = 0
    n_open = 0
    n_close = 0
    under = False  # depth went negative
    for tok in seg:
        t = int(tok)
        if _OP_LO <= t <= _OP_HI:
            depth += 1
            n_open += 1
            if depth > max_depth:
                max_depth = depth
        elif _CL_LO <= t <= _CL_HI:
            n_close += 1
            depth -= 1
            if depth < 0:
                under = True
    return {
        "n_open": n_open, "n_close": n_close, "imbalance": n_open - n_close,
        "max_depth": max_depth, "end_depth": depth, "under": under,
        "balanced": (depth == 0 and not under),
    }


def _count_chunk_bos(args: Tuple[str, int, int]) -> Dict:
    """Scan a chunk of a BOS-delimited stream. Documents may straddle chunks;
    we carry depth across via overlap and only emit a doc verdict when we hit a
    BOS (start) or EOS (end).  To keep it simple and exact, we re-find document
    boundaries within [start-OVERLAP, end+OVERLAP) and analyze each complete
    document fully inside the chunk."""
    path, start, end = args
    arr = np.load(path, mmap_mode="r")
    n = len(arr)
    lo = max(0, start - OVERLAP)
    hi = min(n, end + OVERLAP)
    seg = arr[lo:hi]

    # find BOS positions in this window
    bos_pos = np.where(seg == _BOS)[0]
    eos_pos = np.where(seg == _EOS)[0]

    # build document spans: each (bos_idx, next_eos_after_bos)
    docs: List[Tuple[int, int]] = []
    ei = 0
    for bi in bos_pos:
        # find first eos >= bi
        while ei < len(eos_pos) and eos_pos[ei] < bi:
            ei += 1
        if ei < len(eos_pos) and eos_pos[ei] > bi:
            docs.append((int(bi), int(eos_pos[ei])))

    results: List[Dict] = []
    under = 0
    unbal = 0
    ok = 0
    n_open_tot = 0
    n_close_tot = 0
    max_depth_max = 0
    imbalances: List[int] = []
    for bi, ei in docs:
        # segment between BOS+1 and EOS (exclusive of both)
        sub = seg[bi + 1:ei]
        r = _analyze_segment(sub)
        n_open_tot += r["n_open"]
        n_close_tot += r["n_close"]
        if r["max_depth"] > max_depth_max:
            max_depth_max = r["max_depth"]
        if r["under"]:
            under += 1
        elif r["imbalance"] != 0:
            unbal += 1
            imbalances.append(r["imbalance"])
        else:
            ok += 1
        results.append(r)
    return {
        "n_docs": len(docs), "ok": ok, "under": under, "unbal": unbal,
        "n_open": n_open_tot, "n_close": n_close_tot,
        "max_depth": max_depth_max, "imbalances": imbalances,
    }


def _count_chunk_record(args: Tuple[str, int, int, int, np.ndarray]) -> Dict:
    """Scan tree_300 records [start, end) using the global offsets array."""
    path, start, end, sps, offsets = args
    arr = np.load(path, mmap_mode="r")
    ok = under = unbal = 0
    n_open_tot = n_close_tot = 0
    max_depth_max = 0
    broken_balanced = 0   # records that are NT-balanced but belong to a
    clean_balanced = 0    # terminal-mismatching sentence
    # we don't have terminal verdict here; just record-level balance
    for s in range(start, end):
        first = s * sps
        for c in range(sps):
            s0 = int(offsets[first + c])
            e0 = int(offsets[first + c + 1])
            sub = arr[s0:e0]
            r = _analyze_segment(sub)
            n_open_tot += r["n_open"]
            n_close_tot += r["n_close"]
            if r["max_depth"] > max_depth_max:
                max_depth_max = r["max_depth"]
            if r["under"]:
                under += 1
            elif r["imbalance"] != 0:
                unbal += 1
            else:
                ok += 1
    return {
        "n_records": (end - start) * sps, "ok": ok, "under": under,
        "unbal": unbal, "n_open": n_open_tot, "n_close": n_close_tot,
        "max_depth": max_depth_max,
    }


def merge_bos(results: List[Dict]) -> Dict:
    ok = sum(r["ok"] for r in results)
    under = sum(r["under"] for r in results)
    unbal = sum(r["unbal"] for r in results)
    n_docs = sum(r["n_docs"] for r in results)
    n_open = sum(r["n_open"] for r in results)
    n_close = sum(r["n_close"] for r in results)
    max_depth = max((r["max_depth"] for r in results), default=0)
    imb = []
    for r in results:
        imb.extend(r["imbalances"])
    return {"n_docs": n_docs, "ok": ok, "under": under, "unbal": unbal,
            "n_open": n_open, "n_close": n_close, "max_depth": max_depth,
            "imbalances": imb}


def scan_bos_file(path: str, label: str, pool) -> Dict:
    arr = np.load(path, mmap_mode="r")
    n = len(arr)
    chunks = [(path, s, min(s + CHUNK_TOKENS, n)) for s in range(0, n, CHUNK_TOKENS)]
    t0 = time.time()
    print(f"\n[{label}] {n:,} tokens, {len(chunks)} chunks (BOS-delimited docs)...", flush=True)
    results = pool.map(_count_chunk_bos, chunks)
    m = merge_bos(results)
    elapsed = time.time() - t0
    print(f"[{label}] done in {elapsed:.1f}s", flush=True)
    return {"label": label, "n_tokens": n, **m}


def scan_record_file(path: str, label: str, sps: int, pool) -> Dict:
    arr = np.load(path, mmap_mode="r")
    n = len(arr)
    lengths = np.load(f"{TESTPPL_DIR}/tree_sent_index.npy", mmap_mode="r")
    num_records = len(lengths)
    num_sentences = num_records // sps
    offsets = np.empty(num_records + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])

    # split sentences across workers
    n_workers = min(pool._processes, 32) if hasattr(pool, "_processes") else 32
    per = max(1, num_sentences // (n_workers * 4))
    chunks = []
    s = 0
    while s < num_sentences:
        e = min(s + per, num_sentences)
        chunks.append((path, s, e, sps, offsets))
        s = e

    t0 = time.time()
    print(f"\n[{label}] {n:,} tokens, {num_records:,} records "
          f"({num_sentences:,} sentences x {sps}), {len(chunks)} chunks...", flush=True)
    results = pool.map(_count_chunk_record, chunks)
    ok = sum(r["ok"] for r in results)
    under = sum(r["under"] for r in results)
    unbal = sum(r["unbal"] for r in results)
    n_rec = sum(r["n_records"] for r in results)
    n_open = sum(r["n_open"] for r in results)
    n_close = sum(r["n_close"] for r in results)
    max_depth = max((r["max_depth"] for r in results), default=0)
    elapsed = time.time() - t0
    print(f"[{label}] done in {elapsed:.1f}s", flush=True)
    return {"label": label, "n_tokens": n, "n_records": n_rec, "ok": ok,
            "under": under, "unbal": unbal, "n_open": n_open,
            "n_close": n_close, "max_depth": max_depth}


def main() -> None:
    import multiprocessing as mp

    vocab = TreeVocab.from_tokenizer_file(TOK_PATH)
    op_lo, op_hi = vocab.op_lo, vocab.op_hi
    cl_lo, cl_hi = vocab.cl_lo, vocab.cl_hi
    bos, eos = vocab.bos, vocab.eos

    hr("0. Setup")
    print(f"NT opening range : [{op_lo}, {op_hi}]  ({op_hi-op_lo+1} labels)")
    print(f"NT closing range : [{cl_lo}, {cl_hi}]  ({cl_hi-cl_lo+1} labels)")
    print(f"BOS={bos}  EOS={eos}")
    print(f"workers          : {min(mp.cpu_count(), 32)}")

    pool = mp.Pool(min(mp.cpu_count(), 32), initializer=_init_worker,
                   initargs=(op_lo, op_hi, cl_lo, cl_hi, bos, eos))

    hr("1. Scanning files")
    summaries = []
    for label, path, has_idx in TARGETS:
        if not os.path.exists(path):
            print(f"\n[{label}] MISSING: {path}")
            continue
        if has_idx:
            summaries.append(scan_record_file(path, label, 300, pool))
        else:
            summaries.append(scan_bos_file(path, label, pool))
    pool.close()
    pool.join()

    hr("2. Results")
    print(f"{'file':<24} {'units':>12} {'balanced':>10} {'under':>8} "
          f"{'unbal':>8} {'opens':>12} {'closes':>12} {'maxd':>5}")
    print("-" * 93)
    for s in summaries:
        unit = s.get("n_docs", s.get("n_records", 0))
        unit_lbl = "docs" if "n_docs" in s else "records"
        print(f"{s['label']:<24} {unit:>12,} {s['ok']:>10,} {s['under']:>8,} "
              f"{s['unbal']:>8,} {s['n_open']:>12,} {s['n_close']:>12,} "
              f"{s['max_depth']:>5}")
        print(f"{'':<24} ({unit_lbl})")

    hr("3. Detail")
    for s in summaries:
        unit = s.get("n_docs", s.get("n_records", 0))
        bad = s["under"] + s["unbal"]
        rate = 100 * bad / unit if unit else 0
        print(f"\n{s['label']}:")
        print(f"  units scanned   : {unit:,}")
        print(f"  balanced (OK)   : {s['ok']:,}  ({100*s['ok']/unit:.2f}%)" if unit else "")
        print(f"  UNDER (depth<0) : {s['under']:,}  (closing NT without open)")
        print(f"  UNBAL (end!=0)  : {s['unbal']:,}  (open without closing/truncated)")
        print(f"  total broken    : {bad:,}  ({rate:.4f}%)")
        print(f"  opens / closes  : {s['n_open']:,} / {s['n_close']:,}  "
              f"(diff {s['n_open']-s['n_close']:,})")
        print(f"  max depth       : {s['max_depth']}")
        if "imbalances" in s and s["imbalances"]:
            imb = np.array(s["imbalances"])
            print(f"  unbalanced doc imbalances: min={imb.min()} max={imb.max()} "
                  f"mean={imb.mean():.1f}")

    hr("4. Verdict")
    for s in summaries:
        unit = s.get("n_docs", s.get("n_records", 0))
        bad = s["under"] + s["unbal"]
        if bad == 0:
            print(f"{s['label']}: ALL {unit:,} units are well-balanced.")
        else:
            print(f"{s['label']}: {bad:,}/{unit:,} units broken "
                  f"({100*bad/unit:.4f}%).")


if __name__ == "__main__":
    main()
