#!/usr/bin/env python
"""Fix the terminal-misalignment bug in tree_300.npy in-place (no regeneration).

Applies the same logic as ``parse_testppl_standalone.py``'s back-fill, but to
the *existing* corpus: for each sentence, keep only the candidates whose
terminals match candidate 0 AND are legal trees (have a root constituent), drop
the rest (the bare-token / degenerate candidates), and back-fill to exactly 300
using the lowest-score surviving candidate.

This preserves the original K-best sampling — we only discard candidates that
were already invalid (treeless bare tokens) and pad with the lowest-score legal
tree.  doc-PPL stays well-defined: cand 0 (highest score) is unchanged, all 300
share terminals, K=300 uniform.

Pre-step: the ``(ADJ ... ADJ)`` label leak is repaired first (via
``repair_block``) so that leak-corrupted candidates become legal trees again
and are kept rather than dropped.

Outputs (does NOT overwrite input):
  tree_300_fixed.npy            — repaired + back-filled flat token array
  tree_sent_index_fixed.npy     — per-record lengths (changed by ADJ repair)
  (tree_doc_index.npy is unchanged — document grouping is sentence-level)

Run:
  export PYTHONPATH=/home/wangpch/TG-Interpolation
  /home/wangpch/.conda/envs/LLM/bin/python diagnostics/fix_tree300_terminals.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from olmo.data.parse_align import TreeVocab, parse_block_segments  # noqa: E402
from diagnostics.repair_adj_leak import repair_block  # noqa: E402

TREE = f"{REPO}/dataset/testppl_tree/tree_300.npy"
SENT_IDX = f"{REPO}/dataset/testppl_tree/tree_sent_index.npy"
DOC_IDX = f"{REPO}/dataset/testppl_tree/tree_doc_index.npy"
OUT_TREE = f"{REPO}/dataset/testppl_tree/tree_300_fixed.npy"
OUT_SENT = f"{REPO}/dataset/testppl_tree/tree_sent_index_fixed.npy"
SPS = 300


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


# Worker globals
_OPLO = _OPHI = _CLLO = _CLHI = 0
_OFFS = None


def _init_worker(op_lo, op_hi, cl_lo, cl_hi, offsets):
    global _OPLO, _OPHI, _CLLO, _CLHI, _OFFS
    _OPLO, _OPHI, _CLLO, _CLHI = op_lo, op_hi, cl_lo, cl_hi
    _OFFS = offsets


def _terms(block: List[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in block if not (_OPLO <= x <= _CLHI))


def _has_root(block: List[int], vocab: TreeVocab) -> bool:
    segs = parse_block_segments(block, vocab)
    return any(k == "tree" for k, _ in segs)


def _fix_sentence(args: Tuple[str, int, int]) -> dict:
    """Repair + back-fill sentences [start, end). Returns list of 300 records
    (as np.ndarray) + stats."""
    path, start, end = args
    arr = np.load(path, mmap_mode="r")
    vocab = TreeVocab.from_tokenizer_file(
        f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json")
    records: List[np.ndarray] = []
    n_changed = 0
    n_backfilled = 0
    for s in range(start, end):
        first = s * SPS
        # Step 1: ADJ-repair every candidate.
        repaired: List[List[int]] = []
        for c in range(SPS):
            s0 = int(_OFFS[first + c])
            e0 = int(_OFFS[first + c + 1])
            block = arr[s0:e0].astype(np.int64).tolist()
            rb, _, _, _ = repair_block(block)
            repaired.append(rb)

        # Step 2: reference terminals = candidate 0.
        ref = _terms(repaired[0])

        # Step 3: keep candidates that match ref terminals AND are legal trees.
        legit_idx: List[int] = []
        for c in range(SPS):
            if c == 0:
                # cand 0 is always kept (it defines ref); trust its tree.
                legit_idx.append(c)
                continue
            if _terms(repaired[c]) == ref and _has_root(repaired[c], vocab):
                legit_idx.append(c)

        # Step 4: back-fill to 300 using the lowest-score surviving candidate.
        if len(legit_idx) < SPS:
            fill_idx = legit_idx[-1]   # lowest-score legal candidate
            keep = legit_idx + [fill_idx] * (SPS - len(legit_idx))
            n_backfilled += 1
        else:
            keep = legit_idx

        for c in keep:
            records.append(np.asarray(repaired[c], dtype=arr.dtype))

    return {"records": records, "n_backfilled": n_backfilled,
            "n_sentences": end - start, "sent_range": (start, end)}


def main() -> None:
    import multiprocessing as mp

    vocab = TreeVocab.from_tokenizer_file(
        f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json")
    op_lo, op_hi = vocab.op_lo, vocab.op_hi
    cl_lo, cl_hi = vocab.cl_lo, vocab.cl_hi

    lengths = np.load(SENT_IDX, mmap_mode="r")
    num_records = len(lengths)
    num_sentences = num_records // SPS
    offsets = np.empty(num_records + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])

    hr("0. Setup")
    print(f"input  : {TREE}")
    print(f"output : {OUT_TREE} + {OUT_SENT}")
    print(f"sentences: {num_sentences:,}  records: {num_records:,}")

    hr("1. Repair + back-fill (parallel)")
    n_workers = min(mp.cpu_count(), 32)
    pool = mp.Pool(n_workers, initializer=_init_worker,
                   initargs=(op_lo, op_hi, cl_lo, cl_hi, offsets))
    per = max(1, num_sentences // (n_workers * 4))
    chunks = [(TREE, s, min(s + per, num_sentences))
              for s in range(0, num_sentences, per)]
    t0 = time.time()
    print(f"{len(chunks)} chunks across {n_workers} workers...", flush=True)

    all_records: List[np.ndarray] = []
    total_backfilled = 0
    for r in pool.map(_fix_sentence, chunks):
        all_records.extend(r["records"])
        total_backfilled += r["n_backfilled"]
    pool.close()
    pool.join()
    print(f"done in {time.time()-t0:.1f}s", flush=True)
    print(f"sentences back-filled (<300 legit): {total_backfilled:,} / {num_sentences:,}")

    # Build new sent_index from record lengths.
    new_lengths = np.array([len(r) for r in all_records], dtype=np.uint16)
    assert len(new_lengths) == num_records, \
        f"record count {len(new_lengths)} != {num_records}"
    flat = np.concatenate(all_records)

    hr("2. Saving")
    print(f"original tokens: {int(np.sum(lengths)):,}")
    print(f"fixed tokens   : {flat.size:,}  (delta {flat.size - int(np.sum(lengths)):,})")
    np.save(OUT_TREE, flat)
    np.save(OUT_SENT, new_lengths)
    # doc_index is sentence-level, unchanged.
    print(f"saved -> {OUT_TREE} ({flat.nbytes/1e9:.2f} GB), {OUT_SENT}")

    hr("3. Done")
    print("Verify with: diagnostics/verify_repaired.py (point at the fixed files)")


if __name__ == "__main__":
    main()
