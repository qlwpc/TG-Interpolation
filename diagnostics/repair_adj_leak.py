#!/usr/bin/env python
"""Repair the literal ``(ADJ ... ADJ)`` label leak in tree_300.npy.

The defect: some 300-candidate parse variants were produced by a benepar model
whose EWT label set contains the constituent label ``ADJ`` (absent from the
WSJ-only ``TG_GPT2_tokenizer`` NT-bracket set).  When serialized, the ``ADJ``
node could not map to an NT token and was written as *literal surface tokens*:

  open  ' (ADJ' = [357, 2885, 41]   (or no-space '(ADJ' = [7, 2885, 41])
  close ' ADJ)' = [5984, 41, 8]

These leak into the terminal stream, making the candidate's terminals differ
from candidate 0 (job 45195).  The fix replaces each ``(ADJ`` open with the
proper NT token ``<(ADJP>`` (id 50268) and each ``ADJ)`` close with ``<ADJP)>``
(id 50294), restoring them to structural NT-brackets.

This script:
  1. Rebuilds tree_300.npy record-by-record (parallel), replacing matched
     open/close leak triplets with ADJP NT tokens.  Unmatched triplets are
     reported (none expected).
  2. Saves the repaired array to an output path (does NOT overwrite input).
  3. Re-verifies on the repaired data:
       a. NT-bracket balance per record (must stay 100% balanced).
       b. Terminal consistency: all 300 candidates of each sentence now share
          candidate 0's terminal sequence (the 45195 check).
  4. Reports counts: records changed, triplets replaced, sentences still broken.

Run:
  export PYTHONPATH=/home/wangpch/TG-Interpolation
  /home/wangpch/.conda/envs/LLM/bin/python diagnostics/repair_adj_leak.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

TOK_PATH = f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json"
IN_TREE = f"{REPO}/dataset/testppl_tree/tree_300.npy"
SENT_IDX = f"{REPO}/dataset/testppl_tree/tree_sent_index.npy"
OUT_TREE = f"{REPO}/dataset/testppl_tree/tree_300_repaired.npy"

ADJP_OPEN = 50268    # <(ADJP>
ADJP_CLOSE = 50294   # <ADJP)>
SPS = 300

# leak triplets (token id sequences)
OPEN_PATTERNS = ((357, 2885, 41), (7, 2885, 41))   # ' (ADJ' , '(ADJ'
CLOSE_PATTERNS = ((5984, 41, 8),)                  # ' ADJ)'


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def repair_block(block: List[int]) -> Tuple[List[int], int, int, int]:
    """Replace (ADJ...ADJ) leak triplets with ADJP NT tokens, using a stack to
    pair opens with closes (they are guaranteed equal-count per record, but the
    stack makes the pairing explicit and safe against nesting/misordering).

    Returns (repaired_block, n_open_replaced, n_close_replaced, n_unmatched).
    """
    n = len(block)
    out: List[int] = []
    # stack of indices into `out` where we pushed a placeholder open marker.
    # We push ADJP_OPEN immediately; on close we just append ADJP_CLOSE.
    open_stack: List[int] = []
    n_open = 0
    n_close = 0
    n_unmatched = 0
    i = 0
    while i < n:
        matched = False
        # try open patterns
        if i + 2 < n:
            tri = (block[i], block[i + 1], block[i + 2])
            if tri in OPEN_PATTERNS:
                out.append(ADJP_OPEN)
                open_stack.append(len(out) - 1)
                n_open += 1
                i += 3
                matched = True
            elif tri in CLOSE_PATTERNS:
                if open_stack:
                    open_stack.pop()
                    out.append(ADJP_CLOSE)
                    n_close += 1
                else:
                    # unmatched close: leave as-is (surface text) and flag
                    out.extend(block[i:i + 3])
                    n_unmatched += 1
                i += 3
                matched = True
        if not matched:
            out.append(block[i])
            i += 1
    # any leftover unmatched opens -> they had no close; flag them
    n_unmatched += len(open_stack)
    return out, n_open, n_close, n_unmatched


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_LENGTHS = None
_OFFSETS = None
_OP_LO = _OP_HI = _CL_LO = _CL_HI = 0


def _init_worker(lengths, offsets, op_lo, op_hi, cl_lo, cl_hi):
    global _LENGTHS, _OFFSETS, _OP_LO, _OP_HI, _CL_LO, _CL_HI
    _LENGTHS = lengths
    _OFFSETS = offsets
    _OP_LO, _OP_HI, _CL_LO, _CL_HI = op_lo, op_hi, cl_lo, cl_hi


def _repair_chunk(args: Tuple[str, int, int]) -> Dict:
    """Repair sentences [start, end). Returns repaired record lists + stats."""
    path, start, end = args
    arr = np.load(path, mmap_mode="r")
    repaired: List[np.ndarray] = []
    total_open = total_close = total_unmatched = 0
    records_changed = 0
    for s in range(start, end):
        first = s * SPS
        for c in range(SPS):
            s0 = int(_OFFSETS[first + c])
            e0 = int(_OFFSETS[first + c + 1])
            block = arr[s0:e0].astype(np.int64).tolist()
            new_block, no, nc, nu = repair_block(block)
            total_open += no
            total_close += nc
            total_unmatched += nu
            if new_block != block:
                records_changed += 1
            repaired.append(np.asarray(new_block, dtype=arr.dtype))
    return {"repaired": repaired, "n_open": total_open, "n_close": total_close,
            "n_unmatched": total_unmatched, "records_changed": records_changed,
            "sent_range": (start, end)}


def nt_balance_ok(block: List[int]) -> Tuple[bool, int]:
    """Return (balanced, end_depth). NT tokens are in [op_lo,cl_hi]."""
    depth = 0
    for t in block:
        if _OP_LO <= t <= _OP_HI:
            depth += 1
        elif _CL_LO <= t <= _CL_HI:
            depth -= 1
            if depth < 0:
                return False, depth
    return depth == 0, depth


def terminals(block: List[int], op_lo, op_hi, cl_lo, cl_hi) -> List[int]:
    return [int(t) for t in block if not (op_lo <= t <= cl_hi)]


def main() -> None:
    import multiprocessing as mp
    from olmo.data.parse_align import TreeVocab

    vocab = TreeVocab.from_tokenizer_file(TOK_PATH)
    op_lo, op_hi = vocab.op_lo, vocab.op_hi
    cl_lo, cl_hi = vocab.cl_lo, vocab.cl_hi

    lengths = np.load(SENT_IDX, mmap_mode="r")
    num_records = len(lengths)
    num_sentences = num_records // SPS
    offsets = np.empty(num_records + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])

    hr("0. Setup")
    print(f"input  : {IN_TREE}")
    print(f"output : {OUT_TREE}")
    print(f"records: {num_records:,}  sentences: {num_sentences:,}")
    print(f"ADJP NT : open=<{ADJP_OPEN} (<(ADJP>)  close={ADJP_CLOSE} (<ADJP)>)")
    print(f"leak open : {OPEN_PATTERNS}")
    print(f"leak close: {CLOSE_PATTERNS}")

    # ------------------------------------------------------------------ #
    # 1. Repair (parallel)
    # ------------------------------------------------------------------ #
    hr("1. Repairing (ADJ...ADJ) -> <(ADJP>...<ADJP)>")
    n_workers = min(mp.cpu_count(), 32)
    pool = mp.Pool(n_workers, initializer=_init_worker,
                   initargs=(lengths, offsets, op_lo, op_hi, cl_lo, cl_hi))

    per = max(1, num_sentences // (n_workers * 4))
    chunks = [(IN_TREE, s, min(s + per, num_sentences))
              for s in range(0, num_sentences, per)]
    t0 = time.time()
    print(f"{len(chunks)} chunks across {n_workers} workers...", flush=True)
    results = pool.map(_repair_chunk, chunks)
    pool.close()
    pool.join()

    # flatten repaired records in order
    all_repaired: List[np.ndarray] = []
    total_open = total_close = total_unmatched = records_changed = 0
    for r in results:
        all_repaired.extend(r["repaired"])
        total_open += r["n_open"]
        total_close += r["n_close"]
        total_unmatched += r["n_unmatched"]
        records_changed += r["records_changed"]
    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s", flush=True)
    print(f"records changed       : {records_changed:,} / {num_records:,}")
    print(f"open triplets replaced : {total_open:,}")
    print(f"close triplets replaced: {total_close:,}")
    print(f"unmatched triplets    : {total_unmatched:,}")

    # ------------------------------------------------------------------ #
    # 2. Save repaired array
    # ------------------------------------------------------------------ #
    hr("2. Saving repaired array")
    # concatenate (variable-length records) into one flat array
    flat = np.concatenate(all_repaired) if all_repaired else np.array([], dtype=np.uint16)
    print(f"original total tokens : {int(np.sum(lengths)):,}")
    print(f"repaired total tokens : {flat.size:,}")
    # each replaced triplet (3 tokens) -> 1 NT token, so size shrinks
    delta = int(np.sum(lengths)) - flat.size
    expected_delta = (total_open + total_close) * 2  # 3->1 saves 2 per triplet
    print(f"token delta           : {delta:,}  (expected {expected_delta:,})")
    np.save(OUT_TREE, flat)
    print(f"saved -> {OUT_TREE}  ({flat.nbytes/1e9:.2f} GB)")

    # ------------------------------------------------------------------ #
    # 3. Verify: NT balance + terminal consistency on repaired data
    # ------------------------------------------------------------------ #
    hr("3. Verifying repaired data")
    # rebuild offsets for repaired records (lengths changed)
    new_lengths = np.array([len(r) for r in all_repaired], dtype=offsets.dtype)
    new_offsets = np.empty(num_records + 1, dtype=np.uint64)
    new_offsets[0] = 0
    np.cumsum(new_lengths, dtype=np.uint64, out=new_offsets[1:])

    # 3a. NT balance (sample + full count of broken)
    print("\n3a. NT-bracket balance per record...", flush=True)
    t0 = time.time()
    unbal = 0
    under = 0
    checked = 0
    for i in range(num_records):
        s0 = int(new_offsets[i])
        e0 = int(new_offsets[i + 1])
        block = flat[s0:e0].astype(np.int64).tolist()
        ok, depth = nt_balance_ok(block)
        checked += 1
        if not ok:
            if depth < 0:
                under += 1
            else:
                unbal += 1
        if (i + 1) % 5_000_000 == 0:
            print(f"  balanced {i+1}/{num_records}  (broken: {under+unbal})", flush=True)
    print(f"  checked {checked:,}  balanced={checked-under-unbal:,}  "
          f"under={under}  unbal={unbal}  ({time.time()-t0:.1f}s)", flush=True)

    # 3b. Terminal consistency (the 45195 check)
    print("\n3b. Terminal consistency (300 candidates share terminals)...", flush=True)
    t0 = time.time()
    broken_sentences = 0
    bad_cand_total = 0
    for s in range(num_sentences):
        first = s * SPS
        s0 = int(new_offsets[first])
        e0 = int(new_offsets[first + 1])
        ref = terminals(flat[s0:e0].astype(np.int64).tolist(), op_lo, op_hi, cl_lo, cl_hi)
        ref_list = ref
        nbad = 0
        for c in range(1, SPS):
            sc = int(new_offsets[first + c])
            ec = int(new_offsets[first + c + 1])
            cand = terminals(flat[sc:ec].astype(np.int64).tolist(), op_lo, op_hi, cl_lo, cl_hi)
            if cand != ref_list:
                nbad += 1
        if nbad:
            broken_sentences += 1
            bad_cand_total += nbad
        if (s + 1) % 20_000 == 0:
            print(f"  checked {s+1}/{num_sentences} sentences  (broken: {broken_sentences})", flush=True)
    rate = 100 * broken_sentences / num_sentences
    print(f"  broken sentences: {broken_sentences:,} / {num_sentences:,}  ({rate:.2f}%)", flush=True)
    print(f"  avg bad cand per broken: {bad_cand_total/max(broken_sentences,1):.1f}  ({time.time()-t0:.1f}s)", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Verdict
    # ------------------------------------------------------------------ #
    hr("4. Verdict")
    print(f"Replaced {total_open:,} open + {total_close:,} close leak triplets.")
    print(f"Unmatched triplets: {total_unmatched}")
    print(f"NT balance after repair: {checked-under-unbal:,}/{checked:,} balanced "
          f"(under={under}, unbal={unbal})")
    print(f"Terminal consistency after repair: {num_sentences-broken_sentences:,}/{num_sentences:,} "
          f"clean ({rate:.2f}% broken remain)")
    if broken_sentences == 0 and under == 0 and unbal == 0 and total_unmatched == 0:
        print("\n*** PERFECT: repaired tree_300 is fully balanced AND terminal-consistent. ***")
    else:
        print("\n*** Residual defects remain (see above). ***")
    print(f"\nRepaired file: {OUT_TREE}")


if __name__ == "__main__":
    main()
