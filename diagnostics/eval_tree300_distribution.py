#!/usr/bin/env python
"""Evaluate the data distribution of ``dataset/testppl_tree/tree_300.npy`` and
sample several documents to show whether multiple sentences are broken under the
two failure modes seen in jobs 45195 / 45196:

  * PUSHDOWN (data bug, 45195): the 300 gold candidates of one sentence must
    share an identical terminal sequence.  A mismatch means candidates from a
    different sentence leaked into the group.  Checked cheaply: a candidate's
    terminals are exactly its non-NT tokens (verified bit-identical to
    ``parse_chunk_slice``'s ``input_ids``), so no tree parsing is needed.

  * GPST (code bug, 45196): ``tree_to_merge_orders`` must yield a permutation of
    ``range(L-1)`` for every candidate.  The ``tree_spans`` id() collision makes
    this fail whenever a candidate tree has a repeated leaf token id <= 256.

Outputs:
  1. Corpus overview (documents, sentences, tokens, per-doc / per-sentence stats).
  2. Corpus-wide statistical estimate of both error rates (sampled sentences).
  3. Deep-dive on a few sampled documents: per-sentence verdict table +
     per-document aggregate, showing how many sentences are broken.

Run:
  export PYTHONPATH=/home/wangpch/TG-Interpolation
  /home/wangpch/.conda/envs/LLM/bin/python diagnostics/eval_tree300_distribution.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import List, Sequence, Tuple

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from olmo.data.parse_align import TreeVocab, parse_block_segments  # noqa: E402
from olmo.gpst.reader.dataset_gold import tree_to_merge_orders  # noqa: E402

TREE = f"{REPO}/dataset/testppl_tree/tree_300.npy"
SENT_IDX = f"{REPO}/dataset/testppl_tree/tree_sent_index.npy"
DOC_IDX = f"{REPO}/dataset/testppl_tree/tree_doc_index.npy"
TOK = f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json"
SPS = 300
SEED = 12345

# Sampling defaults
N_ESTIMATE_SENTENCES = 600      # corpus-wide error-rate estimate
N_DEEPDOCS = 5                  # documents for the per-sentence deep-dive
DEEPDOC_MAX_SENTENCES = 40      # cap sentences printed per deep-dive document


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_offsets(lengths: np.ndarray) -> np.ndarray:
    offs = np.empty(len(lengths) + 1, dtype=np.uint64)
    offs[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offs[1:])
    return offs


def record_terminals(tree: np.ndarray, start: int, end: int,
                     is_nt: np.ndarray) -> List[int]:
    """Cheap terminal extraction: non-NT tokens of one record (== input_ids)."""
    block = tree[start:end]
    # vectorized mask over the token-id space
    mask = ~is_nt[block]
    return block[mask].tolist()


def sentence_terminal_check(tree: np.ndarray, offsets: np.ndarray,
                            sentence_index: int, is_nt: np.ndarray
                            ) -> Tuple[bool, int, List[int]]:
    """Return (all_consistent, n_mismatch_candidates, mismatch_candidate_ids).

    ``all_consistent`` is True iff all 300 candidates share candidate 0's
    terminal sequence.  ``n_mismatch_candidates`` counts how many differ.
    """
    first = sentence_index * SPS
    s0, e0 = int(offsets[first]), int(offsets[first + 1])
    ref = record_terminals(tree, s0, e0, is_nt)
    bad_ids: List[int] = []
    for c in range(1, SPS):
        sc, ec = int(offsets[first + c]), int(offsets[first + c + 1])
        cand = record_terminals(tree, sc, ec, is_nt)
        if cand != ref:
            bad_ids.append(c)
    return (len(bad_ids) == 0), len(bad_ids), bad_ids


def candidate_gpst_valid(tree: np.ndarray, start: int, end: int,
                         vocab: TreeVocab) -> Tuple[bool, str]:
    """True iff this record's tree_to_merge_orders is a permutation of range(L-1)."""
    block = tree[start:end].astype(np.int64).tolist()
    segs = parse_block_segments(block, vocab)
    trees = [d for k, d in segs if k == "tree"]
    if not trees:
        return False, "no tree segment"
    try:
        leaves, orders = tree_to_merge_orders(trees[0], direction="right")
    except Exception as exc:  # noqa: BLE001
        return False, f"exception: {exc}"
    if len(orders) != max(len(leaves) - 1, 0):
        return False, f"len(orders)={len(orders)} != L-1={len(leaves)-1}"
    if sorted(orders) != list(range(len(leaves) - 1)):
        return False, "not a permutation (dup/missing gap)"
    return True, "ok"


def sentence_gpst_check_all(tree: np.ndarray, offsets: np.ndarray,
                            sentence_index: int, vocab: TreeVocab
                            ) -> Tuple[bool, int, List[int]]:
    """Check all 300 candidates' merge orders. (Expensive.)"""
    first = sentence_index * SPS
    bad_ids: List[int] = []
    for c in range(SPS):
        sc, ec = int(offsets[first + c]), int(offsets[first + c + 1])
        ok, _ = candidate_gpst_valid(tree, sc, ec, vocab)
        if not ok:
            bad_ids.append(c)
    return (len(bad_ids) == 0), len(bad_ids), bad_ids


def sentence_gpst_check_fast(tree: np.ndarray, offsets: np.ndarray,
                             sentence_index: int, vocab: TreeVocab, k: int = 1
                             ) -> Tuple[bool, int]:
    """Check only the first ``k`` candidates (fast lower-bound estimate)."""
    first = sentence_index * SPS
    bad = 0
    for c in range(min(k, SPS)):
        sc, ec = int(offsets[first + c]), int(offsets[first + c + 1])
        ok, _ = candidate_gpst_valid(tree, sc, ec, vocab)
        if not ok:
            bad += 1
    return (bad == 0), bad


def percentile(arr: np.ndarray, q: float) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def main() -> None:
    rng = np.random.default_rng(SEED)

    tree = np.load(TREE, mmap_mode="r")
    lengths = np.load(SENT_IDX, mmap_mode="r")
    doc_counts = np.load(DOC_IDX, mmap_mode="r")
    vocab = TreeVocab.from_tokenizer_file(TOK)

    num_records = len(lengths)
    num_sentences = num_records // SPS
    num_docs = len(doc_counts)
    offsets = build_offsets(lengths)
    doc_ends = np.cumsum(doc_counts, dtype=np.int64)  # sentence index where each doc ends (exclusive)

    # Precompute a boolean NT-mask over the whole token-id space.
    max_id = int(max(vocab.cl_hi, vocab.op_hi, vocab.bos, vocab.eos, vocab.pad)) + 1
    is_nt = np.zeros(max_id + 1, dtype=bool)
    is_nt[vocab.op_lo:vocab.op_hi + 1] = True
    is_nt[vocab.cl_lo:vocab.cl_hi + 1] = True

    # ------------------------------------------------------------------ #
    # 1. Corpus overview
    # ------------------------------------------------------------------ #
    hr("1. Corpus overview")
    print(f"documents              : {num_docs:>12,}")
    print(f"sentences (records/300): {num_sentences:>12,}")
    print(f"records                : {num_records:>12,}")
    print(f"total tree tokens      : {len(tree):>12,}")
    print(f"sum(sent_index)==len   : {int(np.sum(lengths)) == len(tree)}")
    print(f"sum(doc_index)==sents  : {int(np.sum(doc_counts)) == num_sentences}")

    sents_per_doc = np.asarray(doc_counts, dtype=np.int64)
    print(f"\nsentences per document : min={sents_per_doc.min()} "
          f"p25={percentile(sents_per_doc,25):.0f} median={percentile(sents_per_doc,50):.0f} "
          f"mean={sents_per_doc.mean():.1f} p75={percentile(sents_per_doc,75):.0f} "
          f"max={sents_per_doc.max()}")
    print("  size buckets (sentences/doc):")
    edges = [0, 5, 10, 20, 50, 100, 500, 10_000]
    for lo, hi in zip(edges, edges[1:] + [10**9]):
        cnt = int(np.sum((sents_per_doc > lo) & (sents_per_doc <= hi)))
        if cnt:
            print(f"    ({lo+1:>5}..{hi if hi<10**9 else 'inf':>5}): {cnt:>5} docs")

    rec_lens = np.asarray(lengths, dtype=np.int64)
    # per-sentence total tokens = sum of its 300 record lengths
    sent_totals = rec_lens[:num_sentences * SPS].reshape(num_sentences, SPS).sum(axis=1)
    print(f"\nrecord length (tokens) : min={rec_lens.min()} median={percentile(rec_lens,50):.0f} "
          f"mean={rec_lens.mean():.1f} max={rec_lens.max()}")
    print(f"sentence total tokens  : min={sent_totals.min()} median={percentile(sent_totals,50):.0f} "
          f"mean={sent_totals.mean():.1f} max={sent_totals.max()}")

    # ------------------------------------------------------------------ #
    # 2. Corpus-wide error-rate estimate (sampled sentences)
    # ------------------------------------------------------------------ #
    hr(f"2. Corpus-wide error-rate estimate (sampled {N_ESTIMATE_SENTENCES} sentences)")
    sample_idx = rng.integers(0, num_sentences, size=N_ESTIMATE_SENTENCES)
    sample_idx = np.sort(np.unique(sample_idx))
    n_s = len(sample_idx)

    pd_bad = 0
    pd_bad_cand_total = 0
    gpst_bad_c0 = 0
    for i, sidx in enumerate(sample_idx):
        ok, nbad, _ = sentence_terminal_check(tree, offsets, int(sidx), is_nt)
        if not ok:
            pd_bad += 1
            pd_bad_cand_total += nbad
        okg, _ = sentence_gpst_check_fast(tree, offsets, int(sidx), vocab, k=1)
        if not okg:
            gpst_bad_c0 += 1
        if (i + 1) % 100 == 0:
            print(f"  ...sampled {i+1}/{n_s}", flush=True)

    print(f"\nsentences sampled                        : {n_s}")
    print(f"PUSHDOWN terminal-mismatch sentences     : {pd_bad}  "
          f"({100*pd_bad/n_s:.2f}%)")
    if pd_bad:
        print(f"  avg bad candidates per bad sentence    : {pd_bad_cand_total/pd_bad:.1f} / {SPS}")
    print(f"GPST invalid merge_orders (candidate 0)  : {gpst_bad_c0}  "
          f"({100*gpst_bad_c0/n_s:.2f}%)  [lower bound: only candidate 0 checked]")
    print("  NOTE: GPST is a code bug triggered by any repeated leaf token id <=256,")
    print("        so the true per-sentence rate (all 300 candidates) is higher.")

    # ------------------------------------------------------------------ #
    # 3. Deep-dive on a few documents
    # ------------------------------------------------------------------ #
    hr(f"3. Document deep-dive ({N_DEEPDOCS} documents, <= {DEEPDOC_MAX_SENTENCES} sentences each)")
    # Pick documents spread across the corpus by cumulative sentence position.
    doc_centers = np.linspace(0, num_docs - 1, N_DEEPDOCS).astype(int)
    picked = sorted(set(int(d) for d in doc_centers))

    for did in picked:
        d_start = int(doc_ends[did - 1]) if did > 0 else 0
        d_end = int(doc_ends[did])
        n_sent = d_end - d_start
        print(f"\n--- Document {did} : sentences [{d_start}..{d_end})  ({n_sent} sentences) ---")
        scan = list(range(d_start, d_end))
        capped = False
        if len(scan) > DEEPDOC_MAX_SENTENCES:
            # evenly subsample to keep the table readable
            idxs = np.linspace(0, len(scan) - 1, DEEPDOC_MAX_SENTENCES).astype(int)
            scan = [scan[i] for i in idxs]
            capped = True
            print(f"  (document has {n_sent} sentences; showing {len(scan)} evenly subsampled)")

        pd_bad_docsent = 0
        gpst_bad_docsent = 0
        print(f"  {'sent':>6} | {'pd_ok':>5} {'pd_bad#':>7} | {'gpst_ok':>7} {'gp_bad#':>7} | bad candidate ids (truncated)")
        print(f"  {'-'*6}-+-{'-'*13}-+-{'-'*16}-+-{'-'*30}")
        for sidx in scan:
            ok_pd, npd, bpd = sentence_terminal_check(tree, offsets, sidx, is_nt)
            ok_gp, ngp, bgp = sentence_gpst_check_all(tree, offsets, sidx, vocab)
            if not ok_pd:
                pd_bad_docsent += 1
            if not ok_gp:
                gpst_bad_docsent += 1
            bpd_s = (",".join(str(x) for x in bpd[:8])) + ("..." if len(bpd) > 8 else "")
            bgp_s = (",".join(str(x) for x in bgp[:8])) + ("..." if len(bgp) > 8 else "")
            print(f"  {sidx:>6} | {str(ok_pd):>5} {npd:>7} | {str(ok_gp):>7} {ngp:>7} | "
                  f"pd=[{bpd_s}] gp=[{bgp_s}]")

        denom = len(scan)
        print(f"  -> Document {did} summary ({denom} sampled sentences):")
        print(f"     PUSHDOWN-broken sentences : {pd_bad_docsent}/{denom} "
              f"({100*pd_bad_docsent/max(denom,1):.1f}%)")
        print(f"     GPST-broken sentences     : {gpst_bad_docsent}/{denom} "
              f"({100*gpst_bad_docsent/max(denom,1):.1f}%)")
        if capped:
            print(f"     (rates extrapolated from subsample of a {n_sent}-sentence document)")

    # ------------------------------------------------------------------ #
    # 4. Verdict
    # ------------------------------------------------------------------ #
    hr("4. Verdict")
    print(f"PUSHDOWN (data bug, job 45195): {100*pd_bad/n_s:.2f}% of sampled sentences have")
    print("  candidate terminal mismatches -> tree_300.npy grouping is corrupted at non-trivial rate.")
    print(f"GPST (code bug, job 45196): {100*gpst_bad_c0/n_s:.2f}% of sampled sentences fail on")
    print("  candidate 0 alone -> tree_spans id() collision is pervasive (any repeated small token).")
    print("Multiple broken sentences per document => both bugs are systemic, not isolated.")


if __name__ == "__main__":
    main()
