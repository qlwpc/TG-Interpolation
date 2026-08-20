#!/usr/bin/env python
"""Data-integrity diagnosis of ``dataset/testppl_tree/tree_300.npy`` for the
PUSHDOWN bug (job 45195): the 300 gold candidates of each sentence must share
an identical terminal token sequence.  A mismatch means candidates from a
different sentence leaked into the group.

This script is NOT about the GPST ``tree_spans`` id() code bug.  It focuses
entirely on the data defect: how many sentences are broken, how badly, what
the mismatched candidates look like, and whether the corruption clusters by
document or by record-position-within-the-300-block.

Terminal extraction is cheap and exact: a candidate's terminals are exactly
its non-NT tokens (verified bit-identical to ``parse_chunk_slice`` input_ids).

Outputs:
  1. Corpus overview.
  2. Corpus-wide integrity scan (ALL 148,836 sentences, not sampled):
     broken-sentence count, rate, and distribution of #bad-candidates.
  3. How brokenness distributes across the 300 candidate slots: is there a
     systematic block of candidate ids that is always wrong (a fixed offset
     / grouping artifact)?
  4. Per-document aggregation: is the corruption uniform or clustered?
  5. Sampled broken sentences: show the actual diverging terminal sequences
     (candidate 0 vs first bad candidate) to characterize the leak.

Run:
  export PYTHONPATH=/home/wangpch/TG-Interpolation
  /home/wangpch/.conda/envs/LLM/bin/python diagnostics/diag_tree300_terminals.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import List, Tuple

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from olmo.data.parse_align import TreeVocab  # noqa: E402

TREE = f"{REPO}/dataset/testppl_tree/tree_300.npy"
SENT_IDX = f"{REPO}/dataset/testppl_tree/tree_sent_index.npy"
DOC_IDX = f"{REPO}/dataset/testppl_tree/tree_doc_index.npy"
TOK = f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json"
SPS = 300


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_offsets(lengths: np.ndarray) -> np.ndarray:
    offs = np.empty(len(lengths) + 1, dtype=np.uint64)
    offs[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offs[1:])
    return offs


def record_terminals(tree: np.ndarray, start: int, end: int,
                     is_nt: np.ndarray) -> np.ndarray:
    """Cheap exact terminal extraction: non-NT tokens of one record."""
    block = tree[start:end]
    mask = ~is_nt[block]
    return block[mask]


def sentence_terminal_check(tree: np.ndarray, offsets: np.ndarray,
                            sentence_index: int, is_nt: np.ndarray
                            ) -> Tuple[bool, int, np.ndarray, List[int]]:
    """Return (consistent, n_bad, ref_terminals, bad_candidate_ids).

    consistent=True iff all 300 candidates share candidate 0's terminal array.
    """
    first = sentence_index * SPS
    s0, e0 = int(offsets[first]), int(offsets[first + 1])
    ref = record_terminals(tree, s0, e0, is_nt)
    ref_list = ref.tolist()
    bad_ids: List[int] = []
    for c in range(1, SPS):
        sc, ec = int(offsets[first + c]), int(offsets[first + c + 1])
        cand = record_terminals(tree, sc, ec, is_nt)
        if cand.tolist() != ref_list:
            bad_ids.append(c)
    return (len(bad_ids) == 0), len(bad_ids), ref, bad_ids


def main() -> None:
    tree = np.load(TREE, mmap_mode="r")
    lengths = np.load(SENT_IDX, mmap_mode="r")
    doc_counts = np.load(DOC_IDX, mmap_mode="r")
    vocab = TreeVocab.from_tokenizer_file(TOK)

    num_records = len(lengths)
    num_sentences = num_records // SPS
    num_docs = len(doc_counts)
    offsets = build_offsets(lengths)
    doc_ends = np.cumsum(doc_counts, dtype=np.int64)  # exclusive end sentence index per doc

    # NT mask over token-id space
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

    # ------------------------------------------------------------------ #
    # 2. Full corpus integrity scan
    # ------------------------------------------------------------------ #
    hr("2. Full-corpus terminal-consistency scan (ALL sentences)")
    broken_count = 0
    bad_cand_counts: List[int] = []          # #bad candidates per broken sentence
    bad_slot_counter: Counter = Counter()     # which candidate ids (0..299) are bad
    broken_sentence_ids: List[int] = []
    for s in range(num_sentences):
        ok, nbad, _, bad_ids = sentence_terminal_check(tree, offsets, s, is_nt)
        if not ok:
            broken_count += 1
            bad_cand_counts.append(nbad)
            broken_sentence_ids.append(s)
            for cid in bad_ids:
                bad_slot_counter[cid] += 1
        if (s + 1) % 20000 == 0:
            print(f"  scanned {s+1}/{num_sentences}  (broken so far: {broken_count})", flush=True)

    rate = 100.0 * broken_count / num_sentences
    print(f"\nsentences scanned        : {num_sentences:,}")
    print(f"broken sentences         : {broken_count:,}  ({rate:.2f}%)")
    print(f"clean sentences          : {num_sentences - broken_count:,}  ({100-rate:.2f}%)")

    if bad_cand_counts:
        bcc = np.array(bad_cand_counts)
        print(f"\n#bad candidates per broken sentence:")
        print(f"  min={bcc.min()}  p25={np.percentile(bcc,25):.0f}  median={np.percentile(bcc,50):.0f}  "
              f"mean={bcc.mean():.1f}  p75={np.percentile(bcc,75):.0f}  max={bcc.max()}")
        # histogram buckets
        print("  distribution:")
        buckets = [(0, 0), (1, 1), (2, 5), (6, 20), (21, 100), (101, 299), (300, 300)]
        for lo, hi in buckets:
            if lo == hi:
                cnt = int(np.sum(bcc == lo))
                label = f"=={lo}"
            else:
                cnt = int(np.sum((bcc >= lo) & (bcc <= hi)))
                label = f"{lo}-{hi}"
            if cnt:
                print(f"    bad-cand {label:>9}: {cnt:>6} broken sentences "
                      f"({100*cnt/broken_count:.1f}% of broken)")

    # ------------------------------------------------------------------ #
    # 3. Candidate-slot distribution: is a fixed block of ids always bad?
    # ------------------------------------------------------------------ #
    hr("3. Which candidate slots (0..299) are bad? (grouping artifact check)")
    if bad_slot_counter:
        slots = np.array([bad_slot_counter.get(i, 0) for i in range(SPS)])
        print(f"candidate slots scanned : 0..{SPS-1}")
        print(f"slots ever bad          : {int(np.sum(slots > 0))} / {SPS}")
        print(f"slot bad-count: min={slots.min()} median={np.median(slots):.0f} "
              f"mean={slots.mean():.1f} max={slots.max()}")
        print("\nslots bad in >5% of broken sentences:")
        thr = 0.05 * broken_count
        hot = [(i, int(c)) for i, c in enumerate(slots) if c > thr]
        if hot:
            for i, c in hot[:40]:
                print(f"  candidate {i:>3}: bad in {c} sentences ({100*c/broken_count:.1f}% of broken)")
            if len(hot) > 40:
                print(f"  ... and {len(hot)-40} more")
        else:
            print("  (none — corruption is NOT concentrated in fixed candidate slots)")
        # contiguous-block detection
        print("\nlongest contiguous run of always-bad slots (>10% broken):")
        run_start = None; best = (0, 0); cur_start = None; cur_len = 0
        for i in range(SPS):
            if slots[i] > 0.10 * broken_count:
                if cur_start is None:
                    cur_start = i; cur_len = 1
                else:
                    cur_len += 1
            else:
                if cur_len > best[1]:
                    best = (cur_start, cur_len)
                cur_start = None; cur_len = 0
        if cur_len > best[1]:
            best = (cur_start, cur_len)
        print(f"  start={best[0]} length={best[1]}" if best[1] else "  (no contiguous always-bad run)")

    # ------------------------------------------------------------------ #
    # 4. Per-document aggregation
    # ------------------------------------------------------------------ #
    hr("4. Per-document brokenness (is it uniform or clustered?)")
    # assign each broken sentence to its document
    broken_arr = np.array(broken_sentence_ids, dtype=np.int64) if broken_sentence_ids else np.empty(0, dtype=np.int64)
    doc_of_broken = np.searchsorted(doc_ends, broken_arr, side="right") if len(broken_arr) else np.empty(0, dtype=np.int64)
    doc_broken = np.bincount(doc_of_broken, minlength=num_docs)
    doc_total = np.asarray(doc_counts, dtype=np.int64)
    doc_rate = doc_broken / np.maximum(doc_total, 1)
    docs_with_any = int(np.sum(doc_broken > 0))
    print(f"documents with >=1 broken sentence : {docs_with_any} / {num_docs} "
          f"({100*docs_with_any/num_docs:.1f}%)")
    print(f"documents with ALL sentences broken: {int(np.sum(doc_broken == doc_total))} / {num_docs}")
    print(f"documents with 0 broken sentences  : {int(np.sum(doc_broken == 0))} / {num_docs} "
          f"({100*np.sum(doc_broken==0)/num_docs:.1f}%)")
    print(f"\nper-document broken rate distribution:")
    nz = doc_rate[doc_total > 0]
    print(f"  min={nz.min():.2f} p25={np.percentile(nz,25):.2f} median={np.median(nz):.2f} "
          f"mean={nz.mean():.2f} p75={np.percentile(nz,75):.2f} max={nz.max():.2f}")
    print("\nrate buckets (broken sentences / total in doc):")
    for lo, hi in [(0.0,0.0),(0.01,0.25),(0.26,0.50),(0.51,0.75),(0.76,0.99),(1.0,1.0)]:
        if lo == hi:
            cnt = int(np.sum(nz == lo)); label = f"{lo:.2f}"
        else:
            cnt = int(np.sum((nz >= lo) & (nz <= hi))); label = f"{lo:.2f}-{hi:.2f}"
        if cnt:
            print(f"    rate {label:>11}: {cnt:>5} docs ({100*cnt/num_docs:.1f}%)")

    # ------------------------------------------------------------------ #
    # 5. Sampled broken sentences: the actual diverging terminals
    # ------------------------------------------------------------------ #
    hr("5. Sampled broken sentences: diverging terminal sequences")
    rng = np.random.default_rng(12345)
    n_show = 6
    if len(broken_sentence_ids) >= n_show:
        show_ids = rng.choice(broken_sentence_ids, size=n_show, replace=False)
    else:
        show_ids = np.array(broken_sentence_ids)
    show_ids = sorted(int(x) for x in show_ids)

    for s in show_ids:
        first = s * SPS
        ok, nbad, ref, bad_ids = sentence_terminal_check(tree, offsets, s, is_nt)
        doc_id = int(np.searchsorted(doc_ends, s, side="right"))
        print(f"\n--- sentence {s} (document {doc_id}): {nbad}/300 candidates differ ---")
        # candidate 0
        print(f"  cand   0: len={len(ref)}  tokens[:16]={ref[:16].tolist()}")
        # show first 2 bad candidates
        for cid in bad_ids[:2]:
            sc, ec = int(offsets[first + cid]), int(offsets[first + cid + 1])
            cand = record_terminals(tree, sc, ec, is_nt)
            # find first divergence
            m = min(len(ref), len(cand))
            k = next((p for p in range(m) if int(ref[p]) != int(cand[p])), m)
            print(f"  cand {cid:>3}: len={len(cand)}  tokens[:16]={cand[:16].tolist()}")
            print(f"            first divergence at terminal idx {k}: "
                  f"ref={int(ref[k]) if k<len(ref) else 'END'} vs cand={int(cand[k]) if k<len(cand) else 'END'}")
            # is the bad candidate's terminals == some OTHER candidate in this block?
            # (cheap check: compare to a few neighbours)
            match = None
            for other in range(SPS):
                if other == cid:
                    continue
                so, eo = int(offsets[first + other]), int(offsets[first + other + 1])
                if record_terminals(tree, so, eo, is_nt).tolist() == cand.tolist():
                    match = other
                    break
            if match is not None:
                print(f"            bad cand {cid} terminals == cand {match} (duplicate group!)")

    # ------------------------------------------------------------------ #
    # 6. Verdict
    # ------------------------------------------------------------------ #
    hr("6. Verdict")
    print(f"tree_300.npy terminal-consistency defect:")
    print(f"  {broken_count:,} / {num_sentences:,} sentences ({rate:.2f}%) have >=1 candidate")
    print(f"  whose terminal sequence differs from candidate 0.")
    print(f"  Corruption is {'SYSTEMIC' if rate > 5 else 'SPORADIC'} and spans {docs_with_any}/{num_docs} documents.")
    if bad_cand_counts:
        med = np.median(bad_cand_counts)
        if med <= 5:
            print(f"  Most broken sentences have only a few bad candidates (median {med:.0f}/300) ->")
            print("  likely a small number of mis-grouped records per sentence.")
        else:
            print(f"  Broken sentences have many bad candidates (median {med:.0f}/300) ->")
            print("  large contiguous groups of records belong to the wrong sentence.")
    print("\nThis is a DATA defect in tree_300.npy (2025-08-12), independent of the")
    print("GPST tree_spans code bug.  Fix requires regenerating tree_300.npy or")
    print("filtering broken sentences at eval time.")


if __name__ == "__main__":
    main()
