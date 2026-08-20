#!/usr/bin/env python
"""Diagnose dataset/testppl_tree consistency for jobs 45195 (pushdown) & 45196 (gpst).

Reproduces both failing checks against the SAME data the eval jobs read, without
loading any model:

  * Pushdown (45195): ``sentence 4 gold candidates have different terminals``
    -> Section B dumps sentence 4's 300 candidate terminal sequences and the
       first divergence, plus per-record bracket balance (a non-zero residual
       depth means the sent_index offsets are misaligned with real tree
       boundaries -- the smoking gun for an indexing bug).

  * GPST (45196): ``grouped merge orders are not a global gap permutation``
    -> Section C checks each candidate's per-segment merge orders are a clean
       permutation of range(L-1) (the property the collator's grouped check
       ultimately requires).  Section D drives the real GoldTreeCollator on
       context+candidate items exactly like evaluate_gold_tree_document_ppl and
       dumps the first failing item's segments / splits / orders.

Run:  /home/wangpch/.conda/envs/LLM/bin/python diagnostics/diag_tree300_consistency.py
"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from olmo.data.parse_align import (  # noqa: E402
    TreeVocab,
    parse_block_segments,
    parse_chunk_slice,
)
from olmo.gpst.reader.dataset_gold import tree_to_merge_orders  # noqa: E402
from olmo.gpst.eval.document_ppl import (  # noqa: E402
    GoldTree300Corpus,
    GoldTreeCollator,
    _as_collator_item,
    _count_tokens,
    parse_gold_tree_candidate,
)

TREE = f"{REPO}/dataset/testppl_tree/tree_300.npy"
SENT_IDX = f"{REPO}/dataset/testppl_tree/tree_sent_index.npy"
DOC_IDX = f"{REPO}/dataset/testppl_tree/tree_doc_index.npy"
TOK = f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json"
SPS = 300


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def bracket_depth(block: Sequence[int], vocab: TreeVocab) -> int:
    """Net NT depth after walking a record. 0 == a balanced, complete tree(s)."""
    depth = 0
    for tok in block:
        t = int(tok)
        if vocab.is_opening(t):
            depth += 1
        elif vocab.is_closing(t):
            depth -= 1
    return depth


def pushdown_terminals(block: Sequence[int], vocab: TreeVocab) -> List[int]:
    """Faithful reproduction of PushdownGoldCandidate.tokens (parse_chunk_slice
    defaults: BOS/EOS/whitespace + tree content leaves, in order)."""
    parsed = parse_chunk_slice(
        block, vocab, direction="right", binarize=True,
        collapse_unary=True, drop_singleton_spans=True,
    )
    return parsed["input_ids"].tolist()


def main() -> None:
    tree = np.load(TREE, mmap_mode="r")
    lengths = np.load(SENT_IDX, mmap_mode="r")
    doc_counts = np.load(DOC_IDX, mmap_mode="r")
    vocab = TreeVocab.from_tokenizer_file(TOK)

    num_records = len(lengths)
    num_sentences = num_records // SPS

    # ---- Section A: index-level sanity ---------------------------------- #
    hr("A. Index sanity")
    print(f"tree_300.npy total tokens       : {len(tree):>12,}")
    print(f"tree_sent_index.npy records     : {num_records:>12,}")
    print(f"sum(sent_index)                 : {int(np.sum(lengths)):>12,}")
    print(f"len(sent_index) % 300           : {num_records % SPS:>12}")
    print(f"num_sentences (records // 300)  : {num_sentences:>12,}")
    print(f"sum(doc_index)                  : {int(np.sum(doc_counts)):>12,}")
    print(f"doc_index covers num_sentences? : {int(np.sum(doc_counts)) >= num_sentences}")
    print(f"sum(sent_index) == len(tree)?   : {int(np.sum(lengths)) == len(tree)}")

    offsets = np.empty(num_records + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])

    # ---- Section B: pushdown sentence 4 terminals ---------------------- #
    hr("B. Pushdown: sentence 4 candidate terminals (job 45195)")
    sent = 4
    first = sent * SPS
    cand_tokens: List[List[int]] = []
    cand_depths: List[int] = []
    for i in range(first, first + SPS):
        start, end = int(offsets[i]), int(offsets[i + 1])
        block = tree[start:end]
        cand_depths.append(bracket_depth(block, vocab))
        cand_tokens.append(pushdown_terminals(block, vocab))

    ref = cand_tokens[0]
    print(f"sentence {sent}: record lengths = {[int(lengths[i]) for i in range(first, first+SPS)][:8]}{' ...' if SPS>8 else ''}")
    print(f"sentence {sent}: bracket-balance residual depth (first 12) = {cand_depths[:12]}")
    print(f"sentence {sent}: #candidates with depth != 0 = {sum(1 for d in cand_depths if d != 0)} / {SPS}")
    print(f"sentence {sent}: terminal lengths (first 12) = {[len(t) for t in cand_tokens[:12]]}")
    mism = [(j, t) for j, t in enumerate(cand_tokens) if t != ref]
    print(f"sentence {sent}: #candidates whose terminals != candidate0 = {len(mism)} / {SPS}")
    if mism:
        j, t = mism[0]
        print(f"  first mismatch at candidate {j}: len(ref)={len(ref)}, len(cand)={len(t)}")
        # first diverging position
        k = next((p for p in range(min(len(ref), len(t))) if ref[p] != t[p]), min(len(ref), len(t)))
        lo, hi = max(0, k - 4), min(min(len(ref), len(t)), k + 5)
        print(f"  divergence at terminal index {k}")
        print(f"    ref  [{lo}:{hi}] = {ref[lo:hi]}")
        print(f"    cand [{lo}:{hi}] = {t[lo:hi]}")
        # also show raw token ids around the record boundary to spot offset drift
        s0, e0 = int(offsets[first]), int(offsets[first + 1])
        sj, ej = int(offsets[first + j]), int(offsets[first + j + 1])
        print(f"  candidate0 record tokens [{s0}:{e0}] (first 24): {tree[s0:e0][:24].tolist()}")
        print(f"  candidate{j} record tokens [{sj}:{ej}] (first 24): {tree[sj:ej][:24].tolist()}")
    else:
        print("  (no terminal mismatch in sentence 4 -- defect may be elsewhere)")

    # ---- Section C: per-segment merge-order permutation validity ------- #
    hr("C. GPST: per-segment merge_orders permutation check (probe for 45196)")
    print("For each candidate: each segment's merge_orders must be a permutation of")
    print("range(len(tokens)-1). The collator's grouped check fails otherwise.")
    bad_seg = None
    checked = 0
    for s in range(min(num_sentences, 40)):
        f = s * SPS
        for c in range(SPS):
            start, end = int(offsets[f + c]), int(offsets[f + c + 1])
            block = tree[start:end].astype(np.int64).tolist()
            try:
                segs = parse_gold_tree_candidate(block, vocab, direction="right")
            except Exception as exc:  # noqa: BLE001
                bad_seg = ("parse_error", s, c, -1, str(exc))
                break
            for si, seg in enumerate(segs):
                checked += 1
                L = len(seg.tokens)
                if L == 0:
                    continue
                want = list(range(L - 1))
                got = sorted(seg.merge_orders)
                if got != want or len(seg.merge_orders) != L - 1:
                    missing = sorted(set(want) - set(seg.merge_orders))
                    dups = sorted({x for x in seg.merge_orders if seg.merge_orders.count(x) > 1})
                    bad_seg = ("perm", s, c, si, {
                        "L": L, "orders": list(seg.merge_orders),
                        "missing": missing, "duplicates": dups,
                    })
                    break
            if bad_seg:
                break
        if bad_seg:
            break
    print(f"segments checked before stopping: {checked}")
    if bad_seg:
        kind, s, c, si, info = bad_seg
        print(f"FIRST INVALID: sentence {s}, candidate {c}, segment {si}  [{kind}]")
        print(f"  details: {info}")
    else:
        print("All checked segments are valid permutations (defect is NOT here).")

    # ---- Section D: real collator on context+candidate (faithful) ------ #
    hr("D. GPST: drive GoldTreeCollator on context+candidate (exact 45196 path)")
    corpus = GoldTree300Corpus(TREE, SENT_IDX, DOC_IDX, TOK, samples_per_sentence=SPS)
    collator = GoldTreeCollator()
    prefix: Tuple = ()
    prev_doc: int | None = None
    found = False
    for s, (doc_id, candidates) in enumerate(corpus):
        if doc_id != prev_doc:
            prefix = ()
            prev_doc = doc_id
        current = candidates[0]
        # batch the 300 candidates in chunks of 4, exactly like the eval
        for cs in range(0, len(candidates), 4):
            chunk = candidates[cs:cs + 4]
            items = [_as_collator_item(prefix + tuple(cand)) for cand in chunk]
            try:
                collator(items)
            except ValueError as exc:
                print(f"FIRST COLLATOR FAILURE: sentence {s}, doc {doc_id}, cand chunk {cs}..{cs+len(chunk)-1}")
                print(f"  error: {exc}")
                # dump the offending item's segment layout
                it = items[0]
                seg_lens = [len(o) for o in it["merge_orders"]]
                seg_toks = [len(list(_seg.tokens)) for _seg in (prefix + tuple(chunk[0]))]
                print(f"  item0 text_len={len(it['text'])} splits={it['sentence_splits']} "
                      f"seg_merge_counts={seg_lens}")
                print(f"  context prefix segments={len(prefix)} (token lens per seg shown above)")
                # show per-segment merge orders of item0
                for si, o in enumerate(it["merge_orders"]):
                    arr = o if isinstance(o, np.ndarray) else np.asarray(o)
                    L = seg_lens[si] + 1  # leaves = merges+1 (if valid)
                    print(f"    seg{si}: len(orders)={len(arr)} orders={arr.tolist()[:20]}{' ...' if len(arr)>20 else ''}")
                found = True
                break
        if found:
            break
        # OLMo commits candidate 0 to the shared document cache
        prefix = prefix + tuple(candidates[0])
        if s >= 60:
            print(f"  (no collator failure in first {s+1} sentences; stopping scan)")
            break
    if not found:
        print("No collator failure reproduced in scanned range.")


if __name__ == "__main__":
    main()
