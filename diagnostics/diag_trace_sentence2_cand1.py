#!/usr/bin/env python
"""Trace tree_to_merge_orders on sentence 2, candidate 1 (the GPST-invalid case).

Prints the raw tree, collapse_unary -> binarize -> tree_spans, then pinpoints
which spans produce the duplicate gap (9) and why gap 5 is missing.
"""
from __future__ import annotations

import os
import sys

REPO = "/home/wangpch/TG-Interpolation"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from olmo.data.parse_align import (  # noqa: E402
    TreeVocab,
    parse_block_segments,
    collapse_unary_tree,
    binarize_tree,
    tree_spans,
)
from olmo.gpst.reader.dataset_gold import tree_to_merge_orders  # noqa: E402

TREE = f"{REPO}/dataset/testppl_tree/tree_300.npy"
SENT_IDX = f"{REPO}/dataset/testppl_tree/tree_sent_index.npy"
TOK = f"{REPO}/dataset/bbc-news/TG_GPT2_tokenizer.json"
SPS = 300
SENT, CAND = 2, 1


def fmt_tree(node, depth: int = 0) -> str:
    if isinstance(node, int):
        return f"L{node}"
    label, children = node
    inner = " ".join(fmt_tree(c, depth + 1) for c in children)
    return f"({label} {inner})"


def main() -> None:
    tree = np.load(TREE, mmap_mode="r")
    lengths = np.load(SENT_IDX, mmap_mode="r")
    vocab = TreeVocab.from_tokenizer_file(TOK)

    offsets = np.empty(len(lengths) + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])

    first = SENT * SPS + CAND
    start, end = int(offsets[first]), int(offsets[first + 1])
    block = tree[start:end].astype(np.int64).tolist()
    print(f"record [{start}:{end}] length={len(block)}")

    segs = parse_block_segments(block, vocab)
    trees = [data for kind, data in segs if kind == "tree"]
    print(f"segments: {[(k, ('tree' if k=='tree' else len(d))) for k,d in segs]}")
    print(f"#top-level trees: {len(trees)}")
    if len(trees) != 1:
        print("NOTE: expected exactly 1 tree; using trees[0]")
    raw = trees[0]

    print("\n--- RAW TREE (first 1200 chars) ---")
    print(fmt_tree(raw)[:1200])

    collapsed = collapse_unary_tree(raw)
    print("\n--- AFTER collapse_unary (first 1200 chars) ---")
    print(fmt_tree(collapsed)[:1200])

    binned = binarize_tree(collapsed, direction="right")
    print("\n--- AFTER binarize (right) (first 1600 chars) ---")
    print(fmt_tree(binned)[:1600])

    leaves, spans = tree_spans(binned)
    print(f"\n--- tree_spans on BINARIZED tree ---")
    print(f"#leaves={len(leaves)}  #spans={len(spans)}")
    print("spans (left, split, right) [post-order]:")
    for i, (l, sp, r) in enumerate(spans):
        flag = ""
        if l == r:
            flag = "  <- single-leaf (preterminal/unary), filtered by _l!=_r"
        print(f"  [{i:2d}] l={l:2d} split={sp:2d} r={r:2d}{flag}")

    merge_orders = [sp for (_l, sp, _r) in spans if _l != _r]
    print(f"\nmerge_orders (len={len(merge_orders)}, expect L-1={len(leaves)-1}):")
    print(f"  {merge_orders}")
    want = list(range(len(leaves) - 1))
    got = sorted(merge_orders)
    print(f"sorted      : {got}")
    print(f"want range  : {want}")
    print(f"missing     : {sorted(set(want) - set(merge_orders))}")
    from collections import Counter
    dups = {k: v for k, v in Counter(merge_orders).items() if v > 1}
    print(f"duplicates  : {dups}")

    # pin which spans share the duplicated split
    for dup_gap, cnt in dups.items():
        owners = [i for i, (l, sp, r) in enumerate(spans) if sp == dup_gap and l != r]
        print(f"\nduplicate gap {dup_gap} (x{cnt}) owned by spans: {owners}")
        for i in owners:
            l, sp, r = spans[i]
            print(f"  span[{i}] l={l} split={sp} r={r}  -> covers leaves [{l}..{r}], bifurcates at {sp}")

    # call the real function to confirm
    L, orders = tree_to_merge_orders(raw, direction="right")
    print(f"\ntree_to_merge_orders confirms: L={L}, orders={orders}")


if __name__ == "__main__":
    main()
