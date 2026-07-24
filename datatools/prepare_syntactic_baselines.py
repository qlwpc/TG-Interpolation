#!/usr/bin/env python
"""Build a chunk index for the Pushdown / TreeReg syntactic-LM baselines.

Both baselines train on the *terminal* token sequence plus an aligned constituency
parse (from ``tree/*.npy``). Training chunks must be **whole-tree units** (no parse
tree split across a chunk boundary), so we cannot use the terminal baseline's fixed
2048-stride chunking. Instead we pack whole top-level parse trees into chunks of
``<= max_len`` terminal leaves.

This script scans ``tree/*.npy`` once (a bracket-depth walk, O(stream)) and emits a
chunk index ``chunk_index.npy`` of shape ``(n_chunks, 2)`` int64: ``(tree_start,
tree_len)`` — the tree-token slice for each chunk. The data loader
(:class:`olmo.data.parse_align.ParseAlignedDataset`) mmaps ``tree/*.npy`` + this
index, slices each chunk's tree tokens, and parses them on the fly (~8.5 ms/chunk)
to produce the terminal leaves (``input_ids`` — bit-identical to ``terminal/*.npy``)
and the constituent spans. The Pushdown depth matrix is computed on the GPU.

Usage:
    python datatools/prepare_syntactic_baselines.py \
        --tree_npy  dataset/bbc-news/tree/train.npy \
        --tokenizer dataset/bbc-news/TG_GPT2_tokenizer.json \
        --max_len 2048 \
        --out       dataset/bbc-news/parse_aligned/train
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from olmo.data.parse_align import TreeVocab, iter_tree_chunks


def build_chunk_index(tree_path, tok_path, max_len, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vocab = TreeVocab.from_tokenizer_file(tok_path)
    tree = np.load(tree_path, mmap_mode="r")
    print(f"scanning {tree_path} ({tree.shape[0]} tree tokens)...")
    chunks = iter_tree_chunks(tree, vocab, direction="left", max_len=max_len)
    idx = np.asarray(chunks, dtype=np.int64)
    out_idx = os.path.join(out_dir, "chunk_index.npy")
    np.save(out_idx, idx)
    # Stats: terminal leaves per chunk, computed once with the same NT predicate
    # the data loader uses (op_lo..cl_hi inclusive). Covers ALL chunks — the old
    # `chunks[:1]` min only inspected the first chunk.
    leaf_counts = []
    over = 0
    for (s, l) in chunks:
        sub = np.asarray(tree[s:s + l])
        is_nt = (sub >= vocab.op_lo) & (sub <= vocab.cl_hi)
        nl = int((~is_nt).sum())
        leaf_counts.append(nl)
        if nl > max_len:
            over += 1
    total_leaves = sum(leaf_counts)
    print(f"wrote {len(chunks)} chunks -> {out_idx}")
    print(f"  terminal leaves covered: {total_leaves}")
    if leaf_counts:
        print(f"  chunk leaf count: min {min(leaf_counts)} max {max(leaf_counts)} "
              f"mean {total_leaves / len(leaf_counts):.1f}")
    print(f"  chunks exceeding max_len (atomic trees > {max_len}): {over}")
    return out_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree_npy", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--out", required=True, help="output dir for chunk_index.npy")
    args = ap.parse_args()
    build_chunk_index(args.tree_npy, args.tokenizer, args.max_len, args.out)


if __name__ == "__main__":
    main()
