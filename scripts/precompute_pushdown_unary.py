#!/usr/bin/env python
"""Build paper-faithful Pushdown parse-aligned data.

The Murty et al. preprocessing collapses unary chains and converts each parse
to Chomsky normal form before deriving attachment labels and stack tapes. This
script applies the equivalent operations to this repository's tokenized
``tree/*.npy`` streams:

  1. collapse internal unary chains;
  2. left- or right-binarize n-ary nodes;
  3. add the paper's BOS/ROOT-to-EOS sentence attachment;
  4. save terminal input ids and binary constituent spans;
  5. optionally convert spans to int16 after validating their range.

It writes to a new output directory and never removes an existing raw-Pushdown
dataset.

Example:
    python scripts/precompute_pushdown_unary.py \
      --split dev --direction left --workers 4 \
      --out-dir dataset/bbc-news/parse_aligned/dev_pushdown_unary
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from olmo.data.parse_align import PrecomputedParseDataset, preprocess_split


def _convert_spans_to_int16(data_dir: str) -> str:
    src = os.path.join(data_dir, "spans.npy")
    dst = os.path.join(data_dir, "spans_int16.npy")
    spans = np.load(src, mmap_mode="r")
    if spans.size:
        lo = int(spans.min())
        hi = int(spans.max())
        info = np.iinfo(np.int16)
        if lo < info.min or hi > info.max:
            raise ValueError(f"span range [{lo}, {hi}] does not fit int16")
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.int16, shape=spans.shape)
    rows = spans.shape[0]
    for start in range(0, rows, 65536):
        end = min(start + 65536, rows)
        out[start:end] = spans[start:end]
    out.flush()
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="split name, e.g. train/dev/test")
    parser.add_argument("--tree-dir", default="dataset/tree")
    parser.add_argument("--tokenizer", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--direction", choices=("left", "right"), default="left")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--pad-token-id", type=int, default=50258)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scan-workers", type=int, default=1)
    parser.add_argument("--max-spans", type=int)
    parser.add_argument("--load-tree-to-ram", action="store_true")
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument("--save-depth", action="store_true",
                        help="save the O(n^2) int8 depth matrix (normally unnecessary)")
    parser.add_argument("--keep-int32", action="store_true",
                        help="also retain spans.npy after making spans_int16.npy")
    args = parser.parse_args()

    tree_npy = os.path.join(args.tree_dir, f"{args.split}.npy")
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.out_dir}"
        )
    preprocess_split(
        tree_npy=tree_npy,
        tokenizer_path=args.tokenizer,
        direction=args.direction,
        out_dir=args.out_dir,
        max_len=args.max_len,
        pad_token_id=args.pad_token_id,
        save_depth_matrix=args.save_depth,
        workers=args.workers,
        scan_workers=args.scan_workers,
        warm_cache=args.warm_cache,
        max_spans_override=args.max_spans,
        load_tree_to_ram=args.load_tree_to_ram,
        binarize=True,
        collapse_unary=True,
        add_boundary_root=True,
    )
    int16_path = _convert_spans_to_int16(args.out_dir)
    if not args.keep_int32:
        os.remove(os.path.join(args.out_dir, "spans.npy"))

    dataset = PrecomputedParseDataset(args.out_dir, pad_token_id=args.pad_token_id)
    first = dataset[0]
    print(
        f"verified {args.out_dir}: chunks={len(dataset)}, "
        f"span_dtype={dataset.spans.dtype}, first_spans={len(first['tree_spans'])}, "
        f"int16={int16_path}"
    )


if __name__ == "__main__":
    main()
