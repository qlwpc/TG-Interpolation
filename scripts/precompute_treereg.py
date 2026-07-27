#!/usr/bin/env python
"""Build paper-faithful parse-aligned data for Tree Regularization.

TreeReg consumes terminal token IDs, binary constituent spans, exact
first-subword masks, and complete top-level-tree IDs. Unary chains are
collapsed before left binarization, matching the upstream preprocessing.

Example:
    python scripts/precompute_treereg.py \
      --split dev --workers 4 \
      --out-dir dataset/bbc-news/parse_aligned/dev_treereg
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
    out = np.lib.format.open_memmap(
        dst, mode="w+", dtype=np.int16, shape=spans.shape
    )
    for start in range(0, spans.shape[0], 65536):
        end = min(start + 65536, spans.shape[0])
        out[start:end] = spans[start:end]
    out.flush()
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="split name: train/dev/test")
    parser.add_argument("--tree-dir", default="dataset/bbc-news/tree")
    parser.add_argument(
        "--tokenizer", default="dataset/bbc-news/TG_GPT2_tokenizer.json"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--direction",
        choices=("left", "right"),
        default="left",
        help="upstream TreeReg uses left binarization",
    )
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--pad-token-id", type=int, default=50258)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scan-workers", type=int, default=1)
    parser.add_argument("--max-spans", type=int)
    parser.add_argument("--load-tree-to-ram", action="store_true")
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument(
        "--keep-int32",
        action="store_true",
        help="retain spans.npy after making the smaller spans_int16.npy",
    )
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
        save_depth_matrix=False,
        workers=args.workers,
        scan_workers=args.scan_workers,
        warm_cache=args.warm_cache,
        max_spans_override=args.max_spans,
        load_tree_to_ram=args.load_tree_to_ram,
        binarize=True,
        collapse_unary=True,
        add_boundary_root=False,
        save_treereg_metadata=True,
    )
    int16_path = _convert_spans_to_int16(args.out_dir)
    if not args.keep_int32:
        os.remove(os.path.join(args.out_dir, "spans.npy"))

    dataset = PrecomputedParseDataset(
        args.out_dir,
        pad_token_id=args.pad_token_id,
        require_treereg_metadata=True,
    )
    first = dataset[0]
    sentence_ids = first["treereg_sentence_ids"]
    word_boundaries = first["treereg_word_boundaries"]
    valid = sentence_ids >= 0
    starts = valid.clone()
    starts[1:] &= sentence_ids[1:] != sentence_ids[:-1]
    if torch_count := int(starts.sum().item()):
        if not bool(torch_all := word_boundaries[starts].all().item()):
            raise AssertionError(
                f"top-level tree starts are not word boundaries: {torch_all}"
            )
    print(
        f"verified {args.out_dir}: chunks={len(dataset)}, "
        f"span_dtype={dataset.spans.dtype}, "
        f"first_spans={len(first['tree_spans'])}, "
        f"first_trees={torch_count}, "
        f"first_word_starts={int(word_boundaries.sum().item())}, "
        f"int16={int16_path}"
    )


if __name__ == "__main__":
    main()
