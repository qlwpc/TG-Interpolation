#!/usr/bin/env python
"""Precompute Pushdown spans (NO binarization) + store as int16.

Pipeline per split:
  1. preprocess_split(binarize=False, save_depth_matrix=False)
     -> input_ids.npy, spans.npy (int32), span_counts.npy, chunk_index.npy
  2. convert_spans_to_int16 -> spans_int16.npy (int16, lossless; values < 2048)
  3. delete the int32 spans.npy (loader prefers spans_int16.npy) -> int16-only storage
  4. verify PrecomputedParseDataset reads the int16 mmap + upcasts to int64

The Pushdown stack tape (compute_depth_matrix) uses only (l, r); binarization
injects artificial X|</X|> constituents whose sub-ranges inflate the depth, so
Pushdown MUST use raw (no-binarize) spans. int16 halves disk vs int32.

Usage:
    python scripts/precompute_pushdown_int16.py <split> [workers] [scan_workers]
    python scripts/precompute_pushdown_int16.py train 32 8
    python scripts/precompute_pushdown_int16.py dev 4 1
    LOAD_TREE_TO_RAM=0 python scripts/precompute_pushdown_int16.py train 32 8
"""
import os
import sys
import time

PYTHONPATH = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{PYTHONPATH}:{os.path.expanduser('~/TG-Interpolation')}"
sys.path.insert(0, os.path.expanduser("~/TG-Interpolation"))

import numpy as np
from olmo.data.parse_align import preprocess_split, PrecomputedParseDataset
from datatools.convert_spans_to_int16 import convert_one

TREE = "dataset/tree"
TOK = "dataset/bbc-news/TG_GPT2_tokenizer.json"
PA = "dataset/bbc-news/parse_aligned"


def run_split(split: str, workers: int, scan_workers: int, load_ram: bool):
    out_dir = f"{PA}/{split}_pushdown"
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    print(f"\n=== {split}: preprocess (no_binarize, no depth) -> {out_dir} ===",
          flush=True)
    preprocess_split(
        tree_npy=f"{TREE}/{split}.npy", tokenizer_path=TOK,
        direction="left", out_dir=out_dir,
        max_len=2048, pad_token_id=50258,
        save_depth_matrix=False,
        workers=workers, scan_workers=scan_workers,
        load_tree_to_ram=load_ram, binarize=False,
    )
    print(f"  preprocess: {time.time()-t0:.1f}s", flush=True)

    # int32 spans.npy -> int16 spans_int16.npy, then drop int32.
    t1 = time.time()
    convert_one(out_dir, block=65536)
    int32_path = os.path.join(out_dir, "spans.npy")
    int16_path = os.path.join(out_dir, "spans_int16.npy")
    sz32 = os.path.getsize(int32_path)
    os.remove(int32_path)
    print(f"  int16 convert: {time.time()-t1:.1f}s; dropped spans.npy "
          f"({sz32/1e9:.1f} GB), kept spans_int16.npy "
          f"({os.path.getsize(int16_path)/1e9:.1f} GB)", flush=True)

    # Verify the loader reads int16 + upcasts.
    ds = PrecomputedParseDataset(out_dir, pad_token_id=50258, load_depth=False)
    assert ds.spans.dtype == np.int16, f"expected int16 mmap, got {ds.spans.dtype}"
    item = ds[0]
    sp = item["tree_spans"]
    print(f"  loader OK: n_chunks={ds.n_chunks}; spans mmap dtype={ds.spans.dtype}; "
          f"item[0] tree_spans {tuple(sp.shape)} dtype={sp.dtype}; "
          f"n_spans={int(item['tree_span_mask'].sum())}", flush=True)
    print(f"=== {split} DONE ({time.time()-t0:.1f}s total) ===", flush=True)


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    scan_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    load_ram = os.environ.get("LOAD_TREE_TO_RAM", "1") != "0"
    run_split(split, workers, scan_workers, load_ram)
