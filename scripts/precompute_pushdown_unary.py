#!/usr/bin/env python
"""Build unary-collapsed, CNF Pushdown parse-aligned data.

The Murty et al. preprocessing collapses unary chains and converts each parse
to Chomsky normal form before deriving attachment labels and stack tapes. This
script applies the equivalent operations to this repository's tokenized
``tree/*.npy`` streams:

  1. collapse internal unary chains;
  2. left- or right-binarize n-ary nodes;
  3. wrap every top-level tree as ``[ROOT] leaves [EOS]``;
  4. save terminal input ids and binary constituent spans;
  5. write ``spans.npy`` directly as int16 after validating every row.

It writes to a new output directory and never removes an existing dataset.

The default ROOT is tokenizer token 50260 (``<|CLS|>``); EOS is 50256
(``<|endoftext|>``). Document-level boundary leaves outside parsed trees are
discarded, so every emitted sequence follows the paper's sentence convention.

Example:
    python scripts/precompute_pushdown_unary.py \
      --split dev --direction left --workers 4 \
      --out-dir dataset/bbc-news/parse_aligned/dev_pushdown_unary
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Keep each parser process single-threaded; process-level parallelism is managed
# explicitly below.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olmo.data.parse_align import PrecomputedParseDataset, preprocess_split
from scripts.precompute_treereg import (
    GIB,
    _acquire_run_lock,
    _memory_capacity,
    _physical_core_count,
    _resolve_workers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, help="split name, e.g. train/dev/test")
    parser.add_argument("--tree-dir", default="dataset/tree")
    parser.add_argument("--tokenizer", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--direction", choices=("left", "right"), default="right",
                        help="binarization spine direction (NLTK factor name): "
                             "'right'=right-recursive (default, TreeReg/NLTK), 'left'=left-recursive")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--pad-token-id", type=int, default=50258)
    parser.add_argument("--root-token-id", type=int, default=50260,
                        help="ROOT token prepended to every top-level tree")
    parser.add_argument("--sentence-eos-token-id", type=int, default=50256,
                        help="EOS token appended to every top-level tree")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="parser processes; 0 chooses a conservative automatic value",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="hard safety cap; also capped at one quarter of physical cores",
    )
    parser.add_argument("--scan-workers", type=int, default=1)
    parser.add_argument("--max-spans", type=int)
    parser.add_argument("--load-tree-to-ram", action="store_true")
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument(
        "--pin-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pin parser processes to distinct allowed physical cores",
    )
    parser.add_argument("--min-free-disk-gb", type=float, default=256.0)
    parser.add_argument("--min-free-disk-percent", type=float, default=10.0)
    parser.add_argument("--min-free-memory-gb", type=float, default=64.0)
    parser.add_argument("--min-free-memory-percent", type=float, default=20.0)
    parser.add_argument("--save-depth", action="store_true",
                        help="save the O(n^2) int8 depth matrix (normally unnecessary)")
    args = parser.parse_args()

    tree_npy = os.path.join(args.tree_dir, f"{args.split}.npy")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative (0 means automatic)")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")
    if args.scan_workers < 1:
        raise ValueError("--scan-workers must be at least 1")
    if args.min_free_disk_gb < 0 or args.min_free_memory_gb < 0:
        raise ValueError("free-space reserves in GB must be non-negative")
    if not 0 <= args.min_free_disk_percent <= 100:
        raise ValueError("--min-free-disk-percent must be between 0 and 100")
    if not 0 <= args.min_free_memory_percent <= 100:
        raise ValueError("--min-free-memory-percent must be between 0 and 100")

    workers, safe_worker_limit = _resolve_workers(args.workers, args.max_workers)
    scan_workers = min(args.scan_workers, 2)
    if args.workers > workers:
        print(
            f"capping requested workers={args.workers} to safe workers={workers} "
            f"(limit={safe_worker_limit})",
            flush=True,
        )
    if args.scan_workers > scan_workers:
        print(f"capping scan_workers={args.scan_workers} to {scan_workers}", flush=True)

    output_parent = os.path.dirname(os.path.abspath(args.out_dir))
    run_lock = _acquire_run_lock(
        output_parent,
        f"unary-pushdown split={args.split} direction={args.direction} "
        f"out={args.out_dir}",
    )
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory: {args.out_dir}"
        )

    disk = shutil.disk_usage(output_parent)
    mem_total, mem_available = _memory_capacity()
    disk_reserve = max(
        int(args.min_free_disk_gb * GIB),
        int(disk.total * args.min_free_disk_percent / 100),
    )
    memory_reserve = max(
        int(args.min_free_memory_gb * GIB),
        int(mem_total * args.min_free_memory_percent / 100),
    )
    print(
        "resource plan: "
        f"workers={workers}/{_physical_core_count()} physical cores, "
        f"allowed_logical_cpus={len(os.sched_getaffinity(0))}, "
        f"memory_available={mem_available / GIB:.1f}/{mem_total / GIB:.1f} GiB, "
        f"disk_free={disk.free / GIB:.1f}/{disk.total / GIB:.1f} GiB, "
        f"memory_reserve={memory_reserve / GIB:.1f} GiB, "
        f"disk_reserve={disk_reserve / GIB:.1f} GiB",
        flush=True,
    )

    preprocess_split(
        tree_npy=tree_npy,
        tokenizer_path=args.tokenizer,
        direction=args.direction,
        out_dir=args.out_dir,
        max_len=args.max_len,
        pad_token_id=args.pad_token_id,
        save_depth_matrix=args.save_depth,
        workers=workers,
        scan_workers=scan_workers,
        warm_cache=args.warm_cache,
        max_spans_override=args.max_spans,
        load_tree_to_ram=args.load_tree_to_ram,
        binarize=True,
        collapse_unary=True,
        add_boundary_root=False,
        wrap_toplevel_trees=True,
        root_token_id=args.root_token_id,
        sentence_eos_token_id=args.sentence_eos_token_id,
        input_dtype=np.uint16,
        span_dtype=np.int16,
        pin_workers=args.pin_workers,
        min_free_disk_bytes=disk_reserve,
        min_free_memory_bytes=memory_reserve,
    )

    dataset = PrecomputedParseDataset(
        args.out_dir,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.sentence_eos_token_id,
        require_pushdown_root_token_id=args.root_token_id,
        expected_binarize_direction=args.direction,
    )
    first = dataset[0]
    manifest_path = os.path.join(args.out_dir, "preprocessing.json")
    if not os.path.exists(manifest_path):
        raise AssertionError(f"missing preprocessing manifest: {manifest_path}")
    if dataset.spans.dtype != np.int16:
        raise AssertionError(f"expected int16 spans.npy, got {dataset.spans.dtype}")
    print(
        f"verified {args.out_dir}: chunks={len(dataset)}, "
        f"span_dtype={dataset.spans.dtype}, first_spans={len(first['tree_spans'])}, "
        f"spans={os.path.join(args.out_dir, 'spans.npy')}"
    )
    run_lock.close()


if __name__ == "__main__":
    main()
