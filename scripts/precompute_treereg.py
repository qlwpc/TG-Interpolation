#!/usr/bin/env python
"""Build paper-faithful parse-aligned data for Tree Regularization.

TreeReg consumes terminal token IDs, binary constituent spans, exact
first-subword masks, and complete top-level-tree IDs. Unary chains are
collapsed before right-recursive binarization (``direction=right`` under the
NLTK factor naming convention), matching the upstream preprocessing.

Example:
    python scripts/precompute_treereg.py \
      --split dev --workers 4 \
      --out-dir dataset/bbc-news/parse_aligned/dev_treereg
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
from pathlib import Path

# Prevent hidden BLAS/OpenMP thread pools inside each parser process.
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

import numpy as np

# Make the documented ``python scripts/precompute_treereg.py`` invocation work
# without requiring an editable install or a manually set PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olmo.data.parse_align import PrecomputedParseDataset, preprocess_split


GIB = 2**30


def _memory_capacity() -> tuple[int, int]:
    values = {}
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            key, value, *_ = line.split()
            values[key.rstrip(":")] = int(value) * 1024
    return values["MemTotal"], values["MemAvailable"]


def _physical_core_count() -> int:
    allowed = sorted(os.sched_getaffinity(0))
    cores = set()
    for cpu in allowed:
        try:
            with open(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/physical_package_id",
                encoding="utf-8",
            ) as f:
                package = int(f.read())
            with open(
                f"/sys/devices/system/cpu/cpu{cpu}/topology/core_id",
                encoding="utf-8",
            ) as f:
                core = int(f.read())
            cores.add((package, core))
        except (OSError, ValueError):
            cores.add((0, cpu))
    return max(len(cores), 1)


def _resolve_workers(requested: int, max_workers: int) -> tuple[int, int]:
    physical = _physical_core_count()
    safe_limit = max(1, min(max_workers, physical // 4))
    workers = min(requested, safe_limit) if requested > 0 else safe_limit
    return max(workers, 1), safe_limit


def _acquire_run_lock(output_parent: str, description: str):
    os.makedirs(output_parent, exist_ok=True)
    lock_path = os.path.join(output_parent, ".treereg_preprocess.lock")
    lock = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError(
            "another TreeReg preprocessing run holds "
            f"{lock_path}; run one direction at a time"
        ) from exc
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid={os.getpid()} {description}\n")
    lock.flush()
    return lock


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
        default="right",
        help="binarization spine direction (NLTK chomsky_normal_form factor name): "
             "'right'=right-recursive (default; matches official TreeReg + NLTK default), "
             "'left'=left-recursive",
    )
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--pad-token-id", type=int, default=50258)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="parser processes; 0 chooses a conservative automatic value",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
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
    parser.add_argument(
        "--min-free-disk-gb",
        type=float,
        default=256.0,
        help="minimum disk space that must remain after estimated output",
    )
    parser.add_argument(
        "--min-free-disk-percent",
        type=float,
        default=10.0,
        help="minimum filesystem percentage that must remain after output",
    )
    parser.add_argument(
        "--min-free-memory-gb",
        type=float,
        default=64.0,
        help="memory reserve when --load-tree-to-ram is used",
    )
    parser.add_argument(
        "--min-free-memory-percent",
        type=float,
        default=20.0,
        help="minimum host-memory percentage to reserve",
    )
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
        print(
            f"capping scan_workers={args.scan_workers} to {scan_workers}",
            flush=True,
        )

    output_parent = os.path.dirname(os.path.abspath(args.out_dir))
    run_lock = _acquire_run_lock(
        output_parent,
        f"split={args.split} direction={args.direction} out={args.out_dir}",
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
        save_depth_matrix=False,
        workers=workers,
        scan_workers=scan_workers,
        warm_cache=args.warm_cache,
        max_spans_override=args.max_spans,
        load_tree_to_ram=args.load_tree_to_ram,
        binarize=True,
        collapse_unary=True,
        add_boundary_root=False,
        save_treereg_metadata=True,
        input_dtype=np.uint16,
        span_dtype=np.int16,
        pin_workers=args.pin_workers,
        min_free_disk_bytes=disk_reserve,
        min_free_memory_bytes=memory_reserve,
    )

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
        f"spans={os.path.join(args.out_dir, 'spans.npy')}"
    )
    run_lock.close()


if __name__ == "__main__":
    main()
