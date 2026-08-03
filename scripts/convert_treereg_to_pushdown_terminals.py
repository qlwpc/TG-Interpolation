#!/usr/bin/env python
"""Convert completed TreeReg arrays into primal-stream Pushdown arrays.

TreeReg and Pushdown use the same unary-collapsed, CNF constituency trees.  The
TreeReg arrays retain the exact primal terminal stream, including document
BOS/EOS and whitespace positions marked with ``sentence_id == -1``.  This
converter preserves that stream and its packed row positions bit-for-bit.  It
only removes singleton/preterminal spans because Algorithm 1 treats them as
SHIFT rather than REDUCE operations, and excludes spans from an incomplete tree
that crosses a packed-row boundary.

No ROOT or sentence-level EOS token is synthesized.  EOS tokens already present
in the primal stream are deliberately retained.  Exact parse-tree boundaries
are saved as ``pushdown_sentence_ids.npy`` for attachment masking.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Tuple

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ranges(n: int, parts: int) -> list[Tuple[int, int]]:
    parts = max(1, min(parts, n))
    q, r = divmod(n, parts)
    out = []
    lo = 0
    for i in range(parts):
        hi = lo + q + (i < r)
        if lo < hi:
            out.append((lo, hi))
        lo = hi
    return out


def _convert_range(src_dir: str, out_dir: str, pad_id: int, root_id: int,
                   eos_id: int, row_range: Tuple[int, int]) -> int:
    src_ids = np.load(os.path.join(src_dir, "input_ids.npy"), mmap_mode="r")
    src_spans = np.load(os.path.join(src_dir, "spans.npy"), mmap_mode="r")
    src_counts = np.load(os.path.join(src_dir, "span_counts.npy"), mmap_mode="r")
    src_sentences = np.load(
        os.path.join(src_dir, "treereg_sentence_ids.npy"), mmap_mode="r"
    )
    out_spans = np.lib.format.open_memmap(
        os.path.join(out_dir, "spans.npy"), mode="r+"
    )
    out_counts = np.lib.format.open_memmap(
        os.path.join(out_dir, "span_counts.npy"), mode="r+"
    )

    lo, hi = row_range
    width = src_ids.shape[1]
    for row in range(lo, hi):
        sentence_ids = np.asarray(src_sentences[row])
        tokens = np.asarray(src_ids[row])
        valid_tokens = tokens != pad_id
        if np.any(tokens[valid_tokens] == root_id):
            raise ValueError(
                f"row {row}: primal stream unexpectedly contains ROOT={root_id}"
            )

        count = int(src_counts[row])
        spans = np.asarray(src_spans[row, :count], dtype=np.int32)
        if len(spans):
            left, split, right = spans[:, 0], spans[:, 1], spans[:, 2]
            valid = ((left >= 0) & (right < width) & (split >= left)
                     & (split <= right) & (left < right))
            # Index only bounds-checked coordinates.  A complete constituent
            # must live within one parsed top-level tree; sentence_id == -1
            # marks primal BOS/EOS/whitespace, padding, or a boundary-crossing
            # incomplete tree.
            bounded = np.flatnonzero(valid)
            if len(bounded):
                l_sid = sentence_ids[left[bounded]]
                s_sid = sentence_ids[split[bounded]]
                r_sid = sentence_ids[right[bounded]]
                valid[bounded] &= (
                    (l_sid >= 0) & (l_sid == s_sid) & (l_sid == r_sid)
                )
            spans = spans[valid]

        # Each complete top-level tree must retain its native full-terminal span.
        for sentence_id in np.unique(sentence_ids[sentence_ids >= 0]):
            sent_pos = np.flatnonzero(sentence_ids == sentence_id)
            if not len(sent_pos):
                continue
            start, end = int(sent_pos[0]), int(sent_pos[-1])
            if start < end and not np.any(
                (spans[:, 0] == start) & (spans[:, 2] == end)
            ):
                raise ValueError(
                    f"row {row}: sentence {int(sentence_id)} lacks full span "
                    f"({start}, {end})"
                )

        if len(spans) > out_spans.shape[1]:
            raise ValueError(
                f"row {row}: {len(spans)} spans exceed capacity {out_spans.shape[1]}"
            )
        out_spans[row, :, :] = -1
        if len(spans):
            out_spans[row, : len(spans)] = spans
        out_counts[row] = len(spans)
    return hi - lo


def _convert_star(args) -> int:
    return _convert_range(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--direction", choices=("left", "right"), default="right")
    parser.add_argument("--pad-token-id", type=int, default=50258)
    parser.add_argument("--root-token-id", type=int, default=50260)
    parser.add_argument("--eos-token-id", type=int, default=50256)
    parser.add_argument("--min-free-disk-gb", type=float, default=256.0)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    src_dir = os.path.abspath(args.source_dir)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    lock_path = os.path.join(os.path.dirname(out_dir), ".pushdown_terminal_conversion.lock")
    lock = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another terminal conversion holds {lock_path}") from exc

    src_ids = np.load(os.path.join(src_dir, "input_ids.npy"), mmap_mode="r")
    src_spans = np.load(os.path.join(src_dir, "spans.npy"), mmap_mode="r")
    src_counts = np.load(os.path.join(src_dir, "span_counts.npy"), mmap_mode="r")
    src_sentence_path = os.path.join(src_dir, "treereg_sentence_ids.npy")
    if not os.path.exists(src_sentence_path):
        raise FileNotFoundError(f"missing exact sentence metadata: {src_sentence_path}")
    if src_ids.ndim != 2 or src_spans.ndim != 3 or src_spans.shape[2] != 3:
        raise ValueError("source arrays have unexpected shapes")
    if len(src_ids) != len(src_spans) or len(src_ids) != len(src_counts):
        raise ValueError("source row counts disagree")

    n_rows, width = src_ids.shape
    max_spans = src_spans.shape[1]
    estimated = (
        n_rows * width * np.dtype(np.uint16).itemsize
        + n_rows * max_spans * 3 * np.dtype(np.int16).itemsize
        + n_rows * np.dtype(np.int32).itemsize
        + n_rows * width * np.dtype(np.int16).itemsize
        + n_rows * 2 * np.dtype(np.int64).itemsize
    )
    disk = shutil.disk_usage(os.path.dirname(out_dir))
    reserve = int(args.min_free_disk_gb * 2**30)
    if disk.free - estimated < reserve:
        raise RuntimeError(
            f"conversion needs about {estimated/2**30:.1f} GiB and would leave "
            f"less than the {args.min_free_disk_gb:.1f} GiB reserve"
        )
    allocated_cpus = len(os.sched_getaffinity(0))
    # Keep at least 30% of the Slurm allocation available for the coordinator,
    # kernel writeback, and other node services. This also makes the resource
    # policy explicit instead of relying on os.cpu_count(), which may expose the
    # entire node rather than the task's cpuset.
    worker_cap = max(1, int(allocated_cpus * 0.70))
    workers = min(args.workers, 32, worker_cap)
    print(
        f"source={src_dir} rows={n_rows} width={width} max_spans={max_spans}; "
        f"workers={workers}/{allocated_cpus} allocated CPUs; "
        f"estimated={estimated/2**30:.1f} GiB; "
        f"disk_free={disk.free/2**30:.1f} GiB",
        flush=True,
    )

    incomplete = os.path.join(out_dir, "PREPROCESSING_INCOMPLETE")
    complete = os.path.join(out_dir, "PREPROCESSING_COMPLETE")
    Path(incomplete).write_text("Output is incomplete until PREPROCESSING_COMPLETE exists.\n")
    # Byte-for-byte copies are both faster and stronger than reconstructing
    # these arrays row by row: the Pushdown LM sees precisely the same primal
    # stream and padding layout as TreeReg/the base corpus.
    shutil.copyfile(
        os.path.join(src_dir, "input_ids.npy"),
        os.path.join(out_dir, "input_ids.npy"),
    )
    np.lib.format.open_memmap(
        os.path.join(out_dir, "spans.npy"), mode="w+", dtype=np.int16,
        shape=src_spans.shape,
    )
    np.lib.format.open_memmap(
        os.path.join(out_dir, "span_counts.npy"), mode="w+", dtype=np.int32,
        shape=src_counts.shape,
    )
    shutil.copyfile(
        src_sentence_path,
        os.path.join(out_dir, "pushdown_sentence_ids.npy"),
    )
    shutil.copyfile(
        os.path.join(src_dir, "chunk_index.npy"),
        os.path.join(out_dir, "chunk_index.npy"),
    )

    task_ranges = _ranges(n_rows, workers * 32)
    done = 0
    report_every = max(n_rows // 100, 1)
    next_report = report_every
    worker_args = [
        (src_dir, out_dir, args.pad_token_id, args.root_token_id, args.eos_token_id, rng)
        for rng in task_ranges
    ]
    if workers == 1:
        iterator = map(lambda x: _convert_range(*x), worker_args)
        for converted in iterator:
            done += converted
            if done >= next_report or done == n_rows:
                print(f"converted {done}/{n_rows}", flush=True)
                next_report += report_every
    else:
        with multiprocessing.Pool(workers) as pool:
            for converted in pool.imap_unordered(_convert_star, worker_args, chunksize=1):
                done += converted
                if done >= next_report or done == n_rows:
                    print(f"converted {done}/{n_rows}", flush=True)
                    next_report += report_every

    manifest = {
        "format_version": 3,
        "source_dir": src_dir,
        "direction": args.direction,
        "sentence_format": "terminals",
        "binarize": True,
        "collapse_unary": True,
        "extract_toplevel_trees": False,
        "wrap_toplevel_trees": False,
        "preserve_primal_boundaries": True,
        "preserve_native_eos": True,
        "synthesize_sentence_eos": False,
        "root_token_id": None,
        # This means no EOS is synthesized per parse tree. Native EOS tokens in
        # input_ids are retained from the primal corpus.
        "sentence_eos_token_id": None,
        "native_eos_token_id": args.eos_token_id,
        "drop_singleton_spans": True,
        "sentence_ids_file": "pushdown_sentence_ids.npy",
        "pad_token_id": args.pad_token_id,
        "input_dtype": "uint16",
        "span_dtype": "int16",
        "n_chunks": n_rows,
        "max_len": width,
        "max_spans": max_spans,
    }
    with open(os.path.join(out_dir, "preprocessing.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    Path(complete).write_text("ok\n")
    os.remove(incomplete)
    print(f"verified terminal-only output: {out_dir}", flush=True)
    lock.close()


if __name__ == "__main__":
    main()
