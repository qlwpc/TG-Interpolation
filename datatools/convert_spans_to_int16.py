#!/usr/bin/env python
"""Convert ``spans.npy`` (int32) -> ``spans_int16.npy`` (int16), lossless.

Constituent spans are terminal indices in ``[0, max_sequence_length)`` (=2048 here,
max valid value 2047), plus ``-1`` for padding. Both fit in int16 (range
``[-32768, 32767]``), so the cast is lossless and halves the file size — the train
split's spans drop from ~149 GB to ~75 GB. This directly cuts the cgroup page-cache
pressure that caused the ``num_workers>0`` OOM under slurm (the dataset mmaps spans
by random chunk index; halving the file halves the resident pages).

``PrecomputedParseDataset`` prefers ``spans_int16.npy`` when present and upcasts to
int64 at tensor-build time, so no other code changes are needed.

Streams block-by-block (does NOT load the whole file): reads ``block`` rows of the
int32 mmap, casts to int16, writes into a preallocated int16 output memmap. Safe to
interrupt — re-running overwrites the partial output.

Usage:
    python -m datatools.convert_spans_to_int16 <parse_aligned_dir> [<dir> ...] [--block 65536]
    python -m datatools.convert_spans_to_int16 dataset/bbc-news/parse_aligned/train_right
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def convert_one(data_dir: str, block: int = 65536) -> None:
    src = os.path.join(data_dir, "spans.npy")
    dst = os.path.join(data_dir, "spans_int16.npy")
    if not os.path.exists(src):
        print(f"  SKIP {data_dir}: spans.npy not found", flush=True)
        return
    a = np.load(src, mmap_mode="r")
    n, m, c = a.shape
    print(
        f"  {data_dir}: spans.npy {a.shape} {a.dtype} ({a.nbytes / 1e9:.1f} GB) "
        f"-> int16 ({a.nbytes / 2 / 1e9:.1f} GB)",
        flush=True,
    )
    # Sanity: verify int16-safety on a sample (the loader relies on this).
    sample = a[: min(n, 65536)]
    sample_valid = sample[sample >= 0]
    if len(sample_valid) and int(sample_valid.max()) > 32767:
        raise ValueError(
            f"{data_dir}: max span value {int(sample_valid.max())} > 32767, "
            "int16 is NOT safe — aborting (re-run without int16)."
        )
    # Preallocate the int16 output memmap with the same shape.
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.int16, shape=(n, m, c))
    try:
        for i in range(0, n, block):
            j = min(i + block, n)
            out[i:j] = a[i:j].astype(np.int16, copy=False)
            if (i // block) % 32 == 0:
                print(f"    {i:>10}/{n} ({100 * i / n:5.1f}%)", flush=True)
        out.flush()
    finally:
        del out
    print(f"  DONE {dst}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", help="parse_aligned dirs containing spans.npy")
    p.add_argument("--block", type=int, default=65536, help="row block size for streaming")
    args = p.parse_args(argv)
    for d in args.dirs:
        d = str(Path(d).expanduser())
        if not os.path.isdir(d):
            print(f"SKIP {d}: not a directory", file=sys.stderr)
            continue
        convert_one(d, block=args.block)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
