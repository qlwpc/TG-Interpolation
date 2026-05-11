"""
Compute which memmap files will be accessed during the decay stage.

Simulates the IterableDataset shuffle to determine the first N indices,
then maps them back to the source .npy files.

Usage:
    python scripts/compute_decay_data_files.py \
        --data_dir /path/to/dataset/fineweb-edu-v2/tree \
        --glob_pattern "*-00*.npy" \
        --max_duration 30000 \
        --global_batch_size 832 \
        --seed 6198 \
        --world_size 8 \
        --chunk_size 2048 \
        --memmap_dtype uint32 \
        --drop_last
"""

import argparse
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True, help="Directory containing .npy memmap files")
    p.add_argument("--glob_pattern", default="*.npy", help="Glob pattern for .npy files")
    p.add_argument("--max_duration", type=int, default=30000, help="Decay steps")
    p.add_argument("--global_batch_size", type=int, default=832)
    p.add_argument("--seed", type=int, default=6198)
    p.add_argument("--world_size", type=int, default=8)
    p.add_argument("--chunk_size", type=int, default=2048, help="Tokens per example")
    p.add_argument("--memmap_dtype", default="uint32")
    p.add_argument("--drop_last", action="store_true", default=True)
    p.add_argument("--epoch", type=int, default=0, help="Epoch for shuffle seed (0 for decay)")
    return p.parse_args()


def get_file_example_count(path, chunk_size, dtype):
    """Number of chunk_size examples in a memmap file."""
    item_size = np.dtype(dtype).itemsize
    file_bytes = os.path.getsize(path)
    return file_bytes // (item_size * chunk_size)


def main():
    args = parse_args()

    # 1. Discover files and their example counts
    pattern = os.path.join(args.data_dir, args.glob_pattern)
    all_files = sorted(glob.glob(pattern))
    print(f"Found {len(all_files)} files matching pattern")

    dtype = np.dtype(args.memmap_dtype)
    file_examples = {}
    total_examples = 0
    for fpath in all_files:
        n = get_file_example_count(fpath, args.chunk_size, dtype)
        file_examples[fpath] = n
        total_examples += n

    print(f"Total examples across all files: {total_examples:,}")
    print(f"Total tokens: {total_examples * args.chunk_size:,}")

    # 2. Compute total_size (matches IterableDataset logic)
    if args.drop_last and total_examples % args.world_size != 0:
        num_samples = math.ceil((total_examples - args.world_size) / args.world_size)
    else:
        num_samples = math.ceil(total_examples / args.world_size)
    total_size = num_samples * args.world_size
    print(f"total_size (after drop_last): {total_size:,}")
    print(f"Dropped examples: {total_examples - total_size:,}")

    # 3. Number of examples needed for decay
    examples_needed = args.max_duration * args.global_batch_size
    examples_needed = min(examples_needed, total_size)
    print(f"Examples needed for decay: {examples_needed:,}")
    print(f"Fraction of total: {examples_needed / total_examples * 100:.1f}%")

    # 4. Simulate the deterministic shuffle
    rng = np.random.Generator(np.random.PCG64(seed=args.seed + args.epoch))
    indices = np.arange(total_examples, dtype=np.uint32)
    rng.shuffle(indices)
    indices = indices[:total_size]  # drop_last truncation
    needed_indices = indices[:examples_needed]

    # 5. Map indices to files using cumulative offsets
    # Build offset table matching MemMapDataset.offsets
    offsets = []
    start = 0
    for fpath in all_files:
        n = file_examples[fpath]
        end = start + n
        offsets.append((start, end, fpath))
        start = end

    file_usage = defaultdict(int)
    for idx in needed_indices:
        for start_off, end_off, fpath in offsets:
            if start_off <= idx < end_off:
                file_usage[fpath] += 1
                break

    used_files = sorted(file_usage.keys())
    unused_files = sorted(set(all_files) - set(used_files))

    # 6. Report
    print(f"\n{'='*80}")
    print(f"FILES NEEDED for decay: {len(used_files)}")
    print(f"FILES NOT NEEDED (can delete): {len(unused_files)}")
    print(f"{'='*80}")

    # Group by common patterns for readability
    total_used_tokens = sum(file_examples[f] for f in used_files) * args.chunk_size
    total_unused_tokens = sum(file_examples[f] for f in unused_files) * args.chunk_size
    print(f"\nStorage to keep:    {total_used_tokens / 1e9:.2f}B tokens ({total_used_tokens * 4 / 1e9:.2f} GB)")
    print(f"Storage to free:     {total_unused_tokens / 1e9:.2f}B tokens ({total_unused_tokens * 4 / 1e9:.2f} GB)")

    print(f"\n--- Files to KEEP ({len(used_files)}) ---")
    for f in used_files:
        examples = file_examples[f]
        used = file_usage[f]
        pct = used / examples * 100 if examples > 0 else 0
        print(f"  {f}  ({used:,}/{examples:,} examples used, {pct:.1f}%)")

    if unused_files:
        print(f"\n--- Files that can be DELETED ({len(unused_files)}) ---")
        # Group by number ranges for readability
        for f in unused_files:
            print(f"  {f}")

    # 7. Save the file lists for scripting
    keep_path = os.path.join(args.data_dir, "decay_files_to_keep.txt")
    delete_path = os.path.join(args.data_dir, "decay_files_to_delete.txt")
    with open(keep_path, "w") as f:
        for fpath in used_files:
            f.write(f"{fpath}\n")
    with open(delete_path, "w") as f:
        for fpath in unused_files:
            f.write(f"{fpath}\n")
    print(f"\nFile lists saved to: {keep_path} and {delete_path}")


if __name__ == "__main__":
    main()
