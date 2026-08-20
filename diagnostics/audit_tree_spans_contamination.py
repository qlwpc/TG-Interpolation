#!/usr/bin/env python
"""Compare precomputed TreeReg spans with spans from the fixed parser.

The audit streams raw tree chunks and compares the existing ``spans.npy`` row
with a fresh parse. It does not write a replacement span array. Besides raw
span differences, it classifies whether a changed split was used as a wrong
TreeReg gold decision or caused the intended decision to be dropped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from olmo.data.parse_align import TreeVocab, parse_chunk_slice  # noqa: E402


_TREE = None
_CHUNKS = None
_OLD_SPANS = None
_OLD_COUNTS = None
_VOCAB = None
_MAX_LEN = 2048


def _init_worker(tree_path: str, data_dir: str, tokenizer_path: str, max_len: int) -> None:
    global _TREE, _CHUNKS, _OLD_SPANS, _OLD_COUNTS, _VOCAB, _MAX_LEN
    _TREE = np.load(tree_path, mmap_mode="r")
    _CHUNKS = np.load(os.path.join(data_dir, "chunk_index.npy"), mmap_mode="r")
    _OLD_SPANS = np.load(os.path.join(data_dir, "spans.npy"), mmap_mode="r")
    _OLD_COUNTS = np.load(os.path.join(data_dir, "span_counts.npy"), mmap_mode="r")
    _VOCAB = TreeVocab.from_tokenizer_file(tokenizer_path)
    _MAX_LEN = int(max_len)


def _empty_stats() -> Dict[str, Any]:
    return {
        "chunks": 0,
        "affected_chunks": 0,
        "tree_reg_affected_chunks": 0,
        "total_spans": 0,
        "non_singleton_spans": 0,
        "changed_spans": 0,
        "changed_split_spans": 0,
        "changed_left_right_spans": 0,
        "count_mismatch_chunks": 0,
        "new_tree_reg_decisions": 0,
        "old_tree_reg_decisions": 0,
        "corrupted_intended_decisions": 0,
        "wrong_gold_decisions": 0,
        "dropped_intended_decisions": 0,
        "old_only_decisions": 0,
        "split_shift_sum": 0,
        "split_shift_abs_sum": 0,
        "split_shift_min": None,
        "split_shift_max": None,
        "split_shift_histogram": Counter(),
        "examples": [],
    }


def _tree_reg_eligible(
    spans: np.ndarray,
    sentence_ids: np.ndarray,
    word_boundaries: np.ndarray,
) -> np.ndarray:
    """Mirror the span-selection portion of ``compute_treereg_loss``."""
    if not len(spans):
        return np.zeros(0, dtype=np.bool_)
    n = len(sentence_ids)
    left = spans[:, 0].astype(np.int64, copy=False)
    split = spans[:, 1].astype(np.int64, copy=False)
    right = spans[:, 2].astype(np.int64, copy=False)
    eligible = (
        (left >= 0)
        & (left <= split)
        & (split < right)
        & (right < n)
    )
    for row in np.flatnonzero(eligible):
        l, s, r = int(left[row]), int(split[row]), int(right[row])
        sid = int(sentence_ids[l])
        if sid < 0 or int(sentence_ids[s]) != sid or int(sentence_ids[r]) != sid:
            eligible[row] = False
            continue
        candidates = int(np.count_nonzero(word_boundaries[l + 1 : r + 1]))
        if candidates < 2 or not bool(word_boundaries[s + 1]):
            eligible[row] = False
    return eligible


def _fixed_chunk(chunk_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start, length = map(int, _CHUNKS[chunk_index])
    out = parse_chunk_slice(
        np.asarray(_TREE[start : start + length]),
        _VOCAB,
        direction="right",
        binarize=True,
        collapse_unary=True,
        add_boundary_root=False,
    )
    spans = out["spans"]
    word_boundaries = out["word_boundaries"]
    sentence_ids = out["sentence_ids"]
    if len(out["input_ids"]) > _MAX_LEN:
        full_sentence_ids = sentence_ids
        spans = spans[spans[:, 2] < _MAX_LEN] if len(spans) else spans
        word_boundaries = word_boundaries[:_MAX_LEN]
        sentence_ids = sentence_ids[:_MAX_LEN]
        crossing_id = int(sentence_ids[-1]) if len(sentence_ids) else -1
        if crossing_id >= 0 and np.any(full_sentence_ids[_MAX_LEN:] == crossing_id):
            crossing = sentence_ids == crossing_id
            sentence_ids = sentence_ids.copy()
            word_boundaries = word_boundaries.copy()
            sentence_ids[crossing] = -1
            word_boundaries[crossing] = False
    return spans, sentence_ids, word_boundaries


def _audit_range(bounds: Tuple[int, int]) -> Dict[str, Any]:
    lo, hi = bounds
    stats = _empty_stats()
    stats["chunks"] = hi - lo
    for chunk_index in range(lo, hi):
        new, sentence_ids, word_boundaries = _fixed_chunk(chunk_index)
        old_count = int(_OLD_COUNTS[chunk_index])
        old = np.asarray(_OLD_SPANS[chunk_index, :old_count], dtype=np.int32)
        stats["total_spans"] += len(new)
        if len(new):
            stats["non_singleton_spans"] += int(np.count_nonzero(new[:, 0] < new[:, 2]))

        if len(old) != len(new):
            stats["count_mismatch_chunks"] += 1
            if len(stats["examples"]) < 20:
                stats["examples"].append({
                    "chunk": chunk_index,
                    "kind": "count_mismatch",
                    "old_count": len(old),
                    "new_count": len(new),
                })
            continue

        changed = np.any(old != new, axis=1) if len(new) else np.zeros(0, dtype=np.bool_)
        split_changed = (old[:, 1] != new[:, 1]) if len(new) else changed
        lr_changed = (
            np.any(old[:, (0, 2)] != new[:, (0, 2)], axis=1)
            if len(new) else changed
        )
        n_changed = int(np.count_nonzero(changed))
        stats["changed_spans"] += n_changed
        stats["changed_split_spans"] += int(np.count_nonzero(split_changed))
        stats["changed_left_right_spans"] += int(np.count_nonzero(lr_changed))
        if n_changed:
            stats["affected_chunks"] += 1

        old_eligible = _tree_reg_eligible(old, sentence_ids, word_boundaries)
        new_eligible = _tree_reg_eligible(new, sentence_ids, word_boundaries)
        stats["old_tree_reg_decisions"] += int(np.count_nonzero(old_eligible))
        stats["new_tree_reg_decisions"] += int(np.count_nonzero(new_eligible))
        corrupted = changed & new_eligible
        wrong = corrupted & old_eligible
        dropped = corrupted & ~old_eligible
        old_only = old_eligible & ~new_eligible
        stats["corrupted_intended_decisions"] += int(np.count_nonzero(corrupted))
        stats["wrong_gold_decisions"] += int(np.count_nonzero(wrong))
        stats["dropped_intended_decisions"] += int(np.count_nonzero(dropped))
        stats["old_only_decisions"] += int(np.count_nonzero(old_only))
        if np.any(corrupted):
            stats["tree_reg_affected_chunks"] += 1

        for row in np.flatnonzero(split_changed):
            shift = int(old[row, 1]) - int(new[row, 1])
            stats["split_shift_sum"] += shift
            stats["split_shift_abs_sum"] += abs(shift)
            stats["split_shift_min"] = shift if stats["split_shift_min"] is None else min(stats["split_shift_min"], shift)
            stats["split_shift_max"] = shift if stats["split_shift_max"] is None else max(stats["split_shift_max"], shift)
            stats["split_shift_histogram"][str(shift)] += 1
            if len(stats["examples"]) < 20:
                stats["examples"].append({
                    "chunk": chunk_index,
                    "span_row": int(row),
                    "old": old[row].tolist(),
                    "new": new[row].tolist(),
                    "old_tree_reg_eligible": bool(old_eligible[row]),
                    "new_tree_reg_eligible": bool(new_eligible[row]),
                })
    return stats


def _merge_stats(total: Dict[str, Any], part: Dict[str, Any]) -> None:
    scalar_keys = set(total) - {"split_shift_min", "split_shift_max", "split_shift_histogram", "examples"}
    for key in scalar_keys:
        total[key] += part[key]
    for key, fn in (("split_shift_min", min), ("split_shift_max", max)):
        value = part[key]
        if value is not None:
            total[key] = value if total[key] is None else fn(total[key], value)
    total["split_shift_histogram"].update(part["split_shift_histogram"])
    if len(total["examples"]) < 20:
        total["examples"].extend(part["examples"][: 20 - len(total["examples"])])


def _ranges(n: int, pieces: int) -> Iterable[Tuple[int, int]]:
    pieces = max(1, min(pieces, n))
    q, r = divmod(n, pieces)
    lo = 0
    for i in range(pieces):
        hi = lo + q + (i < r)
        if lo < hi:
            yield lo, hi
        lo = hi


def _stratified_block_ranges(
    population: int,
    sample_chunks: int,
    sample_blocks: int,
    seed: int,
) -> list[Tuple[int, int]]:
    """Select contiguous blocks distributed across the full chunk population."""
    sample_chunks = min(sample_chunks, population)
    sample_blocks = max(1, min(sample_blocks, sample_chunks))
    if sample_chunks == population:
        return list(_ranges(population, sample_blocks))
    rng = np.random.default_rng(seed)
    block_lengths = np.full(sample_blocks, sample_chunks // sample_blocks, dtype=np.int64)
    block_lengths[: sample_chunks % sample_blocks] += 1
    edges = np.linspace(0, population, sample_blocks + 1, dtype=np.int64)
    out = []
    for i, length in enumerate(block_lengths):
        lo, hi = int(edges[i]), int(edges[i + 1])
        latest = hi - int(length)
        start = lo if latest <= lo else int(rng.integers(lo, latest + 1))
        out.append((start, start + int(length)))
    return out


def _with_rates(stats: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(stats)
    out["split_shift_histogram"] = dict(sorted(stats["split_shift_histogram"].items(), key=lambda x: int(x[0])))
    denominators = {
        "affected_chunk_rate": (stats["affected_chunks"], stats["chunks"]),
        "tree_reg_affected_chunk_rate": (stats["tree_reg_affected_chunks"], stats["chunks"]),
        "changed_span_rate": (stats["changed_spans"], stats["total_spans"]),
        "changed_non_singleton_span_rate": (stats["changed_spans"], stats["non_singleton_spans"]),
        "corrupted_tree_reg_decision_rate": (stats["corrupted_intended_decisions"], stats["new_tree_reg_decisions"]),
        "wrong_gold_decision_rate": (stats["wrong_gold_decisions"], stats["new_tree_reg_decisions"]),
        "dropped_intended_decision_rate": (stats["dropped_intended_decisions"], stats["new_tree_reg_decisions"]),
    }
    out["rates"] = {
        key: (numerator / denominator if denominator else None)
        for key, (numerator, denominator) in denominators.items()
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument(
        "--sample-chunks",
        type=int,
        help="audit this many chunks in stratified contiguous blocks across the full split",
    )
    parser.add_argument("--sample-blocks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_chunks is not None and args.sample_chunks is not None:
        raise ValueError("--max-chunks and --sample-chunks are mutually exclusive")
    if args.sample_chunks is not None and args.sample_chunks < 1:
        raise ValueError("--sample-chunks must be positive")
    if args.sample_blocks < 1:
        raise ValueError("--sample-blocks must be positive")

    chunk_index = np.load(os.path.join(args.data_dir, "chunk_index.npy"), mmap_mode="r")
    population = len(chunk_index)
    if args.sample_chunks is not None:
        tasks = _stratified_block_ranges(
            population, args.sample_chunks, args.sample_blocks, args.seed
        )
        sampling = {
            "design": "stratified_contiguous_blocks",
            "population_chunks": population,
            "sample_chunks": sum(hi - lo for lo, hi in tasks),
            "sample_blocks": len(tasks),
            "seed": args.seed,
            "ranges": tasks,
        }
    else:
        n_prefix = population if args.max_chunks is None else min(population, args.max_chunks)
        tasks = list(_ranges(n_prefix, args.workers * 16))
        sampling = {
            "design": "full" if n_prefix == population else "prefix",
            "population_chunks": population,
            "sample_chunks": n_prefix,
            "sample_blocks": len(tasks),
            "seed": None,
        }
    n = sum(hi - lo for lo, hi in tasks)
    total = _empty_stats()
    started = time.time()
    initializer = (args.tree, args.data_dir, args.tokenizer, args.max_len)
    if args.workers == 1:
        _init_worker(*initializer)
        iterator = map(_audit_range, tasks)
        for part in iterator:
            _merge_stats(total, part)
    else:
        with mp.Pool(args.workers, initializer=_init_worker, initargs=initializer) as pool:
            done = 0
            for part in pool.imap_unordered(_audit_range, tasks, chunksize=1):
                _merge_stats(total, part)
                done += part["chunks"]
                if done == n or done % max(n // 20, 1) < part["chunks"]:
                    print(f"audited {done}/{n} chunks", flush=True)

    report = {
        "schema_version": 1,
        "claim_type": "computed",
        "tree": os.path.abspath(args.tree),
        "data_dir": os.path.abspath(args.data_dir),
        "tokenizer": os.path.abspath(args.tokenizer),
        "max_len": args.max_len,
        "requested_workers": args.workers,
        "sampling": sampling,
        "elapsed_seconds": time.time() - started,
        "statistics": _with_rates(total),
        "validation": {
            "span_counts_match": total["count_mismatch_chunks"] == 0,
            "left_right_unchanged": total["changed_left_right_spans"] == 0,
            "all_changes_are_split_only": total["changed_spans"] == total["changed_split_spans"] and total["changed_left_right_spans"] == 0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
