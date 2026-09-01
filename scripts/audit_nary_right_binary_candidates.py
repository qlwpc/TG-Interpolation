#!/usr/bin/env python
"""Audit the BPE-spliced right-binary Pushdown n-ary candidate support."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from olmo.eval.gpst_binary_pushdown_document_ppl import (  # noqa: E402
    gpst_merge_orders_to_pushdown_spans,
    right_binarize_native_nary_spans,
)
from olmo.eval.native_model_topk_corpus import NativeModelTopKCorpus  # noqa: E402


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank == 1:
        return "rank_1"
    if rank <= 10:
        return "rank_2_10"
    if rank <= 50:
        return "rank_11_50"
    if rank <= 100:
        return "rank_51_100"
    return "rank_101_300"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-data",
        default="dataset/bbc-news/testppl/native_model_topk_300_v2",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    corpus = NativeModelTopKCorpus(args.native_data)
    word_offsets = tuple(
        np.load(shard.path / "word_offsets.npy", mmap_mode="r")
        for shard in corpus.shards
    )
    sentence_limit = (
        len(corpus)
        if args.max_sentences is None
        else min(len(corpus), args.max_sentences)
    )
    source_count = unique_count = collision_sentences = 0
    gpst_count = overlap_count = zero_overlap_sentences = candidate0_matches = 0
    sum_log_source_k = sum_log_unique_k = 0.0
    min_unique = 300
    max_unique = 0
    right0_rank_in_direct = {}
    direct0_rank_in_right = {}
    common_same_rank = common_rank_abs_delta = 0
    top_overlap = {10: 0, 50: 0, 100: 0, 300: 0}
    bpe_groups = {
        "all_words_single_bpe": {},
        "has_multi_bpe_word": {},
    }
    for index in range(sentence_limit):
        shard_id = int(np.searchsorted(corpus.ends, index, side="right"))
        shard_start = 0 if shard_id == 0 else int(corpus.ends[shard_id - 1])
        local_index = index - shard_start
        word_count = int(
            word_offsets[shard_id][local_index + 1]
            - word_offsets[shard_id][local_index]
        )
        row = corpus.sentence(index)
        binary, _sources = right_binarize_native_nary_spans(
            row.pushdown_spans,
            row.pushdown_span_counts,
            int(row.content_bounds[0]),
            int(row.content_bounds[1]),
            deduplicate=True,
        )
        source_k = int(row.pushdown_valid_count)
        unique_k = int(binary.shape[0])
        gpst = gpst_merge_orders_to_pushdown_spans(
            row.gpst_merge_orders,
            int(row.content_bounds[0]),
            int(row.content_bounds[1] - row.content_bounds[0]),
            validate=False,
        )
        # Canonical split order makes topology equality independent of the
        # merge/post-order serialization used by the two source axes.
        if gpst.shape[1]:
            order = np.argsort(gpst[:, :, 1], axis=1)
            gpst = np.take_along_axis(gpst, order[:, :, None], axis=1)
        binary_keys = [
            np.ascontiguousarray(candidate).tobytes() for candidate in binary
        ]
        gpst_keys = [np.ascontiguousarray(candidate).tobytes() for candidate in gpst]
        binary_rank = {key: rank for rank, key in enumerate(binary_keys, start=1)}
        gpst_rank = {key: rank for rank, key in enumerate(gpst_keys, start=1)}
        common = binary_rank.keys() & gpst_rank.keys()
        overlap = len(common)
        content_length = int(row.content_bounds[1] - row.content_bounds[0])
        group_name = (
            "all_words_single_bpe"
            if content_length == word_count
            else "has_multi_bpe_word"
        )
        group = bpe_groups[group_name]
        for key, value in (
            ("sentence_count", 1),
            ("source_nary_candidate_count", source_k),
            ("unique_rightbinary_candidate_count", unique_k),
            ("direct_candidate_count", int(gpst.shape[0])),
            ("overlap_candidate_count", overlap),
            ("zero_overlap_sentence_count", int(overlap == 0)),
            ("candidate0_match_sentence_count", int(binary_keys[0] == gpst_keys[0])),
        ):
            group[key] = group.get(key, 0) + value
        right0_other_rank = gpst_rank.get(binary_keys[0])
        direct0_other_rank = binary_rank.get(gpst_keys[0])
        right0_bucket = _rank_bucket(right0_other_rank)
        direct0_bucket = _rank_bucket(direct0_other_rank)
        right0_rank_in_direct[right0_bucket] = (
            right0_rank_in_direct.get(right0_bucket, 0) + 1
        )
        direct0_rank_in_right[direct0_bucket] = (
            direct0_rank_in_right.get(direct0_bucket, 0) + 1
        )
        common_same_rank += sum(
            int(binary_rank[key] == gpst_rank[key]) for key in common
        )
        common_rank_abs_delta += sum(
            abs(binary_rank[key] - gpst_rank[key]) for key in common
        )
        for cutoff in top_overlap:
            top_overlap[cutoff] += len(
                set(binary_keys[:cutoff]) & set(gpst_keys[:cutoff])
            )
        if not 0 < unique_k <= source_k <= 300:
            raise RuntimeError(
                f"invalid candidate counts at sentence {index}: "
                f"source={source_k}, unique={unique_k}"
            )
        source_count += source_k
        unique_count += unique_k
        gpst_count += int(gpst.shape[0])
        overlap_count += overlap
        zero_overlap_sentences += int(overlap == 0)
        candidate0_matches += int(np.array_equal(binary[0], gpst[0]))
        collision_sentences += int(unique_k != source_k)
        sum_log_source_k += math.log(source_k)
        sum_log_unique_k += math.log(unique_k)
        min_unique = min(min_unique, unique_k)
        max_unique = max(max_unique, unique_k)
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"audited {index + 1}/{sentence_limit}", flush=True)

    result = {
        "status": "complete",
        "structure_source": "v2_pushdown_nary_topk_spliced_bpe_right_binarized",
        "binarization": "deterministic_right_cnf_after_bpe_splicing",
        "deduplicated_binary_structures": True,
        "sentence_count": sentence_limit,
        "source_nary_candidate_count": source_count,
        "unique_binary_candidate_count": unique_count,
        "collision_candidate_count": source_count - unique_count,
        "collision_sentence_count": collision_sentences,
        "direct_gpst_candidate_count": gpst_count,
        "cross_axis_overlap_candidate_count": overlap_count,
        "zero_cross_axis_overlap_sentence_count": zero_overlap_sentences,
        "candidate0_topology_match_sentence_count": candidate0_matches,
        "rightbinary_candidate0_rank_in_direct": right0_rank_in_direct,
        "direct_candidate0_rank_in_rightbinary": direct0_rank_in_right,
        "common_topology_same_rank_count": common_same_rank,
        "common_topology_mean_absolute_rank_delta": (
            common_rank_abs_delta / overlap_count if overlap_count else None
        ),
        "cross_axis_same_cutoff_overlap_count": {
            str(cutoff): count for cutoff, count in top_overlap.items()
        },
        "bpe_group_breakdown": bpe_groups,
        "min_unique_candidates": min_unique,
        "max_unique_candidates": max_unique,
        "sum_log_source_k": sum_log_source_k,
        "sum_log_unique_k": sum_log_unique_k,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
