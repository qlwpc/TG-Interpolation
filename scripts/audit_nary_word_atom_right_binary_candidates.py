#!/usr/bin/env python
"""Audit native n-ary candidates in checkpoint-training word-atom right-CNF."""

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
    right_binarize_native_nary_spans_with_word_atoms,
)
from olmo.eval.native_model_topk_corpus import NativeModelTopKCorpus  # noqa: E402


def _keys(rows: np.ndarray) -> list[bytes]:
    if rows.shape[1]:
        order = np.argsort(rows[:, :, 1], axis=1)
        rows = np.take_along_axis(rows, order[:, :, None], axis=1)
    return [np.ascontiguousarray(row).tobytes() for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-data",
        default="dataset/bbc-news/testppl/native_model_topk_300_v2",
    )
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = NativeModelTopKCorpus(args.native_data)
    limit = (
        len(corpus)
        if args.max_sentences is None
        else min(len(corpus), args.max_sentences)
    )
    totals = {
        "sentence_count": 0,
        "source_nary_candidate_count": 0,
        "unique_word_atom_binary_candidate_count": 0,
        "binary_collision_count": 0,
        "direct_gpst_candidate_count": 0,
        "cross_axis_overlap_candidate_count": 0,
        "candidate0_match_sentence_count": 0,
        "zero_overlap_sentence_count": 0,
    }
    groups = {"all_words_single_bpe": {}, "has_multi_bpe_word": {}}
    sum_log_source_k = 0.0
    sum_log_unique_k = 0.0
    common_rank_abs_delta = 0
    common_same_rank = 0
    for index in range(limit):
        row = corpus.sentence(index)
        left, right = map(int, row.content_bounds)
        binary, _sources = right_binarize_native_nary_spans_with_word_atoms(
            row.pushdown_spans,
            row.pushdown_span_counts,
            row.word_starts,
            left,
            right,
            deduplicate=True,
        )
        direct = gpst_merge_orders_to_pushdown_spans(
            row.gpst_merge_orders,
            left,
            right - left,
            validate=False,
        )
        binary_keys = _keys(binary)
        direct_keys = _keys(direct)
        binary_rank = {key: rank for rank, key in enumerate(binary_keys, start=1)}
        direct_rank = {key: rank for rank, key in enumerate(direct_keys, start=1)}
        common = binary_rank.keys() & direct_rank.keys()
        overlap = len(common)
        source_k = int(row.pushdown_valid_count)
        unique_k = len(binary_keys)
        direct_k = int(row.gpst_valid_count)
        values = {
            "sentence_count": 1,
            "source_nary_candidate_count": source_k,
            "unique_word_atom_binary_candidate_count": unique_k,
            "binary_collision_count": source_k - unique_k,
            "direct_gpst_candidate_count": direct_k,
            "cross_axis_overlap_candidate_count": overlap,
            "candidate0_match_sentence_count": int(
                binary_keys[0] == direct_keys[0]
            ),
            "zero_overlap_sentence_count": int(overlap == 0),
        }
        group_name = (
            "all_words_single_bpe"
            if right - left == len(row.word_starts)
            else "has_multi_bpe_word"
        )
        for key, value in values.items():
            totals[key] += value
            groups[group_name][key] = groups[group_name].get(key, 0) + value
        common_same_rank += sum(
            int(binary_rank[key] == direct_rank[key]) for key in common
        )
        common_rank_abs_delta += sum(
            abs(binary_rank[key] - direct_rank[key]) for key in common
        )
        sum_log_source_k += math.log(source_k)
        sum_log_unique_k += math.log(unique_k)
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"audited {index + 1}/{limit}", flush=True)

    overlap_count = totals["cross_axis_overlap_candidate_count"]
    result = {
        "status": "complete",
        "structure_source": "v2_pushdown_nary_topk_word_atom_right_binarized",
        "binarization": "deterministic_right_cnf_with_fixed_word_bpe_atoms",
        "checkpoint_training_evidence": {
            "precomputed_dataset": (
                "dataset/bbc-news/parse_aligned/train_pushdown_unary_terminals"
            ),
            "source_converter": "scripts/convert_treereg_to_pushdown_terminals.py",
            "preterminal_policy": "retained; singleton spans dropped only",
            "multi_bpe_word_policy": "fixed right-recursive preterminal subtree",
        },
        "deduplicated_binary_structures": True,
        **totals,
        "common_topology_same_rank_count": common_same_rank,
        "common_topology_mean_absolute_rank_delta": (
            common_rank_abs_delta / overlap_count if overlap_count else None
        ),
        "sum_log_source_k": sum_log_source_k,
        "sum_log_unique_k": sum_log_unique_k,
        "bpe_group_breakdown": groups,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
