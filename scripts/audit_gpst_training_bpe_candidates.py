#!/usr/bin/env python
"""Audit the superseded strict-binary CKY BPE-splicing diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from olmo.eval.gpst_binary_pushdown_document_ppl import (  # noqa: E402
    gpst_merge_orders_to_pushdown_spans,
    gpst_merge_orders_to_spliced_bpe_spans,
    right_binarize_native_nary_spans,
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
        "source_direct_candidate_count": 0,
        "training_bpe_candidate_count": 0,
        "training_bpe_collision_count": 0,
        "nary_rightbinary_candidate_count": 0,
        "training_bpe_nary_overlap_count": 0,
        "training_bpe_direct_overlap_count": 0,
        "candidate0_match_nary_sentence_count": 0,
        "candidate0_match_direct_sentence_count": 0,
        "zero_nary_overlap_sentence_count": 0,
    }
    groups = {"all_words_single_bpe": {}, "has_multi_bpe_word": {}}
    for index in range(limit):
        row = corpus.sentence(index)
        left, right = map(int, row.content_bounds)
        training, _sources = gpst_merge_orders_to_spliced_bpe_spans(
            row.gpst_merge_orders,
            row.word_starts,
            left,
            right,
            deduplicate=True,
        )
        nary, _ = right_binarize_native_nary_spans(
            row.pushdown_spans,
            row.pushdown_span_counts,
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
        training_keys = _keys(training)
        nary_keys = _keys(nary)
        direct_keys = _keys(direct)
        training_set = set(training_keys)
        nary_set = set(nary_keys)
        direct_set = set(direct_keys)
        nary_overlap = len(training_set & nary_set)
        direct_overlap = len(training_set & direct_set)
        source_k = int(row.gpst_valid_count)
        training_k = len(training_keys)
        values = {
            "sentence_count": 1,
            "source_direct_candidate_count": source_k,
            "training_bpe_candidate_count": training_k,
            "training_bpe_collision_count": source_k - training_k,
            "nary_rightbinary_candidate_count": len(nary_keys),
            "training_bpe_nary_overlap_count": nary_overlap,
            "training_bpe_direct_overlap_count": direct_overlap,
            "candidate0_match_nary_sentence_count": int(
                training_keys[0] == nary_keys[0]
            ),
            "candidate0_match_direct_sentence_count": int(
                training_keys[0] == direct_keys[0]
            ),
            "zero_nary_overlap_sentence_count": int(nary_overlap == 0),
        }
        group_name = (
            "all_words_single_bpe"
            if right - left == len(row.word_starts)
            else "has_multi_bpe_word"
        )
        for key, value in values.items():
            totals[key] += value
            groups[group_name][key] = groups[group_name].get(key, 0) + value
        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(f"audited {index + 1}/{limit}", flush=True)

    result = {
        "status": "complete",
        "structure_source": ("v2_gpst_strict_binary_cky_spliced_bpe_right_binarized"),
        "binarization": ("gpst_word_topology_spliced_bpe_deterministic_right_cnf"),
        "deduplicated_binary_structures": True,
        **totals,
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
