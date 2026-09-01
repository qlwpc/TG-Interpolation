#!/usr/bin/env python
"""Merge disjoint GPST-binary Pushdown document-PPL shard JSON files."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


CONTRACT_KEYS = (
    "protocol_version",
    "structure_source",
    "source_candidate_axis",
    "binarization",
    "deduplicated_binary_structures",
    "prefix_policy",
    "context_truncation",
    "attachment_normalization",
    "candidate_aggregation",
    "divide_by_candidate_count",
    "ppl_denominator",
    "max_sequence_length",
    "max_candidates_per_sentence",
    "checkpoint_model_sha256",
    "native_manifest_sha256",
    "tokenizer_sha256",
)

FULL_BBC_COMMON_INVARIANTS = {
    "start_document": 0,
    "end_document": 4966,
    "document_count": 4966,
    "sentence_count": 148836,
    "terminal_count": 3284061,
    "candidate_slots": 44650800,
    "max_candidates_per_sentence": 300,
}


def validate_full_bbc(
    result: dict,
    *,
    expected_valid_candidate_count: int = 37227054,
    expected_structure_source: str = "v2_gpst_strict_binary_to_pushdown",
) -> None:
    expected = {
        **FULL_BBC_COMMON_INVARIANTS,
        "structure_source": expected_structure_source,
        "valid_candidate_count": expected_valid_candidate_count,
        "model_candidate_forwards": expected_valid_candidate_count,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise ValueError(f"full BBC corpus invariants failed: {mismatches}")


def merge(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("at least one result is required")
    if any(row.get("max_sentences") is not None for row in rows):
        raise ValueError("cannot merge max_sentences smoke/partial-document results")
    reference = rows[0]
    normalization = reference.get("attachment_normalization")
    if normalization not in ("stack_legal", "sentence_causal"):
        raise ValueError(
            f"unsupported attachment normalization in result: {normalization!r}"
        )
    suffix = "v1" if normalization == "stack_legal" else "v2"
    joint_ll_key = f"joint_log_likelihood_{suffix}"
    joint_ppl_key = f"joint_document_perplexity_{suffix}"
    for row in rows[1:]:
        mismatches = [
            key for key in CONTRACT_KEYS if row.get(key) != reference.get(key)
        ]
        if mismatches:
            raise ValueError(f"incompatible result contracts: {mismatches}")
    intervals = sorted(
        (int(row["start_document"]), int(row["end_document"]), row) for row in rows
    )
    for start, end, row in intervals:
        if start >= end:
            raise ValueError(f"empty or reversed document shard [{start},{end})")
        if int(row["document_count"]) != end - start:
            raise ValueError(
                f"shard [{start},{end}) reports document_count={row['document_count']}"
            )
        if int(row["model_candidate_forwards"]) != int(row["valid_candidate_count"]):
            raise ValueError(
                "model_candidate_forwards must equal valid_candidate_count"
            )
        expected_slots = int(row["sentence_count"]) * int(
            row["max_candidates_per_sentence"]
        )
        if int(row["candidate_slots"]) != expected_slots:
            raise ValueError(
                "candidate_slots must equal sentence_count times physical capacity"
            )
        for field in (
            joint_ll_key,
            "candidate0_terminal_log_likelihood",
        ):
            if not math.isfinite(float(row[field])):
                raise ValueError(
                    f"shard [{start},{end}) has non-finite {field}: {row[field]}"
                )
    for (left_start, left_end, _), (right_start, _right_end, _) in zip(
        intervals, intervals[1:]
    ):
        if right_start < left_end:
            raise ValueError(
                f"overlapping document shards [{left_start},{left_end}) and "
                f"[{right_start},...)"
            )
        if right_start != left_end:
            raise ValueError(
                f"non-contiguous document shards end at {left_end} and restart "
                f"at {right_start}"
            )

    terminal_count = sum(int(row["terminal_count"]) for row in rows)
    joint_ll = sum(float(row[joint_ll_key]) for row in rows)
    token_ll = sum(float(row["candidate0_terminal_log_likelihood"]) for row in rows)
    result = {key: reference.get(key) for key in CONTRACT_KEYS}
    for key in ("checkpoint", "native_data", "tokenizer_path"):
        result[key] = reference.get(key)
    for key in (
        "sentence_count",
        "document_count",
        "valid_candidate_count",
        "candidate_slots",
        "model_candidate_forwards",
        "kv_cache_hits",
        "kv_cache_rebuilds",
        "oom_retries",
        "nonfinite_retries",
    ):
        result[key] = sum(int(row.get(key, 0)) for row in rows)
    result.update(
        {
            joint_ll_key: joint_ll,
            "candidate0_terminal_log_likelihood": token_ll,
            "terminal_count": terminal_count,
            joint_ppl_key: math.exp(-joint_ll / terminal_count),
            "candidate0_structured_terminal_perplexity": math.exp(
                -token_ll / terminal_count
            ),
            "start_document": intervals[0][0],
            "end_document": intervals[-1][1],
            "merged_shards": len(rows),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-full-bbc",
        action="store_true",
        help="require the protocol's complete 4,966-document BBC corpus counts",
    )
    parser.add_argument(
        "--expected-valid-candidate-count",
        type=int,
        default=37227054,
        help="exact full-corpus unique candidate count for the selected axis",
    )
    parser.add_argument(
        "--expected-structure-source",
        default="v2_gpst_strict_binary_to_pushdown",
    )
    parser.add_argument(
        "--expected-attachment-normalization",
        choices=("stack_legal", "sentence_causal"),
    )
    args = parser.parse_args()
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    result = merge(rows)
    if (
        args.expected_attachment_normalization is not None
        and result.get("attachment_normalization")
        != args.expected_attachment_normalization
    ):
        raise ValueError(
            "attachment normalization mismatch: expected "
            f"{args.expected_attachment_normalization}, got "
            f"{result.get('attachment_normalization')}"
        )
    if args.require_full_bbc:
        validate_full_bbc(
            result,
            expected_valid_candidate_count=args.expected_valid_candidate_count,
            expected_structure_source=args.expected_structure_source,
        )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
