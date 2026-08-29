#!/usr/bin/env python
"""Merge independent complete-document native doc-PPL shards exactly.

When ``--expected-documents`` is supplied, aggregation is fail-closed: every
document ID in ``[0, N)`` must occur exactly once and every row must satisfy the
same schema/mixture contract. ``--output`` writes the validated aggregate
atomically, so a partial or interrupted merge can never masquerade as final.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


COMMON_FIELDS = {
    "terminal_count", "sentence_count", "document_count", "candidate_slots",
    "model_candidate_forwards", "samples_per_sentence",
}
MODEL_FIELDS = {
    "gpst": {"log_likelihood"},
    "pushdown": {
        "legacy_log_likelihood", "uniform_mixture_log_likelihood",
        "token_only_log_likelihood",
    },
}
PUSHDOWN_PROTOCOL_FIELDS = {
    "protocol_version", "structure_source", "attachment_normalization",
    "prefix_policy", "max_sequence_length",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("gpst", "pushdown"))
    parser.add_argument("directory")
    parser.add_argument("--expected-documents", type=int)
    parser.add_argument("--expected-samples-per-sentence", type=int)
    parser.add_argument("--output", help="atomically write the validated aggregate here")
    return parser.parse_args()


def _load_rows(
    model: str,
    root: Path,
    expected_documents: int | None,
    expected_samples_per_sentence: int | None,
) -> list[dict]:
    document_dir = root / "documents"
    document_paths = sorted(document_dir.glob("document_*.json")) if document_dir.is_dir() else []
    paths = document_paths or sorted(root.glob("shard_*.json"))
    if not paths:
        raise SystemExit("no shard or atomic document result JSON files")

    rows = []
    required = COMMON_FIELDS | MODEL_FIELDS[model]
    if model == "pushdown":
        required |= PUSHDOWN_PROTOCOL_FIELDS
    schema = None
    protocol = None
    ids = []
    samples_per_sentence = None
    for path in paths:
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid result JSON {path}: {error}") from error
        missing_fields = sorted(required - row.keys())
        if missing_fields:
            raise SystemExit(f"{path} is missing required fields: {missing_fields}")
        current_schema = frozenset(row)
        if schema is None:
            schema = current_schema
        elif current_schema != schema:
            raise SystemExit(f"result schema mismatch in {path}")
        if model == "pushdown":
            current_protocol = {field: row[field] for field in PUSHDOWN_PROTOCOL_FIELDS}
            if int(current_protocol["protocol_version"]) != 2:
                raise SystemExit(f"unsupported Pushdown DocPPL protocol in {path}: {current_protocol}")
            if protocol is None:
                protocol = current_protocol
            elif current_protocol != protocol:
                raise SystemExit(f"Pushdown DocPPL protocol mismatch in {path}")
        if int(row["terminal_count"]) <= 0 or int(row["sentence_count"]) <= 0:
            raise SystemExit(f"non-positive terminal/sentence count in {path}")
        if int(row["document_count"]) <= 0:
            raise SystemExit(f"non-positive document_count in {path}")
        if int(row["candidate_slots"]) <= 0 or int(row["model_candidate_forwards"]) <= 0:
            raise SystemExit(f"non-positive candidate count in {path}")
        for field in MODEL_FIELDS[model]:
            if not math.isfinite(float(row[field])):
                raise SystemExit(f"non-finite {field} in {path}")
        row_samples = int(row["samples_per_sentence"])
        if samples_per_sentence is None:
            samples_per_sentence = row_samples
        elif row_samples != samples_per_sentence:
            raise SystemExit(f"samples_per_sentence mismatch in {path}")
        if (expected_samples_per_sentence is not None
                and row_samples != expected_samples_per_sentence):
            raise SystemExit(
                f"unexpected samples_per_sentence in {path}: "
                f"expected={expected_samples_per_sentence} actual={row_samples}"
            )
        if document_paths:
            if int(row["document_count"]) != 1:
                raise SystemExit(f"atomic result must contain exactly one document: {path}")
            document_id = int(row.get("document_id", -1))
            if path.stem != f"document_{document_id:05d}":
                raise SystemExit(f"document filename/ID mismatch: {path} vs {document_id}")
            ids.append(document_id)
        rows.append(row)

    if document_paths:
        if len(ids) != len(set(ids)):
            raise SystemExit("duplicate atomic document results")
        if expected_documents is not None:
            if expected_documents <= 0:
                raise SystemExit("--expected-documents must be positive")
            expected = set(range(expected_documents))
            actual = set(ids)
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            if missing or unexpected:
                preview = lambda values: values[:10] + (["..."] if len(values) > 10 else [])
                raise SystemExit(
                    f"incomplete document result set: expected={expected_documents} "
                    f"actual={len(actual)} missing={preview(missing)} "
                    f"unexpected={preview(unexpected)}"
                )
    elif expected_documents is not None:
        actual_documents = sum(int(row["document_count"]) for row in rows)
        if actual_documents != expected_documents:
            raise SystemExit(
                f"incomplete shard result set: expected={expected_documents} actual={actual_documents}"
            )
    return rows


def main() -> None:
    args = _parse_args()
    model = args.model
    root = Path(args.directory)
    rows = _load_rows(
        model, root, args.expected_documents, args.expected_samples_per_sentence
    )
    total = {key: sum(row[key] for row in rows) for key in (
        "terminal_count", "sentence_count", "document_count", "candidate_slots", "model_candidate_forwards"
    )}
    if model == "gpst":
        ll = sum(row["log_likelihood"] for row in rows)
        output = {**total, "log_likelihood": ll,
                  "perplexity": math.exp(-ll / total["terminal_count"]),
                  "samples_per_sentence": rows[0]["samples_per_sentence"],
                  "candidate_compression_ratio": total["candidate_slots"] / total["model_candidate_forwards"]}
    elif model == "pushdown":
        assert protocol is not None
        legacy = sum(row["legacy_log_likelihood"] for row in rows)
        uniform = sum(row["uniform_mixture_log_likelihood"] for row in rows)
        token = sum(row["token_only_log_likelihood"] for row in rows)
        output = {**total, **protocol, "legacy_log_likelihood": legacy,
                  "uniform_mixture_log_likelihood": uniform, "token_only_log_likelihood": token,
                  "legacy_perplexity": math.exp(-legacy / total["terminal_count"]),
                  "uniform_mixture_perplexity": math.exp(-uniform / total["terminal_count"]),
                  "token_only_perplexity": math.exp(-token / total["terminal_count"]),
                  "samples_per_sentence": rows[0]["samples_per_sentence"],
                  "candidate_compression_ratio": total["candidate_slots"] / total["model_candidate_forwards"]}
    if args.expected_documents is not None:
        output.update({
            "complete": True,
            "validated_document_count": args.expected_documents,
            "validated_document_id_min": 0,
            "validated_document_id_max": args.expected_documents - 1,
        })
    payload = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(payload)
        os.replace(temporary, destination)
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()
