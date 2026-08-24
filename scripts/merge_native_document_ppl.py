#!/usr/bin/env python
"""Merge independent complete-document native doc-PPL shards exactly."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> None:
    model, directory = sys.argv[1:3]
    rows = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("shard_*.json"))]
    if not rows:
        raise SystemExit("no shard result JSON files")
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
        legacy = sum(row["legacy_log_likelihood"] for row in rows)
        uniform = sum(row["uniform_mixture_log_likelihood"] for row in rows)
        token = sum(row["token_only_log_likelihood"] for row in rows)
        output = {**total, "legacy_log_likelihood": legacy,
                  "uniform_mixture_log_likelihood": uniform, "token_only_log_likelihood": token,
                  "legacy_perplexity": math.exp(-legacy / total["terminal_count"]),
                  "uniform_mixture_perplexity": math.exp(-uniform / total["terminal_count"]),
                  "token_only_perplexity": math.exp(-token / total["terminal_count"]),
                  "samples_per_sentence": rows[0]["samples_per_sentence"],
                  "candidate_compression_ratio": total["candidate_slots"] / total["model_candidate_forwards"]}
    else:
        raise SystemExit("model must be gpst or pushdown")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
