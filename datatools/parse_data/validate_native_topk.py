#!/usr/bin/env python
"""Smoke-test native-binary top-K on a real Benepar score chart."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Required by the locally installed Benepar/T5 protobuf combination.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import benepar

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datatools.parse_data.native_topk import decode_labeled_scores_topk  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentence",
        default="The quick brown fox jumps over the lazy dog .",
        help="pre-tokenized whitespace-separated parser words",
    )
    parser.add_argument("--model", default="benepar_en3_large")
    parser.add_argument("--k", type=int, default=300)
    args = parser.parse_args()

    words = args.sentence.split()
    load_start = time.perf_counter()
    beneparser = benepar.Parser(args.model)
    load_seconds = time.perf_counter() - load_start

    sentence = benepar.InputSentence(words=words)
    encoded = [beneparser._with_missing_fields_filled(sentence)]
    score_start = time.perf_counter()
    scores = list(beneparser._parser.parse(encoded, return_scores=True))[0]
    score_seconds = time.perf_counter() - score_start

    result = {
        "sentence": args.sentence,
        "num_words": len(words),
        "score_shape": list(scores.shape),
        "model_load_seconds": load_seconds,
        "benepar_score_seconds": score_seconds,
        "modes": {},
    }
    decoded = {}
    for mode in ("logsumexp", "max"):
        start = time.perf_counter()
        candidates = decode_labeled_scores_topk(
            scores,
            k=args.k,
            mode=mode,
            backend="lazy",
        )
        decoded[mode] = candidates
        result["modes"][mode] = {
            "decode_seconds": time.perf_counter() - start,
            "count": len(candidates),
            "unique": len({candidate.spans for candidate in candidates}),
            "internal_spans_per_tree": sorted(
                {len(candidate.spans) for candidate in candidates}
            ),
            "merge_orders_valid": all(
                sorted(candidate.merge_orders) == list(range(len(words) - 1))
                for candidate in candidates
            ),
            "scores_nonincreasing": all(
                candidates[i].score >= candidates[i + 1].score
                for i in range(len(candidates) - 1)
            ),
            "top1_spans": candidates[0].spans,
        }
    result["mode_overlap"] = len(
        {candidate.spans for candidate in decoded["logsumexp"]}
        & {candidate.spans for candidate in decoded["max"]}
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
