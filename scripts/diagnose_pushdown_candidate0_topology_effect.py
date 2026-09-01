#!/usr/bin/env python
"""Measure candidate-0 topology effects without document-history propagation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from olmo.eval.gpst_binary_pushdown_document_ppl import (  # noqa: E402
    NativeGPSTBinaryPushdownCorpus,
    NativeNaryRightBinarizedPushdownCorpus,
    score_gpst_binary_pushdown_candidates,
)
from olmo.model import OLMo  # noqa: E402


def _signature(candidate) -> tuple:
    return tuple(sorted((int(left), int(split), int(right)) for left, split, right in candidate.spans))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--native-data",
        default="dataset/bbc-news/testppl/native_model_topk_300_v2",
    )
    parser.add_argument(
        "--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json"
    )
    parser.add_argument("--max-sentences", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    direct = NativeGPSTBinaryPushdownCorpus(
        args.native_data, args.tokenizer_path, max_sentences=args.max_sentences
    )
    right = NativeNaryRightBinarizedPushdownCorpus(
        args.native_data, args.tokenizer_path, max_sentences=args.max_sentences
    )
    model = OLMo.from_checkpoint(args.checkpoint, device=args.device).eval()
    groups = {
        "matching_topology": {"sentences": 0, "terminals": 0},
        "different_topology": {"sentences": 0, "terminals": 0},
    }
    rows = []
    for index in range(args.max_sentences):
        direct_candidate = direct.sentence_candidates(index)[0]
        right_candidate = right.sentence_candidates(index)[0]
        matching = _signature(direct_candidate) == _signature(right_candidate)
        direct_scores, _ = score_gpst_binary_pushdown_candidates(
            model, (), (direct_candidate,), args.device
        )
        right_scores, _ = score_gpst_binary_pushdown_candidates(
            model, (), (right_candidate,), args.device
        )
        terminal_count = len(direct_candidate.tokens) - 1
        group = groups["matching_topology" if matching else "different_topology"]
        group["sentences"] += 1
        group["terminals"] += terminal_count
        row = {
            "sentence": index,
            "matching_topology": matching,
            "terminal_count": terminal_count,
        }
        for field in ("joint_nll", "token_nll", "attachment_nll"):
            direct_value = float(getattr(direct_scores, field)[0])
            right_value = float(getattr(right_scores, field)[0])
            if not math.isfinite(direct_value) or not math.isfinite(right_value):
                raise FloatingPointError(f"non-finite {field} at sentence {index}")
            delta = direct_value - right_value
            group[field + "_direct_minus_right"] = (
                group.get(field + "_direct_minus_right", 0.0) + delta
            )
            row[field + "_direct_minus_right"] = delta
        rows.append(row)
        if (index + 1) % 20 == 0:
            print(f"scored {index + 1}/{args.max_sentences}", flush=True)

    result = {
        "status": "complete",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "max_sentences": args.max_sentences,
        "context": "sentence_local_bos_only",
        "comparison": "direct_gpst_binary_minus_nary_right_binary_candidate0_nll",
        "groups": groups,
        "per_sentence": rows,
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
