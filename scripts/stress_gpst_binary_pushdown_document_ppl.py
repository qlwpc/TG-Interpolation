#!/usr/bin/env python
"""Repeat one GPST document in one process to expose transient non-finite scores."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from olmo.eval.gpst_binary_pushdown_document_ppl import (  # noqa: E402
    NativeGPSTBinaryPushdownCorpus,
    evaluate_gpst_binary_pushdown_document_ppl,
)
from olmo.model import OLMo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-data", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--document", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--disable-pushdown-flex", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = OLMo.from_checkpoint(Path(args.checkpoint).resolve(), device="cuda")
    if args.disable_pushdown_flex:
        model.config.pushdown_use_flex = False

    reference = None
    max_absolute_delta = {
        "joint_log_likelihood_v1": 0.0,
        "candidate0_terminal_log_likelihood": 0.0,
    }
    last_result = None
    for repeat in range(args.repeats):
        corpus = NativeGPSTBinaryPushdownCorpus(
            args.native_data,
            args.tokenizer_path,
            start_document=args.document,
            end_document=args.document + 1,
            validate_merge_orders=False,
        )
        last_result = evaluate_gpst_binary_pushdown_document_ppl(
            model,
            corpus,
            "cuda",
            eval_batch_size=args.eval_batch_size,
            use_kv_cache=not args.disable_kv_cache,
            prefetch_sentences=0,
        )
        current = {
            "joint_log_likelihood_v1": last_result.joint_log_likelihood_v1,
            "candidate0_terminal_log_likelihood": (
                last_result.candidate0_terminal_log_likelihood
            ),
        }
        if reference is None:
            reference = current
        for field, value in current.items():
            max_absolute_delta[field] = max(
                max_absolute_delta[field], abs(value - reference[field])
            )
        print(
            json.dumps(
                {"repeat": repeat, **current}, sort_keys=True, allow_nan=False
            ),
            flush=True,
        )

    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "native_data": str(Path(args.native_data).resolve()),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "document": args.document,
        "repeats_completed": args.repeats,
        "eval_batch_size": args.eval_batch_size,
        "kv_cache_enabled": not args.disable_kv_cache,
        "pushdown_flex_enabled": not args.disable_pushdown_flex,
        "reference": reference,
        "max_absolute_delta": max_absolute_delta,
        "last_result": last_result.as_dict() if last_result is not None else None,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
