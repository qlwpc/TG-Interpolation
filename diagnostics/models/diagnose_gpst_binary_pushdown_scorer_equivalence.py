#!/usr/bin/env python
"""Compare every GPST-binary Pushdown scoring path on real model inputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from olmo.eval.gpst_binary_pushdown_document_ppl import (
    NativeGPSTBinaryPushdownCorpus,
    NativeNaryRightBinarizedPushdownCorpus,
    _build_prefix_cache,
    score_gpst_binary_pushdown_candidates,
)
from olmo.eval.pushdown_document_ppl import (
    _drop_leading_bos,
    score_pushdown_gold_candidates,
)
from olmo.model import OLMo


def _delta(reference: torch.Tensor, other: torch.Tensor) -> dict:
    difference = (reference.to(torch.float64) - other.to(torch.float64)).abs()
    return {
        "count": int(difference.numel()),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "allclose_atol_2e-4_rtol_2e-5": bool(
            torch.allclose(reference, other, atol=2e-4, rtol=2e-5)
        ),
    }


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
    parser.add_argument("--sentence", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument(
        "--candidate-source",
        choices=("gpst-strict-binary", "pushdown-nary-spliced-right-binary"),
        default="gpst-strict-binary",
    )
    parser.add_argument(
        "--attachment-normalization",
        choices=("stack_legal", "sentence_causal"),
        default="stack_legal",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.sentence <= 0:
        parser.error("--sentence must be positive so the check has a real prefix")

    device = torch.device(args.device)
    corpus_class = (
        NativeGPSTBinaryPushdownCorpus
        if args.candidate_source == "gpst-strict-binary"
        else NativeNaryRightBinarizedPushdownCorpus
    )
    corpus = corpus_class(
        args.native_data, args.tokenizer_path, max_sentences=args.sentence + 1
    )
    previous = corpus.sentence_candidates(args.sentence - 1)[0]
    original = corpus.sentence_candidates(args.sentence)[: args.candidates]
    candidates = tuple(
        _drop_leading_bos(candidate, corpus.vocab.bos) for candidate in original
    )
    prefix = (previous,)
    model = OLMo.from_checkpoint(args.checkpoint, device=device).eval()

    reference = score_pushdown_gold_candidates(
        model,
        prefix,
        candidates,
        device,
        eval_batch_size=len(candidates),
        include_attachment_probability=True,
        max_batch_tokens=1 << 30,
        attachment_normalization=args.attachment_normalization,
    )
    full, _ = score_gpst_binary_pushdown_candidates(
        model,
        prefix,
        candidates,
        device,
        attachment_normalization=args.attachment_normalization,
    )
    cache = _build_prefix_cache(model, prefix, device)
    cached, _ = score_gpst_binary_pushdown_candidates(
        model,
        prefix,
        candidates,
        device,
        prefix_cache=cache,
        attachment_normalization=args.attachment_normalization,
    )

    comparisons = {}
    finite = True
    equivalent = True
    for field in ("joint_nll", "token_nll", "attachment_nll"):
        expected = getattr(reference, field)
        values = {
            "reference": expected,
            "full": getattr(full, field),
            "cached": getattr(cached, field),
        }
        finite = finite and all(
            bool(torch.isfinite(value).all()) for value in values.values()
        )
        row = {
            "full_vs_reference": _delta(expected, values["full"]),
            "cached_vs_reference": _delta(expected, values["cached"]),
            "cached_vs_full": _delta(values["full"], values["cached"]),
        }
        equivalent = equivalent and all(
            item["allclose_atol_2e-4_rtol_2e-5"] for item in row.values()
        )
        comparisons[field] = row

    result = {
        "status": "complete" if finite and equivalent else "failed",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "native_data": str(Path(args.native_data).resolve()),
        "sentence": args.sentence,
        "candidate_count": len(candidates),
        "candidate_source": args.candidate_source,
        "structure_source": corpus_class.structure_source,
        "attachment_normalization": args.attachment_normalization,
        "finite": finite,
        "equivalent": equivalent,
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if (
        not finite
        or not equivalent
        or any(
            not math.isfinite(value)
            for field in ("joint_nll", "token_nll", "attachment_nll")
            for value in getattr(reference, field).tolist()
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
