#!/usr/bin/env python
"""Numerically compare candidate-0 Pushdown KV scoring with full-prefix scoring."""
from __future__ import annotations

import argparse
import json

import torch

from olmo.eval.pushdown_document_ppl import (
    NativePushdownTopKCorpus,
    _drop_leading_bos,
    score_pushdown_native_candidates,
)
from olmo.model import OLMo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-data", default="dataset/bbc-news/native_model_topk_300_v2")
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda")
    corpus = NativePushdownTopKCorpus(args.native_data, args.tokenizer_path, max_sentences=2)
    if len(corpus) < 2:
        raise SystemExit("need two sentences in one document")
    model = OLMo.from_checkpoint(args.checkpoint, device=device).eval()
    first = corpus.sentence_candidates(0)[:args.candidates]
    second = tuple(_drop_leading_bos(row, corpus.vocab.bos)
                   for row in corpus.sentence_candidates(1)[:args.candidates])
    _, cache = score_pushdown_native_candidates(model, (), first, device, True, None, True)
    cached = score_pushdown_native_candidates(model, (first[0],), second, device, True, cache, False)
    full = score_pushdown_native_candidates(model, (first[0],), second, device, True)
    report = {}
    for field in ("joint_nll", "token_nll", "attachment_nll"):
        delta = (getattr(cached, field) - getattr(full, field)).abs()
        report[field] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean())}
    # Cached Pushdown uses the SDPA rectangular-attention fallback while the
    # full path uses FlexAttention; both are fp32 reductions with a different
    # accumulation order.  3e-5 is below one fp16 ULP at this scale.
    report["allclose_3e-5"] = all(
        value["max_abs"] <= 3e-5 for value in report.values() if isinstance(value, dict)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["allclose_3e-5"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
