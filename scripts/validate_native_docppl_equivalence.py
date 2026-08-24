#!/usr/bin/env python
"""Compare native fast-path NLLs against the retained reference scorers."""
from __future__ import annotations

import argparse
import json

import torch

from olmo.eval.pushdown_document_ppl import (
    NativePushdownTopKCorpus,
    _drop_leading_bos,
    score_pushdown_gold_candidates,
    score_pushdown_native_candidates,
)
from olmo.gpst.eval.document_ppl import (
    NativeGPSTTopKCorpus,
    _as_collator_item,
    _count_actions,
    _count_tokens,
    _score_items,
    _score_native_candidates,
)
from olmo.gpst.model.model_factory import create_model
from olmo.gpst.reader.dataset_gold import GoldTreeCollator
from olmo.model import OLMo


def _difference(reference: torch.Tensor, optimized: torch.Tensor) -> dict:
    delta = (reference - optimized).abs()
    return {"count": int(delta.numel()), "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "allclose_1e-5": bool(torch.allclose(reference, optimized, atol=1e-5, rtol=1e-5))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-data", required=True)
    parser.add_argument("--gpst-checkpoint", required=True)
    parser.add_argument("--pushdown-checkpoint", required=True)
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--sentence", type=int, default=1, help="use one preceding sentence as document prefix")
    parser.add_argument("--r2d2-config", default="olmo/gpst/data/en_config/r2d2_256_4_1.json")
    parser.add_argument("--gpt-config", default="olmo/gpst/data/gpt2-bbc/config.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    report = {}

    gpst_corpus = NativeGPSTTopKCorpus(args.native_data, args.tokenizer_path)
    gpst_model = create_model("r2d2-gen-fast", args.r2d2_config, args.gpt_config, backbone="olmo")
    gpst_model.from_pretrain(args.gpst_checkpoint); gpst_model.to(device).eval()
    gpst_prefix = gpst_corpus.sentence_candidates(args.sentence - 1)[0]
    gpst_candidates = gpst_corpus.sentence_candidates(args.sentence)[:args.candidates]
    full = [gpst_prefix + candidate for candidate in gpst_candidates]
    pt, pa = _count_tokens(gpst_prefix), _count_actions(gpst_prefix)
    ct, ca = _count_tokens(gpst_candidates[0]), _count_actions(gpst_candidates[0])
    reference = _score_items(gpst_model, [_as_collator_item(candidate) for candidate in full],
                             GoldTreeCollator(), device, pt, ct, pa, ca)
    optimized = _score_native_candidates(gpst_model, full, device, pt, ct, pa, ca)
    report["gpst_joint_nll"] = _difference(reference, optimized)
    del gpst_model
    torch.cuda.empty_cache()

    push_corpus = NativePushdownTopKCorpus(args.native_data, args.tokenizer_path)
    push_model = OLMo.from_checkpoint(args.pushdown_checkpoint, device=device).eval()
    push_prefix = (push_corpus.sentence_candidates(args.sentence - 1)[0],)
    candidates = tuple(_drop_leading_bos(candidate, push_corpus.vocab.bos)
                       for candidate in push_corpus.sentence_candidates(args.sentence)[:args.candidates])
    reference = score_pushdown_gold_candidates(push_model, push_prefix, candidates, device,
                                                args.candidates, True)
    optimized = score_pushdown_native_candidates(push_model, push_prefix, candidates, device, True)
    report["pushdown_joint_nll"] = _difference(reference.joint_nll, optimized.joint_nll)
    report["pushdown_token_nll"] = _difference(reference.token_nll, optimized.token_nll)
    report["pushdown_attachment_nll"] = _difference(reference.attachment_nll, optimized.attachment_nll)
    if not all(value["allclose_1e-5"] for value in report.values()):
        raise SystemExit(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
