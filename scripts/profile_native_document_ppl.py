#!/usr/bin/env python
"""Profile one real native document-PPL candidate batch on CUDA.

This deliberately excludes checkpoint loading and mmap construction from the
CUDA region.  It reports wall-clock preparation time separately, so a profile
can distinguish data-path overhead from model execution before a full run.
"""

from __future__ import annotations

import argparse
import time

import torch

from olmo.eval.pushdown_document_ppl import NativePushdownTopKCorpus, score_pushdown_gold_candidates
from olmo.gpst.eval.document_ppl import NativeGPSTTopKCorpus, _as_collator_item, _score_items
from olmo.gpst.reader.dataset_gold import GoldTreeCollator
from olmo.gpst.model.model_factory import create_model
from olmo.model import OLMo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("gpst", "pushdown"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-data", required=True)
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sentence", type=int, default=0)
    parser.add_argument("--r2d2-config", default="olmo/gpst/data/en_config/r2d2_256_4_1.json")
    parser.add_argument("--gpt-config", default="olmo/gpst/data/gpt2-bbc/config.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.model == "gpst":
        corpus = NativeGPSTTopKCorpus(args.native_data, args.tokenizer_path)
        model = create_model("r2d2-gen-fast", args.r2d2_config, args.gpt_config, backbone="olmo")
        model.from_pretrain(args.checkpoint); model.to(device).eval()
        candidates = corpus.sentence_candidates(args.sentence)[:args.batch_size]
        start = time.perf_counter()
        items = [_as_collator_item(candidate) for candidate in candidates]
        preparation = time.perf_counter() - start
        run = lambda: _score_items(model, items, GoldTreeCollator(), device, 0,
                                   len(candidates[0][0].tokens), 0,
                                   candidates[0][0].action_count)
    else:
        corpus = NativePushdownTopKCorpus(args.native_data, args.tokenizer_path)
        model = OLMo.from_checkpoint(args.checkpoint, device=device).eval()
        candidates = corpus.sentence_candidates(args.sentence)[:args.batch_size]
        start = time.perf_counter()
        preparation = time.perf_counter() - start
        run = lambda: score_pushdown_gold_candidates(model, (), candidates, device,
                                                      args.batch_size, True)
    run()  # warm up compiler and allocations outside the measurement.
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run()
        torch.cuda.synchronize()
    print(f"candidate_count={len(candidates)} preparation_seconds={preparation:.6f}")
    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
