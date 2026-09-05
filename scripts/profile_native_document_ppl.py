#!/usr/bin/env python
"""Profile a bounded candidate-0 document context using the native scorers."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import torch
from olmo.eval import pushdown_document_ppl as pushdown
from olmo.gpst.eval import document_ppl as gpst
from olmo.gpst.model.model_factory import create_model
from olmo.model import OLMo


def prepare_batch(model_name, corpus, sentence, batch_size, attention_limit, context_limit):
    """Build history within the selected document, matching evaluator semantics."""
    if min(batch_size, attention_limit, context_limit) <= 0 or not 0 <= sentence < len(corpus):
        raise ValueError("invalid sentence or non-positive batch/context limit")
    ids = corpus.native.document_ids[corpus.start_sentence:corpus.start_sentence + len(corpus)]
    start = sentence
    while start > 0 and ids[start - 1] == ids[sentence]:
        start -= 1
    prefix = ()
    for index in range(start, sentence + 1):
        candidates = corpus.sentence_candidates(index)
        if model_name == "pushdown":
            if index > start:
                candidates = tuple(pushdown._drop_leading_bos(c, corpus.vocab.bos) for c in candidates)
            context = pushdown._trim_prefix(prefix, candidates[0], context_limit)
            prefix = pushdown._retain_prefix_for_any_future_sentence(prefix + (candidates[0],), context_limit)
        else:
            context = gpst._trim_prefix(prefix, candidates[0], context_limit, context_limit)
            prefix = gpst._retain_prefix_for_any_future_sentence(prefix + tuple(candidates[0]), context_limit, context_limit)
    length = (sum(len(c.tokens) for c in context) + len(candidates[0].tokens)
              if model_name == "pushdown" else gpst._count_actions(context + candidates[0]))
    limit = min(batch_size, max(1, attention_limit // max(length * length, 1)))
    return context, candidates[:limit], length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=("gpst", "pushdown"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-data", required=True)
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sentence", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-batch-attention-elements", type=int, default=16777216)
    parser.add_argument("--attachment-normalization", choices=("stack_legal", "sentence_causal"), default="stack_legal")
    parser.add_argument("--r2d2-config", default="olmo/gpst/data/en_config/r2d2_256_4_1.json")
    parser.add_argument("--gpt-config", default="olmo/gpst/data/gpt2-bbc/config.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    if args.model == "gpst":
        corpus = gpst.NativeGPSTTopKCorpus(args.native_data, args.tokenizer_path)
        model = create_model("r2d2-gen-fast", args.r2d2_config, args.gpt_config, backbone="olmo")
        model.from_pretrain(args.checkpoint)
        model.to(device).eval()
    else:
        corpus = pushdown.NativePushdownTopKCorpus(args.native_data, args.tokenizer_path)
        model = OLMo.from_checkpoint(args.checkpoint, device=device).eval()
    start = time.perf_counter()
    context, candidates, length = prepare_batch(args.model, corpus, args.sentence, args.batch_size,
                                                args.max_batch_attention_elements, args.max_sequence_length)
    preparation = time.perf_counter() - start
    if args.model == "gpst":
        run = lambda: gpst._score_native_candidates(
            model, [context + c for c in candidates], device,
            gpst._count_tokens(context), gpst._count_tokens(candidates[0]),
            gpst._count_actions(context), gpst._count_actions(candidates[0]),
        )
    else:
        run = lambda: pushdown.score_pushdown_native_candidates(
            model, context, candidates, device, attachment_normalization=args.attachment_normalization)
    run()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        run()
        torch.cuda.synchronize()
    print(f"candidate_count={len(candidates)} preparation_seconds={preparation:.6f} context_length={length}")
    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
