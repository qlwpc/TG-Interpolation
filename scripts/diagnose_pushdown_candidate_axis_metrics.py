#!/usr/bin/env python
"""Decompose Pushdown likelihood across three candidate-support choices."""

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
    _build_prefix_cache,
    score_gpst_binary_pushdown_candidates,
)
from olmo.eval.pushdown_document_ppl import (  # noqa: E402
    NativePushdownTopKCorpus,
    PushdownCandidateScores,
    _drop_leading_bos,
    _trim_prefix,
)
from olmo.model import OLMo  # noqa: E402


def evaluate_axis(model, corpus, device: str, batch_limit: int) -> dict:
    prefix = ()
    prefix_cache = None
    previous_doc = None
    totals = {
        "joint_truncated_sum_log_likelihood": 0.0,
        "token_truncated_sum_log_likelihood": 0.0,
        "candidate0_joint_log_likelihood": 0.0,
        "candidate0_token_log_likelihood": 0.0,
        "candidate0_attachment_log_likelihood": 0.0,
    }
    terminals = sentences = documents = candidates_total = 0
    sum_log_k = 0.0
    for index, (doc_id, original) in enumerate(corpus):
        first = doc_id != previous_doc
        if first:
            prefix = ()
            prefix_cache = None
            previous_doc = doc_id
            documents += 1
        candidates = original if first else tuple(
            _drop_leading_bos(candidate, corpus.vocab.bos) for candidate in original
        )
        current = candidates[0]
        context = _trim_prefix(prefix, current, 2048)
        active_cache = None
        if context:
            if prefix_cache is not None and prefix_cache.context == context:
                active_cache = prefix_cache
            else:
                active_cache = _build_prefix_cache(model, context, device)
        total_length = sum(len(sentence.tokens) for sentence in context) + len(current.tokens)
        model_length = len(current.tokens) if active_cache is not None else total_length
        attention_cells = model_length * total_length
        batch_size = min(
            batch_limit,
            len(candidates),
            max(1, 65536 // max(model_length, 1)),
            max(1, 16777216 // max(attention_cells, 1)),
        )
        parts = []
        next_cache = None
        for start in range(0, len(candidates), batch_size):
            part, candidate0_cache = score_gpst_binary_pushdown_candidates(
                model,
                context,
                candidates[start : start + batch_size],
                device,
                active_cache,
                return_candidate0_cache=(start == 0),
            )
            for field in ("joint_nll", "token_nll", "attachment_nll"):
                if not bool(torch.isfinite(getattr(part, field)).all()):
                    raise FloatingPointError(
                        f"non-finite {field} on axis={corpus.structure_source} "
                        f"sentence={index} candidates={start}:{start + len(part.joint_nll)}"
                    )
            parts.append(part)
            if start == 0:
                next_cache = candidate0_cache
        scores = PushdownCandidateScores(
            *(torch.cat([getattr(part, field) for part in parts])
              for field in ("joint_nll", "token_nll", "attachment_nll"))
        )
        totals["joint_truncated_sum_log_likelihood"] += float(
            torch.logsumexp(-scores.joint_nll.to(torch.float64), dim=0).item()
        )
        totals["token_truncated_sum_log_likelihood"] += float(
            torch.logsumexp(-scores.token_nll.to(torch.float64), dim=0).item()
        )
        totals["candidate0_joint_log_likelihood"] -= float(scores.joint_nll[0])
        totals["candidate0_token_log_likelihood"] -= float(scores.token_nll[0])
        totals["candidate0_attachment_log_likelihood"] -= float(
            scores.attachment_nll[0]
        )
        sentence_terminals = len(current.tokens) - int(first)
        terminals += sentence_terminals
        sentences += 1
        candidates_total += len(candidates)
        sum_log_k += math.log(len(candidates))
        prefix = context + (current,)
        prefix_cache = next_cache
        if (index + 1) % 20 == 0:
            print(
                f"{corpus.structure_source}: {index + 1}/{len(corpus)}",
                flush=True,
            )

    result = {
        "structure_source": corpus.structure_source,
        "sentence_count": sentences,
        "document_count": documents,
        "terminal_count": terminals,
        "valid_candidate_count": candidates_total,
        "sum_log_k": sum_log_k,
        **totals,
    }
    for key, value in totals.items():
        result[key.replace("log_likelihood", "perplexity")] = math.exp(
            -value / terminals
        )
    result["joint_uniform_average_log_likelihood"] = (
        totals["joint_truncated_sum_log_likelihood"] - sum_log_k
    )
    result["joint_uniform_average_perplexity"] = math.exp(
        -result["joint_uniform_average_log_likelihood"] / terminals
    )
    return result


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
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = OLMo.from_checkpoint(args.checkpoint, device=args.device).eval()
    axes = (
        NativePushdownTopKCorpus(
            args.native_data, args.tokenizer_path, max_sentences=args.max_sentences
        ),
        NativeNaryRightBinarizedPushdownCorpus(
            args.native_data, args.tokenizer_path, max_sentences=args.max_sentences
        ),
        NativeGPSTBinaryPushdownCorpus(
            args.native_data, args.tokenizer_path, max_sentences=args.max_sentences
        ),
    )
    # The legacy corpus predates explicit structure-source metadata.
    axes[0].structure_source = "v2_pushdown_native_nary_topk"
    result = {
        "status": "complete",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "native_data": str(Path(args.native_data).resolve()),
        "max_sentences": args.max_sentences,
        "attachment_normalization": "stack_legal",
        "candidate_aggregation": "truncated_sum_no_divide_by_k",
        "axes": [
            evaluate_axis(model, corpus, args.device, args.eval_batch_size)
            for corpus in axes
        ],
    }
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, args.output)
    print(payload, end="")


if __name__ == "__main__":
    main()
