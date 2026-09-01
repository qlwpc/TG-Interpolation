#!/usr/bin/env python
"""Small sentence-local Pushdown PPL comparison over a 2x2 protocol matrix.

Axes:
  1. parse support: model beam vs supplied teacher-forced parse trees;
  2. attachment probability: v1 stack-legal vs v2 sentence-causal.

All four cells use one trained checkpoint, sum joint ``p(x,y)`` mass without a
``1/K`` factor, and divide the final negative log likelihood by the same number
of sentence-content terminal tokens. See ``docs/PLAN_pushdown_ppl_2x2.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from olmo.attachment import (  # noqa: E402
    ATTACHMENT_NORMALIZATION_V1,
    ATTACHMENT_NORMALIZATION_V2,
    canonical_attachment_normalization,
    derive_gold_attachment_actions,
)
from olmo.eval.pushdown_document_ppl import (  # noqa: E402
    PushdownGold300Corpus,
    PushdownGoldCandidate,
    _compress_candidates,
    score_pushdown_gold_candidates,
)
from olmo.model import OLMo  # noqa: E402


PROTOCOLS = (ATTACHMENT_NORMALIZATION_V1, ATTACHMENT_NORMALIZATION_V2)


def _sentence_local_candidate(
    candidate: PushdownGoldCandidate, bos_token_id: int
) -> PushdownGoldCandidate:
    """Keep parsed content, prepend an LM-only BOS, and rebuild stack actions."""
    content = [index for index, sid in enumerate(candidate.sentence_ids) if sid >= 0]
    if not content:
        raise ValueError("supplied candidate has no parsed sentence content")
    left, right = content[0], content[-1] + 1
    if content != list(range(left, right)):
        raise ValueError("parsed sentence content must be one contiguous token interval")

    tokens = (int(bos_token_id),) + tuple(candidate.tokens[left:right])
    spans = tuple(
        (span_left - left + 1, split - left + 1, span_right - left + 1)
        for span_left, split, span_right in candidate.spans
        if left <= span_left <= span_right < right
    )
    sentence_ids = (-1,) + (0,) * (right - left)
    span_tensor = torch.tensor(
        spans or [(-1, -1, -1)], dtype=torch.long
    ).unsqueeze(0)
    sid_tensor = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0)
    targets, legal = derive_gold_attachment_actions(span_tensor, sid_tensor)
    return PushdownGoldCandidate(
        tokens=tokens,
        spans=spans,
        sentence_ids=sentence_ids,
        attachment_targets=tuple(map(int, targets[0].tolist())),
        legal_attachment_targets=tuple(tuple(row) for row in legal[0]),
    )


def _score_teacher_forced(
    model: OLMo,
    candidates: Sequence[PushdownGoldCandidate],
    device: str,
    eval_batch_size: int,
    max_batch_tokens: int,
    normalization: str,
):
    return score_pushdown_gold_candidates(
        model=model,
        prefix=(),
        candidates=candidates,
        device=device,
        eval_batch_size=eval_batch_size,
        include_attachment_probability=True,
        max_batch_tokens=max_batch_tokens,
        attachment_normalization=normalization,
    )


def evaluate_matrix(
    model: OLMo,
    corpus: PushdownGold300Corpus,
    device: str,
    beam_size: int = 300,
    max_reduce: int | None = None,
    eval_batch_size: int = 32,
    max_batch_tokens: int = 65536,
    training_attachment_normalization: str = ATTACHMENT_NORMALIZATION_V2,
    include_per_sentence: bool = False,
    progress_every: int = 1,
) -> dict:
    """Evaluate the four cells and return a JSON-serializable result."""
    if beam_size <= 0 or eval_batch_size <= 0 or max_batch_tokens <= 0:
        raise ValueError("beam size, eval batch size, and max batch tokens must be positive")
    training_attachment_normalization = canonical_attachment_normalization(
        training_attachment_normalization
    )
    model.eval()
    total_ll: Dict[str, Dict[str, float]] = {
        "beam_search": {protocol: 0.0 for protocol in PROTOCOLS},
        "teacher_forced": {protocol: 0.0 for protocol in PROTOCOLS},
    }
    terminal_count = 0
    supplied_slots = 0
    supplied_unique = 0
    sentence_rows = []
    log = logging.getLogger("pushdown-ppl-2x2")

    for sentence_index in range(len(corpus)):
        original = corpus.sentence_candidates(sentence_index)
        localized = tuple(
            _sentence_local_candidate(candidate, corpus.vocab.bos)
            for candidate in original
        )
        reference_tokens = localized[0].tokens
        if any(candidate.tokens != reference_tokens for candidate in localized[1:]):
            raise ValueError(
                f"sentence {sentence_index} supplied candidates have different terminals"
            )
        unique, _multiplicities = _compress_candidates(localized)
        # Label/unary collisions are one latent Pushdown structure. Counting a
        # duplicate twice would create probability mass merely from serialization.
        supplied_slots += len(localized)
        supplied_unique += len(unique)
        content_ids = torch.tensor(reference_tokens[1:], dtype=torch.long, device=device)
        if content_ids.numel() == 0:
            raise ValueError(f"sentence {sentence_index} has no content terminals")
        terminal_count += int(content_ids.numel())

        teacher_scores = {}
        sentence_result = {
            "sentence_index": sentence_index,
            "terminal_count": int(content_ids.numel()),
            "supplied_slots": len(localized),
            "supplied_unique_structures": len(unique),
            "log_likelihoods": {"beam_search": {}, "teacher_forced": {}},
        }
        for protocol in PROTOCOLS:
            scores = _score_teacher_forced(
                model,
                unique,
                device,
                eval_batch_size,
                max_batch_tokens,
                protocol,
            )
            teacher_scores[protocol] = scores
            sentence_ll = float(torch.logsumexp(-scores.joint_nll, dim=0).item())
            if not math.isfinite(sentence_ll):
                raise RuntimeError(
                    f"non-finite teacher-forced likelihood for sentence {sentence_index}, {protocol}"
                )
            total_ll["teacher_forced"][protocol] += sentence_ll
            sentence_result["log_likelihoods"]["teacher_forced"][protocol] = sentence_ll

        # On identical fixed trees, v1 conditions away probability mass and
        # therefore cannot have a larger attachment/joint NLL than v2.
        if bool(
            (
                teacher_scores[ATTACHMENT_NORMALIZATION_V1].joint_nll
                > teacher_scores[ATTACHMENT_NORMALIZATION_V2].joint_nll + 1e-5
            ).any()
        ):
            raise AssertionError(
                "teacher-forced v1 joint NLL exceeded v2 on a fixed candidate"
            )
        if not torch.allclose(
            teacher_scores[ATTACHMENT_NORMALIZATION_V1].token_nll,
            teacher_scores[ATTACHMENT_NORMALIZATION_V2].token_nll,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise AssertionError("v1/v2 changed teacher-forced token NLL")

        for protocol in PROTOCOLS:
            surprisal = model.pushdown_beam_search(
                eval_input_ids=content_ids,
                beam_size=beam_size,
                max_reduce=max_reduce,
                bos_id=corpus.vocab.bos,
                tag=None,
                use_attachment_head=True,
                attachment_normalization=protocol,
                sentence_local_stack=True,
                require_complete_parse=True,
            )
            sentence_ll = -float(surprisal)
            if not math.isfinite(sentence_ll):
                raise RuntimeError(
                    f"non-finite beam likelihood for sentence {sentence_index}, {protocol}"
                )
            total_ll["beam_search"][protocol] += sentence_ll
            sentence_result["log_likelihoods"]["beam_search"][protocol] = sentence_ll

        if include_per_sentence:
            sentence_rows.append(sentence_result)
        done = sentence_index + 1
        if progress_every > 0 and (done == len(corpus) or done % progress_every == 0):
            log.info("scored %d/%d sentence-local rows", done, len(corpus))

    if terminal_count <= 0:
        raise RuntimeError("the comparison contains no terminal tokens")
    cells = {}
    for source, protocol_values in total_ll.items():
        cells[source] = {}
        for protocol, log_likelihood in protocol_values.items():
            nll = -log_likelihood
            cells[source][protocol] = {
                "log_likelihood": log_likelihood,
                "negative_log_likelihood": nll,
                "perplexity": math.exp(nll / terminal_count),
            }

    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_tier": "auxiliary/dev",
        "experiment": "pushdown_sentence_ppl_2x2",
        "training_attachment_normalization": training_attachment_normalization,
        "axes": {
            "parse_support": ["beam_search", "teacher_forced"],
            "attachment_normalization": list(PROTOCOLS),
        },
        "contract": {
            "context": "sentence_local_bos_lm_context_empty_structural_stack",
            "candidate_aggregation": "unique_truncated_joint_sum_no_divide_by_k",
            "ppl_denominator": "sentence_content_terminal_count",
            "teacher_forced_support": "supplied_tree_300_unique_pushdown_structures",
            "beam_support": "complete_incremental_topk",
        },
        "counts": {
            "sentence_count": len(corpus),
            "terminal_count": terminal_count,
            "supplied_candidate_slots": supplied_slots,
            "supplied_unique_structures": supplied_unique,
            "requested_beam_size": beam_size,
        },
        "cells": cells,
    }
    if include_per_sentence:
        result["per_sentence"] = sentence_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tree-path", default="dataset/bbc-news/testppl_tree/tree_300.npy"
    )
    parser.add_argument(
        "--sentence-index-path",
        default="dataset/bbc-news/testppl_tree/tree_sent_index.npy",
    )
    parser.add_argument(
        "--document-index-path",
        default="dataset/bbc-news/testppl_tree/tree_doc_index.npy",
    )
    parser.add_argument(
        "--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json"
    )
    parser.add_argument("--max-sentences", type=int, default=2)
    parser.add_argument("--beam-size", type=int, default=300)
    parser.add_argument("--max-reduce", type=int)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument(
        "--training-attachment-normalization",
        choices=PROTOCOLS,
        default=ATTACHMENT_NORMALIZATION_V2,
        help="metadata for the already-trained checkpoint; this does not change scoring",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--include-per-sentence", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_sentences <= 0:
        parser.error("--max-sentences must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    corpus = PushdownGold300Corpus(
        args.tree_path,
        args.sentence_index_path,
        args.document_index_path,
        args.tokenizer_path,
        samples_per_sentence=300,
        max_sentences=args.max_sentences,
    )
    model = OLMo.from_checkpoint(args.checkpoint, device=args.device)
    if model.config.transformer_grammar_type != "pushdown":
        raise ValueError("checkpoint is not a Pushdown OLMo model")
    if not getattr(model, "_pushdown_attachment_weights_loaded", True):
        raise RuntimeError("checkpoint has no trained Pushdown attachment head")
    if torch.device(args.device).type == "cpu":
        # Production checkpoints enable the compiled FlexAttention score-mod,
        # whose current PyTorch CPU lowering cannot capture the depth buffers.
        # The model already has an exact dense SDPA fallback used by CPU tests.
        model.config.pushdown_use_flex = False
        for block in model.transformer.blocks:
            block.config.pushdown_use_flex = False
    result = evaluate_matrix(
        model=model,
        corpus=corpus,
        device=args.device,
        beam_size=args.beam_size,
        max_reduce=args.max_reduce,
        eval_batch_size=args.eval_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        training_attachment_normalization=args.training_attachment_normalization,
        include_per_sentence=args.include_per_sentence,
        progress_every=args.progress_every,
    )
    result["checkpoint"] = args.checkpoint
    result["data"] = {
        "tree_path": args.tree_path,
        "sentence_index_path": args.sentence_index_path,
        "document_index_path": args.document_index_path,
        "tokenizer_path": args.tokenizer_path,
    }
    result["run"] = {
        "argv": [sys.executable, *sys.argv],
        "device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
        logging.getLogger("pushdown-ppl-2x2").info("wrote %s", output)
    print(rendered)


if __name__ == "__main__":
    main()
