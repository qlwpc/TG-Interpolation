#!/usr/bin/env python
"""Audit the GPST document-PPL objective on native fixed top-k trees.

This is intentionally a small, exact diagnostic.  It reports the same
unnormalised 300-tree joint marginal used by the legacy evaluator, plus a
token/action decomposition of the MAP tree.  It can also choose the document
prefix tree either by slot 0 (the historical OLMo convention) or by the
model's joint-MAP candidate (the greedy convention in the paper).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Sequence, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
import torch.nn.functional as F

from olmo.gpst.eval.document_ppl import (
    GoldSegment,
    NativeGPSTTopKCorpus,
    _count_actions,
    _count_tokens,
    _fast_gold_batch,
    _move_batch,
    _trim_prefix,
)
from olmo.gpst.model.model_factory import create_model


def retain_prefix(prefix: Sequence[GoldSegment], max_action_nodes: int,
                  max_terminals: int) -> Tuple[GoldSegment, ...]:
    """Keep the largest suffix any later `_trim_prefix` can consume exactly."""
    action_limit, token_limit = max_action_nodes - 1, max_terminals - 1
    kept = []
    actions = tokens = 0
    for segment in reversed(prefix):
        if (actions + segment.action_count > action_limit
                or tokens + len(segment.tokens) > token_limit):
            break
        kept.append(segment)
        actions += segment.action_count
        tokens += len(segment.tokens)
    return tuple(reversed(kept))


@torch.no_grad()
def score_split(model, candidates, device, prefix_tokens, current_tokens,
                prefix_actions, current_actions):
    """Return per-candidate current token and action negative log likelihood."""
    batch = _move_batch(_fast_gold_batch(candidates), device)
    output = model(
        **batch,
        force_gold_tree=True,
        score_token_range=(prefix_tokens, prefix_tokens + current_tokens),
        score_action_range=(prefix_actions, prefix_actions + current_actions),
    )
    if output.logits is None or output.action_logits is None:
        raise RuntimeError("GPST token/action heads are required")
    token_nll = F.cross_entropy(output.logits.transpose(1, 2), output.token_targets,
                                ignore_index=-100, reduction="none").sum(dim=1)
    action_nll = F.cross_entropy(output.action_logits.transpose(1, 2), output.action_targets,
                                 ignore_index=-1, reduction="none").sum(dim=1)
    return token_nll.double().cpu(), action_nll.double().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-data", default="dataset/bbc-news/testppl/native_model_topk_300_v2")
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--r2d2-config", default="olmo/gpst/data/en_config/r2d2_256_4_1.json")
    parser.add_argument("--gpt-config", default="olmo/gpst/data/gpt2-bbc/config.json")
    parser.add_argument("--backbone", choices=("olmo", "gpt2"), default="olmo")
    parser.add_argument("--start-document", type=int, default=0)
    parser.add_argument("--end-document", type=int, default=1)
    parser.add_argument("--prefix-mode", choices=("candidate0", "joint-best"), default="candidate0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-batch-actions", type=int, default=32768)
    parser.add_argument("--max-action-nodes", type=int, default=2048)
    parser.add_argument("--max-terminals", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.eval_batch_size <= 0 or args.max_batch_actions <= 0:
        parser.error("batch limits must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("gpst-objective-diagnostic")
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
    corpus = NativeGPSTTopKCorpus(args.native_data, args.tokenizer_path,
                                  start_document=args.start_document,
                                  end_document=args.end_document)
    model = create_model("r2d2-gen-fast", args.r2d2_config, args.gpt_config,
                         backbone=args.backbone)
    model.from_pretrain(args.checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()

    prefix: Tuple[GoldSegment, ...] = ()
    previous_doc = None
    terminal_count = sentence_count = document_count = 0
    joint_log_likelihood = token_marginal_log_likelihood = 0.0
    map_token_nll = map_action_nll = 0.0
    candidate_slots = candidate_forwards = 0
    selected_slots = []
    for sentence_index, (doc_id, original_candidates) in enumerate(corpus):
        if doc_id != previous_doc:
            prefix = ()
            previous_doc = doc_id
            document_count += 1
        current = original_candidates[0]
        context = _trim_prefix(prefix, current, args.max_action_nodes, args.max_terminals)
        prefix_tokens, prefix_actions = _count_tokens(context), _count_actions(context)
        current_tokens, current_actions = _count_tokens(current), _count_actions(current)
        token_parts, action_parts = [], []
        batch_size = min(args.eval_batch_size, max(1, args.max_batch_actions // max(
            prefix_actions + current_actions, 1
        )))
        for start in range(0, len(original_candidates), batch_size):
            candidates = original_candidates[start:start + batch_size]
            token_nll, action_nll = score_split(
                model, [tuple(context) + tuple(candidate) for candidate in candidates], device,
                prefix_tokens, current_tokens, prefix_actions, current_actions,
            )
            token_parts.append(token_nll)
            action_parts.append(action_nll)
            candidate_forwards += len(candidates)
        token_nll, action_nll = torch.cat(token_parts), torch.cat(action_parts)
        joint_nll = token_nll + action_nll
        joint_log_likelihood += torch.logsumexp(-joint_nll, 0).item()
        token_marginal_log_likelihood += torch.logsumexp(-token_nll, 0).item()
        best_slot = int(torch.argmin(joint_nll).item())
        map_token_nll += token_nll[best_slot].item()
        map_action_nll += action_nll[best_slot].item()
        selected_slots.append(best_slot)
        terminal_count += current_tokens
        sentence_count += 1
        candidate_slots += len(original_candidates)
        selected = original_candidates[0] if args.prefix_mode == "candidate0" else original_candidates[best_slot]
        prefix = retain_prefix(
            tuple(prefix) + tuple(selected), args.max_action_nodes, args.max_terminals
        )
        if args.log_every and (sentence_index + 1) % args.log_every == 0:
            log.info("scored %d/%d sentences", sentence_index + 1, len(corpus))

    result = {
        "prefix_mode": args.prefix_mode,
        "document_range": [args.start_document, args.end_document],
        "document_count": document_count,
        "sentence_count": sentence_count,
        "terminal_count": terminal_count,
        "candidate_slots": candidate_slots,
        "model_candidate_forwards": candidate_forwards,
        "joint_legacy_log_likelihood": joint_log_likelihood,
        "joint_legacy_perplexity": math.exp(-joint_log_likelihood / terminal_count),
        "token_only_legacy_log_likelihood": token_marginal_log_likelihood,
        "token_only_legacy_perplexity": math.exp(-token_marginal_log_likelihood / terminal_count),
        "map_tree_token_perplexity": math.exp(map_token_nll / terminal_count),
        "map_tree_action_nll_per_terminal": map_action_nll / terminal_count,
        "map_tree_joint_perplexity": math.exp((map_token_nll + map_action_nll) / terminal_count),
        "selected_slot_nonzero_count": sum(slot != 0 for slot in selected_slots),
        "selected_slot_first_20": selected_slots[:20],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
