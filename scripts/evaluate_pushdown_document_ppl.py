#!/usr/bin/env python
"""Teacher-forced document PPL over the 300 supplied Pushdown gold trees."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from olmo.attachment import canonical_attachment_normalization  # noqa: E402
from olmo.eval.pushdown_document_ppl import NativePushdownTopKCorpus, evaluate_pushdown_document_ppl  # noqa: E402
from olmo.model import OLMo  # noqa: E402
from scripts.native_document_results import prepare_result_store, selected_document_ids  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-data", default="dataset/bbc-news/testppl/native_model_topk_300_v2")
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--start-document", type=int, default=0)
    parser.add_argument("--end-document", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--token-only", action="store_true")
    parser.add_argument(
        "--attachment-normalization",
        choices=("v1", "stack_legal", "v2", "sentence_causal"),
        default="stack_legal",
        help=(
            "v1/stack_legal renormalizes over stack-reachable targets; "
            "v2/sentence_causal retains probabilities from the full causal row"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-batch-attention-elements", type=int, default=16777216)
    parser.add_argument("--document-result-dir", help="atomically save complete documents for restart")
    parser.add_argument("--resume-document-results", action="store_true")
    parser.add_argument("--no-kv-cache", action="store_true", help="use full-prefix scoring as a correctness reference")
    args = parser.parse_args()
    attachment_normalization = canonical_attachment_normalization(
        args.attachment_normalization
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pushdown-document-ppl")
    log.info(
        "structure_source=native_pushdown_nary_topk beam_search=false context_update=candidate0 "
        "attachment_probability=%s attachment_normalization=%s "
        "mixture_reporting=legacy_and_normalized",
        not args.token_only,
        attachment_normalization,
    )
    store = prepare_result_store(args, "pushdown", {
        "attachment_normalization": attachment_normalization,
        "include_attachment_probability": not args.token_only,
        "max_sequence_length": args.max_sequence_length,
        "use_kv_cache": not args.no_kv_cache,
        "deduplicated_trees": False,
        "prefix_policy": "candidate0", "candidate_aggregation": "truncated_joint_sum",
        "ppl_denominator": "terminal_count",
    })
    corpus = NativePushdownTopKCorpus(args.native_data, args.tokenizer_path, args.max_sentences,
                                      args.start_document, args.end_document)
    expected_ids = selected_document_ids(corpus)
    if store is not None and expected_ids and expected_ids <= store.completed_ids:
        print(json.dumps(store.aggregate(expected_ids), indent=2, sort_keys=True))
        return
    model = OLMo.from_checkpoint(args.checkpoint, device=args.device)
    if model.config.transformer_grammar_type != "pushdown":
        raise ValueError("checkpoint is not a Pushdown OLMo model")
    if not args.token_only and not getattr(model, "_pushdown_attachment_weights_loaded", True):
        raise RuntimeError("checkpoint has no attachment-head weights; rerun with --token-only")
    if not args.token_only and not model.config.pushdown_use_attachment_head_inference:
        log.warning("checkpoint config disables attachment-head inference; evaluating its loaded attachment head nonetheless")
    def progress(done: int, total: int, doc_id: int) -> None:
        if done == total or (args.log_every > 0 and done % args.log_every == 0):
            log.info("scored %d/%d sentences (document %d)", done, total, doc_id)
    result = evaluate_pushdown_document_ppl(
        model,
        corpus,
        args.device,
        args.eval_batch_size,
        args.max_sequence_length,
        False,
        not args.token_only,
        progress,
        args.max_batch_tokens,
        attachment_normalization,
        max_batch_attention_elements=args.max_batch_attention_elements,
        use_kv_cache=not args.no_kv_cache,
        document_complete=store.write if store else None,
        completed_document_ids=store.completed_ids if store else None,
    )
    output = store.aggregate(expected_ids) if store is not None else result.as_dict()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
