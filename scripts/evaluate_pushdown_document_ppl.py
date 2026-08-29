#!/usr/bin/env python
"""Teacher-forced document PPL over the 300 supplied Pushdown gold trees."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from olmo.eval.pushdown_document_ppl import (  # noqa: E402
    PUSHDOWN_DOCUMENT_PPL_PROTOCOL_VERSION,
    NativePushdownTopKCorpus,
    evaluate_pushdown_document_ppl,
)
from olmo.model import OLMo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-data", default="dataset/bbc-news/native_model_topk_300_v2")
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument("--max-batch-attention-elements", type=int, default=16777216,
                        help="cap B * full_context_length^2; protects long-context batches")
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--start-document", type=int, default=0)
    parser.add_argument("--end-document", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--token-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--document-result-dir",
                        help="write each completed document atomically as document_<id>.json")
    parser.add_argument("--resume-document-results", action="store_true",
                        help="skip document IDs already atomically committed in --document-result-dir")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pushdown-document-ppl")
    expected_protocol = {
        "protocol_version": PUSHDOWN_DOCUMENT_PPL_PROTOCOL_VERSION,
        "structure_source": "native_pushdown_nary_topk",
        "attachment_normalization": "none" if args.token_only else "sentence_causal",
        "prefix_policy": "candidate0",
        "max_sequence_length": args.max_sequence_length,
    }
    log.info(
        "structure_source=native_pushdown_nary_topk beam_search=false "
        "context_update=candidate0 kv_cache=true attachment_probability=%s "
        "attachment_normalization=%s mixture_reporting=truncated_marginal_and_uniform_diagnostic",
        not args.token_only, expected_protocol["attachment_normalization"],
    )
    corpus = NativePushdownTopKCorpus(args.native_data, args.tokenizer_path, args.max_sentences,
                                      args.start_document, args.end_document)
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
    result_dir = Path(args.document_result_dir) if args.document_result_dir else None
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
    completed_document_ids = set()
    if args.resume_document_results:
        if result_dir is None:
            parser.error("--resume-document-results requires --document-result-dir")
        for path in result_dir.glob("document_*.json"):
            row = json.loads(path.read_text())
            actual_protocol = {key: row.get(key) for key in expected_protocol}
            if actual_protocol != expected_protocol:
                raise RuntimeError(
                    f"refusing to resume incompatible document result {path}: "
                    f"expected={expected_protocol} actual={actual_protocol}"
                )
            completed_document_ids.add(int(row["document_id"]))
        log.info("resuming with %d complete document results", len(completed_document_ids))
    def document_complete(doc_id: int, row: dict) -> None:
        if result_dir is None:
            return
        output = result_dir / f"document_{doc_id:05d}.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, output)
        log.info("committed document %d to %s", doc_id, output)
    result = evaluate_pushdown_document_ppl(
        model, corpus, args.device, args.eval_batch_size, args.max_sequence_length, False,
        not args.token_only, progress, args.max_batch_tokens, args.max_batch_attention_elements,
        document_complete, completed_document_ids,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
