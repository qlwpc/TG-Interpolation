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
from olmo.eval.pushdown_document_ppl import PushdownGold300Corpus, evaluate_pushdown_document_ppl  # noqa: E402
from olmo.model import OLMo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tree-data", default="dataset/testppl_tree/tree_300.npy")
    parser.add_argument("--sentence-index", default="dataset/testppl_tree/tree_sent_index.npy")
    parser.add_argument("--document-index", default="dataset/testppl_tree/tree_doc_index.npy")
    parser.add_argument("--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--samples-per-sentence", type=int, default=300)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--deduplicate-trees", action="store_true")
    parser.add_argument("--token-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pushdown-document-ppl")
    log.info("structure_source=gold300 beam_search=false context_update=candidate0 attachment_probability=%s mixture_reporting=legacy_and_normalized", not args.token_only)
    corpus = PushdownGold300Corpus(args.tree_data, args.sentence_index, args.document_index, args.tokenizer_path, args.samples_per_sentence, args.max_sentences)
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
    result = evaluate_pushdown_document_ppl(model, corpus, args.device, args.eval_batch_size, args.max_sequence_length, args.deduplicate_trees, not args.token_only, progress)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
