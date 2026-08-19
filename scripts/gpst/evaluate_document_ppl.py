#!/usr/bin/env python
"""Evaluate GPST document perplexity with 300 prescribed gold parse trees.

No parsing beam is run.  Every serialized candidate is converted to its exact
GPST binary merge trajectory and teacher-forced through the model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from olmo.gpst.eval.document_ppl import (  # noqa: E402
    GoldTree300Corpus,
    evaluate_gold_tree_document_ppl,
)
from olmo.gpst.model.model_factory import create_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tree-data", default="dataset/testppl_tree/tree_300.npy"
    )
    parser.add_argument(
        "--sentence-index", default="dataset/testppl_tree/tree_sent_index.npy"
    )
    parser.add_argument(
        "--document-index", default="dataset/testppl_tree/tree_doc_index.npy"
    )
    parser.add_argument(
        "--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json"
    )
    parser.add_argument(
        "--r2d2-config", default="olmo/gpst/data/en_config/r2d2_256_4_1.json"
    )
    parser.add_argument(
        "--gpt-config", default="olmo/gpst/data/gpt2-bbc/config.json"
    )
    parser.add_argument("--backbone", choices=("olmo", "gpt2"), default="olmo")
    parser.add_argument("--samples-per-sentence", type=int, default=300)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--max-action-nodes", type=int, default=2048)
    parser.add_argument("--max-terminals", type=int, default=2048)
    parser.add_argument(
        "--normalize-mixture",
        action="store_true",
        help="subtract log(K) per sentence; default reproduces OLMo's legacy metric",
    )
    parser.add_argument(
        "--deduplicate-trees",
        action="store_true",
        help="collapse candidates that map to the same unlabeled GPST tree",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("gpst-document-ppl")
    # GPST action sequences have length 2*n-1 and are not necessarily a
    # multiple of eight.  The memory-efficient SDPA backend supports them.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)

    log.info("loading fixed gold-tree corpus (beam search is disabled)")
    corpus = GoldTree300Corpus(
        args.tree_data,
        args.sentence_index,
        args.document_index,
        args.tokenizer_path,
        samples_per_sentence=args.samples_per_sentence,
        max_sentences=args.max_sentences,
    )
    log.info("loading GPST checkpoint with %s transformer blocks", args.backbone)
    model = create_model(
        "r2d2-gen-fast", args.r2d2_config, args.gpt_config, backbone=args.backbone
    )
    model.from_pretrain(args.checkpoint)
    device = torch.device(args.device)
    model.to(device)

    def progress(done: int, total: int, doc_id: int) -> None:
        if done == total or (args.log_every > 0 and done % args.log_every == 0):
            log.info("scored %d/%d sentences (document %d)", done, total, doc_id)

    result = evaluate_gold_tree_document_ppl(
        model,
        corpus,
        device=device,
        eval_batch_size=args.eval_batch_size,
        max_action_nodes=args.max_action_nodes,
        max_terminals=args.max_terminals,
        normalize_mixture=args.normalize_mixture,
        deduplicate_trees=args.deduplicate_trees,
        progress=progress,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
