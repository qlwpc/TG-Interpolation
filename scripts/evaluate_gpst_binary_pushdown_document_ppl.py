#!/usr/bin/env python
"""Evaluate Pushdown document PPL on a selected v2 binary candidate support."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from olmo.attachment import canonical_attachment_normalization  # noqa: E402

from olmo.eval.gpst_binary_pushdown_document_ppl import (  # noqa: E402
    NativeGPSTBinaryPushdownCorpus,
    NativeGPSTSplicedBPEPushdownCorpus,
    NativeNaryRightBinarizedPushdownCorpus,
    NativeNaryWordAtomRightBinarizedPushdownCorpus,
    evaluate_gpst_binary_pushdown_document_ppl,
)
from olmo.model import OLMo  # noqa: E402


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a valid v2 candidate axis to binary Pushdown spans and "
            "evaluate truncated-sum document perplexity."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--native-data",
        default="dataset/bbc-news/testppl/native_model_topk_300_v2",
    )
    parser.add_argument(
        "--tokenizer-path", default="dataset/bbc-news/TG_GPT2_tokenizer.json"
    )
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument(
        "--attachment-normalization",
        choices=("stack_legal", "sentence_causal", "v1", "v2"),
        default="stack_legal",
        help=(
            "historical Table-4-compatible stack-legal conditional (v1), or "
            "the checkpoint training objective's sentence-causal softmax (v2)"
        ),
    )
    parser.add_argument(
        "--candidate-source",
        choices=(
            "gpst-strict-binary",
            "gpst-strict-binary-spliced-bpe",
            "pushdown-nary-spliced-right-binary",
            "pushdown-nary-word-atom-right-binary",
        ),
        default="gpst-strict-binary",
        help=(
            "direct training-compatible GPST CKY binary support, diagnostic "
            "BPE-spliced variants, or native n-ary support converted with the "
            "checkpoint's fixed-word-atom right-CNF representation"
        ),
    )
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument("--max-batch-attention-elements", type=int, default=16777216)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--start-document", type=int, default=0)
    parser.add_argument("--end-document", type=int)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--prefetch-sentences",
        type=int,
        default=2,
        help="number of upcoming CPU conversions queued behind GPU scoring",
    )
    parser.add_argument(
        "--disable-kv-cache",
        action="store_true",
        help="correctness reference: recompute the complete bounded prefix",
    )
    parser.add_argument(
        "--skip-merge-order-validation",
        action="store_true",
        help="skip the per-row permutation check (native v2 data is prevalidated)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional aggregate JSON path (stdout is always emitted)",
    )
    parser.add_argument(
        "--document-output-dir",
        type=Path,
        help="optional directory for atomic per-document JSON records",
    )
    args = parser.parse_args()
    attachment_normalization = canonical_attachment_normalization(
        args.attachment_normalization
    )

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_model_path = checkpoint_path / "model.pt"
    native_data_path = Path(args.native_data).resolve()
    native_manifest_path = native_data_path / "manifest.json"
    tokenizer_path = Path(args.tokenizer_path).resolve()
    for label, path in (
        ("checkpoint model", checkpoint_model_path),
        ("native manifest", native_manifest_path),
        ("tokenizer", tokenizer_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    checkpoint_model_sha256 = _sha256(checkpoint_model_path)
    native_manifest_sha256 = _sha256(native_manifest_path)
    tokenizer_sha256 = _sha256(tokenizer_path)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("gpst-binary-pushdown-document-ppl")
    corpus_class = {
        "gpst-strict-binary": NativeGPSTBinaryPushdownCorpus,
        "gpst-strict-binary-spliced-bpe": NativeGPSTSplicedBPEPushdownCorpus,
        "pushdown-nary-spliced-right-binary": NativeNaryRightBinarizedPushdownCorpus,
        "pushdown-nary-word-atom-right-binary": (
            NativeNaryWordAtomRightBinarizedPushdownCorpus
        ),
    }[args.candidate_source]
    log.info(
        "structure_source=%s source_candidate_axis=%s binarization=%s "
        "prefix_policy=candidate0 context_truncation=left_drop_complete_sentences "
        "attachment_normalization=%s candidate_aggregation=truncated_sum "
        "divide_by_candidate_count=false kv_cache=%s",
        corpus_class.structure_source,
        corpus_class.source_candidate_axis,
        corpus_class.binarization,
        attachment_normalization,
        not args.disable_kv_cache,
    )

    corpus = corpus_class(
        args.native_data,
        args.tokenizer_path,
        max_sentences=args.max_sentences,
        start_document=args.start_document,
        end_document=args.end_document,
        validate_merge_orders=not args.skip_merge_order_validation,
    )
    model = OLMo.from_checkpoint(checkpoint_path, device=args.device)
    if model.config.transformer_grammar_type != "pushdown":
        raise ValueError("checkpoint is not a Pushdown OLMo model")
    if not getattr(model, "_pushdown_attachment_weights_loaded", True):
        raise RuntimeError("checkpoint has no loaded Pushdown attachment-head weights")

    def progress(done: int, total: int, doc_id: int) -> None:
        if done == total or (args.log_every > 0 and done % args.log_every == 0):
            log.info("scored %d/%d sentences (document %d)", done, total, doc_id)

    def document_complete(doc_id: int, payload: dict) -> None:
        if args.document_output_dir is not None:
            _atomic_json(
                args.document_output_dir / f"document_{doc_id:06d}.json", payload
            )

    result = evaluate_gpst_binary_pushdown_document_ppl(
        model=model,
        corpus=corpus,
        device=args.device,
        eval_batch_size=args.eval_batch_size,
        max_sequence_length=args.max_sequence_length,
        max_batch_tokens=args.max_batch_tokens,
        max_batch_attention_elements=args.max_batch_attention_elements,
        use_kv_cache=not args.disable_kv_cache,
        prefetch_sentences=args.prefetch_sentences,
        progress=progress,
        document_complete=document_complete,
        attachment_normalization=attachment_normalization,
    )
    payload = result.as_dict()
    payload.update(
        {
            "native_data": str(native_data_path),
            "native_manifest_sha256": native_manifest_sha256,
            "checkpoint": str(checkpoint_path),
            "checkpoint_model_sha256": checkpoint_model_sha256,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_sha256,
            "start_document": corpus.start_document,
            "end_document": corpus.end_document,
            "max_sentences": args.max_sentences,
            "kv_cache_enabled": not args.disable_kv_cache,
        }
    )
    if args.output is not None:
        _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
