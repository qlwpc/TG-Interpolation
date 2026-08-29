#!/usr/bin/env python3
"""Compare the legacy and constrained Pause XSum decoding paths.

This script is intentionally bounded and read-only.  It loads an existing
fine-tuned checkpoint, runs the same sampled XSum examples through the legacy
``word_sync_beam_search + extract_real_tokens`` route and the corrected
absolute-phase ``pause_generate`` route, and records ROUGE, lengths, uniqueness,
runtime, and the raw predictions.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import evaluate
import numpy as np
import torch

from olmo.config import BeamSearchType
from olmo.data.util import extract_real_tokens, is_pause_label
from olmo.eval.downstream import XsumDataset
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/Xsum"))
    parser.add_argument(
        "--vocab", type=Path, default=Path("dataset/bbc-news/TG_GPT2_tokenizer.json")
    )
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--max-real-tokens", type=int, default=150)
    parser.add_argument("--beam-size", type=int, default=6)
    parser.add_argument("--skip-legacy", action="store_true")
    return parser.parse_args()


def decode(tokenizer: Tokenizer, token_ids) -> str:
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    return tokenizer.decode([int(token) for token in token_ids]).strip()


def summarize(predictions: list[str], references: list[str], elapsed: float) -> dict:
    rouge = evaluate.load("rouge").compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_aggregator=True,
    )
    rouge["R-AVG"] = sum(rouge.values()) / 3
    counts = Counter(predictions)
    lengths = [len(prediction.split()) for prediction in predictions]
    rouge.update(
        {
            "num_examples": len(predictions),
            "unique_predictions": len(counts),
            "top1_count": counts.most_common(1)[0][1],
            "mean_words": sum(lengths) / len(lengths),
            "min_words": min(lengths),
            "max_words": max(lengths),
            "elapsed_seconds": elapsed,
            "seconds_per_example": elapsed / len(predictions),
        }
    )
    return rouge


def main() -> None:
    args = parse_args()
    if args.num_examples < 1 or args.max_real_tokens < 1 or args.beam_size < 1:
        raise ValueError("num-examples, max-real-tokens, and beam-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint.resolve()
    model = OLMo.from_checkpoint(checkpoint, device="cuda").eval()
    # Migrated finetune checkpoints retain the source cluster's absolute
    # tokenizer path in config.yaml.  The diagnostic already has an explicit
    # canonical --vocab argument, so use it instead of following a stale path.
    tokenizer = Tokenizer.from_file(
        args.vocab.resolve(),
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )
    grammar = model.config.transformer_grammar_type
    if not grammar.startswith("pause"):
        raise ValueError(f"checkpoint is not a pause model: {grammar!r}")
    pause_spec = model.config.pause_spec
    pause_token_id = model.config.pause_token_id
    dataset = XsumDataset(
        tokenizer=tokenizer,
        dataset_path=str(args.dataset_dir.resolve()),
        split="test",
        model_ctx_len=model.config.max_sequence_length,
        transformer_grammar_type=grammar,
        vocab_path=str(args.vocab.resolve()),
        pause_token_id=pause_token_id,
    )
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(dataset)), args.num_examples))
    methods = ["constrained"] + ([] if args.skip_legacy else ["legacy"])
    predictions = {method: [] for method in methods}
    elapsed = {method: 0.0 for method in methods}
    records = []

    for offset, index in enumerate(indices):
        sample = dataset[index]
        prompt = torch.as_tensor(sample["input_ids"], dtype=torch.long, device="cuda")[None, :]
        record = {
            "dataset_index": index,
            "reference": sample["gold_summary"],
            "expanded_prompt_tokens": int(prompt.shape[1]),
            "pause_token_count": int((prompt == pause_token_id).sum().item())
            if pause_token_id is not None
            else None,
        }

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.pause_generate(
                prompt,
                pause_spec=pause_spec,
                max_real_tokens=args.max_real_tokens,
                pause_token_id=pause_token_id,
                vocab=dataset.vocab,
                eos_token_id=model.config.eos_token_id,
                beam_size=args.beam_size,
                score_pause_tokens=not is_pause_label(grammar),
            )
        torch.cuda.synchronize()
        elapsed["constrained"] += time.perf_counter() - started
        constrained = decode(tokenizer, generated.token_ids[0, 0])
        predictions["constrained"].append(constrained)
        record["constrained"] = constrained

        if "legacy" in methods:
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                beams = model.word_sync_beam_search(
                    vocab=dataset.vocab,
                    past_input=prompt[0].cpu(),
                    max_word_steps=args.max_real_tokens // 2,
                    max_length=args.max_real_tokens,
                    beam_size=args.beam_size,
                    generate_TG_bias=None,
                    strategy=BeamSearchType.default,
                    transformer_grammar_type=grammar,
                )
            torch.cuda.synchronize()
            elapsed["legacy"] += time.perf_counter() - started
            legacy_ids = beams[0]["input_ids"].numpy()
            legacy_ids = extract_real_tokens(
                legacy_ids, pause_spec[0], pause_spec[1], skip_first=True
            )
            legacy_ids = dataset.vocab.convert_treenpy_to_terminal(legacy_ids)
            legacy = decode(tokenizer, np.asarray(legacy_ids))
            predictions["legacy"].append(legacy)
            record["legacy"] = legacy

        records.append(record)
        print(f"pause-xsum diagnostic {offset + 1}/{len(indices)}", flush=True)

    references = [record["reference"] for record in records]
    summary = {
        "checkpoint": str(checkpoint),
        "grammar": grammar,
        "pause_spec": list(pause_spec),
        "pause_token_id": pause_token_id,
        "seed": args.seed,
        "indices": indices,
        "methods": {
            method: summarize(predictions[method], references, elapsed[method])
            for method in methods
        },
    }
    with (args.output_dir / "predictions.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
