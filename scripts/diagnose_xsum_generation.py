#!/usr/bin/env python3
"""Compare XSum decoding paths on one fine-tuned checkpoint.

This is deliberately separate from the training/evaluation implementation so it
can diagnose cached decoding and beam scoring without changing campaign results.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import evaluate
import torch

from olmo.beam_search import LengthNormalizedSequenceLogProbabilityScorer
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
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--uncached-examples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--max-steps", type=int, default=150)
    return parser.parse_args()


def decode(tokenizer: Tokenizer, ids: torch.Tensor) -> str:
    return tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=True).strip()


def uncached_greedy(
    model: OLMo, input_ids: torch.Tensor, max_steps: int, eos_token_id: int
) -> torch.Tensor:
    prefix = input_ids
    generated: list[torch.Tensor] = []
    for _ in range(max_steps):
        output = model(prefix, last_logits_only=True)
        next_token = output.logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token)
        prefix = torch.cat((prefix, next_token[:, None]), dim=1)
        if bool((next_token == eos_token_id).all()):
            break
    return torch.stack(generated, dim=1)[0]


def metric_summary(predictions: list[str], references: list[str]) -> dict[str, float | int]:
    rouge = evaluate.load("rouge")
    values = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_aggregator=True,
    )
    values["R-AVG"] = sum(values.values()) / 3
    counts = Counter(predictions)
    values["unique"] = len(counts)
    values["top1_count"] = counts.most_common(1)[0][1]
    values["mean_words"] = sum(len(x.split()) for x in predictions) / len(predictions)
    return values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = OLMo.from_checkpoint(args.checkpoint.resolve(), device=device).eval()
    tokenizer = Tokenizer.from_checkpoint(args.checkpoint.resolve())
    dataset = XsumDataset(
        tokenizer=tokenizer,
        dataset_path=str(args.dataset_dir.resolve()),
        split="test",
        model_ctx_len=model.config.max_sequence_length,
        transformer_grammar_type="terminal",
        vocab_path=str(args.vocab.resolve()),
    )
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(dataset)), args.num_examples))
    strategies: dict[str, list[str]] = {
        "greedy_cached": [],
        "beam6_sum_logprob": [],
        "beam6_length_normalized": [],
    }
    records = []
    length_scorer = LengthNormalizedSequenceLogProbabilityScorer(length_penalty=1.0)

    for offset, index in enumerate(indices):
        sample = dataset[index]
        input_ids = torch.as_tensor(sample["input_ids"], dtype=torch.long, device=device)[None, :]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            greedy = model.generate(
                input_ids, max_steps=args.max_steps, beam_size=1
            ).token_ids[0, 0]
            beam = model.generate(
                input_ids, max_steps=args.max_steps, beam_size=6
            ).token_ids[0, 0]
            length_beam = model.generate(
                input_ids,
                max_steps=args.max_steps,
                beam_size=6,
                final_sequence_scorer=length_scorer,
            ).token_ids[0, 0]
            no_cache = None
            if offset < args.uncached_examples:
                no_cache = uncached_greedy(
                    model, input_ids, min(args.max_steps, 32), model.config.eos_token_id
                )

        decoded = {
            "greedy_cached": decode(tokenizer, greedy),
            "beam6_sum_logprob": decode(tokenizer, beam),
            "beam6_length_normalized": decode(tokenizer, length_beam),
        }
        for key, value in decoded.items():
            strategies[key].append(value)
        uncached_text = decode(tokenizer, no_cache) if no_cache is not None else None
        records.append(
            {
                "dataset_index": index,
                "reference": sample["gold_summary"],
                **decoded,
                "greedy_uncached_first32": uncached_text,
                "cached_uncached_prefix_match": (
                    decoded["greedy_cached"].startswith(uncached_text)
                    if uncached_text is not None
                    else None
                ),
            }
        )
        print(f"completed {offset + 1}/{len(indices)}", flush=True)

    references = [record["reference"] for record in records]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "indices": indices,
        "num_examples": len(indices),
        "uncached_examples": args.uncached_examples,
        "strategies": {
            name: metric_summary(predictions, references)
            for name, predictions in strategies.items()
        },
        "cache_prefix_matches": sum(
            record["cached_uncached_prefix_match"] is True for record in records
        ),
    }
    with (args.output_dir / "predictions.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
