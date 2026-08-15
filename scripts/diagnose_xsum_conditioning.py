#!/usr/bin/env python3
"""Measure whether an XSum-fine-tuned model uses its article context."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import evaluate
import torch
import torch.nn.functional as F

from olmo.eval.downstream import XsumDataset
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=6198)
    return parser.parse_args()


def target_nll(model: OLMo, item: dict) -> tuple[float, int]:
    ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device="cuda")[None, :]
    mask = torch.as_tensor(item["label_mask"], dtype=torch.bool, device="cuda")[None, :]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids).logits[:, :-1, :].float()
    targets = ids[:, 1:]
    active = mask[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    return float(losses[active].sum().item()), int(active.sum().item())


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    root = Path.cwd()
    model = OLMo.from_checkpoint(checkpoint, device="cuda").eval()
    tokenizer = Tokenizer.from_checkpoint(checkpoint)
    dataset = XsumDataset(
        tokenizer=tokenizer,
        dataset_path=str(root / "dataset/Xsum"),
        split="train",
        model_ctx_len=model.config.max_sequence_length,
        transformer_grammar_type="terminal",
        vocab_path=str(root / "dataset/bbc-news/TG_GPT2_tokenizer.json"),
    )
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.num_examples)
    shuffled = indices[1:] + indices[:1]
    records = []
    correct_total = wrong_total = 0.0
    token_total = 0
    for offset, (index, wrong_index) in enumerate(zip(indices, shuffled)):
        correct = dataset[index]
        correct_loss, tokens = target_nll(model, correct)
        original_passage = dataset.passages[index]
        dataset.passages[index] = dataset.passages[wrong_index]
        try:
            wrong = dataset[index]
        finally:
            dataset.passages[index] = original_passage
        wrong_loss, wrong_tokens = target_nll(model, wrong)
        if tokens != wrong_tokens:
            raise RuntimeError("Target token count changed when only the article was shuffled")
        correct_total += correct_loss
        wrong_total += wrong_loss
        token_total += tokens
        records.append(
            {
                "index": index,
                "wrong_article_index": wrong_index,
                "target_tokens": tokens,
                "correct_article_nll": correct_loss / tokens,
                "wrong_article_nll": wrong_loss / tokens,
            }
        )
        print(f"conditioning {offset + 1}/{len(indices)}", flush=True)

    # Evaluate memorization/generalization on the same sampled training articles,
    # but remove summaries from the input so construction matches test inference.
    saved_train_summary = dataset.train_summary
    dataset.train_summary = None
    predictions = []
    references = []
    try:
        for offset, index in enumerate(indices):
            item = dataset[index]
            ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device="cuda")[None, :]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(ids, max_steps=150, beam_size=6).token_ids[0, 0]
            predictions.append(
                tokenizer.decode(generated.cpu().tolist(), skip_special_tokens=True).strip()
            )
            references.append(item["gold_summary"])
            print(f"train-generation {offset + 1}/{len(indices)}", flush=True)
    finally:
        dataset.train_summary = saved_train_summary

    rouge = evaluate.load("rouge").compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_aggregator=True,
    )
    rouge["R-AVG"] = sum(rouge.values()) / 3
    output = {
        "checkpoint": str(checkpoint),
        "num_examples": len(indices),
        "indices": indices,
        "correct_article_mean_nll": correct_total / token_total,
        "wrong_article_mean_nll": wrong_total / token_total,
        "wrong_minus_correct_nll": (wrong_total - correct_total) / token_total,
        "train_article_generation": rouge,
        "records": records,
        "predictions": predictions,
        "references": references,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in {"records", "predictions", "references"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
