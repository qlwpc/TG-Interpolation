#!/usr/bin/env python3
"""Summarize Validation-8 raw Eq.4/5 decomposition outputs with paired CIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


TASK_FILES = {
    "winogrande": "per_example_winogrande_decomp.json",
    "hellaswag": "per_example_hellaswag_decomp.json",
    "openbook_qa": "per_example_openbook_qa_validation_decomp.json",
    "commonsense_qa": "per_example_commonsense_qa_decomp.json",
    "social_iqa": "per_example_social_iqa_decomp.json",
    "arc_easy": "per_example_arc_easy_decomp.json",
    "arc_challenge": "per_example_arc_challenge_decomp.json",
    "piqa": "per_example_piqa_decomp.json",
}


def percentile(draws):
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]


def load_task(path: Path, expected_docs: int):
    rows = json.loads(path.read_text())
    by_doc = {}
    max_residual = 0.0
    max_relative_residual = 0.0
    for row in rows:
        required = {
            "doc_id", "cont_id", "label", "full_score_raw", "term_score_raw",
            "nt_score_raw", "n_terminal", "n_nonterminal",
            "decomposition_residual",
        }
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        residual = abs(float(row["decomposition_residual"]))
        scale = max(
            1.0,
            abs(float(row["full_score_raw"])),
            abs(float(row["term_score_raw"])),
            abs(float(row["nt_score_raw"])),
        )
        max_residual = max(max_residual, residual)
        max_relative_residual = max(max_relative_residual, residual / scale)
        by_doc.setdefault(int(row["doc_id"]), []).append(row)
    if len(by_doc) != expected_docs:
        raise ValueError(f"{path}: expected {expected_docs} docs, found {len(by_doc)}")
    # Raw sums are accumulated and differenced in float32. Long HellaSwag/PIQA
    # continuations can therefore leave one or two float32 ULPs of cancellation
    # error even though NT was constructed as full - terminal. Enforce both a
    # conservative absolute and relative tolerance.
    if max_residual > 2e-5 or max_relative_residual > 1e-6:
        raise ValueError(
            f"{path}: max decomposition residual {max_residual} "
            f"(relative {max_relative_residual})"
        )

    evidence = []
    for doc_id in sorted(by_doc):
        choices = sorted(by_doc[doc_id], key=lambda row: int(row["cont_id"]))
        choice_ids = [int(row["cont_id"]) for row in choices]
        if choice_ids != list(range(len(choices))):
            raise ValueError(f"{path}: non-contiguous choices for doc {doc_id}")
        labels = {int(row["label"]) for row in choices}
        if len(labels) != 1:
            raise ValueError(f"{path}: inconsistent labels for doc {doc_id}")
        label = next(iter(labels))
        full = np.asarray([row["full_score_raw"] for row in choices], dtype=np.float64)
        term = np.asarray([row["term_score_raw"] for row in choices], dtype=np.float64)
        full_pred = int(np.argmax(full))
        term_pred = int(np.argmax(term))
        full_correct = full_pred == label
        term_correct = term_pred == label
        evidence.append([
            float(full_correct),
            float(term_correct),
            float(full_pred != term_pred),
            float(term_correct and not full_correct),
            float(full_correct and not term_correct),
            float(sum(row["term_score_raw"] for row in choices)),
            float(sum(row["nt_score_raw"] for row in choices)),
            float(sum(row["n_terminal"] for row in choices)),
            float(sum(row["n_nonterminal"] for row in choices)),
        ])
    return (
        np.asarray(evidence, dtype=np.float64),
        len(rows),
        max_residual,
        max_relative_residual,
    )


def point_metrics(evidence):
    full_acc = evidence[:, 0].mean()
    term_acc = evidence[:, 1].mean()
    n_term = evidence[:, 7].sum()
    n_nt = evidence[:, 8].sum()
    gap = (
        evidence[:, 6].sum() / n_nt - evidence[:, 5].sum() / n_term
        if n_term > 0 and n_nt > 0 else float("nan")
    )
    return {
        "full_accuracy": float(full_acc),
        "terminal_accuracy": float(term_acc),
        "delta_accuracy_pp": float(100.0 * (term_acc - full_acc)),
        "flip_rate": float(evidence[:, 2].mean()),
        "flip_to_correct": float(evidence[:, 3].mean()),
        "flip_to_wrong": float(evidence[:, 4].mean()),
        "nt_logp_gap": float(gap),
        "n_terminal_tokens": int(n_term),
        "n_nonterminal_tokens": int(n_nt),
    }


def bootstrap(evidence, rng, resamples, chunk=128):
    draws = {key: [] for key in (
        "full_accuracy", "terminal_accuracy", "delta_accuracy_pp", "flip_rate",
        "flip_to_correct", "flip_to_wrong", "nt_logp_gap",
    )}
    n = len(evidence)
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        indices = rng.integers(0, n, size=(size, n))
        sample = evidence[indices]
        full = sample[:, :, 0].mean(axis=1)
        term = sample[:, :, 1].mean(axis=1)
        draws["full_accuracy"].extend(full)
        draws["terminal_accuracy"].extend(term)
        draws["delta_accuracy_pp"].extend(100.0 * (term - full))
        draws["flip_rate"].extend(sample[:, :, 2].mean(axis=1))
        draws["flip_to_correct"].extend(sample[:, :, 3].mean(axis=1))
        draws["flip_to_wrong"].extend(sample[:, :, 4].mean(axis=1))
        term_sum = sample[:, :, 5].sum(axis=1)
        nt_sum = sample[:, :, 6].sum(axis=1)
        n_term = sample[:, :, 7].sum(axis=1)
        n_nt = sample[:, :, 8].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gap = nt_sum / n_nt - term_sum / n_term
        draws["nt_logp_gap"].extend(gap)
    return {key: np.asarray(values, dtype=np.float64) for key, values in draws.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    expected = {row["task"]: row["evaluation_count"] for row in manifest["tasks"]}
    rng = np.random.default_rng(args.seed)
    metrics = {}
    all_draws = {}
    for task, filename in TASK_FILES.items():
        evidence, choices, residual, relative_residual = load_task(
            args.run_dir / filename, expected[task]
        )
        points = point_metrics(evidence)
        draws = bootstrap(evidence, rng, args.resamples)
        metrics[task] = {
            "documents": len(evidence),
            "choices": choices,
            "max_decomposition_residual": residual,
            "max_relative_decomposition_residual": relative_residual,
            **points,
            "ci95": {key: percentile(value) for key, value in draws.items()},
        }
        all_draws[task] = draws

    macro = {}
    for key in ("full_accuracy", "terminal_accuracy", "delta_accuracy_pp",
                "flip_rate", "flip_to_correct", "flip_to_wrong", "nt_logp_gap"):
        point = float(np.mean([metrics[task][key] for task in TASK_FILES]))
        draws = np.mean(np.stack([all_draws[task][key] for task in TASK_FILES]), axis=0)
        macro[key] = {"value": point, "ci95": percentile(draws)}

    result = {
        "protocol": "paper_eq4_eq5_raw_sum_validation8",
        "seed": args.seed,
        "bootstrap_resamples": args.resamples,
        "tasks": metrics,
        "macro_average": macro,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    fields = ["task", "documents", "full_accuracy", "terminal_accuracy",
              "delta_accuracy_pp", "flip_rate", "flip_to_correct",
              "flip_to_wrong", "nt_logp_gap"]
    with (args.output_dir / "table10_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task, row in metrics.items():
            writer.writerow({key: row.get(key, task if key == "task" else None) for key in fields})

    lines = [
        "# Terminal-format decomposition on Validation-8",
        "",
        "| Task | N | Full Acc | Term Acc | Delta (pp) | Flip | NT log-p gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, row in metrics.items():
        lines.append(
            f"| {task} | {row['documents']} | {row['full_accuracy']:.4f} | "
            f"{row['terminal_accuracy']:.4f} | {row['delta_accuracy_pp']:+.2f} | "
            f"{row['flip_rate']:.4f} | {row['nt_logp_gap']:+.3f} |"
        )
    lines.extend(["", f"Bootstrap: {args.resamples} paired resamples, seed {args.seed}."])
    (args.output_dir / "table10_validation.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"tasks": len(metrics), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
