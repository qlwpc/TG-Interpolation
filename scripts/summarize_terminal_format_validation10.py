#!/usr/bin/env python3
"""Summarize five-model Validation-10 accuracy with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk


TASK_FILES = {
    "winogrande": "per_example_winogrande.json",
    "hellaswag": "per_example_hellaswag.json",
    "boolq": "per_example_boolq.json",
    "mmlu": "per_example_mmlu_validation_holdout.json",
    "openbook_qa": "per_example_openbook_qa_validation.json",
    "commonsense_qa": "per_example_commonsense_qa.json",
    "social_iqa": "per_example_social_iqa.json",
    "arc_easy": "per_example_arc_easy.json",
    "arc_challenge": "per_example_arc_challenge.json",
    "piqa": "per_example_piqa.json",
}
DEFAULT_MODELS = ("terminal", "tree", "tgtree", "pause1", "pause2")


def percentile(draws: np.ndarray) -> list[float]:
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]


def load_correct(path: Path, expected_docs: int) -> np.ndarray:
    rows = json.loads(path.read_text())
    if len(rows) != expected_docs:
        raise ValueError(f"{path}: expected {expected_docs} rows, found {len(rows)}")
    by_doc: dict[int, float] = {}
    for row in rows:
        required = {"doc_id", "pred", "label", "correct", "n_choices"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        doc_id = int(row["doc_id"])
        correct = float(row["correct"])
        if doc_id in by_doc:
            raise ValueError(f"{path}: duplicate doc_id {doc_id}")
        if correct not in (0.0, 1.0):
            raise ValueError(f"{path}: invalid correct value {correct}")
        if (int(row["pred"]) == int(row["label"])) != bool(correct):
            raise ValueError(f"{path}: inconsistent prediction for doc_id {doc_id}")
        by_doc[doc_id] = correct
    doc_ids = sorted(by_doc)
    if doc_ids != list(range(expected_docs)):
        raise ValueError(f"{path}: doc IDs are not contiguous 0..{expected_docs - 1}")
    return np.asarray([by_doc[index] for index in doc_ids], dtype=np.float64)


def bootstrap_task(
    arrays: dict[str, np.ndarray], rng: np.random.Generator, resamples: int,
    groups: np.ndarray | None = None, chunk: int = 128,
) -> dict[str, np.ndarray]:
    names = list(arrays)
    n = len(next(iter(arrays.values())))
    draws = {name: [] for name in names}
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        if groups is None:
            indices = rng.integers(0, n, size=(size, n))
            for name in names:
                draws[name].extend(arrays[name][indices].mean(axis=1))
        else:
            group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
            task_draws = {name: np.zeros(size, dtype=np.float64) for name in names}
            for indices in group_indices:
                sampled = rng.integers(0, len(indices), size=(size, len(indices)))
                for name in names:
                    task_draws[name] += arrays[name][indices][sampled].mean(axis=1)
            for name in names:
                draws[name].extend(task_draws[name] / len(group_indices))
    return {name: np.asarray(values, dtype=np.float64) for name, values in draws.items()}


def mmlu_holdout_groups(path: Path, shots_per_subject: int = 5) -> np.ndarray:
    records = list(load_from_disk(str(path)))
    by_subject: dict[str, list[dict]] = {}
    for row in records:
        by_subject.setdefault(row["subject"], []).append(row)
    shot_keys = {
        (row["subject"], row["question"], tuple(row["choices"]), row["answer"])
        for rows in by_subject.values()
        for row in rows[:shots_per_subject]
    }
    holdout = [
        row for row in records
        if (row["subject"], row["question"], tuple(row["choices"]), row["answer"])
        not in shot_keys
    ]
    return np.asarray([row["subject"] for row in holdout], dtype=object)


def point_accuracy(values: np.ndarray, groups: np.ndarray | None = None) -> float:
    if groups is None:
        return float(values.mean())
    return float(np.mean([
        values[groups == group].mean() for group in np.unique(groups)
    ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--reference", default="terminal")
    parser.add_argument(
        "--mmlu-validation", type=Path, default=Path("dataset/mmlu/validation")
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.reference not in args.models:
        raise ValueError(f"reference {args.reference!r} is not in --models")

    manifest = json.loads(args.manifest.read_text())
    expected = {row["task"]: int(row["evaluation_count"]) for row in manifest["tasks"]}
    rng = np.random.default_rng(args.seed)
    points: dict[str, dict[str, float]] = {model: {} for model in args.models}
    ci95: dict[str, dict[str, list[float]]] = {model: {} for model in args.models}
    differences: dict[str, dict[str, dict[str, object]]] = {
        model: {} for model in args.models if model != args.reference
    }
    all_draws: dict[str, dict[str, np.ndarray]] = {}
    mmlu_groups = mmlu_holdout_groups(args.mmlu_validation)
    if len(mmlu_groups) != expected["mmlu"]:
        raise ValueError(
            f"MMLU group map has {len(mmlu_groups)} rows; expected {expected['mmlu']}"
        )

    for task, filename in TASK_FILES.items():
        arrays = {
            model: load_correct(
                args.artifact_dir / "runs" / f"{model}_validation10" / filename,
                expected[task],
            )
            for model in args.models
        }
        groups = mmlu_groups if task == "mmlu" else None
        draws = bootstrap_task(arrays, rng, args.resamples, groups=groups)
        all_draws[task] = draws
        for model in args.models:
            points[model][task] = point_accuracy(arrays[model], groups)
            ci95[model][task] = percentile(draws[model])
            if model != args.reference:
                delta = 100.0 * (
                    points[model][task] - points[args.reference][task]
                )
                delta_draws = 100.0 * (draws[model] - draws[args.reference])
                differences[model][task] = {
                    "delta_accuracy_pp": float(delta),
                    "ci95": percentile(delta_draws),
                }

    macro: dict[str, dict[str, object]] = {}
    for model in args.models:
        value = float(np.mean(list(points[model].values())))
        draws = np.mean(
            np.stack([all_draws[task][model] for task in TASK_FILES]), axis=0
        )
        macro[model] = {"accuracy": value, "ci95": percentile(draws)}
        if model != args.reference:
            ref_draws = np.mean(
                np.stack([all_draws[task][args.reference] for task in TASK_FILES]),
                axis=0,
            )
            differences[model]["macro_average"] = {
                "delta_accuracy_pp": 100.0 * (
                    value - float(np.mean(list(points[args.reference].values())))
                ),
                "ci95": percentile(100.0 * (draws - ref_draws)),
            }

    result = {
        "protocol": "terminal_format_validation10",
        "task_aggregation": {
            "mmlu": "macro_average_over_57_subject_accuracies",
            "other_tasks": "micro_accuracy",
            "cross_task": "unweighted_macro_average_over_10_tasks",
        },
        "reference_model": args.reference,
        "seed": args.seed,
        "bootstrap_resamples": args.resamples,
        "documents": {task: expected[task] for task in TASK_FILES},
        "accuracy": {
            model: {
                task: {"value": points[model][task], "ci95": ci95[model][task]}
                for task in TASK_FILES
            }
            for model in args.models
        },
        "macro_average": macro,
        "paired_difference_vs_reference": differences,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )

    fields = ["model", *TASK_FILES, "macro_average"]
    with (args.output_dir / "validation10_accuracy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in args.models:
            writer.writerow({
                "model": model,
                **points[model],
                "macro_average": macro[model]["accuracy"],
            })

    lines = [
        "# Terminal-format performance on Validation-10",
        "",
        "| Model | " + " | ".join(TASK_FILES) + " | Macro |",
        "|---|" + "---:|" * (len(TASK_FILES) + 1),
    ]
    for model in args.models:
        values = [f"{100.0 * points[model][task]:.2f}" for task in TASK_FILES]
        lines.append(
            f"| {model} | " + " | ".join(values) +
            f" | {100.0 * float(macro[model]['accuracy']):.2f} |"
        )
    lines.extend([
        "",
        f"Accuracy in percent. Bootstrap: {args.resamples} paired resamples, seed {args.seed}.",
    ])
    (args.output_dir / "validation10_accuracy.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"models": len(args.models), "tasks": len(TASK_FILES),
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
