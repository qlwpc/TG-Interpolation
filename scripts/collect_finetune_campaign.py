#!/usr/bin/env python3
"""Collect Slurm states and final XSum/BoolQ metrics for the multi-seed campaign."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


PAPER_REFERENCE = {
    "terminal": {"xsum_finetune": 21.99, "boolq": 67.40},
    "tree": {"xsum_finetune": 22.82, "boolq": 67.86},
    "tgtree": {"xsum_finetune": 22.69, "boolq": 68.38},
    "tree_noont": {"xsum_finetune": 22.29, "boolq": 68.13},
    "tree_compress": {"xsum_finetune": 22.63, "boolq": 68.04},
    "tree_triplecnt": {"xsum_finetune": 21.18, "boolq": 69.02},
}
MODEL_ORDER = [
    "terminal",
    "tgtree",
    "tree",
    "tree_noont",
    "tree_triplecnt",
    "tree_compress",
]
TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
}
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=Path("artifacts/experiment/finetune_multiseed_20260803"),
    )
    return parser.parse_args()


def slurm_states(job_ids: list[int]) -> dict[int, dict[str, str]]:
    command = [
        "sacct",
        "-X",
        "-j",
        ",".join(map(str, job_ids)),
        "--noheader",
        "--parsable2",
        "--format=JobIDRaw,State,ExitCode,Elapsed,Start,End",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    states: dict[int, dict[str, str]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        job_id = int(parts[0])
        states[job_id] = {
            "state": parts[1].split()[0],
            "exit_code": parts[2],
            "elapsed": parts[3],
            "start": parts[4],
            "end": parts[5],
        }
    return states


def last_float(text: str, key: str) -> float | None:
    matches = re.findall(
        rf"(?m)(?:^|\s){re.escape(key)}(?:=|\s+)({FLOAT})(?:\s|$)", text
    )
    return float(matches[-1]) if matches else None


def parse_metrics(task: str, text: str) -> tuple[dict[str, float], dict[str, object]]:
    details: dict[str, object] = {}
    if task == "boolq":
        values = re.findall(
            rf"eval/downstream/boolq_acc__(?:=|\s+)({FLOAT})(?:\s|$)", text
        )
        metrics = {"boolq_accuracy": float(values[-1])} if values else {}
        details["boolq_metric_occurrences"] = len(values)
        return metrics, details

    metrics = {}
    for key, output_key in (
        ("rouge1", "rouge1"),
        ("rouge2", "rouge2"),
        ("rougeL", "rougeL"),
        ("R-AVG", "r_avg"),
    ):
        value = last_float(text, key)
        if value is not None:
            metrics[output_key] = value
    if all(key in metrics for key in ("rouge1", "rouge2", "rougeL")):
        calculated = statistics.mean(metrics[key] for key in ("rouge1", "rouge2", "rougeL"))
        details["calculated_r_avg"] = calculated
        details["reported_r_avg_delta"] = (
            metrics.get("r_avg", calculated) - calculated
        )
    details["new_passage_records"] = text.count("<New Passage>")
    return metrics, details


def metric_for_summary(row: dict[str, object]) -> float | None:
    metrics = row["metrics"]
    if row["task"] == "boolq":
        return metrics.get("boolq_accuracy")
    return metrics.get("r_avg")


def render_report(rows: list[dict[str, object]]) -> str:
    state_counts = Counter(row["slurm"]["state"] for row in rows)
    completed_metrics = sum(metric_for_summary(row) is not None for row in rows)
    lines = [
        "# Multi-seed XSum and BoolQ results",
        "",
        f"- Runs in manifest: {len(rows)}",
        f"- Runs with canonical metrics: {completed_metrics}",
        f"- Slurm states: {dict(sorted(state_counts.items()))}",
        "- Metric values in tables are percentages (raw evaluator values multiplied by 100).",
        "",
    ]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["task"])].append(row)

    for task, title in (("boolq", "BoolQ validation accuracy"), ("xsum_finetune", "XSum R-AVG")):
        lines.extend([f"## {title}", "", "| Model | Seed metrics | Mean ± SD | Paper | Delta |", "|---|---:|---:|---:|---:|"])
        for model in MODEL_ORDER:
            model_rows = sorted(grouped[(model, task)], key=lambda row: row["seed"])
            values = [metric_for_summary(row) for row in model_rows]
            shown = ["—" if value is None else f"{100 * value:.3f}" for value in values]
            finite = [100 * value for value in values if value is not None and math.isfinite(value)]
            if finite:
                mean = statistics.mean(finite)
                sd = statistics.stdev(finite) if len(finite) > 1 else 0.0
                summary = f"{mean:.3f} ± {sd:.3f}"
                paper = PAPER_REFERENCE[model][task]
                delta = f"{mean - paper:+.3f}"
            else:
                summary = delta = "—"
                paper = PAPER_REFERENCE[model][task]
            lines.append(
                f"| {model} | {', '.join(shown)} | {summary} | {paper:.2f} | {delta} |"
            )
        lines.append("")

    failed = [row for row in rows if row["slurm"]["state"] in TERMINAL_STATES - {"COMPLETED"}]
    lines.extend(["## Run audit", ""])
    if failed:
        for row in failed:
            lines.append(
                f"- `{row['run_name']}`: {row['slurm']['state']} "
                f"(exit {row['slurm']['exit_code']}), log `{row['log_path']}`"
            )
    else:
        lines.append("- No terminal failures recorded.")
    lines.extend(
        [
            "",
            "Raw per-run metrics, job timing, configs, checkpoints, and evaluation-log paths are in `results.json` and `results.csv`. Full passage/prediction and evaluator details remain in each run's Slurm log.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    campaign_dir = parse_args().campaign_dir.resolve()
    jobs = json.loads((campaign_dir / "jobs.json").read_text())
    states = slurm_states([int(job["job_id"]) for job in jobs])
    rows = []
    for job in jobs:
        job_id = int(job["job_id"])
        run_dir = Path(job["config"]).parent
        logs = sorted((run_dir / "logs").glob(f"*-{job_id}.out"))
        log_path = logs[-1] if logs else run_dir / "logs" / f"missing-{job_id}.out"
        text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        metrics, evaluation_details = parse_metrics(job["task"], text)
        rows.append(
            {
                **job,
                "log_path": str(log_path),
                "log_bytes": log_path.stat().st_size if log_path.is_file() else 0,
                "slurm": states.get(
                    job_id,
                    {"state": "UNKNOWN", "exit_code": "", "elapsed": "", "start": "", "end": ""},
                ),
                "metrics": metrics,
                "evaluation_details": evaluation_details,
            }
        )

    (campaign_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "run_name",
            "model",
            "task",
            "seed",
            "job_id",
            "state",
            "exit_code",
            "elapsed",
            "boolq_accuracy",
            "rouge1",
            "rouge2",
            "rougeL",
            "r_avg",
            "log_path",
            "config",
            "checkpoint",
            "output_dir",
        ],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "run_name": row["run_name"],
                "model": row["model"],
                "task": row["task"],
                "seed": row["seed"],
                "job_id": row["job_id"],
                "state": row["slurm"]["state"],
                "exit_code": row["slurm"]["exit_code"],
                "elapsed": row["slurm"]["elapsed"],
                "boolq_accuracy": row["metrics"].get("boolq_accuracy", ""),
                "rouge1": row["metrics"].get("rouge1", ""),
                "rouge2": row["metrics"].get("rouge2", ""),
                "rougeL": row["metrics"].get("rougeL", ""),
                "r_avg": row["metrics"].get("r_avg", ""),
                "log_path": row["log_path"],
                "config": row["config"],
                "checkpoint": row["checkpoint"],
                "output_dir": row["output_dir"],
            }
        )
    (campaign_dir / "results.csv").write_text(output.getvalue())
    (campaign_dir / "REPORT.md").write_text(render_report(rows))
    print(f"collected {len(rows)} runs; canonical metrics present for {sum(metric_for_summary(row) is not None for row in rows)}")


if __name__ == "__main__":
    main()
