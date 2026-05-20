"""Experiments 2-3: MC flip analysis and statistical tests.

Processes evaluation outputs from the _decomp evaluators to:
  - Exp 2: Quantify flip rates, accuracy deltas, per-task breakdown,
           binomial tests on flip direction asymmetry
  - Exp 3: Placeholder for per-token predictability gradient extension

Usage:
    python analysis/nonterminal_noise/run_analysis.py \
        --results_json /tmp/a800-tree-1B-decomp/eval_results.json \
        --output_dir analysis-output/exp2/a800_tree_1B \
        --model_name "Tree-1B (A800)"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger(__name__)

TASKS = [
    "hellaswag", "winogrande", "arc_easy", "arc_challenge",
    "piqa", "social_iqa", "commonsense_qa", "openbook_qa",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze decomposed MC evaluation results"
    )
    p.add_argument("--results_json", required=True,
                   help="Path to eval_results JSON from a _decomp evaluation run")
    p.add_argument("--output_dir", required=True,
                   help="Directory for output files and figures")
    p.add_argument("--model_name", default="unknown",
                   help="Model identifier for plot labels")
    p.add_argument("--compare_with", nargs="*", default=None,
                   help="Additional results JSONs for multi-model comparison plots")
    p.add_argument("--compare_names", nargs="*", default=None,
                   help="Names for comparison models")
    return p.parse_args()


def load_results(path):
    """Load evaluation results JSON."""
    with open(path) as f:
        return json.load(f)


def binomial_flip_test(n_flip_to_correct, n_flip_to_wrong):
    """Two-sided binomial test: H0 is flip directions are symmetric."""
    n = n_flip_to_correct + n_flip_to_wrong
    if n == 0:
        return 1.0
    result = stats.binomtest(n_flip_to_correct, n, p=0.5, alternative="two-sided")
    return result.pvalue


def extract_metrics(results):
    """Extract per-task metrics from evaluation results."""
    rows = []
    for task in TASKS:
        key = f"{task}_decomp"
        if key not in results:
            log.warning(f"Task '{key}' not found in results, skipping")
            continue

        r = results[key]
        acc_full = r.get("_", float("nan"))
        acc_term = r.get("_term", float("nan"))
        flip_rate = r.get("_flip_rate", float("nan"))
        ftc_rate = r.get("_flip_to_correct", 0.0)
        ftw_rate = r.get("_flip_to_wrong", 0.0)
        total = int(r.get("_total", 0))

        ftc = int(ftc_rate * total)
        ftw = int(ftw_rate * total)
        delta = acc_term - acc_full
        p_binom = binomial_flip_test(ftc, ftw)

        rows.append({
            "task": task,
            "acc_full": acc_full,
            "acc_term": acc_term,
            "delta": delta,
            "flip_rate": flip_rate,
            "flip_to_correct": ftc,
            "flip_to_wrong": ftw,
            "total": total,
            "p_binomial": p_binom,
        })
    return rows


def summarize(rows, model_name):
    """Compute summary statistics across tasks."""
    deltas = [r["delta"] for r in rows if not np.isnan(r["delta"])]
    flips = [r["flip_rate"] for r in rows if not np.isnan(r["flip_rate"])]
    ftc_total = sum(r["flip_to_correct"] for r in rows)
    ftw_total = sum(r["flip_to_wrong"] for r in rows)

    return {
        "model": model_name,
        "n_tasks": len(rows),
        "mean_accuracy_delta": float(np.mean(deltas)) if deltas else None,
        "std_accuracy_delta": float(np.std(deltas)) if deltas else None,
        "mean_flip_rate": float(np.mean(flips)) if flips else None,
        "std_flip_rate": float(np.std(flips)) if flips else None,
        "total_flip_to_correct": ftc_total,
        "total_flip_to_wrong": ftw_total,
        "aggregate_p_binomial": binomial_flip_test(ftc_total, ftw_total),
        "tasks": rows,
    }


def print_report(summary):
    """Print a formatted report to stdout."""
    print(f"\n{'='*70}")
    print(f"  Model: {summary['model']}")
    print(f"  Tasks evaluated: {summary['n_tasks']}")
    print(f"  Mean accuracy delta (term - full): {summary['mean_accuracy_delta']:+.4f}")
    print(f"  Mean flip rate:                     {summary['mean_flip_rate']:.4f}")
    print(f"  Total flip-to-correct:  {summary['total_flip_to_correct']}")
    print(f"  Total flip-to-wrong:    {summary['total_flip_to_wrong']}")
    print(f"  Aggregate p(binomial):  {summary['aggregate_p_binomial']:.4f}")
    print(f"{'='*70}\n")

    print(f"{'Task':<20} {'Acc(full)':>10} {'Acc(term)':>10} {'Delta':>8} {'Flip%':>8} {'FTC':>5} {'FTW':>5} {'p(binom)':>10}")
    print("-" * 78)
    for r in summary["tasks"]:
        print(f"{r['task']:<20} {r['acc_full']:>10.4f} {r['acc_term']:>10.4f} "
              f"{r['delta']:>+8.4f} {r['flip_rate']:>8.4f} "
              f"{r['flip_to_correct']:>5d} {r['flip_to_wrong']:>5d} "
              f"{r['p_binomial']:>10.4f}")
    print("-" * 78)


def plot_results(all_summaries, output_dir):
    """Generate figures comparing model results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper",
                  palette="colorblind", font_scale=1.1)
    out = Path(output_dir)

    models = [s["model"] for s in all_summaries]
    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#5B9BD5"]

    # ---- Figure 1: Accuracy delta bar plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Per-task accuracy delta
    ax = axes[0]
    task_data = {}
    for s in all_summaries:
        for r in s["tasks"]:
            task_data.setdefault(r["task"], {})[s["model"]] = r["delta"]

    task_list = sorted(task_data.keys())
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(len(task_list))

    for i, s in enumerate(all_summaries):
        deltas = [task_data[t].get(s["model"], 0) for t in task_list]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=s["model"],
                      color=colors[i % len(colors)])
        # Highlight significant tasks (flip rate > 0)
        for j, (t, d) in enumerate(zip(task_list, deltas)):
            if d > 0.005:
                ax.text(x[j] + offset, d + 0.002, "*", ha="center",
                        fontsize=14, fontweight="bold", color=colors[i % len(colors)])

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in task_list], fontsize=8)
    ax.set_ylabel("Accuracy Delta (term − full)")
    ax.set_title("Per-Task Accuracy Gain from Terminal-Only Scoring")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # Panel B: Flip rate comparison (swarm + mean)
    ax = axes[1]
    all_flips = []
    all_labels = []
    for s in all_summaries:
        for r in s["tasks"]:
            if not np.isnan(r["flip_rate"]):
                all_flips.append(r["flip_rate"])
                all_labels.append(s["model"])

    # Grouped scatter
    for i, s in enumerate(all_summaries):
        flips = [r["flip_rate"] for r in s["tasks"] if not np.isnan(r["flip_rate"])]
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(flips))
        ax.scatter(np.full(len(flips), i) + jitter, flips,
                   alpha=0.5, s=40, color=colors[i % len(colors)])
        mean_f = np.mean(flips)
        ax.scatter([i], [mean_f], color="red", s=120, zorder=5,
                   marker="X", edgecolors="darkred", linewidth=1)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Flip Rate")
    ax.set_title("Per-Task Flip Rates (full vs term scoring)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(bottom=-0.01)

    fig.suptitle("Non-Terminal Noise in MC Benchmarks", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "figure_accuracy_delta.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved Figure 1 to {out / 'figure_accuracy_delta.pdf'}")

    # ---- Figure 2: Flip direction asymmetry ----
    if len(all_summaries) == 1:
        s = all_summaries[0]
        tasks_with_data = [r for r in s["tasks"]
                           if (r["flip_to_correct"] + r["flip_to_wrong"]) > 0]

        if tasks_with_data:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(tasks_with_data))
            width = 0.35

            ftc_vals = [r["flip_to_correct"] for r in tasks_with_data]
            ftw_vals = [r["flip_to_wrong"] for r in tasks_with_data]

            ax.bar(x - width / 2, ftc_vals, width, label="Flip to Correct",
                   color="#4472C4")
            ax.bar(x + width / 2, ftw_vals, width, label="Flip to Wrong",
                   color="#ED7D31")

            # Binomial significance stars
            for i, r in enumerate(tasks_with_data):
                if r["p_binomial"] < 0.05:
                    y = max(ftc_vals[i], ftw_vals[i]) + 1
                    marker = "**" if r["p_binomial"] < 0.01 else "*"
                    ax.text(i, y, marker, ha="center", fontsize=14, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels([r["task"].replace("_", "\n") for r in tasks_with_data],
                               fontsize=8)
            ax.set_ylabel("Number of Flips")
            ax.set_title(f"Flip Direction Asymmetry — {s['model']}")
            ax.legend()

            fig.tight_layout()
            fig.savefig(out / "figure_flip_asymmetry.pdf", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            log.info(f"Saved Figure 2 to {out / 'figure_flip_asymmetry.pdf'}")

    # ---- Figure 3: Model comparison (if multiple models) ----
    if len(all_summaries) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(models))
        width = 0.35

        means = [s["mean_accuracy_delta"] for s in all_summaries]
        stds = [s["std_accuracy_delta"] for s in all_summaries]
        flip_means = [s["mean_flip_rate"] for s in all_summaries]

        bars1 = ax.bar(x - width / 2, means, width, label="Mean Acc. Delta",
                       yerr=stds, capsize=5, color="#4472C4")
        ax.set_ylabel("Accuracy Delta (term − full)")
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width / 2, flip_means, width,
                        label="Mean Flip Rate", color="#ED7D31")
        ax2.set_ylabel("Flip Rate")

        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title("Model Comparison: Non-Terminal Noise Impact")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        fig.tight_layout()
        fig.savefig(out / "figure_model_comparison.pdf", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved Figure 3 to {out / 'figure_model_comparison.pdf'}")


def main():
    args = parse_args()

    # Load primary results
    results = load_results(args.results_json)
    rows = extract_metrics(results)
    summary = summarize(rows, args.model_name)
    print_report(summary)

    all_summaries = [summary]

    # Load comparison results if provided
    if args.compare_with:
        for path, name in zip(args.compare_with,
                              args.compare_names or args.compare_with):
            comp_results = load_results(path)
            comp_rows = extract_metrics(comp_results)
            comp_summary = summarize(comp_rows, name)
            print_report(comp_summary)
            all_summaries.append(comp_summary)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Saved summary to {out / 'summary.json'}")

    # Generate figures
    plot_results(all_summaries, args.output_dir)

    return summary


if __name__ == "__main__":
    main()
