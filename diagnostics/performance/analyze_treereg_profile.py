"""Parse durable TreeReg Slurm logs into comparable timing/throughput JSON."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


STEP_RE = re.compile(
    r"\[step_profile\] step=(?P<step>\d+) "
    r"data_ms=(?P<data>[0-9.]+) "
    r"step_total_ms=(?P<total>[0-9.]+) "
    r"gpu_tr_ms=(?P<gpu>[0-9.]+)"
)
TOKENS_RE = re.compile(r"throughput/device/tokens_per_second=([0-9,.]+)")
BATCHES_RE = re.compile(r"throughput/device/batches_per_second=([0-9,.]+)")
MEMORY_RE = re.compile(r"System/Peak GPU Memory \(MB\)=([0-9,.]+)")


def _values(pattern: re.Pattern[str], text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in pattern.findall(text)]


def parse_profile(path: Path, treereg_every_k: int, first_step: int = 3) -> dict:
    text = path.read_text(errors="replace")
    steps = [
        {
            "step": int(match.group("step")),
            "data_ms": float(match.group("data")),
            "step_total_ms": float(match.group("total")),
            "gpu_train_ms": float(match.group("gpu")),
        }
        for match in STEP_RE.finditer(text)
    ]
    measured = [row for row in steps if row["step"] >= first_step]
    treereg = [
        row
        for row in measured
        if treereg_every_k <= 0 or row["step"] % treereg_every_k == 0
    ]
    ordinary = [row for row in measured if row not in treereg]

    def median(rows: list[dict], key: str) -> float | None:
        return statistics.median(row[key] for row in rows) if rows else None

    ordinary_median = median(ordinary, "step_total_ms")
    treereg_median = median(treereg, "step_total_ms")
    tokens = _values(TOKENS_RE, text)
    batches = _values(BATCHES_RE, text)
    memory = _values(MEMORY_RE, text)
    result = {
        "log_path": str(path),
        "completed_steps": len(steps),
        "measurement_first_step": first_step,
        "treereg_every_k": treereg_every_k,
        "ordinary_step_count": len(ordinary),
        "treereg_step_count": len(treereg),
        "ordinary_step_median_ms": ordinary_median,
        "treereg_step_median_ms": treereg_median,
        "treereg_incremental_median_ms": (
            treereg_median - ordinary_median
            if treereg_median is not None and ordinary_median is not None
            else None
        ),
        "measured_step_mean_ms": (
            statistics.mean(row["step_total_ms"] for row in measured)
            if measured
            else None
        ),
        "final_tokens_per_second_per_device": tokens[-1] if tokens else None,
        "final_batches_per_second_per_device": batches[-1] if batches else None,
        "peak_gpu_memory_mb": max(memory) if memory else None,
        "steps": steps,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--treereg-every-k", type=int, default=10)
    parser.add_argument("--first-step", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse_profile(args.log, args.treereg_every_k, args.first_step)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
