"""Parse durable Pushdown Slurm logs into comparable timing/throughput JSON."""

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
FIX2_RE = re.compile(r"\[fix2_bwd\] (?P<parts>.+)")
PART_RE = re.compile(r"(?P<name>[a-z_]+)=(?P<value>[0-9.]+)ms")
ATTACHMENT_FORWARD_RE = re.compile(r"\[pushdown_attachment_forward\] ms=([0-9.]+)")
FORWARD_RE = re.compile(
    r"\[pushdown_forward\] model_including_attachment_ms=([0-9.]+) "
    r"lm_and_attachment_loss_ms=([0-9.]+)"
)
BACKWARD_RE = re.compile(r"\[pushdown_backward\] total_ms=([0-9.]+)")
NONFINITE_RE = re.compile(r"\[pushdown_nonfinite_grads\] count=(\d+)")


def _values(pattern: re.Pattern[str], text: str) -> list[float]:
    return [float(value.replace(",", "")) for value in pattern.findall(text)]


def parse_profile(path: Path, first_step: int = 3) -> dict:
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
    fix2_rows = [
        {m.group("name"): float(m.group("value")) for m in PART_RE.finditer(line)}
        for line in FIX2_RE.findall(text)
    ]
    fix2_medians = {
        key: statistics.median(row[key] for row in fix2_rows if key in row)
        for key in sorted({key for row in fix2_rows for key in row})
    }
    tokens = _values(TOKENS_RE, text)
    batches = _values(BATCHES_RE, text)
    memory = _values(MEMORY_RE, text)
    attachment_forward = _values(ATTACHMENT_FORWARD_RE, text)
    forward_rows = [tuple(map(float, row)) for row in FORWARD_RE.findall(text)]
    backward = _values(BACKWARD_RE, text)
    nonfinite = [int(value) for value in NONFINITE_RE.findall(text)]
    # The same warmup convention used for step rows: phase probe call i maps to
    # training step i, so drop the first `first_step - 1` values.
    phase_offset = max(0, first_step - 1)
    attachment_forward = attachment_forward[phase_offset:]
    forward_rows = forward_rows[phase_offset:]
    backward = backward[phase_offset:]
    return {
        "log_path": str(path),
        "completed_steps": len(steps),
        "measurement_first_step": first_step,
        "measured_step_count": len(measured),
        "step_median_ms": (
            statistics.median(row["step_total_ms"] for row in measured)
            if measured else None
        ),
        "gpu_train_median_ms": (
            statistics.median(row["gpu_train_ms"] for row in measured)
            if measured else None
        ),
        "final_tokens_per_second_per_device": tokens[-1] if tokens else None,
        "final_batches_per_second_per_device": batches[-1] if batches else None,
        "peak_gpu_memory_mb": max(memory) if memory else None,
        "fix2_backward_calls": len(fix2_rows),
        "fix2_phase_median_ms_per_layer": fix2_medians,
        "attachment_forward_median_ms": (
            statistics.median(attachment_forward) if attachment_forward else None
        ),
        "model_forward_median_ms": (
            statistics.median(row[0] for row in forward_rows) if forward_rows else None
        ),
        "lm_and_attachment_loss_median_ms": (
            statistics.median(row[1] for row in forward_rows) if forward_rows else None
        ),
        "backward_median_ms": statistics.median(backward) if backward else None,
        "max_nonfinite_gradient_parameter_count": max(nonfinite) if nonfinite else None,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--first-step", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = parse_profile(args.log, args.first_step)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
