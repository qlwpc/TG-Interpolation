#!/usr/bin/env python3
"""Fail closed when an OLMo checkpoint contains non-finite or extreme tensors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--max-abs", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported checkpoint payload: {type(payload)!r}")
    return {name: value for name, value in payload.items() if torch.is_tensor(value)}


def main() -> int:
    args = parse_args()
    current = state_dict(args.model)
    reference = state_dict(args.reference) if args.reference else None
    total = 0
    nonfinite = 0
    sumsq = 0.0
    delta_sumsq = 0.0
    reference_sumsq = 0.0
    worst_name = None
    worst_abs = 0.0
    extreme_tensors: list[dict[str, object]] = []

    for name, tensor in current.items():
        value = tensor.detach().float()
        finite = torch.isfinite(value)
        total += value.numel()
        nonfinite += int((~finite).sum().item())
        finite_value = value[finite]
        tensor_max = float(finite_value.abs().max().item()) if finite_value.numel() else math.inf
        if tensor_max > worst_abs:
            worst_abs = tensor_max
            worst_name = name
        count_extreme = int((finite_value.abs() > args.max_abs).sum().item())
        if count_extreme or not finite.all():
            extreme_tensors.append({"name": name, "max_abs": tensor_max, "count_above_limit": count_extreme})
        sumsq += float(finite_value.double().square().sum().item())
        if reference is not None:
            if name not in reference or reference[name].shape != tensor.shape:
                raise ValueError(f"reference tensor mismatch: {name}")
            ref = reference[name].detach().float()
            delta_sumsq += float((value - ref).double().square().sum().item())
            reference_sumsq += float(ref.double().square().sum().item())

    healthy = nonfinite == 0 and worst_abs <= args.max_abs
    report: dict[str, object] = {
        "model": str(args.model),
        "healthy": healthy,
        "tensor_count": len(current),
        "parameter_count": total,
        "nonfinite_count": nonfinite,
        "max_abs": worst_abs,
        "max_abs_tensor": worst_name,
        "max_abs_limit": args.max_abs,
        "l2_norm": math.sqrt(sumsq),
        "extreme_tensors": extreme_tensors,
    }
    if reference is not None:
        report["reference"] = str(args.reference)
        report["delta_l2_norm"] = math.sqrt(delta_sumsq)
        report["relative_delta_l2"] = math.sqrt(delta_sumsq / reference_sumsq) if reference_sumsq else None
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
