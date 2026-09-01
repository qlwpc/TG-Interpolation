"""Reproduce and validate compiled FlexAttention backward at head_dim=128.

This is intentionally independent of the third-party ``flash-attn`` package.
It exercises ``torch.nn.attention.flex_attention`` on the same Ampere-class
shape reported to fail during backward, then writes a small JSON evidence file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

import torch
import triton
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


REPO = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO / "validation" / "slurm" / "results"


def causal_mask(_batch, _head, query_index, key_index):
    return query_index >= key_index


def tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor).all().item()),
        "abs_max": float(tensor.float().abs().max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--sequence-length", type=int, default=2048)
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    result_path = RESULT_DIR / f"flex-attention-128-{args.label}_{job_id}.json"
    result: dict[str, object] = {
        "label": args.label,
        "job_id": job_id,
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "python": os.sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "shape": [1, 16, args.sequence_length, 128],
        "dtype": "torch.bfloat16",
        "num_stages": args.num_stages,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the validation job")

        torch.manual_seed(20260901)
        tensors = [
            torch.randn(
                1,
                16,
                args.sequence_length,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        grad_output = torch.randn_like(tensors[0])
        block_mask = create_block_mask(
            causal_mask,
            B=1,
            H=1,
            Q_LEN=args.sequence_length,
            KV_LEN=args.sequence_length,
            device="cuda",
        )
        kernel_options = None
        if args.num_stages is not None:
            kernel_options = {
                "fwd_num_stages": args.num_stages,
                "bwd_num_stages": args.num_stages,
            }

        compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
        started = time.perf_counter()
        output = compiled_flex_attention(
            *tensors,
            block_mask=block_mask,
            kernel_options=kernel_options,
        )
        output.backward(grad_output)
        torch.cuda.synchronize()
        result["elapsed_seconds"] = time.perf_counter() - started
        result["output"] = tensor_summary(output)
        result["gradients"] = [tensor_summary(tensor.grad) for tensor in tensors]
        if not result["output"]["finite"] or not all(
            gradient["finite"] for gradient in result["gradients"]
        ):
            raise RuntimeError("FlexAttention produced a non-finite output or gradient")
        result["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        exit_code = 1

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"result_path={result_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
