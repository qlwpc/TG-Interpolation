"""Validate compiled FlexAttention backward around the 128-token boundary.

Each invocation runs exactly one shape because a CUDA device-side assertion can
poison the process.  The defaults follow the upstream ``N < 128`` report:
multiple attention heads, a per-head BlockMask, and BF16 Q/K/V tensors.  Both
the BlockMask builder and FlexAttention are compiled, matching the reproducer.
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
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--block-mask-heads", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--key-value-heads", type=int)
    parser.add_argument("--key-value-length", type=int)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--mask-mode", choices=("causal", "eagle3"), default="causal")
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    if args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")
    if args.block_mask_heads <= 0:
        parser.error("--block-mask-heads must be positive")
    if args.block_mask_heads not in (1, args.query_heads):
        parser.error("--block-mask-heads must be 1 or equal --query-heads")
    key_value_heads = args.key_value_heads or args.query_heads
    key_value_length = args.key_value_length or args.sequence_length
    if key_value_heads <= 0 or args.query_heads % key_value_heads != 0:
        parser.error("--key-value-heads must be a positive divisor of --query-heads")
    if key_value_length <= 0:
        parser.error("--key-value-length must be positive")
    if args.num_stages is not None and args.num_stages <= 0:
        parser.error("--num-stages must be positive")

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    result_path = RESULT_DIR / f"flex-attention-short-sequence-{args.label}_{job_id}.json"
    query_shape = [args.batch_size, args.query_heads, args.sequence_length, args.head_dim]
    key_value_shape = [args.batch_size, key_value_heads, key_value_length, args.head_dim]
    dtype = getattr(torch, args.dtype)
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
        "query_shape": query_shape,
        "key_value_shape": key_value_shape,
        "dtype": str(dtype),
        "block_mask_heads": args.block_mask_heads,
        "block_size": 128,
        "block_mask_compiled": True,
        "enable_gqa": key_value_heads != args.query_heads,
        "mask_mode": args.mask_mode,
        "num_stages": args.num_stages,
        "forward_only": args.forward_only,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the validation job")

        torch.manual_seed(20260901)
        query = torch.randn(*query_shape, device="cuda", dtype=dtype, requires_grad=True)
        key = torch.randn(*key_value_shape, device="cuda", dtype=dtype, requires_grad=True)
        value = torch.randn(*key_value_shape, device="cuda", dtype=dtype, requires_grad=True)
        tensors = [query, key, value]
        grad_output = torch.randn_like(query)

        mask_mod = causal_mask
        if args.mask_mode == "eagle3":
            sequence_lengths = torch.tensor(
                [args.sequence_length - 30] * args.batch_size,
                device="cuda",
                dtype=torch.int32,
            )

            def eagle3_mask(batch, _head, query_index, key_index):
                causal = (query_index - 4 >= key_index) & (key_index < sequence_lengths[batch])
                suffix = (
                    (key_index >= args.sequence_length)
                    & (key_index % args.sequence_length < sequence_lengths[batch])
                    & ((key_index - query_index) % args.sequence_length == 0)
                )
                return causal | suffix

            mask_mod = eagle3_mask

        phase_started = time.perf_counter()
        result["phase"] = "create_block_mask"
        print("phase=create_block_mask", flush=True)
        compiled_create_block_mask = torch.compile(create_block_mask)
        block_mask = compiled_create_block_mask(
            mask_mod,
            B=args.batch_size,
            H=args.block_mask_heads,
            Q_LEN=args.sequence_length,
            KV_LEN=key_value_length,
            device="cuda",
            BLOCK_SIZE=128,
        )
        torch.cuda.synchronize()
        result["block_mask_seconds"] = time.perf_counter() - phase_started

        result["phase"] = "compiled_forward_backward"
        print("phase=compiled_forward", flush=True)
        compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
        kernel_options = None
        if args.num_stages is not None:
            kernel_options = {
                "fwd_num_stages": args.num_stages,
                "bwd_num_stages": args.num_stages,
            }
        phase_started = time.perf_counter()
        output = compiled_flex_attention(
            *tensors,
            block_mask=block_mask,
            enable_gqa=key_value_heads != args.query_heads,
            kernel_options=kernel_options,
        )
        torch.cuda.synchronize()
        result["forward_seconds"] = time.perf_counter() - phase_started
        if not args.forward_only:
            result["phase"] = "compiled_backward"
            print("phase=compiled_backward", flush=True)
            phase_started = time.perf_counter()
            output.backward(grad_output)
            torch.cuda.synchronize()
            result["backward_seconds"] = time.perf_counter() - phase_started
        result["output"] = tensor_summary(output)
        if not args.forward_only:
            result["gradients"] = [tensor_summary(tensor.grad) for tensor in tensors]
        if not result["output"]["finite"] or (
            not args.forward_only
            and not all(gradient["finite"] for gradient in result["gradients"])
        ):
            raise RuntimeError("FlexAttention produced a non-finite output or gradient")
        result["phase"] = "complete"
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
