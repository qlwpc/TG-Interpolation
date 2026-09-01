"""Benchmark FlexAttention against production-style SDPA on one fixed shape.

The benchmark separates FlexAttention's one-time compile cost, per-forward
BlockMask construction, and steady-state kernel time.  Each process runs only
one sequence length, mask mode, and workload so the cold timing is not polluted
by an earlier compiled shape.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch.nn.functional as F
import triton
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


REPO = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO / "validation" / "slurm" / "results"


def summarize(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def cuda_average_ms(step, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        step()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("causal", "structured", "tg", "tgnomask"),
        required=True,
    )
    parser.add_argument("--workload", choices=("eval", "train"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()

    if min(args.sequence_length, args.batch_size, args.heads, args.head_dim, args.layers) <= 0:
        parser.error("shape and layer arguments must be positive")
    if min(args.warmup, args.repeats) <= 0:
        parser.error("warmup and repeats must be positive")

    if args.iterations is None:
        if args.sequence_length <= 128:
            args.iterations = 100 if args.workload == "eval" else 40
        elif args.sequence_length <= 512:
            args.iterations = 30 if args.workload == "eval" else 12
        else:
            args.iterations = 8 if args.workload == "eval" else 4

    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    result_path = RESULT_DIR / f"flex-vs-sdpa-{args.label}-{args.workload}_{job_id}.json"
    shape = [args.batch_size, args.heads, args.sequence_length, args.head_dim]
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
        "shape": shape,
        "dtype": "torch.bfloat16",
        "mode": args.mode,
        "workload": args.workload,
        "layers": args.layers,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "iterations_per_repeat": args.iterations,
    }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the benchmark job")

        torch.manual_seed(20260901)
        requires_grad = args.workload == "train"
        q = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)
        k = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)
        v = torch.randn(*shape, device="cuda", dtype=torch.bfloat16, requires_grad=requires_grad)
        grad_output = torch.randn_like(q) if requires_grad else None

        positions = torch.arange(args.sequence_length, device="cuda")
        query_index = positions[:, None]
        key_index = positions[None, :]
        dense_bool = None
        additive_mask = None

        if args.mode == "causal":
            def mask_mod(_batch, _head, q_idx, kv_idx):
                return q_idx >= kv_idx

            block_mask_heads = 1
        elif args.mode == "structured":
            # Per-head causal windows approximate the heterogeneous support of TG
            # masks while guaranteeing that every query has at least one key.
            window = 64 * (1 + torch.arange(args.heads, device="cuda") % 4)
            per_head = (
                (query_index >= key_index).unsqueeze(0)
                & ((query_index - key_index).unsqueeze(0) <= window[:, None, None])
            )
            dense_bool = per_head.unsqueeze(0).expand(args.batch_size, -1, -1, -1).contiguous()
            # OLMoBlock._cast_attn_bias converts the additive bias to the Q/K/V
            # dtype before calling SDPA. Reproduce that production behavior.
            additive_mask = torch.zeros_like(dense_bool, dtype=q.dtype)
            additive_mask.masked_fill_(~dense_bool, torch.finfo(q.dtype).min)

            def mask_mod(batch, head, q_idx, kv_idx):
                return dense_bool[batch, head, q_idx, kv_idx]

            block_mask_heads = args.heads
        else:
            import numpy as np

            from olmo.data import KProximal_TG_attention_bias, TG_attention_bias

            token_data = np.load(REPO / "dataset" / "bbc-news" / "tg" / "test.npy", mmap_mode="r")
            masks = []
            for batch_index in range(args.batch_size):
                start = batch_index * 2048
                tokens = torch.from_numpy(
                    np.array(token_data[start : start + args.sequence_length], copy=True)
                ).long()
                if args.mode == "tg":
                    generator = TG_attention_bias(
                        str(REPO / "dataset" / "bbc-news" / "TG_GPT2_tokenizer.json"),
                        2048,
                    )
                else:
                    generator = KProximal_TG_attention_bias(
                        str(REPO / "dataset" / "bbc-news" / "TG_GPT2_tokenizer.json"),
                        2048,
                        2048,
                        False,
                    )
                mask, _ = generator(tokens)
                masks.append(mask.to(dtype=torch.bool))
            dense_bool = torch.stack(masks, dim=0).unsqueeze(1).to(device="cuda")
            additive_mask = torch.zeros_like(dense_bool, dtype=q.dtype)
            additive_mask.masked_fill_(~dense_bool, torch.finfo(q.dtype).min)
            flex_mask = dense_bool.squeeze(1)

            def mask_mod(batch, _head, q_idx, kv_idx):
                # Match OLMo.forward exactly: indexing a squeezed (B, Q, KV)
                # mask is vmap-safe, while mixing the batched indices with a
                # literal head index on (B, 1, Q, KV) is not in torch 2.7.
                return flex_mask[batch, q_idx, kv_idx]

            block_mask_heads = 1

        if dense_bool is not None:
            result["mask_density"] = float(dense_bool.float().mean().item())

        def sdpa_forward():
            if args.mode == "causal":
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask)

        block_started = time.perf_counter()
        block_mask = create_block_mask(
            mask_mod,
            B=args.batch_size,
            H=block_mask_heads,
            Q_LEN=args.sequence_length,
            KV_LEN=args.sequence_length,
            device="cuda",
            BLOCK_SIZE=128,
        )
        torch.cuda.synchronize()
        result["block_mask_build_cold_ms"] = (time.perf_counter() - block_started) * 1000

        # create_block_mask runs once per OLMo.forward and is reused by all
        # transformer layers. Separate its first-call setup from steady rebuilds.
        block_build_iterations = 20 if args.sequence_length <= 512 else 5
        for _ in range(2):
            block_mask = create_block_mask(
                mask_mod,
                B=args.batch_size,
                H=block_mask_heads,
                Q_LEN=args.sequence_length,
                KV_LEN=args.sequence_length,
                device="cuda",
                BLOCK_SIZE=128,
            )
        torch.cuda.synchronize()
        block_started = time.perf_counter()
        for _ in range(block_build_iterations):
            block_mask = create_block_mask(
                mask_mod,
                B=args.batch_size,
                H=block_mask_heads,
                Q_LEN=args.sequence_length,
                KV_LEN=args.sequence_length,
                device="cuda",
                BLOCK_SIZE=128,
            )
        torch.cuda.synchronize()
        result["block_mask_build_steady_ms"] = (
            (time.perf_counter() - block_started) * 1000 / block_build_iterations
        )
        compiled_flex = torch.compile(flex_attention, dynamic=False)

        def flex_forward():
            return compiled_flex(q, k, v, block_mask=block_mask)

        def clear_grads():
            q.grad = None
            k.grad = None
            v.grad = None

        def make_step(forward):
            if args.workload == "eval":
                def eval_step():
                    with torch.no_grad():
                        forward()

                return eval_step

            def train_step():
                output = forward()
                output.backward(grad_output)
                clear_grads()

            return train_step

        sdpa_step = make_step(sdpa_forward)
        flex_step = make_step(flex_forward)

        cold_started = time.perf_counter()
        sdpa_output = sdpa_forward()
        if requires_grad:
            sdpa_output.backward(grad_output)
            sdpa_grads = [tensor.grad.detach().clone() for tensor in (q, k, v)]
            clear_grads()
        torch.cuda.synchronize()
        result["sdpa_cold_ms"] = (time.perf_counter() - cold_started) * 1000

        cold_started = time.perf_counter()
        flex_output = flex_forward()
        if requires_grad:
            flex_output.backward(grad_output)
            flex_grads = [tensor.grad.detach().clone() for tensor in (q, k, v)]
            clear_grads()
        torch.cuda.synchronize()
        result["flex_compile_and_cold_ms"] = (time.perf_counter() - cold_started) * 1000

        result["output_max_abs_diff"] = float((flex_output - sdpa_output).float().abs().max().item())
        result["output_finite"] = bool(torch.isfinite(flex_output).all().item())
        if requires_grad:
            result["gradient_max_abs_diff"] = [
                float((flex_grad - sdpa_grad).float().abs().max().item())
                for flex_grad, sdpa_grad in zip(flex_grads, sdpa_grads)
            ]
            result["gradients_finite"] = all(torch.isfinite(grad).all().item() for grad in flex_grads)
            del flex_grads, sdpa_grads

        for _ in range(args.warmup):
            sdpa_step()
            flex_step()
        torch.cuda.synchronize()

        sdpa_samples = [cuda_average_ms(sdpa_step, args.iterations) for _ in range(args.repeats)]
        flex_samples = [cuda_average_ms(flex_step, args.iterations) for _ in range(args.repeats)]
        result["sdpa_steady"] = summarize(sdpa_samples)
        result["flex_steady"] = summarize(flex_samples)
        sdpa_median = result["sdpa_steady"]["median_ms"]
        flex_median = result["flex_steady"]["median_ms"]
        result["flex_over_sdpa_ratio"] = flex_median / sdpa_median
        result["twelve_layer_estimate_ms"] = {
            "sdpa": args.layers * sdpa_median,
            "flex_kernel_only": args.layers * flex_median,
            "flex_with_one_steady_block_mask_build": (
                args.layers * flex_median + result["block_mask_build_steady_ms"]
            ),
        }

        if not result["output_finite"] or (requires_grad and not result["gradients_finite"]):
            raise RuntimeError("FlexAttention produced non-finite output or gradients")
        result["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        exit_code = 1

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"result_path={result_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
