#!/usr/bin/env python3
"""Validate pad-to-128 on the real BBC TG and tgnomask masks at N=127."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from olmo.config import ModelConfig
from olmo.data import KProximal_TG_attention_bias, TG_attention_bias
from olmo.model import OLMo


REPO = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO / "validation" / "slurm" / "results"
SEQUENCE_LENGTH = 127
PADDED_LENGTH = 128
BATCH_SIZE = 4
HEADS = 12
HEAD_DIM = 64


def metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_flat = actual.detach().float().reshape(-1)
    reference_flat = reference.detach().float().reshape(-1)
    delta = actual_flat - reference_flat
    denominator = torch.linalg.vector_norm(reference_flat).clamp_min(1e-12)
    return {
        "max_abs": float(delta.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(delta) / denominator),
        "cosine": float(F.cosine_similarity(actual_flat, reference_flat, dim=0, eps=1e-12)),
    }


def real_mask(mode: str, device: torch.device) -> torch.Tensor:
    data = np.load(REPO / "dataset" / "bbc-news" / "tg" / "test.npy", mmap_mode="r")
    vocabulary = str(REPO / "dataset" / "bbc-news" / "TG_GPT2_tokenizer.json")
    masks = []
    for batch_index in range(BATCH_SIZE):
        start = batch_index * 2048
        tokens = torch.from_numpy(
            np.array(data[start : start + SEQUENCE_LENGTH], copy=True)
        ).long()
        if mode == "tg":
            generator = TG_attention_bias(vocabulary, 2048)
        else:
            generator = KProximal_TG_attention_bias(vocabulary, 2048, 2048, False)
        mask, _ = generator(tokens)
        masks.append(mask.to(dtype=torch.bool))
    return torch.stack(masks).unsqueeze(1).to(device=device)


def validate(mode: str, compiled_flex, device: torch.device) -> dict[str, object]:
    mask = real_mask(mode, device)
    padded_mask = torch.zeros(
        (BATCH_SIZE, 1, PADDED_LENGTH, PADDED_LENGTH),
        dtype=torch.bool,
        device=device,
    )
    padded_mask[..., :SEQUENCE_LENGTH, :SEQUENCE_LENGTH] = mask
    padded_mask[..., SEQUENCE_LENGTH, SEQUENCE_LENGTH] = True
    squeezed_mask = padded_mask.squeeze(1)

    def mask_mod(batch, _head, q_idx, kv_idx):
        return squeezed_mask[batch, q_idx, kv_idx]

    block_mask = create_block_mask(
        mask_mod,
        B=BATCH_SIZE,
        H=None,
        Q_LEN=PADDED_LENGTH,
        KV_LEN=PADDED_LENGTH,
        device=device,
        BLOCK_SIZE=128,
    )

    generator = torch.Generator(device=device).manual_seed(20260901)
    base = [
        torch.randn(
            (BATCH_SIZE, HEADS, SEQUENCE_LENGTH, HEAD_DIM),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    ]
    upstream = torch.randn(
        (BATCH_SIZE, HEADS, SEQUENCE_LENGTH, HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    q_ref, k_ref, v_ref = [tensor.detach().clone().requires_grad_(True) for tensor in base]
    additive_mask = torch.zeros_like(mask, dtype=torch.bfloat16)
    additive_mask.masked_fill_(~mask, torch.finfo(torch.bfloat16).min)
    reference_output = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=additive_mask)
    reference_output.backward(upstream)

    q_pad, k_pad, v_pad = [
        F.pad(tensor, (0, 0, 0, 1)).detach().requires_grad_(True) for tensor in base
    ]
    padded_output = compiled_flex(q_pad, k_pad, v_pad, block_mask=block_mask)
    padded_output[..., :SEQUENCE_LENGTH, :].backward(upstream)
    torch.cuda.synchronize()

    flex_gradients = [tensor.grad for tensor in (q_pad, k_pad, v_pad)]
    reference_gradients = [tensor.grad for tensor in (q_ref, k_ref, v_ref)]
    result: dict[str, object] = {
        "mode": mode,
        "sequence_length": SEQUENCE_LENGTH,
        "padded_length": PADDED_LENGTH,
        "mask_density": float(mask.float().mean()),
        "output_finite": bool(torch.isfinite(padded_output).all()),
        "gradients_finite": all(bool(torch.isfinite(gradient).all()) for gradient in flex_gradients),
        "padded_tail_gradients_zero": all(
            bool((gradient[..., SEQUENCE_LENGTH:, :] == 0).all()) for gradient in flex_gradients
        ),
        "output": metrics(padded_output[..., :SEQUENCE_LENGTH, :], reference_output),
        "gradients": [
            metrics(flex_gradient[..., :SEQUENCE_LENGTH, :], reference_gradient)
            for flex_gradient, reference_gradient in zip(flex_gradients, reference_gradients)
        ],
    }
    result["passed"] = bool(
        result["output_finite"]
        and result["gradients_finite"]
        and result["padded_tail_gradients_zero"]
        and result["output"]["rel_l2"] <= 0.02
        and all(gradient["rel_l2"] <= 0.05 for gradient in result["gradients"])
        and all(gradient["cosine"] >= 0.999 for gradient in result["gradients"])
    )
    return result


def model_config(*, flex: bool) -> ModelConfig:
    return ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        mlp_hidden_size=128,
        vocab_size=256,
        embedding_size=256,
        max_sequence_length=128,
        rope=True,
        flash_attention=False,
        flex_attention=flex,
        flex_attention_train_min_sequence_length=0,
        flex_attention_eval_min_sequence_length=0,
        flex_attention_pad_to_multiple=128,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        init_device="cpu",
        init_fn="normal",
        init_std=0.02,
        transformer_grammar_type="tg",
        weight_tying=True,
    )


def flattened_gradients(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            parameter.grad.detach().float().reshape(-1)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
    )


def validate_model_router(mode: str, device: torch.device) -> dict[str, object]:
    """Exercise OLMo's selected-Flex padding route with a real N=127 mask."""
    torch.manual_seed(20260902)
    sdpa_model = OLMo(model_config(flex=False)).to(device=device, dtype=torch.bfloat16).train()
    flex_model = OLMo(model_config(flex=True)).to(device=device, dtype=torch.bfloat16).train()
    flex_model.load_state_dict(sdpa_model.state_dict())
    mask = real_mask(mode, device)
    generator = torch.Generator(device=device).manual_seed(20260902)
    input_ids = torch.randint(
        0,
        256,
        (BATCH_SIZE, SEQUENCE_LENGTH),
        generator=generator,
        device=device,
    )

    sdpa_logits = sdpa_model(input_ids=input_ids, attention_bias=mask).logits
    sdpa_loss = sdpa_logits.float().square().mean()
    sdpa_loss.backward()

    flex_logits = flex_model(input_ids=input_ids, attention_bias=mask).logits
    flex_loss = flex_logits.float().square().mean()
    flex_loss.backward()
    torch.cuda.synchronize()

    gradient_metrics = metrics(
        flattened_gradients(flex_model),
        flattened_gradients(sdpa_model),
    )
    logits_metrics = metrics(flex_logits, sdpa_logits)
    gradients_finite = all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in flex_model.parameters()
        if parameter.grad is not None
    )
    result: dict[str, object] = {
        "kind": "model_router",
        "mode": mode,
        "sequence_length": SEQUENCE_LENGTH,
        "effective_flex_length": PADDED_LENGTH,
        "output_shape": list(flex_logits.shape),
        "sdpa_loss": float(sdpa_loss),
        "flex_loss": float(flex_loss),
        "loss_abs_diff": float((flex_loss - sdpa_loss).abs()),
        "logits": logits_metrics,
        "parameter_gradients": gradient_metrics,
        "gradients_finite": gradients_finite,
    }
    result["passed"] = bool(
        gradients_finite
        and flex_logits.shape[1] == SEQUENCE_LENGTH
        and result["loss_abs_diff"] <= 0.01
        and logits_metrics["rel_l2"] <= 0.03
        and gradient_metrics["rel_l2"] <= 0.08
        and gradient_metrics["cosine"] >= 0.995
    )
    return result


def main() -> int:
    job_id = os.environ.get("SLURM_JOB_ID", "manual")
    output_path = RESULT_DIR / f"real-tg-pad128_{job_id}.json"
    summary: dict[str, object] = {
        "job_id": job_id,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        device = torch.device("cuda")
        compiled_flex = torch.compile(flex_attention, dynamic=False)
        cases = [validate(mode, compiled_flex, device) for mode in ("tg", "tgnomask")]
        cases.extend(validate_model_router(mode, device) for mode in ("tg", "tgnomask"))
        summary["cases"] = cases
        summary["status"] = "passed" if all(case["passed"] for case in cases) else "failed"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"result_path={output_path}", flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
