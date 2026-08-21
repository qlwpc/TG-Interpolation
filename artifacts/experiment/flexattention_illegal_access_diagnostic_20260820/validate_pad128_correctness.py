#!/usr/bin/env python3
"""Numerically validate the FlexAttention pad-to-128 workaround on CUDA.

The validation has two levels:

1. Compare padded FlexAttention against an unpadded dense masked-attention
   reference, including q/k/v gradients, for head-independent and per-head
   masks on both sides of a 128-token boundary.
2. Compare a padded FlexAttention OLMo against an identically initialized
   dense-SDPA OLMo, including logits, language-model loss, and parameter
   gradients.

The script exits non-zero when a semantic metric misses its declared bound.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from olmo.config import ModelConfig
from olmo.model import OLMo


SEED = 20260821
DTYPE = torch.bfloat16
KERNEL_OPTIONS = {"fwd_num_stages": 1, "bwd_num_stages": 1}


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    a = actual.detach().float().reshape(-1)
    r = reference.detach().float().reshape(-1)
    delta = a - r
    denom = torch.linalg.vector_norm(r).clamp_min(1e-12)
    if a.numel() == 0:
        cosine = 1.0
    else:
        cosine = float(F.cosine_similarity(a, r, dim=0, eps=1e-12))
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rel_l2": float(torch.linalg.vector_norm(delta) / denom),
        "cosine": cosine,
    }


def _make_mask(batch: int, heads: int, length: int, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(SEED + length + heads)
    q_idx = torch.arange(length, device=device)[:, None]
    kv_idx = torch.arange(length, device=device)[None, :]
    causal = kv_idx <= q_idx
    random_keep = torch.rand((batch, heads, length, length), generator=generator, device=device) > 0.25
    mask = random_keep & causal
    diagonal = torch.arange(length, device=device)
    mask[..., diagonal, diagonal] = True
    return mask


def _block_mask(mask: torch.Tensor):
    batch, heads, q_len, kv_len = mask.shape
    if heads == 1:
        squeezed = mask[:, 0]

        def mask_mod(b, h, q_idx, kv_idx):
            return squeezed[b, q_idx, kv_idx]

        block_heads = None
    else:

        def mask_mod(b, h, q_idx, kv_idx):
            return mask[b, h, q_idx, kv_idx]

        block_heads = heads
    return create_block_mask(
        mask_mod=mask_mod,
        B=batch,
        H=block_heads,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=mask.device,
    )


def _dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v.float())


def _padded_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, multiple: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = q.shape[-2]
    padded_length = math.ceil(length / multiple) * multiple
    tail = padded_length - length
    q_pad = F.pad(q, (0, 0, 0, tail))
    k_pad = F.pad(k, (0, 0, 0, tail))
    v_pad = F.pad(v, (0, 0, 0, tail))
    mask_pad = torch.zeros(
        (*mask.shape[:-2], padded_length, padded_length), dtype=torch.bool, device=mask.device
    )
    mask_pad[..., :length, :length] = mask
    padded_positions = torch.arange(length, padded_length, device=mask.device)
    mask_pad[..., padded_positions, padded_positions] = True
    return q_pad, k_pad, v_pad, mask_pad


def validate_operator_case(compiled_flex, *, length: int, mask_heads: int) -> Dict[str, object]:
    device = torch.device("cuda")
    batch, query_heads, head_dim = 2, 4, 64
    generator = torch.Generator(device=device).manual_seed(SEED + 1000 * length + mask_heads)
    tensors = [
        torch.randn((batch, query_heads, length, head_dim), generator=generator, device=device, dtype=DTYPE)
        for _ in range(3)
    ]
    mask = _make_mask(batch, mask_heads, length, device=device)
    upstream = torch.randn(
        (batch, query_heads, length, head_dim), generator=generator, device=device, dtype=DTYPE
    )

    q_ref, k_ref, v_ref = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
    out_ref = _dense_attention(q_ref, k_ref, v_ref, mask)
    (out_ref * upstream.float()).sum().backward()

    q_pad, k_pad, v_pad, mask_pad = _padded_inputs(*tensors, mask)
    q_pad, k_pad, v_pad = [tensor.detach().requires_grad_(True) for tensor in (q_pad, k_pad, v_pad)]
    out_pad = compiled_flex(
        q_pad,
        k_pad,
        v_pad,
        block_mask=_block_mask(mask_pad),
        kernel_options=KERNEL_OPTIONS,
    )
    (out_pad[..., :length, :].float() * upstream.float()).sum().backward()
    torch.cuda.synchronize()

    # The exact non-divisible forward is safe in the reproducer. Comparing it
    # with the padded forward isolates padding from dense-vs-fused roundoff.
    with torch.no_grad():
        out_exact_flex = compiled_flex(
            tensors[0],
            tensors[1],
            tensors[2],
            block_mask=_block_mask(mask),
            kernel_options=KERNEL_OPTIONS,
        )
    torch.cuda.synchronize()

    result: Dict[str, object] = {
        "kind": "operator",
        "length": length,
        "padded_length": q_pad.shape[-2],
        "mask_heads": mask_heads,
        "forward_vs_dense": _metrics(out_pad[..., :length, :], out_ref),
        "forward_vs_exact_flex": _metrics(out_pad[..., :length, :], out_exact_flex),
        "q_grad": _metrics(q_pad.grad[..., :length, :], q_ref.grad),
        "k_grad": _metrics(k_pad.grad[..., :length, :], k_ref.grad),
        "v_grad": _metrics(v_pad.grad[..., :length, :], v_ref.grad),
    }
    checks = [
        result["forward_vs_dense"]["rel_l2"] <= 0.02,
        result["forward_vs_exact_flex"]["rel_l2"] <= 0.01,
        result["q_grad"]["rel_l2"] <= 0.04,
        result["k_grad"]["rel_l2"] <= 0.04,
        result["v_grad"]["rel_l2"] <= 0.04,
        result["q_grad"]["cosine"] >= 0.999,
        result["k_grad"]["cosine"] >= 0.999,
        result["v_grad"]["cosine"] >= 0.999,
    ]
    result["passed"] = all(checks)
    return result


def _model_config(*, flex: bool) -> ModelConfig:
    return ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_ratio=2,
        rope=True,
        flash_attention=False,
        flex_attention=flex,
        attention_dropout=0.0,
        residual_dropout=0.0,
        embedding_dropout=0.0,
        max_sequence_length=256,
        vocab_size=256,
        embedding_size=256,
        weight_tying=False,
        init_device="cuda",
        init_fn="normal",
        init_std=0.02,
        transformer_grammar_type="tg",
    )


def _flatten_gradients(model: torch.nn.Module) -> torch.Tensor:
    gradients: Iterable[torch.Tensor] = (
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return torch.cat(tuple(gradients))


def validate_model_case(*, mask_heads: int) -> Dict[str, object]:
    device = torch.device("cuda")
    batch, length = 2, 63
    torch.manual_seed(SEED + mask_heads)
    dense_model = OLMo(_model_config(flex=False), init_params=True).to(device=device, dtype=DTYPE)
    flex_model = OLMo(replace(_model_config(flex=False), flex_attention=True), init_params=True).to(
        device=device, dtype=DTYPE
    )
    flex_model.load_state_dict(dense_model.state_dict())
    dense_model.eval()
    flex_model.eval()

    generator = torch.Generator(device=device).manual_seed(SEED + 100 + mask_heads)
    input_ids = torch.randint(0, 256, (batch, length), generator=generator, device=device)
    labels = torch.randint(0, 256, (batch, length - 1), generator=generator, device=device)
    mask = _make_mask(batch, mask_heads, length, device=device)

    os.environ.pop("OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE", None)
    dense_logits = dense_model(input_ids=input_ids, attention_bias=mask).logits
    dense_loss = F.cross_entropy(dense_logits[:, :-1].float().reshape(-1, 256), labels.reshape(-1))
    dense_loss.backward()

    os.environ["OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE"] = "128"
    flex_logits = flex_model(input_ids=input_ids, attention_bias=mask).logits
    flex_loss = F.cross_entropy(flex_logits[:, :-1].float().reshape(-1, 256), labels.reshape(-1))
    flex_loss.backward()
    torch.cuda.synchronize()

    logits_metrics = _metrics(flex_logits, dense_logits)
    gradient_metrics = _metrics(_flatten_gradients(flex_model), _flatten_gradients(dense_model))
    result: Dict[str, object] = {
        "kind": "model",
        "length": length,
        "padded_length": 128,
        "mask_heads": mask_heads,
        "dense_loss": float(dense_loss),
        "padded_flex_loss": float(flex_loss),
        "loss_abs_diff": float((flex_loss - dense_loss).abs()),
        "logits": logits_metrics,
        "parameter_gradients": gradient_metrics,
    }
    result["passed"] = bool(
        result["loss_abs_diff"] <= 0.01
        and logits_metrics["rel_l2"] <= 0.03
        and gradient_metrics["rel_l2"] <= 0.08
        and gradient_metrics["cosine"] >= 0.995
    )
    return result


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlexAttention correctness validation")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.environ["OLMO_FLEX_ATTENTION_NUM_STAGES"] = "1"
    compiled_flex = torch.compile(flex_attention)

    print(
        json.dumps(
            {
                "kind": "environment",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "dtype": str(DTYPE),
                "seed": SEED,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results = []
    for length, mask_heads in ((63, 1), (127, 4), (129, 1), (191, 4)):
        result = validate_operator_case(compiled_flex, length=length, mask_heads=mask_heads)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    for mask_heads in (1, 4):
        result = validate_model_case(mask_heads=mask_heads)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    passed = all(bool(result["passed"]) for result in results)
    print(json.dumps({"kind": "summary", "passed": passed, "cases": len(results)}, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
