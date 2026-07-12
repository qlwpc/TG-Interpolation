"""Parity test: Pushdown FlexAttention ``score_mod`` path vs the SDPA additive-mask path.

The flex path (``pushdown_use_flex=True`` + ``flex_attention=True``) is the intended
training fast path — it fuses causality+padding into a ``block_mask`` and the depth
bias into a ``score_mod``, replacing the ``(B, n_h, n, n)`` fp32 additive mask +
math-backend SDPA (the ~100x slowdown). This test asserts the two paths produce
numerically equal attention outputs on identical input (same weights, padded batch,
real constituent spans), so flipping the config flag is safe.

GPU-only: ``create_block_mask`` and ``flex_attention`` require CUDA (the CPU vmap
backend rejects the tensor-indexed ``mask_mod`` with a data-dependent-control-flow
error). Skipped when ``torch.cuda.is_available()`` is False.
"""

import pytest
import torch

from olmo.config import (
    ModelConfig,
    BlockType,
    LayerNormType,
    ActivationType,
    InitFnType,
)


def _make_cfg(flex: bool) -> ModelConfig:
    return ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_ratio=4,
        mlp_hidden_size=256,
        vocab_size=50320,
        embedding_size=50320,
        max_sequence_length=32,
        block_type=BlockType.sequential,
        layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu,
        rope=True,
        flash_attention=True,          # kept on; SDPA fallback / no-parse path uses it
        flex_attention=flex,           # compiles flex_attention (needed for the flex path)
        attention_dropout=0.0,
        init_device="cpu",
        init_fn=InitFnType.normal,
        init_std=0.02,
        transformer_grammar_type="pushdown",
        pushdown_max_depth=16,
        pushdown_use_flex=flex,
        weight_tying=True,
        eos_token_id=50256,
        pad_token_id=50258,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")
def test_pushdown_flex_matches_sdpa():
    """The flex score_mod path and the SDPA additive-mask path must agree."""
    from olmo.model import OLMo

    torch.manual_seed(0)
    # Build the SDPA-path model, then clone its state into the flex-path model so
    # the two are weight-identical (independent inits would diverge on comparison).
    m_sdpa = OLMo(_make_cfg(flex=False)).eval()
    m_flex = OLMo(_make_cfg(flex=True)).eval()
    m_flex.load_state_dict(m_sdpa.state_dict())
    m_sdpa = m_sdpa.cuda()
    m_flex = m_flex.cuda()

    B, n = 2, 16
    input_ids = torch.randint(0, 50000, (B, n), device="cuda")
    # Padded batch: sequence 0 is full-length, sequence 1 has 4 pad tokens at the
    # end — exercises the causal+pad block_mask on the flex side.
    input_ids[1, n - 4 :] = 50258  # pad_token_id
    attn = input_ids != 50258  # (B, n) bool
    # Real constituent spans (terminal indices < n). Two spans per sequence.
    spans = torch.tensor(
        [[[0, 2, 5], [6, 8, 11]], [[0, 3, 7], [3, 5, 9]]],
        dtype=torch.long, device="cuda",
    )

    with torch.no_grad():
        out_sdpa = m_sdpa(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits
        out_flex = m_flex(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits

    # bf16-fused flex vs fp32-ish SDPA: allow a generous but bounded tolerance.
    assert torch.allclose(out_flex, out_sdpa, atol=2e-3, rtol=2e-3), (
        f"flex vs SDPA pushdown outputs differ: "
        f"max abs diff = {(out_flex - out_sdpa).abs().max().item()}"
    )

    # Sanity: real spans must change the output vs no parse (depth bias is nonzero).
    empty = torch.tensor([[[-1, -1, -1]]], dtype=torch.long, device="cuda").expand(B, 1, 3).contiguous()
    with torch.no_grad():
        out_none = m_flex(input_ids=input_ids, attention_mask=attn, tree_spans=empty).logits
    assert not torch.allclose(out_flex, out_none, atol=1e-5), (
        "real spans should change the output vs empty/no-parse (depth bias nonzero)"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")
def test_pushdown_flex_backward():
    """Gradients must flow through the flex score_mod path."""
    from olmo.model import OLMo

    torch.manual_seed(1)
    m = OLMo(_make_cfg(flex=True)).cuda().train()
    B, n = 2, 16
    input_ids = torch.randint(0, 50000, (B, n), device="cuda")
    input_ids[1, n - 4 :] = 50258
    attn = input_ids != 50258
    spans = torch.tensor([[[0, 2, 5], [6, 8, 11]], [[0, 3, 7], [3, 5, 9]]],
                         dtype=torch.long, device="cuda")
    out = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits
    out.sum().backward()
    # The per-layer depth embedding must receive a gradient.
    depth_emb_grads = [
        b.attention.pushdown_depth_bias.depth_emb.weight.grad
        for b in m.transformer.blocks
    ]
    assert all(g is not None for g in depth_emb_grads), "depth_emb should get gradients"
    assert all(g.abs().sum().item() > 0 for g in depth_emb_grads), "depth_emb grad nonzero"
