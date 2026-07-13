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

    # With the correct score_mod signature (score first), the flex forward matches
    # SDPA exactly on CPU fp32 (0.0 diff). On GPU under bf16 inductor expect small
    # fused-kernel drift. A wrong signature (score last) gathers the bias at wrong
    # indices -> diff >> 1.0, so this tolerance catches that regression.
    max_diff = (out_flex - out_sdpa).abs().max().item()
    assert max_diff < 0.1, (
        f"flex vs SDPA pushdown outputs diverge too far: max abs diff = {max_diff} "
        f"(expected <0.1; a larger diff signals a score_mod signature/indexing bug)"
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
    # The per-layer depth embedding must receive a gradient. (pushdown_depth_bias is a
    # direct child of OLMoBlock — block.attention is a METHOD, not a submodule.)
    depth_emb_grads = [
        b.pushdown_depth_bias.depth_emb.weight.grad
        for b in m.transformer.blocks
    ]
    assert all(g is not None for g in depth_emb_grads), "depth_emb should get gradients"
    assert all(g.abs().sum().item() > 0 for g in depth_emb_grads), "depth_emb grad nonzero"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")
def test_pushdown_flex_gather_by_depth_matches_dense_db(monkeypatch):
    """The gather-by-depth score_mod (default) must match the dense-db form.

    The default flex path gathers ``P`` by the depth tape ``Dc`` INSIDE score_mod
    (captured buffer = the ~77 MB ``P`` (B,n_h,n,Dmax)), instead of pre-materializing
    the dense ``db = take_along_dim(P, Dc)`` (B,n_h,n,n) fp32 (~2.4 GB/layer) and
    capturing THAT. Both lower cleanly and must produce numerically equal outputs
    AND equal depth_emb gradients — the gather-by-depth form is the ~20x/step speedup
    fix (the dense-db backward scatters into 2.4 GB; the gather-by-depth backward
    scatters into 77 MB). This test guards against a regression that silently swaps
    the representations or breaks the in-score_mod gather's indexing.
    """
    from olmo.model import OLMo

    def _run():
        torch.manual_seed(2)
        m = OLMo(_make_cfg(flex=True)).cuda().train()
        B, n = 2, 16
        input_ids = torch.randint(0, 50000, (B, n), device="cuda")
        input_ids[1, n - 4 :] = 50258
        attn = input_ids != 50258
        spans = torch.tensor([[[0, 2, 5], [6, 8, 11]], [[0, 3, 7], [3, 5, 9]]],
                             dtype=torch.long, device="cuda")
        out = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits
        out.sum().backward()
        grads = [b.pushdown_depth_bias.depth_emb.weight.grad.clone()
                 for b in m.transformer.blocks]
        return out.detach(), grads

    # Default path: gather-by-depth (captured P, 77 MB).
    out_gather, grads_gather = _run()
    # Fallback path: dense-db (captured db, 2.4 GB) via the env switch.
    monkeypatch.setenv("OLMO_PUSHDOWN_DENSE_DB", "1")
    out_dense, grads_dense = _run()
    monkeypatch.delenv("OLMO_PUSHDOWN_DENSE_DB", raising=False)

    max_diff = (out_gather - out_dense).abs().max().item()
    assert max_diff < 1e-4, (
        f"gather-by-depth vs dense-db outputs diverge: max abs diff = {max_diff} "
        f"(they must be numerically equal — a diff signals an indexing bug in the "
        f"in-score_mod gather P[b,h,q, Dc[b,q,kv]])"
    )
    for i, (g_g, g_d) in enumerate(zip(grads_gather, grads_dense)):
        gdiff = (g_g - g_d).abs().max().item()
        assert gdiff < 1e-4, (
            f"gather-by-depth vs dense-db depth_emb grad diverge at layer {i}: "
            f"max abs diff = {gdiff} (backward scatter into P must match scatter into db)"
        )
