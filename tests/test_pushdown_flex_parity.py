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
    """The gather-by-depth score_mod must match the dense-db form in the FORWARD.

    The dense-db flex path (default) pre-materializes ``db = take_along_dim(P, Dc)``
    (B,n_h,n,n) fp32 (~2.4 GB/layer) and captures THAT. The gather-by-depth path
    (OLMO_PUSHDOWN_GATHER_BY_DEPTH=1) gathers ``P`` by the depth tape ``Dc`` INSIDE
    score_mod (captured buffer = the ~77 MB ``P``). Both must produce numerically
    equal FORWARD outputs — the gather-by-depth form eliminates the 2.4 GB/layer
    materialization. This test guards against a regression that silently breaks the
    in-score_mod gather's indexing.

    NOTE: gather-by-depth is currently EXPERIMENTAL — on GPU under amp_bf16+DDP its
    backward does NOT deliver grad to depth_emb (data-dependent two-level index
    defeats inductor's zeros_and_scatter; see olmo/model.py comment). So this test
    asserts FORWARD parity only; the backward-parity assertion is skipped until the
    grad path is fixed (the dense-db path's backward is the known-working reference).
    """
    from olmo.model import OLMo

    def _run_fwd(env_gather: bool):
        if env_gather:
            monkeypatch.setenv("OLMO_PUSHDOWN_GATHER_BY_DEPTH", "1")
        else:
            monkeypatch.delenv("OLMO_PUSHDOWN_GATHER_BY_DEPTH", raising=False)
        torch.manual_seed(2)
        m = OLMo(_make_cfg(flex=True)).cuda().eval()
        B, n = 2, 16
        input_ids = torch.randint(0, 50000, (B, n), device="cuda")
        input_ids[1, n - 4 :] = 50258
        attn = input_ids != 50258
        spans = torch.tensor([[[0, 2, 5], [6, 8, 11]], [[0, 3, 7], [3, 5, 9]]],
                             dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits
        return out

    out_dense = _run_fwd(env_gather=False)
    out_gather = _run_fwd(env_gather=True)
    monkeypatch.delenv("OLMO_PUSHDOWN_GATHER_BY_DEPTH", raising=False)

    max_diff = (out_gather - out_dense).abs().max().item()
    assert max_diff < 1e-4, (
        f"gather-by-depth vs dense-db forward outputs diverge: max abs diff = {max_diff} "
        f"(they must be numerically equal — a diff signals an indexing bug in the "
        f"in-score_mod gather P[b,h,q, Dc[b,q,kv]])"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")
def test_pushdown_flex_fix2_forward_and_grad(monkeypatch):
    """Fix #2 (OLMO_PUSHDOWN_FIX2): forward parity vs dense-db AND grad to depth_emb.

    Fix #2 = gather-by-depth score_mod with P.detach() (fast forward, no 2.4GB
    scatter) + _DepthBiasGradP custom autograd Function (manual grad_P from grad_out
    via recompute pT + scatter_add into 77MB P). This is the ~10x speedup path.

    Asserts:
      (1) FORWARD output matches dense-db (the bias values are identical — fix #2
          only changes the backward, not the forward).
      (2) depth_emb receives a gradient (the whole point — DETACH/gather-by-depth
          drop it; fix #2 restores it via the custom Function).
      (3) depth_emb grad is nonzero and finite.
    The _DepthBiasGradP math is validated vs autograd to 1e-7 on CPU; this test
    guards the flex integration on GPU (block_mask, bf16, GQA).
    """
    from olmo.model import OLMo

    B, n = 2, 16

    def _build():
        torch.manual_seed(3)
        return OLMo(_make_cfg(flex=True)).cuda().train()

    def _run(env_fix2, want_grad=False):
        if env_fix2:
            monkeypatch.setenv("OLMO_PUSHDOWN_FIX2", "1")
        else:
            monkeypatch.delenv("OLMO_PUSHDOWN_FIX2", raising=False)
        m = _build()
        input_ids = torch.randint(0, 50000, (B, n), device="cuda")
        input_ids[1, n - 4 :] = 50258
        attn = input_ids != 50258
        spans = torch.tensor([[[0, 2, 5], [6, 8, 11]], [[0, 3, 7], [3, 5, 9]]],
                             dtype=torch.long, device="cuda")
        if want_grad:
            out = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits
            out.sum().backward()
            grads = [b.pushdown_depth_bias.depth_emb.weight.grad.clone()
                     for b in m.transformer.blocks]
            return grads
        else:
            with torch.no_grad():
                return m(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits

    # (1) Forward parity vs dense-db.
    out_dense = _run(env_fix2=False, want_grad=False)
    out_fix2 = _run(env_fix2=True, want_grad=False)
    monkeypatch.delenv("OLMO_PUSHDOWN_FIX2", raising=False)
    max_diff = (out_fix2 - out_dense).abs().max().item()
    assert max_diff < 1e-3, (
        f"fix #2 vs dense-db forward outputs diverge: max abs diff = {max_diff} "
        f"(fix #2 forward must equal dense-db — it only changes the backward)"
    )

    # (2)+(3) depth_emb receives a nonzero, finite gradient under fix #2.
    grads = _run(env_fix2=True, want_grad=True)
    monkeypatch.delenv("OLMO_PUSHDOWN_FIX2", raising=False)
    assert all(g is not None for g in grads), (
        "fix #2: depth_emb should receive grad via _DepthBiasGradP (unlike DETACH)"
    )
    for i, g in enumerate(grads):
        assert torch.isfinite(g).all(), f"fix #2: depth_emb grad not finite at layer {i}"
        assert g.abs().sum().item() > 0, f"fix #2: depth_emb grad zero at layer {i}"
