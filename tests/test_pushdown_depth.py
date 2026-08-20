"""Tests for olmo.pushdown (Pushdown depth matrix + depth bias)."""

import numpy as np
import torch
import pytest

from olmo.pushdown import compute_depth_matrix_gpu, PushdownDepthBias, _DepthBiasGradP
from olmo.data.parse_align import compute_depth_matrix


def test_depth_matrix_matches_numpy_and_paper():
    # Paper example: [[The dog][is happy]] -> spans (0,1,1),(2,3,3),(0,3,3).
    spans = torch.tensor([[[0, 1, 1], [2, 3, 3], [0, 3, 3]]], dtype=torch.long)
    S = compute_depth_matrix_gpu(spans, 4)
    ref = compute_depth_matrix([(0, 1, 1), (2, 3, 3), (0, 3, 3)], 4)
    assert np.array_equal(S[0].numpy(), ref)
    # Row 3 (all four tokens): [2,2,2,2]; row 2 (prefix of 3): [1,1,0,0].
    assert S[0, 3].tolist() == [2, 2, 2, 2]
    assert S[0, 2].tolist() == [1, 1, 0, 0]
    assert S[0, 0].tolist() == [0, 0, 0, 0]


def test_depth_matrix_lower_triangular():
    spans = torch.tensor([[[0, 2, 4], [0, 1, 2], [3, 3, 4], [0, 4, 4]]], dtype=torch.long)
    S = compute_depth_matrix_gpu(spans, 5)
    assert torch.equal(S, torch.tril(S))
    # Depth of a fixed key is non-decreasing in the query.
    for j in range(5):
        col = S[0, :, j]
        assert all(col[k + 1] >= col[k] for k in range(4))


def test_depth_matrix_ignores_padded_spans():
    # Padded spans (-1) must contribute zero depth.
    spans = torch.tensor([[[0, 1, 1], [-1, -1, -1]]], dtype=torch.long)
    S = compute_depth_matrix_gpu(spans, 4)
    # Only span (0,1) contributes: rows k>=1, cols 0,1 get +1.
    assert S[0, 3].tolist() == [1, 1, 0, 0]
    # All-padded -> all zero.
    S0 = compute_depth_matrix_gpu(torch.tensor([[[-1, -1, -1]]], dtype=torch.long), 4)
    assert int(S0.max()) == 0


def test_depth_matrix_ignores_split_coordinate():
    fixed = torch.tensor([[[1, 1, 4], [0, 0, 4]]], dtype=torch.long)
    corrupted = torch.tensor([[[1, 3, 4], [0, 4, 4]]], dtype=torch.long)
    assert torch.equal(
        compute_depth_matrix_gpu(fixed, 5),
        compute_depth_matrix_gpu(corrupted, 5),
    )


def test_pushdown_depth_bias_shape():
    pdb = PushdownDepthBias(max_depth=16, d_model=64, n_heads=4)
    # key_weight: (n_kv_h * d_head, d_model) = (4*16, 64).
    kw = torch.randn(64, 64)
    E = pdb(kw)
    assert E.shape == (4, 17, 16)  # (n_heads, max_depth+1, d_head)


def test_depth_bias_manual_grad_reuses_flex_output_exactly():
    """Output-assisted grad_P matches dense autograd with document masks."""
    torch.manual_seed(41)
    batch, heads, n, width, depths = 2, 3, 9, 5, 7
    inv = width ** -0.5
    q = torch.randn(batch, heads, n, width)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    dc = torch.randint(0, depths, (batch, n, n))
    valid_keys = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    doc_id = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 2, 2]]
    )
    positions = torch.arange(n)
    grad_mask = (
        valid_keys[:, None, :]
        & (positions[:, None] >= positions[None, :]).unsqueeze(0)
        & (doc_id[:, :, None] == doc_id[:, None, :])
    )
    grad_output = torch.randn_like(q)

    p_reference = torch.randn(batch, heads, n, depths, requires_grad=True)
    scores = torch.einsum("bhni,bhmi->bhnm", q, k) * inv
    bias = torch.take_along_dim(p_reference, dc.unsqueeze(1), dim=3) * inv
    post = (scores + bias).masked_fill(~grad_mask[:, None], float("-inf"))
    flex_lse = torch.logsumexp(post, dim=-1)
    safe_lse = torch.where(
        torch.isfinite(flex_lse), flex_lse, torch.zeros_like(flex_lse)
    )
    probabilities = torch.exp(post - safe_lse.unsqueeze(-1)).masked_fill(
        ~grad_mask[:, None], 0.0
    )
    flex_out = torch.einsum("bhnm,bhmi->bhni", probabilities, v)
    (flex_out * grad_output).sum().backward()
    expected = p_reference.grad.detach().clone()

    p_manual = p_reference.detach().clone().requires_grad_(True)
    auxiliary = _DepthBiasGradP.apply(
        q, k, v, p_manual, dc, ~grad_mask, ~grad_mask.any(dim=-1),
        valid_keys, inv, flex_out.detach(),
    )
    (auxiliary * grad_output).sum().backward()
    assert torch.allclose(p_manual.grad, expected, atol=2e-6, rtol=2e-6)


def test_depth_matrix_saturates_before_int8_cast():
    # Duplicate spans are malformed but possible after noisy/unary parsing.
    # They must saturate at 127 rather than wrap into a negative int8 index.
    spans = torch.tensor([[[0, 0, 0]] * 130], dtype=torch.long)
    depth = compute_depth_matrix_gpu(spans, 1)
    assert depth.dtype == torch.int8
    assert depth.item() == 127


def test_pushdown_model_forward_parity():
    """Empty spans (no parse) must equal no-spans; real spans must differ."""
    from olmo.config import (ModelConfig, BlockType, LayerNormType, ActivationType, InitFnType)
    from olmo.model import OLMo
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_layers=2, mlp_ratio=4, mlp_hidden_size=256,
        vocab_size=50320, embedding_size=50320, max_sequence_length=32,
        block_type=BlockType.sequential, layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu, rope=True, flash_attention=False,
        attention_dropout=0.0, init_device="cpu", init_fn=InitFnType.normal, init_std=0.02,
        transformer_grammar_type="pushdown", pushdown_max_depth=16,
        weight_tying=True, eos_token_id=50256, pad_token_id=50258,
    )
    m = OLMo(cfg).eval()
    torch.manual_seed(0)
    B, n = 2, 16
    input_ids = torch.randint(0, 50000, (B, n))
    attn = torch.ones(B, n, dtype=torch.bool)
    with torch.no_grad():
        o_none = m(input_ids=input_ids, attention_mask=attn, tree_spans=None).logits
        empty = torch.tensor([[[-1, -1, -1]]], dtype=torch.long).expand(B, 1, 3).contiguous()
        o_empty = m(input_ids=input_ids, attention_mask=attn, tree_spans=empty).logits
        spans = torch.tensor([[[2, 4, 7]], [[1, 3, 6]]], dtype=torch.long)
        o_real = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits
    assert torch.allclose(o_none, o_empty, atol=1e-5), "empty spans should equal no spans"
    assert not torch.allclose(o_none, o_real, atol=1e-5), "real spans should change the output"
    # Backward flows (in train mode).
    m.train()
    o_real = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans).logits
    o_real.sum().backward()


def test_pushdown_depth_matrix_is_forward_scoped():
    """The per-forward depth-tape memo must not leak across forwards: two
    forwards with DIFFERENT tree_spans must produce DIFFERENT outputs (a stale
    cached S from the previous forward would make them equal). Also, the memo
    must be repopulated each forward (the cache is empty again at forward end)."""
    from olmo.config import (ModelConfig, BlockType, LayerNormType, ActivationType, InitFnType)
    from olmo.model import OLMo
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_layers=3, mlp_ratio=4, mlp_hidden_size=256,
        vocab_size=50320, embedding_size=50320, max_sequence_length=32,
        block_type=BlockType.sequential, layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu, rope=True, flash_attention=False,
        attention_dropout=0.0, init_device="cpu", init_fn=InitFnType.normal, init_std=0.02,
        transformer_grammar_type="pushdown", pushdown_max_depth=16,
        weight_tying=True, eos_token_id=50256, pad_token_id=50258,
    )
    m = OLMo(cfg).eval()
    torch.manual_seed(1)
    B, n = 2, 16
    input_ids = torch.randint(0, 50000, (B, n))
    attn = torch.ones(B, n, dtype=torch.bool)
    spans_a = torch.tensor([[[2, 4, 7]], [[1, 3, 6]]], dtype=torch.long)
    spans_b = torch.tensor([[[0, 5, 15]], [[3, 8, 12]]], dtype=torch.long)
    with torch.no_grad():
        # First forward with spans_a populates the memo; layers 2,3 reuse it.
        o_a = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans_a).logits
        # The memo must have been invalidated for this second forward (different
        # spans_b), so the output reflects spans_b, NOT a stale spans_a entry.
        o_b = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans_b).logits
        # And a repeat of spans_a must match the first spans_a forward (memo
        # repopulated correctly, not stuck on spans_b).
        o_a2 = m(input_ids=input_ids, attention_mask=attn, tree_spans=spans_a).logits
    assert not torch.allclose(o_a, o_b, atol=1e-5), "different spans must yield different outputs"
    assert torch.allclose(o_a, o_a2, atol=1e-5), "repeating spans_a must reproduce o_a"
    # The memo is invalidated at the START of each forward (pop-then-repopulate),
    # so a stale S from a previous forward can never be read. The output-difference
    # checks above are the real invariant; this confirms the memo path actually ran.
    assert "pushdown_depth_matrix" in m._OLMo__cache, "memo should be populated after a pushdown forward"
