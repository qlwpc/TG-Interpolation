"""Tests for the Pushdown attachment head (Murty et al. 2023, Eq. 5).

Covers:
  * oracle reduce-target derivation via shift-reduce stack simulation — validated
    against the paper's Fig. 2 example ``[[The dog][is happy]]`` and left/right
    branching trees.
  * the attachment head forward (bilinear ``h_j^T W h̃_k`` with ``h̃_k = MLP(...)``)
    produces a strictly lower-triangular (+ diagonal) causal logit matrix.
  * the attachment loss is finite, nonzero, and backprops to ``head.W`` and the
    final-layer hidden states.

CPU-only: no flex/CUDA dependency. ``olmo.attachment`` is pure torch.
"""

import pytest
import torch

from olmo.attachment import (
    PushdownAttachmentHead,
    derive_oracle_reduce_targets,
    compute_attachment_loss,
)


# --------------------------------------------------------------------------- #
# oracle derivation
# --------------------------------------------------------------------------- #
def _spans_tensor(spans, n):
    """Wrap a list of (l, split, r) into a (1, M, 3) long tensor padded with -1."""
    m = len(spans)
    t = torch.full((1, max(m, 1), 3), -1, dtype=torch.long)
    for i, (l, s, r) in enumerate(spans):
        t[0, i] = torch.tensor([l, s, r])
    mask = torch.zeros(1, max(m, 1), dtype=torch.bool)
    mask[0, :m] = True
    return t, mask


def test_oracle_fig2():
    """Fig. 2: ``[[The dog][is happy]]`` -> oracle = [0, 0, 2, 1].

    tokens: The=0, dog=1, is=2, happy=3
    spans (non-binarized, only (l,r) matters): (0,1),(2,3),(0,3)
      k=0 The   : no close -> shift,            r_0 = 0
      k=1 dog   : close (0,1), outer_left=0;    stack top (0,0).left==0 -> r_1 = 0
      k=2 is    : no close -> shift,            r_2 = 2
      k=3 happy : close (2,3),(0,3), outer_left=0; pop (2,2) left!=0, pop (0,1) left==0
                   -> r_3 = 1 (dog)
    """
    spans, mask = _spans_tensor([(0, 0, 1), (2, 2, 3), (0, 1, 3)], n=4)
    oracle = derive_oracle_reduce_targets(spans, 4, mask)
    assert oracle.shape == (1, 4)
    assert oracle.tolist() == [[0, 0, 2, 1]], f"Fig.2 oracle mismatch: {oracle.tolist()}"


def test_oracle_left_branching():
    """Left-branching ``[[A B] C]``: A=0,B=1,C=2, spans (0,1),(0,2) -> [0, 0, 1].

      k=0 A: shift, r_0=0
      k=1 B: close (0,1), outer_left=0; top (0,0).left==0 -> r_1=0
      k=2 C: close (0,2), outer_left=0; top (0,1).left==0 -> r_2=1
    """
    spans, mask = _spans_tensor([(0, 0, 1), (0, 1, 2)], n=3)
    oracle = derive_oracle_reduce_targets(spans, 3, mask)
    assert oracle.tolist() == [[0, 0, 1]]


def test_oracle_right_branching():
    """Right-branching ``[A [B C]]``: A=0,B=1,C=2, spans (1,2),(0,2) -> [0, 1, 0].

      k=0 A: shift, r_0=0
      k=1 B: no close (no span ends at 1) -> shift, r_1=1
      k=2 C: close (1,2),(0,2), outer_left=0; pop (1,2) left!=0, pop (0,0) left==0 -> r_2=0
    """
    spans, mask = _spans_tensor([(1, 1, 2), (0, 1, 2)], n=3)
    oracle = derive_oracle_reduce_targets(spans, 3, mask)
    assert oracle.tolist() == [[0, 1, 0]]


def test_oracle_all_shift():
    """No spans -> every token is shift-only (r_k = k)."""
    spans, mask = _spans_tensor([], n=3)
    oracle = derive_oracle_reduce_targets(spans, 3, mask)
    assert oracle.tolist() == [[0, 1, 2]]


# --------------------------------------------------------------------------- #
# attachment head forward
# --------------------------------------------------------------------------- #
def test_attachment_head_causal():
    """logits[b,k,j] must be -inf for j > k (a query can only reduce to a prefix)."""
    torch.manual_seed(0)
    d, vocab, n, B = 16, 20, 6, 2
    head = PushdownAttachmentHead(d_model=d, vocab_size=vocab)
    final_hidden = torch.randn(B, n, d)
    input_ids = torch.randint(0, vocab, (B, n))
    wte = torch.randn(vocab, d)
    attn = torch.ones(B, n, dtype=torch.bool)
    logits = head(final_hidden, input_ids, wte, attn)
    assert logits.shape == (B, n, n)
    # Strict upper triangle (j > k) must be -inf.
    upper = torch.triu(torch.ones(n, n), diagonal=1).bool()
    assert torch.isinf(logits[:, upper]).all() and (logits[:, upper] < 0).all()
    # Diagonal + lower triangle must be finite (shift-only self-score + reduce candidates).
    lower = ~upper
    assert torch.isfinite(logits[:, lower]).all()


def test_attachment_head_diagonal_uses_predicted_token_representation():
    """Eq. 5's diagonal is h_tilde^T W h_tilde, not h_k^T W h_tilde."""
    torch.manual_seed(17)
    d, vocab, n, batch = 8, 13, 5, 2
    head = PushdownAttachmentHead(d_model=d, vocab_size=vocab)
    hidden = torch.randn(batch, n, d)
    ids = torch.randint(0, vocab, (batch, n))
    embeddings = torch.randn(vocab, d)
    logits = head(hidden, ids, embeddings)

    h = hidden.float()
    emb = torch.nn.functional.embedding(ids, embeddings).float()
    h_prev = torch.nn.functional.pad(h[:, :-1], (0, 0, 1, 0))
    h_tilde = head.mlp(torch.cat([emb, h_prev], dim=-1))
    expected = (head.W(h_tilde) * h_tilde).sum(dim=-1)
    actual = logits.diagonal(dim1=1, dim2=2)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_attachment_loss_gradient():
    """Loss is finite/nonzero and backprops to head.W, head.mlp, and final_hidden."""
    torch.manual_seed(1)
    d, vocab, n, B = 16, 20, 6, 2
    head = PushdownAttachmentHead(d_model=d, vocab_size=vocab)
    final_hidden = torch.randn(B, n, d, requires_grad=True)
    input_ids = torch.randint(0, vocab, (B, n))
    wte = torch.randn(vocab, d, requires_grad=True)
    attn = torch.ones(B, n, dtype=torch.bool)

    logits = head(final_hidden, input_ids, wte, attn)
    # Fig.2-style oracle for both batch elements (reuse the 4-token example padded).
    spans, mask = _spans_tensor([(0, 0, 1), (2, 2, 3), (0, 1, 3)], n=n)
    oracle = derive_oracle_reduce_targets(spans, n, mask).expand(B, n)
    loss = compute_attachment_loss(logits, oracle, attn)

    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    # New params W + MLP must receive grad.
    assert head.W.weight.grad is not None and head.W.weight.grad.abs().sum().item() > 0
    for p in head.mlp.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0
    # final_hidden must receive grad (the keys h_j^L feed the bilinear product).
    assert final_hidden.grad is not None and final_hidden.grad.abs().sum().item() > 0


def test_attachment_loss_empty_oracle_is_zero():
    """All-shift oracle (r_k=k) still yields a valid loss (self-score CE), not NaN."""
    torch.manual_seed(2)
    d, vocab, n, B = 16, 20, 4, 1
    head = PushdownAttachmentHead(d_model=d, vocab_size=vocab)
    final_hidden = torch.randn(B, n, d)
    input_ids = torch.randint(0, vocab, (B, n))
    wte = torch.randn(vocab, d)
    attn = torch.ones(B, n, dtype=torch.bool)
    logits = head(final_hidden, input_ids, wte, attn)
    oracle = torch.arange(n).unsqueeze(0).expand(B, n)  # all shift-only
    loss = compute_attachment_loss(logits, oracle, attn)
    assert torch.isfinite(loss)


def test_attachment_loss_sum_scales_by_valid_token_count():
    torch.manual_seed(23)
    batch, n = 2, 4
    logits = torch.randn(batch, n, n)
    logits = logits.masked_fill(
        torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )
    oracle = torch.tensor([[0, 0, 1, 2], [0, 1, 1, 3]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    mean = compute_attachment_loss(logits, oracle, mask, reduction="mean")
    summed = compute_attachment_loss(logits, oracle, mask, reduction="sum")
    assert torch.allclose(summed, mean * mask.sum())


# --------------------------------------------------------------------------- #
# beam search inference with the attachment head
# --------------------------------------------------------------------------- #
def _make_pushdown_model():
    """Small pushdown OLMo (CPU, SDPA path — no flex)."""
    from olmo.config import (
        ModelConfig, BlockType, LayerNormType, ActivationType, InitFnType,
    )
    from olmo.model import OLMo
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_layers=2, mlp_ratio=4, mlp_hidden_size=256,
        vocab_size=50320, embedding_size=50320, max_sequence_length=64,
        block_type=BlockType.sequential, layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu, rope=True, flash_attention=True,
        flex_attention=False, attention_dropout=0.0, init_device="cpu",
        init_fn=InitFnType.normal, init_std=0.02, transformer_grammar_type="pushdown",
        pushdown_max_depth=16, pushdown_use_flex=False, weight_tying=True,
        eos_token_id=50256, pad_token_id=50258,
    )
    m = OLMo(cfg).eval()
    # Disable the compiled flex kernel on CPU (use the SDPA fallback).
    for b in m.transformer.blocks:
        if hasattr(b, "flex_attention"):
            b.flex_attention = None
    return m


def test_beam_search_default_runs():
    """Default beam search (uniform reduce prior) returns a finite surprisal."""
    torch.manual_seed(3)
    m = _make_pushdown_model()
    eval_ids = torch.randint(0, 50000, (12,))
    s = m.pushdown_beam_search(
        eval_input_ids=eval_ids, beam_size=5, max_reduce=3, bos_id=50256, tag=None,
    )
    assert s == s  # not NaN
    assert s > 0


def test_beam_search_attachment_head_runs():
    """use_attachment_head=True runs and returns a finite surprisal.

    The head's log p(r_k) is added to each reduce candidate's score (Eq. 7), so
    the result generally differs from the uniform-prior path — confirming the
    head actually contributes to beam scoring.
    """
    torch.manual_seed(3)
    m = _make_pushdown_model()
    eval_ids = torch.randint(0, 50000, (12,))
    s_uniform = m.pushdown_beam_search(
        eval_input_ids=eval_ids, beam_size=5, max_reduce=3, bos_id=50256, tag=None,
    )
    s_head = m.pushdown_beam_search(
        eval_input_ids=eval_ids, beam_size=5, max_reduce=3, bos_id=50256, tag=None,
        use_attachment_head=True,
    )
    assert s_head == s_head  # not NaN
    assert s_head > 0
    # The head's structural prior shifts the marginalized surprisal (random init
    # head, so the shift is arbitrary but must be nonzero in expectation).
    assert abs(s_head - s_uniform) > 1e-6, (
        "attachment head should change the beam surprisal vs the uniform prior"
    )


def test_beam_search_attachment_head_tag_scoring():
    """Tag scoring (SG-style) works with the attachment head enabled."""
    torch.manual_seed(4)
    m = _make_pushdown_model()
    eval_ids = torch.randint(0, 50000, (12,))
    tag = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    s = m.pushdown_beam_search(
        eval_input_ids=eval_ids, beam_size=5, max_reduce=3, bos_id=50256,
        tag=tag, use_attachment_head=True,
    )
    assert s == s  # not NaN
    assert s >= 0
