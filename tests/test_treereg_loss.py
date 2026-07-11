"""Tests for olmo.treereg (TreeReg SCIN auxiliary loss)."""

import torch
import pytest

from olmo.treereg import compute_treereg_loss


def test_treereg_loss_shape_and_finite():
    torch.manual_seed(0)
    B, n, d = 2, 16, 32
    hidden = torch.randn(B, n, d, requires_grad=True)
    # One gold constituent per example: (left, split, right) = (2, 4, 7).
    spans = torch.tensor([[[2, 4, 7], [-1, -1, -1]], [[1, 3, 6], [-1, -1, -1]]], dtype=torch.long)
    mask = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
    loss = compute_treereg_loss(hidden, spans, mask, n_heads_subset=2, d_head=d // 8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_treereg_no_spans_returns_zero():
    hidden = torch.randn(2, 8, 16, requires_grad=True)
    spans = torch.full((2, 3, 3), -1, dtype=torch.long)
    mask = torch.zeros((2, 3), dtype=torch.bool)
    loss = compute_treereg_loss(hidden, spans, mask)
    assert float(loss) == 0.0
    loss.backward()  # should not error


def test_treereg_favors_gold_split_when_aligned():
    """If the hidden states already make the gold split maximally independent, the
    CE loss should be low. Construct hidden states where token groups are orthogonal
    by block: the gold split separates two orthogonal blocks -> low loss."""
    B, n, d = 1, 8, 16
    # Make left half [0..3] and right half [4..7] orthogonal blocks.
    h = torch.zeros(B, n, d)
    h[0, 0:4] = torch.randn(d)
    h[0, 4:8] = torch.randn(d)
    # Gold span (0, 3, 7): split at 3 -> left [0..3], right [4..7] (orthogonal).
    spans = torch.tensor([[[0, 3, 7]]], dtype=torch.long)
    mask = torch.tensor([[True]], dtype=torch.bool)
    loss_aligned = compute_treereg_loss(h.clone(), spans, mask, n_heads_subset=0)
    # A misaligned gold split (0, 1, 7): split at 1 -> left [0..1], right [2..7]
    # (right block is not internally orthogonal) -> should be higher loss.
    spans_mis = torch.tensor([[[0, 1, 7]]], dtype=torch.long)
    loss_mis = compute_treereg_loss(h.clone(), spans_mis, mask, n_heads_subset=0)
    assert float(loss_aligned) < float(loss_mis)
