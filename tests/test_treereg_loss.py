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


def test_treereg_favors_gold_split_when_orthogonal():
    """Under the orthogonality metric, a gold split that separates two mutually
    orthogonal blocks should score highest (the two halves are maximally
    independent) and thus have the lowest CE loss."""
    torch.manual_seed(1)
    B, n, d = 1, 8, 16
    # Left block [0..3] along e1, right block [4..7] along e2, e1 ⟂ e2.
    h = torch.zeros(B, n, d)
    e1 = torch.randn(d)
    e2 = torch.randn(d)
    e2 = e2 - (e2 @ e1) / (e1 @ e1) * e1  # make e2 orthogonal to e1
    h[0, 0:4] = e1
    h[0, 4:8] = e2
    # Gold span (0, 3, 7): split at 3 -> left [0..3], right [4..7] (orthogonal halves).
    spans = torch.tensor([[[0, 3, 7]]], dtype=torch.long)
    mask = torch.tensor([[True]], dtype=torch.bool)
    loss_aligned = compute_treereg_loss(h.clone(), spans, mask, n_heads_subset=0)
    # Misaligned gold split (0, 1, 7): split at 1 -> right child [2..7] spans across
    # the block boundary, so the gold split is NOT at the orthogonal boundary ->
    # lower score -> higher CE.
    spans_mis = torch.tensor([[[0, 1, 7]]], dtype=torch.long)
    loss_mis = compute_treereg_loss(h.clone(), spans_mis, mask, n_heads_subset=0)
    assert float(loss_aligned) < float(loss_mis)


def test_treereg_orthogonal_metric_matches_formula():
    """O[i,j] must equal ||H[j] - proj_{Hn[i]}(H[j])|| (component of H[j] orthogonal
    to H[i]), matching scin_computer.get_all_orthogonal_scores."""
    torch.manual_seed(2)
    H = torch.randn(1, 3, 4)
    i, j = 0, 2
    hi, hj = H[0, i], H[0, j]
    expected = (hj - (hj @ hi) / (hi @ hi) * hi).norm()
    # Reconstruct O[i,j] from the loss internals (build_chart step).
    circuit = H.float()
    Hn = torch.nn.functional.normalize(circuit, dim=-1)
    proj = torch.bmm(Hn, circuit.transpose(1, 2))
    orth = circuit.unsqueeze(1) - proj.unsqueeze(-1) * Hn.unsqueeze(2)
    O = orth.norm(dim=-1)
    assert torch.allclose(O[0, i, j], expected, atol=1e-4)
    # A vector's component orthogonal to itself is zero: O[i,i] = 0.
    assert torch.allclose(O[0, i, i], torch.tensor(0.0), atol=1e-5)


def test_treereg_macro_reduction():
    """Loss must be the macro mean (per-sentence span-CE mean, then mean over
    sentences), NOT a flat mean over all spans."""
    torch.manual_seed(3)
    B, n, d = 2, 12, 8
    hidden = torch.randn(B, n, d, requires_grad=True)
    # Sentence 0: two spans; sentence 1: one span.
    spans = torch.tensor([
        [[0, 2, 5], [3, 4, 8], [-1, -1, -1]],
        [[1, 4, 9], [-1, -1, -1], [-1, -1, -1]],
    ], dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)
    loss_macro = compute_treereg_loss(hidden, spans, mask, n_heads_subset=0)

    # Per-span CE computed independently (B=1 each), then macro-averaged.
    def span_ce(b, st, p, en):
        h1 = hidden[b:b + 1].clone().detach().requires_grad_(True)
        sp = torch.tensor([[[st, p, en]]], dtype=torch.long)
        mk = torch.tensor([[True]], dtype=torch.bool)
        return float(compute_treereg_loss(h1, sp, mk, n_heads_subset=0))

    sent0 = (span_ce(0, 0, 2, 5) + span_ce(0, 3, 4, 8)) / 2
    sent1 = span_ce(1, 1, 4, 9)
    expected = (sent0 + sent1) / 2
    assert abs(float(loss_macro) - expected) < 1e-4
