"""Gold-tree Pushdown document-PPL optimization regressions."""

import pytest
import torch

from olmo.eval.pushdown_document_ppl import (
    PushdownGoldCandidate,
    _attachment_nll_from_logits,
    _compress_candidates,
    _weighted_logsumexp,
)


def _candidate(spans):
    return PushdownGoldCandidate(
        tokens=(0, 10, 20),
        spans=spans,
        sentence_ids=(-1, 0, 0),
        attachment_targets=(-1, 1, 1),
        legal_attachment_targets=((), (1,), (2, 1)),
    )


def test_structure_compression_preserves_original_candidate_mass():
    left = _candidate(((1, 1, 2),))
    right = _candidate(((1, 2, 2),))
    unique, counts = _compress_candidates((left, right, left, left, right))
    assert unique == (left, right)
    assert counts.tolist() == [3, 2]

    unique_nll = torch.tensor([0.75, 3.0], dtype=torch.float64)
    expanded_nll = torch.tensor([0.75, 3.0, 0.75, 0.75, 3.0], dtype=torch.float64)
    assert _weighted_logsumexp(unique_nll, counts).item() == pytest.approx(
        torch.logsumexp(-expanded_nll, 0).item()
    )


def test_attachment_nll_uses_trained_full_causal_normalization():
    # All three causal positions participate in the trained head's denominator,
    # even if a decoder would consider only positions 0 and 1 stack-legal.
    logits = torch.zeros((1, 1, 3), dtype=torch.float32)
    targets = torch.tensor([[0]], dtype=torch.long)

    nll = _attachment_nll_from_logits(logits, targets)

    assert nll.item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
    legal_conditioned_nll = torch.log(torch.tensor(2.0)).item()
    assert nll.item() != pytest.approx(legal_conditioned_nll)


def test_attachment_nll_ignores_invalid_queries_before_softmax():
    logits = torch.tensor(
        [[[float("-inf"), float("-inf")], [0.0, 0.0]]], dtype=torch.float32
    )
    targets = torch.tensor([[-100, 1]], dtype=torch.long)

    nll = _attachment_nll_from_logits(logits, targets)

    assert torch.isfinite(nll).all()
    assert nll.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
