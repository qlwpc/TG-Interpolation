"""Gold-tree Pushdown document-PPL optimization regressions."""

import pytest
import torch

from olmo.eval.pushdown_document_ppl import (
    PushdownGoldCandidate,
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
