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


def test_teacher_forced_v1_conditions_on_legal_targets_but_v2_does_not():
    logits = torch.tensor(
        [[
            [0.0, -torch.inf, -torch.inf],
            [4.0, 1.0, -torch.inf],
        ]]
    )
    targets = torch.tensor([[0, 1]])
    legal = torch.tensor(
        [[
            [True, False, False],
            [False, True, False],
        ]]
    )
    v1 = _attachment_nll_from_logits(logits, targets, legal, "stack_legal")
    v2 = _attachment_nll_from_logits(logits, targets, legal, "sentence_causal")
    assert v1.item() == pytest.approx(0.0)
    assert v2.item() == pytest.approx(
        -torch.log_softmax(logits[0, 1], dim=0)[1].item()
    )
    assert v2.item() > v1.item()


def test_teacher_forced_protocol_rejects_illegal_gold_target():
    logits = torch.zeros(1, 1, 2)
    targets = torch.tensor([[1]])
    legal = torch.tensor([[[True, False]]])
    with pytest.raises(ValueError, match="outside its legal action set"):
        _attachment_nll_from_logits(logits, targets, legal, "sentence_causal")
