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


def test_gold_history_uses_individual_probability_not_duplicate_mass(monkeypatch):
    from types import SimpleNamespace
    from olmo.eval import pushdown_document_ppl as ppl
    left, right = _candidate(((1, 1, 2),)), _candidate(((1, 2, 2),))
    class Corpus:
        vocab = SimpleNamespace(bos=99)
        samples_per_sentence = 300
        def __len__(self):
            return 2
        def __iter__(self):
            return iter([(0, (left,) * 299 + (right,))] * 2)
    seen = []
    def score(_model, prefix, candidates, *args):
        seen.append(prefix)
        assert candidates == (left, right)
        nll = torch.tensor([2., 1.], dtype=torch.float64)
        return ppl.PushdownCandidateScores(nll, nll, torch.zeros_like(nll))
    monkeypatch.setattr(ppl, "score_pushdown_gold_candidates", score)
    result = ppl.evaluate_pushdown_document_ppl(SimpleNamespace(eval=lambda: None), Corpus(), "cpu",
                                               include_attachment_probability=False)
    assert seen == [(), (right,)]
    assert result.non_candidate0_count == 2
    assert result.non_candidate0_ratio == 1
    assert result.legacy_log_likelihood == pytest.approx(2 * torch.logsumexp(torch.tensor([-2.] * 299 + [-1.], dtype=torch.float64), 0).item())
