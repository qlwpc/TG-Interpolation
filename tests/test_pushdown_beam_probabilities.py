"""Probability-semantics regressions for Pushdown beam inference."""

from types import SimpleNamespace

import torch

from olmo.model import OLMo


class UniformToy:
    def __init__(self, vocab_size: int, eos: int):
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(eos_token_id=eos, pad_token_id=99)
        self.prefix_lengths = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        tree_spans=None,
        last_logits_only=False,
        compute_attachment_logits=False,
    ):
        self.prefix_lengths.append(int(input_ids.shape[1]))
        batch, length = input_ids.shape
        logits = torch.zeros(batch, 1, self.vocab_size)
        attachment = None
        if compute_attachment_logits:
            attachment = torch.zeros(batch, length, length)
        return SimpleNamespace(logits=logits, attachment_logits=attachment)

    pushdown_beam_search = OLMo.pushdown_beam_search


def test_present_bos_is_not_scored_or_duplicated():
    model = UniformToy(vocab_size=10, eos=2)
    score = model.pushdown_beam_search(
        torch.tensor([0, 1, 2]), bos_id=0, beam_size=20, max_reduce=4
    )
    # Only token 1 and EOS are predicted. Structural probabilities are
    # normalized, and EOS root closure is deterministic.
    assert abs(score - 2 * torch.log(torch.tensor(10.0)).item()) < 1e-6
    assert model.prefix_lengths[0] == 1


def test_uniform_attachment_prior_cannot_create_probability_mass():
    model = UniformToy(vocab_size=1, eos=0)
    # No word uncertainty. Regardless of how many parse branches are explored,
    # their normalized joint mass is <= 1, so surprisal cannot be negative.
    score = model.pushdown_beam_search(
        torch.tensor([0, 0, 0, 0]), bos_id=0, beam_size=100, max_reduce=4
    )
    assert score >= -1e-10
    assert abs(score) < 1e-8


def test_tag_score_uses_original_bos_aware_coordinates():
    model = UniformToy(vocab_size=7, eos=2)
    score = model.pushdown_beam_search(
        torch.tensor([0, 3, 4, 2]),
        bos_id=0,
        beam_size=50,
        max_reduce=4,
        tag=[0, 1, 0, 0],
    )
    assert abs(score - torch.log(torch.tensor(7.0)).item()) < 1e-6


class GenerationToy(UniformToy):
    pushdown_generate = OLMo.pushdown_generate

    def __init__(self):
        super().__init__(vocab_size=2, eos=1)
        self.seen_real_span = False

    def pushdown_beam_search(self, *args, **kwargs):
        return 0.0, torch.tensor([[0, 1, 1]], dtype=torch.long)

    def forward(self, *args, tree_spans=None, **kwargs):
        if tree_spans is not None and bool((tree_spans[..., 0] >= 0).any()):
            self.seen_real_span = True
        return super().forward(*args, tree_spans=tree_spans, **kwargs)


def test_generation_starts_from_inferred_prompt_parse():
    model = GenerationToy()
    output = model.pushdown_generate(
        torch.tensor([[0, 0, 0]]),
        max_steps=3,
        beam_size=2,
        max_reduce=None,
        eos_token_id=1,
        pad_token_id=99,
    )
    assert output.token_ids.shape == (1, 2, 3)
    assert output.scores.shape == (1, 2)
    assert model.seen_real_span
    assert output.scores[0, 0] <= 0


def test_generation_uses_gold_prompt_spans_without_prompt_beam_search():
    model = GenerationToy()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("gold prompt spans must bypass pushdown_beam_search")

    model.pushdown_beam_search = fail_if_called
    output = model.pushdown_generate(
        torch.tensor([[0, 0, 0]]),
        prompt_spans=torch.tensor([[[0, 1, 2], [-1, -1, -1]]]),
        max_steps=3,
        beam_size=2,
        max_reduce=None,
        eos_token_id=1,
        pad_token_id=99,
    )
    assert output.token_ids.shape == (1, 2, 3)
    assert model.seen_real_span
