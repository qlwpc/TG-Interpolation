"""Probability-semantics regressions for Pushdown beam inference."""

from types import SimpleNamespace

import torch

from olmo.attachment import attachment_log_probs_for_targets
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


def test_attachment_target_log_probs_distinguish_v1_and_v2():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    legal = torch.tensor([1, 3])
    v1 = attachment_log_probs_for_targets(logits, legal, "v1")
    v2 = attachment_log_probs_for_targets(logits, legal, "v2")
    assert torch.allclose(v1, torch.log_softmax(logits[legal], dim=0))
    assert torch.allclose(v2, torch.log_softmax(logits, dim=0)[legal])
    # Conditioning on the legal subset can only increase each retained target's
    # probability relative to the full causal row.
    assert bool((v1 > v2).all())


class SentenceLocalBeamToy(UniformToy):
    """Uniform words/attachments with the same sentence mask as the real head."""

    def forward(
        self,
        input_ids,
        attention_mask=None,
        tree_spans=None,
        pushdown_sentence_ids=None,
        last_logits_only=False,
        compute_attachment_logits=False,
    ):
        del attention_mask, tree_spans, last_logits_only
        batch, length = input_ids.shape
        logits = torch.zeros(batch, 1, self.vocab_size)
        attachment = None
        if compute_attachment_logits:
            attachment = torch.zeros(batch, length, length)
            causal = torch.tril(
                torch.ones(length, length, dtype=torch.bool)
            ).unsqueeze(0).expand(batch, -1, -1)
            if pushdown_sentence_ids is not None:
                sid = pushdown_sentence_ids
                same = sid[:, :, None] == sid[:, None, :]
                same &= (sid[:, :, None] >= 0) & (sid[:, None, :] >= 0)
                causal = causal & same
            attachment = attachment.masked_fill(~causal, float("-inf"))
        return SimpleNamespace(logits=logits, attachment_logits=attachment)


def test_sentence_local_complete_beam_supports_both_probability_protocols():
    model = SentenceLocalBeamToy(vocab_size=1, eos=9)
    kwargs = dict(
        eval_input_ids=torch.tensor([0, 0, 0]),
        bos_id=5,
        beam_size=100,
        max_reduce=None,
        use_attachment_head=True,
        sentence_local_stack=True,
        require_complete_parse=True,
    )
    v1 = model.pushdown_beam_search(
        **kwargs, attachment_normalization="stack_legal"
    )
    v2 = model.pushdown_beam_search(
        **kwargs, attachment_normalization="sentence_causal"
    )
    assert torch.isfinite(torch.tensor(v1))
    assert torch.isfinite(torch.tensor(v2))
    # With identical logits, v2 keeps probability assigned to causal positions
    # that are no longer stack-reachable, so its retained complete-tree mass is
    # strictly smaller on this three-token example.
    assert v2 > v1


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


class SentenceLocalGenerationToy(UniformToy):
    pushdown_generate = OLMo.pushdown_generate

    def __init__(self):
        super().__init__(vocab_size=3, eos=2)
        self.prompt_len = 5
        self.observed_spans = []

    def forward(
        self,
        input_ids,
        attention_mask=None,
        tree_spans=None,
        last_logits_only=False,
        compute_attachment_logits=False,
        **kwargs,
    ):
        del attention_mask, last_logits_only, kwargs
        batch, length = input_ids.shape
        if tree_spans is not None:
            for row in tree_spans.reshape(-1, 3).tolist():
                if row[0] >= 0:
                    self.observed_spans.append(tuple(map(int, row)))

        # Generate two ordinary tokens, then EOS.
        logits = torch.full((batch, 1, self.vocab_size), -20.0)
        logits[:, 0, 1 if length < self.prompt_len + 2 else self.config.eos_token_id] = 20.0

        attachment = None
        if compute_attachment_logits:
            attachment = torch.zeros(batch, length, length)
            query = length - 1
            # Strongly prefer a prompt constituent if the decoder incorrectly
            # exposes it. The strongest valid sentence-local alternative is the
            # previous generated token.
            attachment[:, query, self.prompt_len - 1] = 20.0
            if query > self.prompt_len:
                attachment[:, query, query - 1] = 10.0
        return SimpleNamespace(logits=logits, attachment_logits=attachment)


def test_generation_attachment_stack_is_root_free_and_sentence_local():
    model = SentenceLocalGenerationToy()
    output = model.pushdown_generate(
        torch.tensor([[0, 0, 0, 0, 0]]),
        prompt_spans=torch.tensor([[[1, 1, 2], [3, 3, 4]]]),
        max_steps=3,
        beam_size=1,
        max_reduce=None,
        eos_token_id=2,
        pad_token_id=99,
        use_attachment_head=True,
    )

    assert output.token_ids[0, 0].tolist() == [1, 1, 2]
    # The second generated token may naturally reduce to the first one.
    assert (model.prompt_len, model.prompt_len + 1, model.prompt_len + 1) in model.observed_spans
    # Prompt spans remain available as attention history, but no newly closed
    # constituent may cross the prompt/summary sentence boundary.
    assert not any(
        left < model.prompt_len <= right
        for left, _, right in model.observed_spans
    )
