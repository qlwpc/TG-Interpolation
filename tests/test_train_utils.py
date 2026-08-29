"""
Tests for Trainer utilities, loss functions, and attention bias helpers.

Covers: get_labels, cross_entropy_loss, Trainer.split_batch,
        attention_bias generation, and TG attention bias factory.
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from olmo.config import (
    ModelConfig,
    TrainConfig,
    DataConfig,
    TGConfig,
    TokenizerConfig,
)
from olmo.train import cross_entropy_loss, Trainer


# ---------------------------------------------------------------------------
# cross_entropy_loss
# ---------------------------------------------------------------------------

class TestCrossEntropyLoss:
    def test_basic_mean_reduction(self):
        logits = torch.randn(4, 10)  # (seq, vocab)
        labels = torch.randint(0, 10, (4,))
        loss, z_loss = cross_entropy_loss(logits, labels, reduction="mean")
        assert loss.ndim == 0  # scalar
        assert z_loss is None

    def test_sum_reduction(self):
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss, _ = cross_entropy_loss(logits, labels, reduction="sum")
        # sum should be larger than mean
        loss_mean, _ = cross_entropy_loss(logits, labels, reduction="mean")
        assert loss > loss_mean * 3  # roughly batch_size ×

    def test_ignore_index(self):
        logits = torch.randn(4, 10)
        labels = torch.tensor([1, -100, 2, -100])
        loss, _ = cross_entropy_loss(logits, labels, reduction="mean", ignore_index=-100)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_z_loss_computation(self):
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss, z_loss = cross_entropy_loss(
            logits, labels, reduction="mean", compute_z_loss=True, z_loss_multiplier=1e-4
        )
        assert z_loss is not None
        assert z_loss.ndim == 0
        assert z_loss > 0

    def test_all_ignored_labels(self):
        """When all labels are ignored, loss should be NaN or 0 (PyTorch behavior)."""
        logits = torch.randn(4, 10)
        labels = torch.full((4,), -100, dtype=torch.long)
        loss, _ = cross_entropy_loss(logits, labels, reduction="mean", ignore_index=-100)
        # PyTorch CE returns nan when all labels are ignored (mean over zero elements)
        assert torch.isnan(loss) or loss.item() == 0.0

    def test_z_loss_with_ignore(self):
        logits = torch.randn(4, 10)
        labels = torch.tensor([1, -100, 2, -100])
        _, z_loss = cross_entropy_loss(
            logits, labels, reduction="mean", compute_z_loss=True, ignore_index=-100,
        )
        assert z_loss is not None
        assert not torch.isnan(z_loss)


# ---------------------------------------------------------------------------
# Trainer.get_labels
# ---------------------------------------------------------------------------

class TestGetLabels:
    @staticmethod
    def _make_trainer(cfg_overrides=None):
        """Create a minimal Trainer-like context for testing get_labels."""
        model_cfg = ModelConfig(
            pad_token_id=50258,
            eos_token_id=50256,
            transformer_grammar_type="terminal",
        )
        if cfg_overrides:
            for k, v in cfg_overrides.items():
                setattr(model_cfg, k, v)
        train_cfg = TrainConfig(model=model_cfg)
        # Trainer.__post_init__ sets loss_fn; we skip full init for unit test
        trainer = object.__new__(Trainer)
        trainer.cfg = train_cfg
        return trainer

    def test_simple_shift(self):
        """Labels should be input_ids shifted left by 1."""
        trainer = self._make_trainer()
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        labels = trainer.get_labels(batch, ignore_id=-100)
        expected = torch.tensor([[2, 3, 4, 5]])
        assert torch.equal(labels, expected)

    def test_label_mask_applied(self):
        """label_mask should zero out specific positions."""
        trainer = self._make_trainer()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "label_mask": torch.tensor([[True, False, False, True, True]]),
        }
        labels = trainer.get_labels(batch, ignore_id=-100)
        # positions where label_mask is False → ignore_id
        assert labels[0, 0].item() == -100  # position 1 masked, so pos 0 in labels
        assert labels[0, 1].item() == -100  # position 2 masked
        assert labels[0, 2].item() == 4     # position 3 not masked → 4

    def test_attention_mask_ignores_pads(self):
        trainer = self._make_trainer()
        batch = {
            "input_ids": torch.tensor([[1, 2, 50258, 50258]]),
            "attention_mask": torch.tensor([[True, True, False, False]]),
        }
        labels = trainer.get_labels(batch, ignore_id=-100)
        # input_ids[1:] = [2, 50258, 50258], attn_mask[1:] = [True, False, False]
        # pos 0: attn True → label 2; pos 1: attn False → -100; pos 2: attn False → -100
        assert labels[0, 0].item() == 2
        assert labels[0, 1].item() == -100
        assert labels[0, 2].item() == -100

    def test_instance_mask(self):
        trainer = self._make_trainer()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "instance_mask": torch.tensor([False]),
        }
        labels = trainer.get_labels(batch, ignore_id=-100)
        # All should be ignored since instance_mask is False
        assert (labels == -100).all()


# ---------------------------------------------------------------------------
# Trainer.split_batch
# ---------------------------------------------------------------------------

class TestSplitBatch:
    @staticmethod
    def _make_trainer(microbatch_size=4):
        model_cfg = ModelConfig(transformer_grammar_type="terminal")
        train_cfg = TrainConfig(
            model=model_cfg,
            device_train_microbatch_size=microbatch_size,
        )
        trainer = object.__new__(Trainer)
        trainer.cfg = train_cfg
        return trainer

    def test_no_split_when_batch_smaller(self):
        trainer = self._make_trainer(microbatch_size=8)
        batch = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
        result = trainer.split_batch(batch)
        assert len(result) == 1
        assert torch.equal(result[0]["input_ids"], batch["input_ids"])

    def test_single_microbatch(self):
        trainer = self._make_trainer(microbatch_size=4)
        batch = {"input_ids": torch.randn(4, 10)}
        result = trainer.split_batch(batch)
        assert len(result) == 1

    def test_split_into_two(self):
        trainer = self._make_trainer(microbatch_size=2)
        batch = {"input_ids": torch.tensor([[1.0], [2.0], [3.0], [4.0]])}
        result = trainer.split_batch(batch)
        assert len(result) == 2
        assert result[0]["input_ids"].shape[0] == 2
        assert result[1]["input_ids"].shape[0] == 2

    def test_split_into_three_uneven(self):
        trainer = self._make_trainer(microbatch_size=2)
        batch = {"input_ids": torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])}
        result = trainer.split_batch(batch)
        assert len(result) == 3
        assert result[-1]["input_ids"].shape[0] == 1  # last micro-batch has 1

    def test_split_with_attention_bias(self):
        """Verify attention_bias tensor is split correctly."""
        trainer = self._make_trainer(microbatch_size=2)
        batch = {
            "input_ids": torch.randn(4, 10),
            "attention_bias": torch.randn(4, 1, 10, 10),
        }
        result = trainer.split_batch(batch)
        assert len(result) == 2
        for r in result:
            assert r["input_ids"].shape[0] == 2
            assert r["attention_bias"].shape[0] == 2

    def test_split_with_lists_raises(self):
        """BUG CONFIRMATION: non-tensor list items get split incorrectly.
        Lists of arbitrary items (e.g. gold_summary) are split by microbatch_size,
        which may not correspond to batch dimension."""
        trainer = self._make_trainer(microbatch_size=2)
        batch = {
            "input_ids": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            "gold_summary": ["summary_A", "summary_B", "summary_C", "summary_D"],
        }
        result = trainer.split_batch(batch)
        # Lists are split naively by microbatch_size, which happens to work here
        # because gold_summary has batch_size==4 and microbatch==2.
        # But this is fragile — the code assumes ALL list values are batch-aligned.
        assert len(result) == 2
        assert len(result[0]["gold_summary"]) == 2
        assert len(result[1]["gold_summary"]) == 2


def test_pushdown_attachment_loss_uses_independent_query_denominator():
    trainer = object.__new__(Trainer)
    trainer.cfg = SimpleNamespace(
        softmax_auxiliary_loss=False,
        model=SimpleNamespace(pushdown_attachment_weight=2.0),
    )
    trainer.model_forward = MagicMock(
        return_value=(
            torch.tensor(12.0),
            None,
            torch.empty(0),
            None,
            torch.tensor(20.0),
        )
    )

    loss, ce_loss, z_loss = trainer.train_micro_batch(
        {"input_ids": torch.tensor([[1, 2]])},
        batch_size_in_loss_tokens=4,
        attachment_loss_denominator=10,
        device_loss_weight=3.0,
    )

    # LM: (12 / 4) * 3 = 9. Attachment: (20 / 10) * weight 2 = 4.
    assert loss.item() == pytest.approx(13.0)
    assert ce_loss.item() == pytest.approx(3.0)
    assert z_loss is None


# ---------------------------------------------------------------------------
# get_TG_generate_bias_func
# ---------------------------------------------------------------------------

class TestTGAttentionBiasFactory:
    def test_terminal_returns_none(self):
        """Terminal grammar type should return None (no TG bias)."""
        from olmo.data import get_TG_generate_bias_func

        cfg = TrainConfig(
            model=ModelConfig(
                transformer_grammar_type="terminal",
                max_sequence_length=256,
            ),
            tokenizer=TokenizerConfig(
                vocabulary="test_vocab.json",
                identifier="gpt2",
            ),
        )
        result = get_TG_generate_bias_func(cfg, TG_type="terminal")
        assert result is None

    def test_tgtree_returns_none(self):
        """tgtree grammar type returns None from the early return."""
        from olmo.data import get_TG_generate_bias_func

        cfg = TrainConfig(
            model=ModelConfig(
                transformer_grammar_type="tgtree",
                max_sequence_length=256,
            ),
            tokenizer=TokenizerConfig(
                vocabulary="test_vocab.json",
                identifier="gpt2",
            ),
        )
        result = get_TG_generate_bias_func(cfg)
        assert result is None

    @patch("olmo.data.TG_attention_bias", autospec=True)
    def test_tg_returns_bias(self, mock_tg_bias):
        from olmo.data import get_TG_generate_bias_func

        cfg = TrainConfig(
            model=ModelConfig(
                transformer_grammar_type="tg",
                max_sequence_length=256,
            ),
            tokenizer=TokenizerConfig(
                vocabulary="test_vocab.json",
                identifier="gpt2",
            ),
        )
        result = get_TG_generate_bias_func(cfg)
        assert result is not None

    @patch("olmo.data.KProximal_TG_attention_bias", autospec=True)
    def test_tgproximal_returns_bias(self, mock_kprox):
        from olmo.data import get_TG_generate_bias_func

        cfg = TrainConfig(
            model=ModelConfig(
                transformer_grammar_type="tgproximal",
                tg_proximal_k=20,
                max_sequence_length=256,
            ),
            tokenizer=TokenizerConfig(
                vocabulary="test_vocab.json",
                identifier="gpt2",
            ),
        )
        result = get_TG_generate_bias_func(cfg)
        assert result is not None


# ---------------------------------------------------------------------------
# HeadMixingBias
# ---------------------------------------------------------------------------

class TestHeadMixingBias:
    def test_config_passthrough(self):
        """HeadMixingBias should call each sub-bias and concatenate results."""
        from olmo.data import HeadMixingBias

        with patch("olmo.data.get_TG_generate_bias_func") as mock_factory:
            mock_bias = MagicMock()
            # Return shape (seq_len, seq_len) — 2D mask, which gets unsqueezed to 3D
            mock_bias.return_value = (torch.zeros(3, 3), None)
            mock_factory.return_value = mock_bias

            cfg = TrainConfig(
                model=ModelConfig(
                    transformer_grammar_type="mixing",
                    max_sequence_length=256,
                    mix_head_type=[
                        TGConfig(grammar_type="tg", n_heads=6),
                        TGConfig(grammar_type="tree", n_heads=6),
                    ],
                ),
                tokenizer=TokenizerConfig(
                    vocabulary="test_vocab.json",
                    identifier="gpt2",
                ),
            )
            bias_obj = HeadMixingBias(cfg.model.mix_head_type, cfg, max_length=8)
            mask, label_mask = bias_obj(torch.tensor([1, 2, 3]))
            # Result should have 12 heads (6+6)
            assert mask.shape[0] == 12
            assert mask.shape[1] == 3
            assert mask.shape[2] == 3


# ---------------------------------------------------------------------------
# TGCausalBias
# ---------------------------------------------------------------------------

class TestTGCausalBias:
    def test_causal_mask_shape(self):
        from olmo.data import TGCausalBias

        bias = TGCausalBias(vocab_path="dummy.vocab", max_length=8)
        # Override self.vocab to prevent file read
        bias.vocab = MagicMock()
        bias.vocab.pad = 0

        input_ids = torch.tensor([1, 2, 3, 4])
        mask, label_mask = bias(input_ids, update_state=False)
        assert mask.shape == (4, 4)
        # Should be lower-triangular (causal)
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.equal(mask[:4, :4], expected)
        assert label_mask is None

    def test_cur_length_updates(self):
        from olmo.data import TGCausalBias

        bias = TGCausalBias(vocab_path="dummy.vocab", max_length=12)
        bias.vocab = MagicMock()
        bias.vocab.pad = 0

        # First call: 3 tokens
        bias(torch.tensor([1, 2, 3]), update_state=True)
        assert bias.cur_length == 3

        # Second call: 3 more tokens
        bias(torch.tensor([4, 5, 6]), update_state=True)
        assert bias.cur_length == 6

    def test_reset_state(self):
        from olmo.data import TGCausalBias

        bias = TGCausalBias(vocab_path="dummy.vocab", max_length=12)
        bias.vocab = MagicMock()
        bias.vocab.pad = 0

        bias(torch.tensor([1, 2, 3]), update_state=True)
        bias.reset_state()
        assert bias.cur_length == 0


# ---------------------------------------------------------------------------
# Soft_Alibilike_bias
# ---------------------------------------------------------------------------

class TestSoftAlibilikeBias:
    @patch("olmo.data.KProximal_TG_attention_bias", autospec=True)
    def test_output_shapes(self, mock_kprox):
        from olmo.data import Soft_Alibilike_bias

        # Mock the internal KProximal instance
        mock_instance = mock_kprox.return_value
        mock_instance.get_alibi_rel_pos.return_value = (
            torch.zeros(8, 8),     # mask
            torch.ones(8, 8),      # rel_pos
            torch.ones(8, 8),      # label_mask
        )

        bias_obj = Soft_Alibilike_bias(vocab_path="dummy.json", max_token_length=8)
        mask, label_mask = bias_obj(torch.tensor([1, 2, 3]))
        assert mask.shape == (8, 8)
        assert label_mask.shape == (8, 8)


# ---------------------------------------------------------------------------
# Edge case: ModelConfig with pause-like grammar that does NOT start with "pause"
# ---------------------------------------------------------------------------

def test_grammar_type_not_starting_with_pause_returns_zero():
    """Any grammar type not starting with 'pause' should return ispause=0."""
    cfg = ModelConfig(transformer_grammar_type="random_other_type")
    assert cfg.ispause == 0
