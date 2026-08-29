"""
Tests for configuration classes in olmo.config.

Covers: ModelConfig validation, ispause edge cases, scheduler boundaries,
        data config validation, and OmegaConf resolvers.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from olmo.config import (
    ModelConfig,
    TrainConfig,
    SchedulerConfig,
    SchedulerType,
    DataConfig,
    MemMapFileFormat,
    TGConfig,
    PaddingDirection,
    InitFnType,
    OptimizerType,
    SchedulerUnits,
)
from olmo.exceptions import OLMoConfigurationError


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_default_values(self):
        cfg = ModelConfig()
        assert cfg.d_model == 768
        assert cfg.n_heads == 12
        assert cfg.n_layers == 12
        assert cfg.vocab_size == 50257
        assert cfg.max_sequence_length == 1024

    def test_effective_n_kv_heads_default(self):
        cfg = ModelConfig()
        assert cfg.effective_n_kv_heads == 12  # equals n_heads

    def test_effective_n_kv_heads_explicit(self):
        cfg = ModelConfig(n_kv_heads=4)
        assert cfg.effective_n_kv_heads == 4

    def test_effective_n_kv_heads_mqa(self):
        cfg = ModelConfig(multi_query_attention=True)
        assert cfg.effective_n_kv_heads == 1

    def test_effective_n_kv_heads_gqa_conflict_raises(self):
        cfg = ModelConfig(n_kv_heads=4, multi_query_attention=True)
        with pytest.raises(OLMoConfigurationError):
            _ = cfg.effective_n_kv_heads

    # ---- ispause property ----

    @pytest.mark.parametrize("grammar_type,expected", [
        ("terminal", 0),
        ("tg", 0),
        ("tree", 0),
        ("tgproximal", 0),
        ("tgnomask", 0),
        ("tgheight", 0),
        ("pause1", 1),
        ("pause2", 2),
        ("pause5", 5),
        ("pause10", 10),
        ("pause1/2", 1),     # special case: "1/2" → 1
        ("pause1/2_label", 1),  # special case
        ("tree_shuffle", 0),
    ])
    def test_ispause_known_types(self, grammar_type, expected):
        """Verify ispause returns correct values for known types, no crash."""
        cfg = ModelConfig(transformer_grammar_type=grammar_type)
        assert cfg.ispause == expected, f"Failed for type={grammar_type}"

    def test_ispause_bare_pause_defaults_to_one(self):
        """Bare 'pause' (no number suffix) is treated as 'pause1' → ispause == 1.

        See pause_spec_from_grammar_type (olmo/data/util.py): a bare "pause"
        no longer crashes; it parses as the (1, 1) spec.
        """
        cfg = ModelConfig(transformer_grammar_type="pause")
        assert cfg.ispause == 1

    def test_ispause_short_string_ok(self):
        """Short strings like 'tg', 'tree' should not accidentally match 'pause'."""
        cfg = ModelConfig(transformer_grammar_type="tg")
        assert cfg.ispause == 0

    def test_ispause_truthiness_edge(self):
        """ispause returns an int, so bool(0)==False but bool(N)>0 is True."""
        cfg = ModelConfig(transformer_grammar_type="pause3")
        assert cfg.ispause == 3
        assert bool(cfg.ispause) is True

    def test_mix_head_type_valid(self):
        cfg = ModelConfig(
            mix_head_type=[TGConfig(grammar_type="tg", n_heads=6)]
        )
        assert len(cfg.mix_head_type) == 1

    def test_legacy_null_grammar_loads_as_terminal(self, tmp_path):
        path = tmp_path / "legacy_terminal.yaml"
        path.write_text("model:\n  transformer_grammar_type: null\n")
        cfg = TrainConfig.load(path, validate_paths=False)
        assert cfg.model.transformer_grammar_type == "terminal"

    def test_legacy_self_referential_workspace_loads_relative(self, tmp_path):
        path = tmp_path / "legacy_workspace.yaml"
        path.write_text(
            "workspace: ${workspace}\n"
            "tokenizer:\n"
            "  vocabulary: ${workspace}/tokenizer.json\n"
        )
        cfg = TrainConfig.load(path, validate_paths=False)
        assert cfg.workspace == "."
        assert cfg.tokenizer.vocabulary == "./tokenizer.json"

    @pytest.mark.parametrize(
        "mix_head_type,match",
        [
            ([], "requires an explicit"),
            ([TGConfig(grammar_type="tg", n_heads=6)], "allocates 6 heads"),
        ],
    )
    def test_mixing_requires_complete_head_allocation(self, mix_head_type, match):
        from olmo.data import get_TG_generate_bias_func

        cfg = TrainConfig(
            model=ModelConfig(
                transformer_grammar_type="mixing",
                n_heads=12,
                mix_head_type=mix_head_type,
            )
        )
        with pytest.raises(OLMoConfigurationError, match=match):
            get_TG_generate_bias_func(cfg)

    # ---- Validation ----

    def test_embedding_size_smaller_than_vocab_raises_in_olmo(self):
        """embedding_size < vocab_size is rejected by OLMo.__init__."""
        from olmo.model import OLMo

        cfg = ModelConfig(
            vocab_size=1000,
            embedding_size=500,
            block_group_size=1,
            init_device="meta",
        )
        with pytest.raises(OLMoConfigurationError, match="embedding"):
            OLMo(cfg, init_params=False)

    def test_alibi_and_rope_mutually_exclusive_in_olmo(self):
        from olmo.model import OLMo

        cfg = ModelConfig(
            alibi=True, rope=True, block_group_size=1, init_device="meta",
            flash_attention=False, flex_attention=False,
        )
        with pytest.raises(OLMoConfigurationError):
            OLMo(cfg, init_params=False)

    def test_block_group_size_must_divide_n_layers(self):
        from olmo.model import OLMo

        cfg = ModelConfig(n_layers=12, block_group_size=5, init_device="meta")
        with pytest.raises(OLMoConfigurationError, match="divisible"):
            OLMo(cfg, init_params=False)

    def test_block_group_size_zero_raises(self):
        from olmo.model import OLMo

        cfg = ModelConfig(n_layers=12, block_group_size=0, init_device="meta")
        with pytest.raises(OLMoConfigurationError):
            OLMo(cfg, init_params=False)

    def test_treereg_layer_is_one_based_and_range_checked(self):
        from olmo.model import OLMo

        cfg = ModelConfig(
            transformer_grammar_type="treereg",
            n_layers=2,
            treereg_layer=0,
            treereg_n_heads=1,
            init_device="meta",
        )
        with pytest.raises(OLMoConfigurationError, match="1-based"):
            OLMo(cfg, init_params=False)

    def test_treereg_rejects_block_groups(self):
        from olmo.model import OLMo

        cfg = ModelConfig(
            transformer_grammar_type="treereg",
            n_layers=2,
            block_group_size=2,
            treereg_layer=1,
            treereg_n_heads=1,
            init_device="meta",
        )
        with pytest.raises(OLMoConfigurationError, match="block_group_size=1"):
            OLMo(cfg, init_params=False)


# ---------------------------------------------------------------------------
# Scheduler boundary conditions
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_cosine_warmup_decreasing(self):
        """LR should decrease monotonically after warmup."""
        from olmo.optim import CosWithWarmup

        sched = CosWithWarmup(
            warmup_steps=100, alpha_f=0.1,
            grad_clip_warmup_steps=None, grad_clip_warmup_factor=None,
            warmup_min_lr=None, min_lr=None,
        )
        initial_lr = 1e-3
        max_steps = 1000
        # After warmup, LR should be non-increasing
        prev = sched.get_lr(initial_lr, 101, max_steps)
        for step in range(150, max_steps, 50):
            cur = sched.get_lr(initial_lr, step, max_steps)
            assert cur <= prev + 1e-10, f"LR increased at step {step}: {cur} > {prev}"
            prev = cur

    def test_cosine_hits_min_lr_at_max_steps(self):
        from olmo.optim import CosWithWarmup

        sched = CosWithWarmup(
            warmup_steps=100, alpha_f=0.1,
            grad_clip_warmup_steps=None, grad_clip_warmup_factor=None,
            warmup_min_lr=1e-6, min_lr=1e-6,
        )
        lr_at_end = sched.get_lr(1e-3, 10000, 10000)
        assert lr_at_end == pytest.approx(1e-6)

    def test_linear_with_warmup_zero_min_lr(self):
        from olmo.optim import LinearWithWarmup

        sched = LinearWithWarmup(
            warmup_steps=10, alpha_f=0.0, min_lr=0.0,
            grad_clip_warmup_steps=None, grad_clip_warmup_factor=None,
            warmup_min_lr=None,
        )
        assert sched.get_lr(1.0, 100000, 100) == 0.0

    def test_inv_sqrt_never_exceeds_initial(self):
        from olmo.optim import InvSqrtWithWarmup

        sched = InvSqrtWithWarmup(
            warmup_steps=10,
            grad_clip_warmup_steps=None, grad_clip_warmup_factor=None,
            warmup_min_lr=None,
        )
        for step in range(20, 1000):
            assert sched.get_lr(1.0, step, 10000) <= 1.0

    def test_constant_with_warmup(self):
        from olmo.optim import ConstantWithWarmupScheduler

        sched = ConstantWithWarmupScheduler(
            warmup_steps=100,
            grad_clip_warmup_steps=None, grad_clip_warmup_factor=None,
            warmup_min_lr=None,
        )
        # In warmup
        assert sched.get_lr(1.0, 0, 1000) < 1.0
        assert sched.get_lr(1.0, 50, 1000) < 1.0
        # Post warmup
        assert sched.get_lr(1.0, 100, 1000) == pytest.approx(1.0)
        assert sched.get_lr(1.0, 500, 1000) == pytest.approx(1.0)

    def test_warmup_min_lr_zero(self):
        """When warmup_min_lr=0, first step should have near-zero LR."""
        from olmo.optim import CosWithWarmup

        sched = CosWithWarmup(
            warmup_steps=100, alpha_f=0.1,
            warmup_min_lr=0.0,
            grad_clip_warmup_steps=None, grad_clip_warmup_factor=None,
            min_lr=None,
        )
        assert sched.get_lr(1e-3, 0, 1000) == 0.0
        assert sched.get_lr(1e-3, 50, 1000) == pytest.approx(5e-4)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_effective_memmap_dtype_valid(self):
        cfg = DataConfig(memmap_dtype="uint16")
        import numpy as np
        assert cfg.effective_memmap_dtype == np.uint16

    def test_effective_memmap_dtype_invalid_raises(self):
        cfg = DataConfig(memmap_dtype="nonexistent_type")
        with pytest.raises(TypeError):
            _ = cfg.effective_memmap_dtype

    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.num_workers == 0
        assert cfg.pad_direction == PaddingDirection.right
        assert cfg.memmap_format == MemMapFileFormat.auto
        assert cfg.generate_attention_mask is False

    def test_memmap_format_accepts_enum_value(self):
        cfg = DataConfig(memmap_format=MemMapFileFormat.raw)
        assert str(cfg.memmap_format) == "raw"


# ---------------------------------------------------------------------------
# SchedulerConfig validation
# ---------------------------------------------------------------------------

class TestSchedulerConfig:
    def test_units_default(self):
        cfg = SchedulerConfig()
        assert cfg.units == SchedulerUnits.steps

    def test_invalid_t_warmup_type_converted(self):
        """t_warmup is Union[int, float], OmegaConf handles conversion."""
        cfg = SchedulerConfig(t_warmup=100.5)
        assert cfg.t_warmup == 100.5

    def test_cosine_default_alpha_f(self):
        cfg = SchedulerConfig(name=SchedulerType.cosine_with_warmup)
        assert cfg.alpha_f == 0.1
