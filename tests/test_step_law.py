"""Tests for scripts/step_law.py — paper Eq.(8) Step-Law audit tool.

The paper (Li et al. 2025) claims lr and batch size follow
    lr(N, D) = 1.79 * N^-0.713 * D^0.307
    B(D)     = 0.58  * D^0.571          (batch size in TOKENS)
This module implements the formula plus helpers that derive N (non-embedding
params) from a ModelConfig and D (training tokens) from data paths.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch  # noqa: F401  (ensures torch is importable before olmo imports)

from scripts.step_law import (
    audit_config,
    count_data_tokens,
    count_non_embedding_params,
    step_law_batch_tokens,
    step_law_lr,
)


# ---------------------------------------------------------------------------
# Formula properties (non-circular: power-law scaling + loose literal checks)
# ---------------------------------------------------------------------------

class TestStepLawFormula:
    def test_lr_scales_with_N_exponent(self):
        # lr(N*10, D) / lr(N, D) == 10^-0.713 exactly for a power law
        base = step_law_lr(1e8, 1e10)
        scaled = step_law_lr(1e9, 1e10)
        assert scaled == pytest.approx(base * 10 ** -0.713, rel=1e-12)

    def test_lr_scales_with_D_exponent(self):
        base = step_law_lr(1e8, 1e10)  # noqa: F841
        scaled = step_law_lr(1e8, 1e11)
        assert scaled == pytest.approx(base * 10 ** 0.307, rel=1e-12)

    def test_batch_scales_with_D_exponent(self):
        base = step_law_batch_tokens(1e10)
        assert step_law_batch_tokens(1e11) == pytest.approx(base * 10 ** 0.571, rel=1e-12)

    def test_lr_literal_value(self):
        # hand-checked: 1.79 * 1e8^-0.713 * 1e10^0.307 ≈ 4.158e-3
        assert step_law_lr(1e8, 1e10) == pytest.approx(0.0041582, rel=1e-3)

    def test_batch_literal_value(self):
        # hand-checked: 0.58 * 1e10^0.571 ≈ 297,482 tokens ≈ 145.25 seqs of 2048
        assert step_law_batch_tokens(1e10) == pytest.approx(297482, rel=1e-3)
        assert step_law_batch_tokens(1e10) / 2048 == pytest.approx(145.25, rel=1e-3)


# ---------------------------------------------------------------------------
# N from a ModelConfig — count must match a real OLMo build
# ---------------------------------------------------------------------------

def _tiny_model_cfg(**overrides):
    from olmo.config import ModelConfig

    defaults = dict(
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=128,
        mlp_hidden_size=128,
        flash_attention=False,
        flex_attention=False,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _build_olmo(model_cfg):
    from olmo.model import OLMo

    return OLMo(model_cfg)


class TestCountNonEmbeddingParams:
    def test_matches_real_model_weight_tying(self):
        cfg = _tiny_model_cfg()
        model = _build_olmo(cfg)
        actual = model.num_params(include_embedding=False)
        assert count_non_embedding_params(cfg) == actual
        assert actual > 0

    def test_matches_real_model_no_tying_mlp_ratio(self):
        cfg = _tiny_model_cfg(weight_tying=False, mlp_hidden_size=None, mlp_ratio=4)
        model = _build_olmo(cfg)
        actual = model.num_params(include_embedding=False)
        assert count_non_embedding_params(cfg) == actual
        assert actual > 0

    def test_scales_with_layers(self):
        n2 = count_non_embedding_params(_tiny_model_cfg(n_layers=2))
        n4 = count_non_embedding_params(_tiny_model_cfg(n_layers=4))
        assert n4 > 2 * n2 - 1000  # strictly more than double minus norm terms

    def test_100m_scale_is_plausible(self):
        # Paper Table 8 "100M" = d768/12L/12H. This repo's SwiGLU uses gate/up
        # of hidden/2 each, so the non-embedding count is ~70.9M (embedding
        # 50320×768 ≈ 38.6M brings the total to ~109M ≈ "100M").  This exact
        # N is what makes the Step Law reproduce terminal.yaml's lr=0.005323
        # at D≈10.06B: 1.79·N^-0.713·D^0.307 ≈ 0.005322.
        cfg = _tiny_model_cfg(d_model=768, n_layers=12, n_heads=12, mlp_hidden_size=3072,
                              vocab_size=50320)
        n = count_non_embedding_params(cfg)
        assert 6.9e7 < n < 7.3e7


# ---------------------------------------------------------------------------
# D from data paths
# ---------------------------------------------------------------------------

class TestCountDataTokens:
    def test_uint16_npy(self, tmp_path):
        p = tmp_path / "train.npy"
        np.save(p, np.arange(1000, dtype=np.uint16))
        assert count_data_tokens([str(p)]) == 1000

    def test_multiple_files_summed(self, tmp_path):
        a = tmp_path / "a.npy"
        b = tmp_path / "b.npy"
        np.save(a, np.arange(300, dtype=np.uint16))
        np.save(b, np.arange(500, dtype=np.uint16))
        assert count_data_tokens([str(a), str(b)]) == 800

    def test_glob_pattern(self, tmp_path):
        for i in range(3):
            np.save(tmp_path / f"shard-{i:05d}.npy", np.arange(100, dtype=np.uint16))
        assert count_data_tokens([str(tmp_path / "shard-*.npy")]) == 300

    def test_raw_bin_supported(self, tmp_path):
        p = tmp_path / "raw.bin"
        np.arange(64, dtype=np.uint16).tofile(p)
        assert count_data_tokens([str(p)]) == 64

    def test_missing_path_raises_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no data files"):
            count_data_tokens([str(tmp_path / "nope" / "*.npy")])


# ---------------------------------------------------------------------------
# audit_config: end-to-end on an in-memory TrainConfig
# ---------------------------------------------------------------------------

class TestAuditConfig:
    def test_audit_fields(self, tmp_path):
        from olmo.config import DataConfig, OptimizerConfig, TrainConfig

        p = tmp_path / "train.npy"
        np.save(p, np.arange(10_000, dtype=np.uint16))
        cfg = TrainConfig(
            model=_tiny_model_cfg(vocab_size=128),
            data=DataConfig(paths=[str(p)]),
            optimizer=OptimizerConfig(learning_rate=0.004),
            global_train_batch_size=145,
        )
        cfg.model.max_sequence_length = 2048

        out = audit_config(cfg)

        assert out["N"] == count_non_embedding_params(cfg.model)
        assert out["D"] == 10_000
        assert out["lr_pred"] == pytest.approx(step_law_lr(out["N"], out["D"]))
        assert out["batch_tokens_pred"] == pytest.approx(step_law_batch_tokens(out["D"]))
        assert out["batch_seq_pred"] == pytest.approx(step_law_batch_tokens(out["D"]) / 2048)
        assert out["lr_actual"] == 0.004
        assert out["batch_actual"] == 145
        # implied D that would reproduce the actual lr, holding N fixed
        assert out["implied_D_from_lr"] > 0
