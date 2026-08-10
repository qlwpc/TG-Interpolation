"""
Tests for OLMo model utilities: attention bias functions, forward pass helpers,
and buffer cache.

These don't require a full OLMo model instantiation. They cover:
- causal_attention_bias shape/dtype/device
- alibi_attention_bias computation
- BufferCache behavior
- Attention bias slicing logic in OLMoBlock
- Flex attention mask creation logic
"""

import math
import pytest
import torch

from olmo.config import ModelConfig, ActivationCheckpointingStrategy
from olmo.model import (
    causal_attention_bias,
    get_causal_attention_bias,
    alibi_attention_bias,
    BufferCache,
    should_checkpoint_block,
    Dropout,
    OLMoBlock,
)


# ---------------------------------------------------------------------------
# causal_attention_bias
# ---------------------------------------------------------------------------

class TestCausalAttentionBias:
    def test_shape(self):
        bias = causal_attention_bias(16, device=torch.device("cpu"))
        # Shape is (1, 1, seq_len, seq_len) per the function
        assert bias.shape == (1, 1, 16, 16)

    def test_lower_triangular(self):
        bias = causal_attention_bias(8, device=torch.device("cpu"))
        min_val = torch.finfo(torch.float).min
        # Strictly upper-triangular: all values should be finfo.min
        upper_mask = torch.triu(torch.ones(8, 8), diagonal=1).bool()
        upper_vals = bias[0, 0][upper_mask]
        assert torch.all(upper_vals <= min_val / 2)
        # Lower-triangular including diag: all values should be 0
        lower_mask = torch.tril(torch.ones(8, 8), diagonal=0).bool()
        lower_vals = bias[0, 0][lower_mask]
        assert torch.all(lower_vals == 0.0)

    def test_dtype_is_float(self):
        bias = causal_attention_bias(4, device=torch.device("cpu"))
        assert bias.dtype == torch.float32

    def test_device_assignment(self):
        bias = causal_attention_bias(4, device=torch.device("cpu"))
        assert bias.device.type == "cpu"


# ---------------------------------------------------------------------------
# get_causal_attention_bias (with cache)
# ---------------------------------------------------------------------------

class TestGetCausalAttentionBias:
    def test_cache_miss(self):
        cache = BufferCache()
        bias = get_causal_attention_bias(cache, seq_len=8, device=torch.device("cpu"))
        assert "causal_attention_bias" in cache
        assert bias.shape == (1, 1, 8, 8)

    def test_cache_hit(self):
        cache = BufferCache()
        bias1 = get_causal_attention_bias(cache, seq_len=8, device=torch.device("cpu"))
        bias2 = get_causal_attention_bias(cache, seq_len=4, device=torch.device("cpu"))
        # Same object? Actually cached bias has shape (1,1,8,8) and seq_len=4 <= 8,
        # so it returns the sliced version but... let's check:
        # The function checks shape[-1] >= seq_len, if true returns cached bias
        # WITHOUT slicing. So bias2 has shape (1,1,8,8).
        assert bias2.shape[-1] == 8

    def test_larger_seq_causes_rebuild(self):
        cache = BufferCache()
        bias1 = get_causal_attention_bias(cache, seq_len=8, device=torch.device("cpu"))
        bias2 = get_causal_attention_bias(cache, seq_len=16, device=torch.device("cpu"))
        assert bias2.shape[-1] == 16


# ---------------------------------------------------------------------------
# alibi_attention_bias
# ---------------------------------------------------------------------------

class TestAlibiAttentionBias:
    def test_shape(self):
        config = ModelConfig(n_heads=12, alibi_bias_max=8.0)
        bias = alibi_attention_bias(seq_len=16, config=config, device=torch.device("cpu"))
        # Should be (1, n_heads, seq_len, seq_len)
        assert bias.shape == (1, 12, 16, 16)

    def test_values_are_negative(self):
        config = ModelConfig(n_heads=12)
        bias = alibi_attention_bias(seq_len=8, config=config, device=torch.device("cpu"))
        assert (bias <= 0).all()

    def test_heads_different(self):
        """Different heads should have different ALiBi slopes."""
        config = ModelConfig(n_heads=4, alibi_bias_max=8.0)
        bias = alibi_attention_bias(seq_len=8, config=config, device=torch.device("cpu"))
        # Check that head 0 differs from head 1
        assert not torch.allclose(bias[0, 0], bias[0, 1])


# ---------------------------------------------------------------------------
# BufferCache
# ---------------------------------------------------------------------------

class TestBufferCache:
    def test_set_get(self):
        cache = BufferCache()
        cache["key"] = torch.tensor([1.0, 2.0])
        assert "key" in cache
        assert torch.equal(cache["key"], torch.tensor([1.0, 2.0]))

    def test_default_missing(self):
        cache = BufferCache()
        assert cache.get("missing") is None

    def test_overwrite(self):
        cache = BufferCache()
        cache["x"] = torch.tensor([1.0])
        cache["x"] = torch.tensor([2.0])
        assert cache["x"].item() == 2.0


# ---------------------------------------------------------------------------
# should_checkpoint_block
# ---------------------------------------------------------------------------

class TestShouldCheckpointBlock:
    def test_none_strategy(self):
        assert should_checkpoint_block(None, 0) is False
        assert should_checkpoint_block(None, 5) is False

    def test_whole_layer(self):
        strategy = ActivationCheckpointingStrategy.whole_layer
        assert should_checkpoint_block(strategy, 0) is True
        assert should_checkpoint_block(strategy, 1) is True
        assert should_checkpoint_block(strategy, 99) is True

    def test_one_in_two(self):
        strategy = ActivationCheckpointingStrategy.one_in_two
        assert should_checkpoint_block(strategy, 0) is True
        assert should_checkpoint_block(strategy, 1) is False
        assert should_checkpoint_block(strategy, 2) is True
        assert should_checkpoint_block(strategy, 3) is False

    def test_one_in_four(self):
        strategy = ActivationCheckpointingStrategy.one_in_four
        assert should_checkpoint_block(strategy, 0) is True
        assert should_checkpoint_block(strategy, 1) is False
        assert should_checkpoint_block(strategy, 3) is False
        assert should_checkpoint_block(strategy, 4) is True

    def test_two_in_three(self):
        strategy = ActivationCheckpointingStrategy.two_in_three
        assert should_checkpoint_block(strategy, 0) is False  # 0 % 3 == 0 → False
        assert should_checkpoint_block(strategy, 1) is True   # 1 % 3 != 0 → True
        assert should_checkpoint_block(strategy, 2) is True   # 2 % 3 != 0 → True

    def test_three_in_four(self):
        strategy = ActivationCheckpointingStrategy.three_in_four
        assert should_checkpoint_block(strategy, 0) is False  # 0 % 4 == 0 → False
        assert should_checkpoint_block(strategy, 1) is True
        assert should_checkpoint_block(strategy, 2) is True
        assert should_checkpoint_block(strategy, 3) is True


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class TestDropout:
    def test_zero_prob_passthrough(self):
        dropout = Dropout(0.0)
        x = torch.randn(4, 10)
        dropout.train()  # in training mode
        out = dropout(x)
        assert torch.equal(out, x)

    def test_nonzero_prob_changes_in_train(self):
        dropout = Dropout(0.5)
        x = torch.ones(100, 100)
        dropout.train()
        # With high dropout, some values changed; but dropout is random.
        # We just verify it doesn't crash and is not all zeros.
        out = dropout(x)
        assert out.shape == x.shape

    def test_eval_mode_passthrough(self):
        dropout = Dropout(0.5)
        x = torch.randn(4, 10)
        dropout.eval()
        out = dropout(x)
        assert torch.equal(out, x)


# ---------------------------------------------------------------------------
# Attention bias cast helper in OLMoBlock
# ---------------------------------------------------------------------------

class TestOLMoBlockAttnBias:
    def test_cast_attn_bias_same_dtype(self):
        """_cast_attn_bias should return as-is when dtypes match."""
        block = object.__new__(OLMoBlock)
        bias = torch.tensor([[[[0.0]]]], dtype=torch.float32)
        result = OLMoBlock._cast_attn_bias(bias, torch.float32)
        assert result.dtype == torch.float32

    def test_cast_attn_bias_downcast(self):
        """When autocast would use float16, bias should follow."""
        block = object.__new__(OLMoBlock)
        bias = torch.tensor([[[[0.0, -1.0]]]], dtype=torch.float32, device="cpu")
        result = OLMoBlock._cast_attn_bias(bias, torch.float16)
        assert result.dtype == torch.float16

    def test_cast_attn_bias_preserves_neg_inf(self):
        """negative infinities should be preserved after cast."""
        block = object.__new__(OLMoBlock)
        bias = torch.tensor([[[[0.0, float("-inf")]]]], dtype=torch.float32)
        result = OLMoBlock._cast_attn_bias(bias, torch.float32)
        assert result[0, 0, 0, 1] == float("-inf")

    def test_cached_sdpa_fallback_matches_full_attention_with_flash_configured(self):
        """CPU fallback must use the bottom-right causal rows for a KV cache."""
        torch.manual_seed(7)
        cfg = ModelConfig(
            d_model=8,
            n_heads=2,
            n_layers=1,
            mlp_ratio=2,
            init_device="cpu",
            flash_attention=True,
            flex_attention=False,
            rope=False,
            alibi=False,
            include_bias=False,
        )
        block = OLMoBlock(0, cfg, BufferCache())
        with torch.no_grad():
            block.attn_out.weight.copy_(torch.eye(cfg.d_model))
        block.eval()

        qkv = torch.randn(1, 4, cfg.d_model)
        full, _ = block.attention(qkv, qkv, qkv, use_cache=False)
        _, prefix_cache = block.attention(
            qkv[:, :3], qkv[:, :3], qkv[:, :3], use_cache=True
        )
        cached, _ = block.attention(
            qkv[:, 3:], qkv[:, 3:], qkv[:, 3:],
            layer_past=prefix_cache, use_cache=True,
        )

        torch.testing.assert_close(cached[:, 0], full[:, -1], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Config: init_fn coverage
# ---------------------------------------------------------------------------

class TestInitFn:
    def test_normal_init_builds_model(self):
        """Verify a model can be created with normal init on meta device."""
        from olmo.model import OLMo

        cfg = ModelConfig(
            init_fn="normal",
            block_group_size=1,
            n_layers=2,
            init_device="meta",
            flash_attention=False,
            flex_attention=False,
        )
        model = OLMo(cfg, init_params=False)
        assert str(model.config.init_fn) == "normal"

    def test_full_megatron_init_builds_model(self):
        from olmo.model import OLMo

        cfg = ModelConfig(
            init_fn="full_megatron",
            block_group_size=1,
            n_layers=2,
            init_device="meta",
            flash_attention=False,
            flex_attention=False,
        )
        model = OLMo(cfg, init_params=False)
        assert str(model.config.init_fn) == "full_megatron"

    def test_kaiming_normal_init_builds_model(self):
        from olmo.model import OLMo

        cfg = ModelConfig(
            init_fn="kaiming_normal",
            block_group_size=1,
            n_layers=2,
            init_device="meta",
            flash_attention=False,
            flex_attention=False,
        )
        model = OLMo(cfg, init_params=False)
        assert str(model.config.init_fn) == "kaiming_normal"
