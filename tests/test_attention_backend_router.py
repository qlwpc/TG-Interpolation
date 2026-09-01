"""Attention backend routing and Flex tail-safety regressions."""

from __future__ import annotations

import pytest
import torch

from olmo.config import (
    ActivationType,
    BlockType,
    InitFnType,
    LayerNormType,
    ModelConfig,
)
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo, _use_flex_for_structured_attention


def _routing_config(**overrides) -> ModelConfig:
    values = {
        "flex_attention": True,
        "transformer_grammar_type": "tg",
    }
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize(
    "training,length,expected",
    [
        (True, 127, False),
        (True, 512, False),
        (True, 1023, False),
        (True, 1024, True),
        (False, 1024, False),
        (False, 2047, False),
        (False, 2048, True),
    ],
)
def test_structured_flex_length_thresholds(training, length, expected):
    config = _routing_config()
    assert _use_flex_for_structured_attention(
        config,
        length,
        training=training,
        has_attention_bias=True,
        has_past_key_values=False,
        use_cache=False,
    ) is expected


def test_structured_flex_requires_bias_and_no_kv_cache():
    config = _routing_config()
    assert not _use_flex_for_structured_attention(
        config,
        2048,
        training=True,
        has_attention_bias=False,
        has_past_key_values=False,
        use_cache=False,
    )
    assert not _use_flex_for_structured_attention(
        config,
        2048,
        training=True,
        has_attention_bias=True,
        has_past_key_values=True,
        use_cache=False,
    )
    assert not _use_flex_for_structured_attention(
        config,
        2048,
        training=False,
        has_attention_bias=True,
        has_past_key_values=False,
        use_cache=True,
    )
    assert not _use_flex_for_structured_attention(
        _routing_config(flex_attention=False),
        2048,
        training=True,
        has_attention_bias=True,
        has_past_key_values=False,
        use_cache=False,
    )


def test_pushdown_uses_its_dedicated_router():
    config = _routing_config(transformer_grammar_type="pushdown")
    assert not _use_flex_for_structured_attention(
        config,
        2048,
        training=True,
        has_attention_bias=True,
        has_past_key_values=False,
        use_cache=False,
    )


def _tiny_model_config(**overrides) -> ModelConfig:
    values = {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 1,
        "mlp_ratio": 2,
        "mlp_hidden_size": 128,
        "vocab_size": 256,
        "embedding_size": 256,
        "max_sequence_length": 2048,
        "block_type": BlockType.sequential,
        "layer_norm_type": LayerNormType.rms,
        "activation_type": ActivationType.swiglu,
        "rope": True,
        "flash_attention": False,
        "flex_attention": True,
        "attention_dropout": 0.0,
        "residual_dropout": 0.0,
        "embedding_dropout": 0.0,
        "init_device": "cpu",
        "init_fn": InitFnType.normal,
        "init_std": 0.02,
        "transformer_grammar_type": "tg",
        "weight_tying": True,
    }
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize("multiple", [0, 64, 129, -128])
def test_flex_padding_multiple_validation(multiple):
    with pytest.raises(OLMoConfigurationError, match="positive multiple of 128"):
        OLMo(_tiny_model_config(flex_attention_pad_to_multiple=multiple), init_params=False)


@pytest.mark.parametrize(
    "field_name",
    [
        "flex_attention_train_min_sequence_length",
        "flex_attention_eval_min_sequence_length",
    ],
)
def test_flex_threshold_validation(field_name):
    with pytest.raises(OLMoConfigurationError, match=field_name):
        OLMo(_tiny_model_config(**{field_name: -1}), init_params=False)


def _structured_inputs(device: torch.device):
    batch, length = 2, 127
    generator = torch.Generator(device=device).manual_seed(20260901)
    input_ids = torch.randint(0, 256, (batch, length), generator=generator, device=device)
    positions = torch.arange(length, device=device)
    causal_window = (
        (positions[:, None] >= positions[None, :])
        & ((positions[:, None] - positions[None, :]) <= 31)
    )
    # A batch-broadcast bias plus a batch-specific key mask exercises both
    # normalization and the formerly broken TG bias + attention_mask branch.
    attention_bias = causal_window[None, None]
    attention_mask = torch.ones((batch, length), dtype=torch.bool, device=device)
    attention_mask[1, -8:] = False
    return input_ids, attention_bias, attention_mask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_short_structured_training_routes_to_sdpa():
    model = OLMo(_tiny_model_config()).cuda().train()

    def unexpected_flex(*_args, **_kwargs):
        raise AssertionError("N=127 should route to SDPA under the default training threshold")

    for block in model.transformer.blocks:
        block.flex_attention = unexpected_flex

    input_ids, attention_bias, attention_mask = _structured_inputs(torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids,
            attention_bias=attention_bias,
            attention_mask=attention_mask,
        ).logits
        loss = logits.float().square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_forced_short_flex_pads_and_supports_attention_mask():
    model = OLMo(
        _tiny_model_config(
            flex_attention_train_min_sequence_length=0,
            flex_attention_eval_min_sequence_length=0,
            flex_attention_pad_to_multiple=128,
        )
    ).cuda().train()
    calls = {"flex": 0}
    for block in model.transformer.blocks:
        compiled_flex = block.flex_attention

        def counted_flex(*args, _compiled_flex=compiled_flex, **kwargs):
            calls["flex"] += 1
            return _compiled_flex(*args, **kwargs)

        block.flex_attention = counted_flex

    input_ids, attention_bias, attention_mask = _structured_inputs(torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids,
            attention_bias=attention_bias,
            attention_mask=attention_mask,
        ).logits
        loss = logits.float().square().mean()
    assert logits.shape[:2] == input_ids.shape
    loss.backward()
    assert calls["flex"] == 1
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention needs CUDA")
def test_nondivisible_flex_without_padding_safely_falls_back_to_sdpa():
    model = OLMo(
        _tiny_model_config(
            flex_attention_train_min_sequence_length=0,
            flex_attention_pad_to_multiple=None,
        )
    ).cuda().train()

    def unexpected_flex(*_args, **_kwargs):
        raise AssertionError("unsafe N=127 Flex backward must fall back to SDPA")

    for block in model.transformer.blocks:
        block.flex_attention = unexpected_flex

    input_ids, attention_bias, attention_mask = _structured_inputs(torch.device("cuda"))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=input_ids,
            attention_bias=attention_bias,
            attention_mask=attention_mask,
        ).logits
        loss = logits.float().square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
