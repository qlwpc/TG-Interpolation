"""Model factory for GPST pre-training.

Trimmed from ant-research/StructuredLM_RTDT model_factory.py: only the
pre-training model types are retained (``r2d2-gen-fast`` and the ``gpt``
baseline). Fine-tuning wrappers (GLUE/XSum/GPT2Wrapper) are omitted.

Port note: ``GPT2Model`` is our lightweight HF-GPT2 wrapper (SDPA) rather than
the vendored 1200-line file; it exposes ``no_embedding`` /
``no_extra_embedding`` / ``no_layer_norm`` and an ``action_layer_num`` split for
the type/token layer separation.
"""
from copy import deepcopy

from transformers import AutoConfig

from olmo.config import (
    ActivationType, BlockType, InitFnType, LayerNormType, ModelConfig,
)
from olmo.gpst.model.r2d2_insideoutside import InsideOutsideModule
from olmo.gpst.model.gpt2_flash_attn import GPT2Model


def _build_r2d2(r2d2_config_path, gpt_config_path):
    gpt_config = AutoConfig.from_pretrained(gpt_config_path)
    vocab_size = gpt_config.vocab_size
    r2d2_config = AutoConfig.from_pretrained(r2d2_config_path)
    r2d2_config.vocab_size = vocab_size
    r2d2 = InsideOutsideModule(r2d2_config)
    return r2d2, gpt_config, r2d2_config, vocab_size


def _gpt_config_to_olmo_model_config(gpt_config, init_device: str = "cpu") -> ModelConfig:
    """Map an HF ``GPT2Config`` to an OLMo ``ModelConfig`` for the GPST stack.

    RoPE/ALiBi are disabled: GPST feeds tree-ordered ``position_ids`` that only
    a *learned* position embedding (``OLMoStack.wpe``) can honour — RoPE derives
    positions internally from sequence length and cannot accept external ids.
    ``activation_type`` defaults to gelu to mirror GPT2; pass ``swiglu`` for the
    native OLMo activation.
    """
    return ModelConfig(
        d_model=gpt_config.n_embd,
        n_heads=gpt_config.n_head,
        n_kv_heads=None,  # plain multi-head attention, like GPT2
        n_layers=gpt_config.n_layer,
        mlp_ratio=gpt_config.n_inner or (4 * gpt_config.n_embd) // gpt_config.n_embd,
        activation_type=ActivationType.gelu,
        block_type=BlockType.sequential,
        rope=False,
        alibi=False,
        flash_attention=False,
        flex_attention=False,
        attention_dropout=getattr(gpt_config, "attn_pdrop", 0.1),
        residual_dropout=getattr(gpt_config, "resid_pdrop", 0.1),
        embedding_dropout=getattr(gpt_config, "embd_pdrop", 0.1),
        attention_layer_norm=False,
        layer_norm_type=LayerNormType.default,
        max_sequence_length=getattr(gpt_config, "n_positions", 1024),
        vocab_size=gpt_config.vocab_size,
        include_bias=getattr(gpt_config, "bias", True),
        weight_tying=False,
        init_fn=InitFnType.normal,
        init_std=gpt_config.initializer_range if hasattr(gpt_config, "initializer_range") else 0.02,
        init_device=init_device,
    )


def create_model(model_type, r2d2_config_path, gpt_config_path,
                 fix_embeddings=False, gradient_checkpoint=False, backbone="gpt2"):
    if model_type == 'r2d2-gen-fast':
        from olmo.gpst.model.generative_r2d2_fast import FastGenerativeR2D2
        r2d2, gpt_config, r2d2_config, vocab_size = _build_r2d2(r2d2_config_path, gpt_config_path)
        total_layer = gpt_config.n_layer
        action_layer_num = gpt_config.action_layer_num
        if backbone == "gpt2":
            # type (action) layers = first action_layer_num; token layers = the rest
            gpt_config.n_layer = action_layer_num
            action_transformers = GPT2Model(gpt_config, no_embedding=True, no_layer_norm=True)
            action_transformers.gradient_checkpointing = gradient_checkpoint
            gpt_config.n_layer = total_layer - action_layer_num
            gpt_transformers = GPT2Model(gpt_config, no_embedding=True, no_extra_embedding=True)
            gpt_transformers.gradient_checkpointing = gradient_checkpoint
        elif backbone == "olmo":
            from olmo.gpst.model.olmo_stack import OLMoStack
            olmo_cfg = _gpt_config_to_olmo_model_config(gpt_config, init_device="cpu")
            # action (type) stack: shallow, no final LN (type layers manage their own)
            action_cfg = deepcopy(olmo_cfg)
            action_cfg.n_layers = action_layer_num
            action_transformers = OLMoStack(
                action_cfg, no_embedding=True, no_layer_norm=True, add_position=True
            )
            # generation (token) stack: the remaining (deep) layers
            generation_cfg = deepcopy(olmo_cfg)
            generation_cfg.n_layers = total_layer - action_layer_num
            gpt_transformers = OLMoStack(
                generation_cfg, no_embedding=True, no_layer_norm=False, add_position=False
            )
            if gradient_checkpoint:
                action_transformers.gradient_checkpointing = True
                gpt_transformers.gradient_checkpointing = True
        else:
            raise ValueError(f"Unknown backbone: {backbone!r} (expected 'gpt2' or 'olmo')")
        r2d2_input_dim = r2d2.input_dim
        gpt_input_dim = gpt_config.n_embd
        return FastGenerativeR2D2(
            r2d2, action_transformers, gpt_transformers, vocab_size,
            r2d2_input_dim, gpt_input_dim,
            ext_vocab_size=r2d2_config.ext_vocab_size,
            fix_embeddings=fix_embeddings)
    elif model_type == 'gpt':
        from olmo.gpst.model.gpt2_flash_attn import GPT2LMHeadModel
        gpt_config = AutoConfig.from_pretrained(gpt_config_path)
        return GPT2LMHeadModel(gpt_config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
