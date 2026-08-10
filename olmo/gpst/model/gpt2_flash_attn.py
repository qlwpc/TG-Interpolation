"""Lightweight GPT-2 backbone for GPST's generative model.

The reference repo ships a ~1200-line vendored HF GPT-2 (``gpt2_flash_attn.py``)
that imports symbols removed from modern ``transformers`` (e.g.
``SequenceSummary`` from ``modeling_utils``, dropped in 4.57). Pre-training only
needs the Transformer *body* — a stack of GPT-2 blocks accepting
``inputs_embeds`` (the composition model's surrogate representations) plus
``position_ids``/``past_key_values``, returning ``last_hidden_state`` +
``past_key_values``.

This module wraps the canonical HuggingFace ``GPT2Model``, which already uses
``scaled_dot_product_attention`` (PyTorch auto-dispatches to the flash/efficient
kernel on CUDA — satisfying "flash-attn if available, else SDPA"; on CPU the
math backend is used). Two small flags adapt it to GPST:

- ``no_embedding``      : skip the token+position embedding layer (inputs are
                          already embeddings from the composition model).
- ``no_layer_norm``     : drop the final LayerNorm (type layers do their own).
- ``action_layer_num``  : if set, build only the first N layers (the "type"
                          layers); the remaining layers form the "token" model.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2Config
from transformers import GPT2Model as _HFGPT2Model
from transformers.cache_utils import DynamicCache

from olmo.gpst.model.backbone_common import ModelOutput as _ModelOutput


class _ZeroPositionEmbedding(nn.Module):
    """Drop-in ``wpe`` replacement for GPST's deep token stack.

    The official implementation's ``no_extra_embedding=True`` path forwards
    gathered type-layer states unchanged. HuggingFace GPT-2 always invokes
    ``wpe``, so this module returns a dtype/device-aware zero view instead.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.anchor.expand(*position_ids.shape, self.hidden_size)


class GPT2Model(nn.Module):
    """Thin wrapper around HF ``GPT2Model`` with embedding-skip flags."""

    def __init__(self, config: GPT2Config, no_embedding: bool = False,
                 no_extra_embedding: bool = False, no_layer_norm: bool = False):
        super().__init__()
        self.no_embedding = no_embedding
        self.no_extra_embedding = no_extra_embedding
        self.no_layer_norm = no_layer_norm

        child = GPT2Config(**config.to_dict())
        child.add_pooling_layer = False
        self.hf = _HFGPT2Model(child)
        if no_embedding:
            # surrogate embeddings are fed directly via inputs_embeds
            self.hf.set_input_embeddings(nn.Identity())
        if no_layer_norm:
            self.hf.ln_f = nn.Identity()
        if no_extra_embedding:
            self.hf.wpe = _ZeroPositionEmbedding(child.hidden_size)

    @property
    def gradient_checkpointing(self):
        return self.hf.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self.hf.gradient_checkpointing = bool(value)

    def forward(self, inputs_embeds=None, position_ids=None,
                past_key_values=None, attention_mask=None, **kwargs):
        if inputs_embeds is not None and position_ids is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
            if position_ids.ndim == 1:
                if batch_size == 1 and position_ids.numel() == seq_len:
                    position_ids = position_ids.unsqueeze(0)
                elif seq_len == 1 and position_ids.numel() == batch_size:
                    position_ids = position_ids.unsqueeze(-1)
                else:
                    raise ValueError(
                        f"ambiguous 1D position_ids for input shape "
                        f"{(batch_size, seq_len)}: {tuple(position_ids.shape)}"
                    )
            if position_ids.shape[0] == 1 and batch_size > 1:
                position_ids = position_ids.expand(batch_size, -1)
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_ids must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(position_ids.shape)}"
                )
        use_cache = kwargs.get("use_cache")
        if use_cache is None:
            use_cache = bool(getattr(self.hf.config, "use_cache", True))
        if isinstance(past_key_values, (list, tuple)):
            # GPST's BeamContext stores the stable legacy per-layer (k, v)
            # representation, while modern Transformers expects Cache objects.
            past_key_values = DynamicCache.from_legacy_cache(tuple(past_key_values))
        out = self.hf(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )
        output_cache = out.past_key_values
        if output_cache is not None and hasattr(output_cache, "to_legacy_cache"):
            output_cache = output_cache.to_legacy_cache()
        return _ModelOutput(last_hidden_state=out.last_hidden_state,
                            past_key_values=output_cache)


class GPT2LMHeadModel(nn.Module):
    """GPT-2 with an LM head — used by the GPT-2 baseline (model_type='gpt')."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        from transformers import GPT2LMHeadModel as HFGPT2LM
        self.gpt = HFGPT2LM(config)
        self.config = config

    def forward(self, input_ids=None, labels=None, return_dict=True, **kwargs):
        return self.gpt(input_ids=input_ids, labels=labels,
                        return_dict=return_dict, **kwargs)
