"""OLMo-architecture backbone for GPST's generative model.

A drop-in alternative to the HF GPT2 wrapper (``gpt2_flash_attn.py:GPT2Model``).
``FastGenerativeR2D2`` instantiates two independent transformer sub-stacks
(shallow "type"/action layers + deep "token"/generation layers) and drives each
with ``inputs_embeds`` (the composition model's surrogate representations)
plus tree-ordered ``position_ids`` and optional ``past_key_values``. Because
the two sub-stacks are called separately with a gather in between, we cannot
reuse ``OLMo.forward`` (which runs all layers end-to-end and owns the embedding
+ position logic). Instead we reuse the *block* primitive —
``olmo.model.OLMoBlock`` — and wrap a configurable number of them behind the
same interface as ``GPT2Model``.

Position handling: GPST's action ``position_ids`` are word offsets ``w_t``;
COMP actions repeat the current offset while GEN actions advance it. HF GPT2
honours them via a *learned* position embedding (``wpe``) that indexes by id.
``OLMoBlock``'s RoPE instead derives positions internally from
``key_len``/``query_len`` and cannot accept external position ids. To preserve
GPST's semantics we therefore disable RoPE/ALiBi and add our own learned
``wpe`` here, applied at the stack entrance — exactly mirroring GPT2.

No changes to ``olmo/model.py`` or ``olmo/config.py``: everything is layered on
top of the public ``OLMoBlock.build`` / ``LayerNorm.build`` APIs.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from olmo.config import ModelConfig
from olmo.model import BufferCache, LayerNorm, OLMoBlock

from olmo.gpst.model.backbone_common import ModelOutput


def _build_causal_pad_bias(
    attention_mask: Optional[torch.Tensor],
    query_len: int,
    key_len: int,
    device: torch.device,
    has_past: bool = False,
) -> Optional[torch.Tensor]:
    """Build a ``(B, 1, Q, K)`` additive attention bias combining causality
    with padding, or return ``None`` (=> block uses ``is_causal=True``).

    ``attention_mask`` is ``(B, T)`` with 1 for valid tokens, 0 for padding
    (HuggingFace convention); it covers the full sequence (past + current).

    Returns ``None`` only when there is no padding AND no past KV cache — in
    that case ``query_len == key_len`` and the block's ``is_causal=True``
    path is correct. When a KV cache is present (``query_len < key_len``),
    ``is_causal=True`` would wrongly mask the cached keys, so a bias is
    mandatory; the bias allows query ``i`` (global position ``past_len + i``)
    to attend to every key ``j <= past_len + i``.
    """
    if attention_mask is None and not has_past:
        return None
    past_len = key_len - query_len
    # key validity: (B, K)
    if attention_mask is not None:
        key_valid = attention_mask.to(torch.bool)[:, :key_len]  # (B, K)
    else:
        key_valid = torch.ones(1, key_len, device=device, dtype=torch.bool)
    B = key_valid.shape[0]
    # allowed[b, i, j] = (j <= past_len + i) AND key_valid[b, j]
    q_pos = (past_len + torch.arange(query_len, device=device)).view(query_len, 1)  # (Q,1)
    k_pos = torch.arange(key_len, device=device).view(1, key_len)                    # (1,K)
    causal = k_pos <= q_pos                                                          # (Q,K)
    allowed = causal.unsqueeze(0) & key_valid.unsqueeze(1)                           # (B,Q,K)
    bias = torch.zeros(B, 1, query_len, key_len, device=device, dtype=torch.float32)
    bias.masked_fill_(~allowed.unsqueeze(1), float("-inf"))
    return bias


class OLMoStack(nn.Module):
    """A stack of ``OLMoBlock`` layers exposing the GPT2-wrapper interface.

    Parameters mirror the flags ``GPT2Model`` accepts:
    - ``no_embedding``   : inputs are already embeddings (always True for GPST);
    - ``no_layer_norm``  : drop the final LayerNorm (type layers manage their own);
    - ``add_position``   : apply a learned ``wpe`` to external ``position_ids``.

    ``n_layers`` is independent of ``config.n_layers`` so the factory can build
    the shallow action stack and the deep token stack from one shared config.
    """

    def __init__(
        self,
        config: ModelConfig,
        n_layers: Optional[int] = None,
        no_embedding: bool = True,
        no_layer_norm: bool = False,
        add_position: bool = True,
    ):
        super().__init__()
        # Force a config copy that disables RoPE/ALiBi (we handle positions
        # ourselves via wpe) and avoids CUDA-only features on CPU.
        self.config = config
        self._cache = BufferCache()
        self.no_embedding = no_embedding
        self.no_layer_norm = no_layer_norm
        self.add_position = add_position
        self._gradient_checkpointing = False

        n = n_layers if n_layers is not None else config.n_layers
        self.blocks = nn.ModuleList(
            [OLMoBlock.build(i, config, self._cache) for i in range(n)]
        )

        if add_position:
            self.wpe = nn.Embedding(config.max_sequence_length, config.d_model)
        else:
            self.wpe = None

        if no_layer_norm:
            self.ln_f: nn.Module = nn.Identity()
        else:
            self.ln_f = LayerNorm.build(config)
        self.emb_drop = nn.Dropout(config.embedding_dropout)

        # init params (OLMoBlock.build does not auto-init when init_device != 'meta')
        self.reset_parameters()

    def reset_parameters(self):
        for blk in self.blocks:
            blk.reset_parameters()
        if self.wpe is not None:
            nn.init.normal_(self.wpe.weight, mean=0.0, std=self.config.init_std)
        if not isinstance(self.ln_f, nn.Identity):
            self.ln_f.reset_parameters()

    @property
    def gradient_checkpointing(self):
        return self._gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool):
        # Whole-layer checkpointing normally lives in ``OLMo.forward``. GPST
        # drives OLMoBlock instances directly, so the wrapper owns that policy.
        self._gradient_checkpointing = bool(value)

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutput:
        if inputs_embeds is None:
            raise ValueError("OLMoStack requires inputs_embeds")
        x = inputs_embeds
        B, T, _ = x.shape

        if self.add_position:
            if position_ids is None:
                position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            elif position_ids.ndim == 1:
                if B == 1 and position_ids.numel() == T:
                    position_ids = position_ids.unsqueeze(0)
                elif T == 1 and position_ids.numel() == B:
                    # Incremental beam search supplies one scalar per beam.
                    position_ids = position_ids.unsqueeze(-1)
                else:
                    raise ValueError(
                        f"ambiguous 1D position_ids for input shape {(B, T)}: "
                        f"{tuple(position_ids.shape)}"
                    )
            if position_ids.shape[0] == 1 and B > 1:
                position_ids = position_ids.expand(B, -1)
            if position_ids.shape != (B, T):
                raise ValueError(
                    f"position_ids must have shape {(B, T)}, got {tuple(position_ids.shape)}"
                )
            x = x + self.wpe(position_ids)
        x = self.emb_drop(x)

        requested_cache = kwargs.get("use_cache")
        use_cache = (not self.training) if requested_cache is None else bool(requested_cache)
        use_cache = use_cache or past_key_values is not None

        present_kvs = []
        for i, blk in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            # key_len must account for the cached keys; the block will concat
            # past+current internally, so the bias is sized to the full key seq.
            past_len = layer_past[0].shape[-2] if layer_past is not None else 0
            bias = _build_causal_pad_bias(
                attention_mask, query_len=T, key_len=past_len + T,
                device=x.device, has_past=layer_past is not None,
            )
            if self._gradient_checkpointing and self.training and layer_past is None and not use_cache:
                def block_forward(hidden_states, block=blk, block_bias=bias):
                    return block(hidden_states, attention_bias=block_bias, use_cache=False)[0]

                x = checkpoint(block_forward, x, use_reentrant=False)
                present = None
            else:
                x, present = blk(
                    x,
                    attention_bias=bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                )
            present_kvs.append(present)

        x = self.ln_f(x)
        return ModelOutput(
            last_hidden_state=x,
            past_key_values=present_kvs if use_cache else None,
        )
