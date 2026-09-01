"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
import gc
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
    Union
)

import os

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

from .aliases import PathOrStr
from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    InitFnType,
    LayerNormType,
    ModelConfig,
    ShardedCheckpointerType,
    TrainConfig,
    BeamSearchType
)
from .exceptions import OLMoConfigurationError
from .initialization import init_normal
from .torch_util import ensure_finite_, get_cumulative_document_lengths, move_to_device
from .data.tg_mask import SentencepieceVocab, TG_attention_bias


def _flex_attention_kernel_options() -> Optional[Dict[str, int]]:
    """Return an optional low-stage FlexAttention override.

    PyTorch 2.7's default Ampere configs can exceed the shared-memory limit of
    SM86/SM89 GPUs for non-standard head dimensions. Keep the library default
    unless the execution environment explicitly requests a pipeline depth.
    """
    raw = os.environ.get("OLMO_FLEX_ATTENTION_NUM_STAGES")
    if raw is None:
        return None
    try:
        stages = int(raw)
    except ValueError as exc:
        raise OLMoConfigurationError(
            "OLMO_FLEX_ATTENTION_NUM_STAGES must be a positive integer"
        ) from exc
    if stages < 1:
        raise OLMoConfigurationError(
            "OLMO_FLEX_ATTENTION_NUM_STAGES must be a positive integer"
        )
    return {"fwd_num_stages": stages, "bwd_num_stages": stages}


def _use_flex_for_structured_attention(
    config: ModelConfig,
    sequence_length: int,
    *,
    training: bool,
    has_attention_bias: bool,
    has_past_key_values: bool,
    use_cache: bool = False,
) -> bool:
    """Route a TG-style structured mask to Flex or the dense-mask SDPA path.

    Pushdown is intentionally excluded: its score-mod path has a different
    performance envelope and is selected separately in ``OLMo.forward``.
    """
    if (
        config.transformer_grammar_type == "pushdown"
        or not config.flex_attention
        or not has_attention_bias
        or has_past_key_values
        or use_cache
    ):
        return False
    threshold = (
        config.flex_attention_train_min_sequence_length
        if training
        else config.flex_attention_eval_min_sequence_length
    )
    return sequence_length >= threshold


def _flex_attention_pad_multiple(config: ModelConfig) -> Optional[int]:
    """Resolve the configured padding multiple, preserving the legacy env override."""
    raw = os.environ.get("OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE")
    if raw is None:
        return config.flex_attention_pad_to_multiple
    try:
        multiple = int(raw)
    except ValueError as exc:
        raise OLMoConfigurationError(
            "OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE must be a positive multiple of 128"
        ) from exc
    if multiple <= 0 or multiple % 128 != 0:
        raise OLMoConfigurationError(
            "OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE must be a positive multiple of 128"
        )
    return multiple

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "OLMoBlock",
    "OLMoSequentialBlock",
    "OLMo",
    "OLMoOutput",
    "OLMoGenerateOutput",
]


log = logging.getLogger(__name__)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


def should_checkpoint_block(strategy: Optional[ActivationCheckpointingStrategy], block_idx: int) -> bool:
    if strategy is None:
        return False
    elif (
        (strategy == ActivationCheckpointingStrategy.whole_layer)
        or (strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_eight and block_idx % 8 == 0)
        or (strategy == ActivationCheckpointingStrategy.two_in_three and block_idx % 3 != 0)
        or (strategy == ActivationCheckpointingStrategy.three_in_four and block_idx % 4 != 0)
    ):
        return True
    else:
        return False


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
    ):
        super().__init__()
        self.config = config
        self.eps = config.layer_norm_eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class OLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn: Optional[Callable] = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        # Pushdown depth-embedding (Murty et al. 2023). Per-layer depth embedding
        # added to attention keys; the per-(q,kv) bias is applied via FlexAttention
        # score_mod (see OLMoBlock.attention). Only constructed for pushdown models.
        self._is_pushdown = config.transformer_grammar_type == "pushdown"
        if self._is_pushdown:
            from olmo.pushdown import PushdownDepthBias
            self.pushdown_depth_bias = PushdownDepthBias(
                max_depth=config.pushdown_max_depth,
                d_model=config.d_model,
                n_heads=config.n_heads,
            )

        self.flash_attn_func = None
        self.flash_attn_varlen_func = None
        if config.flash_attention:
            try:
                from flash_attn import (  # type: ignore
                    flash_attn_func,
                    flash_attn_varlen_func,
                )

                self.flash_attn_func = flash_attn_func
                self.flash_attn_varlen_func = flash_attn_varlen_func
            except ModuleNotFoundError:
                pass

        if config.flex_attention:
            self.flex_attention = torch.compile(flex_attention)
            self.flex_attention_kernel_options = _flex_attention_kernel_options()

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()

        if self.config.init_fn == InitFnType.normal:
            attn_out_std = ff_out_std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor

        elif self.config.init_fn == InitFnType.mitchell:
            attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
            ff_out_std = 1 / (math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1)))
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        elif self.config.init_fn == InitFnType.full_megatron:
            attn_out_std = ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
            cutoff_factor = self.config.init_cutoff_factor or 3.0

        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
        init_normal(self.ff_out, std=ff_out_std, init_cutoff_factor=cutoff_factor)

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = checkpoint_func or activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_enabled('cpu'):
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if max_doc_len is not None and cu_doc_lens is not None:
            assert self.flash_attn_varlen_func is not None, "flash-attn is required for document masking"
            assert attn_mask is None, "attn-mask is currently not supported with document masking"
            B, T, D = q.size(0), q.size(2), q.size(3)
            r = self.flash_attn_varlen_func(
                q.transpose(1, 2).view(B * T, -1, D),
                k.transpose(1, 2).view(B * T, -1, D),
                v.transpose(1, 2).view(B * T, -1, D),
                cu_doc_lens,
                cu_doc_lens,
                max_doc_len,
                max_doc_len,
                dropout_p=dropout_p,
                causal=is_causal,
            )
            return r.view(B, T, -1, D).transpose(1, 2)
        elif self.flash_attn_func is not None and attn_mask is None and self.attn_out.weight.is_cuda and block_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        elif block_mask is not None:
            flex_diagnostics = os.environ.get("OLMO_FLEX_ATTENTION_DIAGNOSTICS") == "1"
            if flex_diagnostics:
                log.warning(
                    "FLEX_DIAG forward_begin layer=%d q_shape=%s k_shape=%s v_shape=%s "
                    "q_stride=%s k_stride=%s v_stride=%s dtype=%s block_mask_shape=%s",
                    self.layer_id,
                    tuple(q.shape),
                    tuple(k.shape),
                    tuple(v.shape),
                    tuple(q.stride()),
                    tuple(k.stride()),
                    tuple(v.stride()),
                    q.dtype,
                    tuple(block_mask.shape),
                )
                torch.cuda.synchronize(q.device)
            result = self.flex_attention(
                q,
                k,
                v,
                block_mask=block_mask,
                kernel_options=self.flex_attention_kernel_options,
            )
            if flex_diagnostics:
                torch.cuda.synchronize(q.device)
                log.warning("FLEX_DIAG forward_end layer=%d output_shape=%s", self.layer_id, tuple(result.shape))
            return result
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        tree_spans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        # FlashAttention aligns a rectangular causal mask to the bottom-right,
        # which is the desired behavior for cached decoding. PyTorch SDPA's
        # ``is_causal=True`` uses an upper-left alignment instead. When the
        # configuration requests FlashAttention but its kernel is unavailable
        # (for example on CPU or when flash-attn is not installed), construct the
        # explicit bottom rows of the square causal bias before falling back.
        flash_kernel_available = (
            self.flash_attn_func is not None
            and q.is_cuda
            and self.attn_out.weight.is_cuda
            and block_mask is None
        )
        if layer_past is not None and attention_bias is None and not flash_kernel_available:
            attention_bias = get_causal_attention_bias(
                self.__cache, key_len, q.device
            )[:, :, key_len - query_len : key_len, :key_len]

        # Pushdown: per-(q,kv) depth bias added to attention scores. The depth
        # matrix S (B, n, n) int8 is computed on the GPU from the spans; the bias
        # q_k . E_l[S[k,j]] is materialized as a full (B, n_h, n, n) additive mask
        # and merged with the causal+pad attention_bias, then SDPA is used (flash
        # cannot take an additive bias). When tree_spans is None (no parse), the
        # depth bias is zero and this falls through to the standard path below.
        if self._is_pushdown and tree_spans is not None:
            att = self._pushdown_attention(q, k, v, tree_spans, attention_bias, key_len,
                                           dropout_p=0.0 if not self.training else self.config.attention_dropout,
                                           block_mask=block_mask)
            att = att.transpose(1, 2).contiguous().view(B, T, C)
            return self.attn_out(att), present

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            if key_len != query_len and attention_bias.shape[-2] == attention_bias.shape[-1]:
                attention_bias = self._cast_attn_bias(
                    attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
                )
            else:
                attention_bias = self._cast_attn_bias(
                    attention_bias[:, :, :query_len, :key_len], dtype
                )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
            block_mask=block_mask,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    def _pushdown_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        tree_spans: torch.Tensor, attention_bias: Optional[torch.Tensor],
        key_len: int, dropout_p: float = 0.0,
        block_mask: Optional[BlockMask] = None,
    ) -> torch.Tensor:
        """Pushdown attention with a per-(q,kv) stale-depth bias.

        Computes the stale depth tape ``S`` (B, n, n) int8 from ``tree_spans`` on the
        GPU, builds a per-head depth embedding ``E_l`` (projecting the depth embedding
        through the key projection), and applies the per-(q,kv) bias
        ``Q[b,h,q] . E_l[h, S[b,q,kv]]``.

        Two paths:

        * **FlexAttention ``score_mod``** (default fast path, ``pushdown_use_flex``):
          ``OLMo.forward`` builds a causal+pad ``block_mask`` from the 1-D
          ``attention_mask`` (mirroring the TG path) and passes it as ``block_mask``;
          the depth bias goes into the ``score_mod``. This is a single fused
          flash-class kernel — no ``(B, n_h, n, n)`` fp32 mask, no math-backend SDPA.
          ``flex_attention`` is compiled independently (``model.py`` L497), so this
          works without ``block.compile()`` (the ``index_put_`` in
          ``compute_depth_matrix_gpu`` graph-breaks ``block.compile`` but runs eagerly
          here, outside the compiled score_mod closure).

        * **SDPA additive-mask fallback**: materializes the full ``(B, n_h, n, n)``
          fp32 bias, merges with the causal+pad ``attention_bias``, and runs
          ``F.scaled_dot_product_attention(is_causal=False)``. Used when flex is off,
          when no ``block_mask`` is present, or during generation (KV cache — flex
          block_masks are fixed-length). Slow (~100x vs flex); not for training.

        When ``tree_spans`` is None (no parse) the depth bias is zero and the caller
        falls through to the standard attention path.
        """
        from olmo.pushdown import (
            compute_depth_matrix_gpu,
            compute_depth_rows_gpu,
        )
        B, nh, query_len, hs = q.shape
        # Stale depth tape over the (causal) key length. tree_spans: (B, M, 3).
        # S depends ONLY on tree_spans (layer-independent), so memoize it across
        # the 12 blocks within ONE forward pass — otherwise compute_depth_matrix_gpu
        # runs 12x per forward, each allocating a (B,n,n) float32 difference array
        # for no reason. Forward-scoped: OLMo.forward pops this key before the block
        # loop, so a later forward (different tree_spans) never reads a stale entry.
        # Safe under activation checkpointing: S is int8 with no grad, so reusing the
        # forward-pass entry during the backward recompute is correct.
        cache = self.__cache
        S = cache.get("pushdown_depth_matrix") if cache is not None else None
        expected_rows = query_len if query_len < key_len else key_len
        if (S is None or S.shape != (B, expected_rows, key_len)
                or S.device != q.device):
            spans = tree_spans.to(q.device)
            if query_len < key_len:
                S = compute_depth_rows_gpu(
                    spans, key_len, key_len - query_len, key_len
                )
            else:
                S = compute_depth_matrix_gpu(spans, key_len)  # (B, k, k)
            if cache is not None:
                cache["pushdown_depth_matrix"] = S
        # During cached decoding q contains only the newly appended token(s),
        # while k/v contain the complete prefix. Select the bottom query rows of
        # the full stale-depth tape. In training/prefill query_len == key_len and
        # this is the original square matrix.
        D = S if expected_rows == query_len else S[:, key_len - query_len : key_len, :key_len]
        D = D.clamp(max=self.config.pushdown_max_depth).long()  # (B, q, k)
        # Per-head depth embedding E_l: (n_heads, max_depth+1, d_head).
        n_kv = k.shape[1]
        kv_dim = n_kv * hs
        key_weight = self.att_proj.weight[self.config.d_model : self.config.d_model + kv_dim]
        E = self.pushdown_depth_bias(key_weight)  # (n_heads, D, hs)
        # P[b,h,q,d] = Q[b,h,q] . E[h,d]; bias = P.gather(D) -> (B, n_h, n, n).
        # Compute in float32 for numerical stability: under amp_bf16, q/E are bf16
        # and a bf16 attn_mask with -inf makes SDPA's internal fp32 cublas gemm
        # raise CUBLAS_STATUS_EXECUTION_FAILED. An fp32 attn_mask is accepted by
        # SDPA alongside bf16 q/k/v.
        P = torch.einsum("bhni,hdi->bhnd", q.float(), E.float())  # (B, n_h, n, D) fp32
        Dmax = P.shape[3] - 1
        # int64: required by take_along_dim in the SDPA fallback below (it rejects
        # int32). The flex score_mod closure gathers with this same Dc and accepts
        # int64 fine. (A (B,n,n) int64 view is ~402 MB at B=12,n=2048, but it is a
        # non-materialized view of the int8 S clamped — the underlying S is 50 MB.)
        Dc = D.clamp(0, Dmax)  # (B, n, n) int64, values in [0, Dmax]

        # FlexAttention score_mod path (fused flash-class kernel) — the intended
        # pushdown fast path. The stale depth bias is per-(query,key) (S[b,q,kv]),
        # so plain flash_attn_func cannot express it; FlexAttention is the only
        # fused path. OLMo.forward builds a causal+pad block_mask from the 1-D
        # attention_mask (mirroring the TG path) and passes it here, fusing both
        # causality and padding into the mask and the depth bias into score_mod.
        # This replaces the (B, n_h, n, n) fp32 additive mask + math-backend SDPA
        # (the ~100x slowdown). The graph-breaking index_put_ lives in
        # compute_depth_matrix_gpu, which runs eagerly and is memoized OUTSIDE this
        # closure, so the compiled flex kernel is not affected by it.
        if (self.config.pushdown_use_flex
                and getattr(self, "flex_attention", None) is not None
                and block_mask is not None):
            # Bias representation. The depth bias is P[b,h,q, Dc[b,q,kv]].
            #
            #   dense-db (DEFAULT, OLMO_PUSHDOWN_GATHER_BY_DEPTH not set):
            #     Pre-materialize db = take_along_dim(P, Dc) as (B, n_h, n, n) fp32
            #     (~2.4 GB/layer) and capture THAT in score_mod. inductor's flex
            #     backward scatters the score_mod grad into each captured buffer that
            #     requires grad (process_joint_outputs -> flex_lib.zeros_and_scatter,
            #     flex_attention.py:2349-2357); for db the scatter index is a SIMPLE
            #     per-cell (b,h,q,kv) gather, which the lowering handles, so grad
            #     reaches depth_emb correctly. This is the KNOWN-WORKING path
            #     (trains on GPU under amp_bf16+DDP). It is ~20x slower than baseline
            #     — see memory pushdown-flex-slowdown-analysis — but correct.
            #
            #   gather-by-depth (OLMO_PUSHDOWN_GATHER_BY_DEPTH=1): EXPERIMENTAL.
            #     Gather P by Dc INSIDE score_mod (captured buffer = the ~77 MB P
            #     (B,n_h,n,Dmax), not 2.4 GB db). Forward output is bit-identical to
            #     dense-db (verified on CPU, max diff 0.0) and the forward eliminates
            #     the 2.4 GB/layer materialization. BUT on GPU under amp_bf16+DDP the
            #     backward does NOT deliver grad to depth_emb -> DDP errors "unused
            #     parameters" (param indices 7,20,...,150, the 12 per-layer depth_emb
            #     Embeddings). Root cause (inferred): the score_mod does a TWO-LEVEL
            #     data-dependent index _P[b,h,q, _Dc[b,q,kv]] (index _Dc to get a
            #     depth, then index _P with it); inductor's zeros_and_scatter grad for
            #     a captured grad-requiring buffer needs a simple per-cell index, and
            #     the data-dependent second index defeats the scatter -> no grad to P
            #     -> no grad to E -> no grad to depth_emb. CPU fp32 trace does populate
            #     P.grad (the CPU dense backward fallback handles it), so this only
            #     manifests on the GPU Triton path. DO NOT enable for training until
            #     the grad path is fixed (planned: detach P in score_mod + an explicit
            #     aux loss that backprops to depth_emb).
            inv_sqrt_hs = 1.0 / (hs ** 0.5)
            _B, _nh, _n, _key_n = B, nh, query_len, key_len
            # Bias representation. DEFAULT = fix #2 (the production path):
            #   gather-by-depth score_mod with P DETACHED (fast forward — flash-fused,
            #   77MB P, no 2.4GB materialization, no zeros_and_scatter) + a custom
            #   autograd Function (_DepthBiasGradP) that manually computes grad_P
            #   (recompute pT, fp32 scatter_add into 77MB P). grad_q/grad_k/grad_v come
            #   from flex's own fused backward. Measured: 14.24s/step single-GPU = 2.9x
            #   over dense-db (41.5s); grad reaches depth_emb (no find_unused_params
            #   needed); scales down with world_size. See memory pushdown-flex-slowdown.
            #
            # OLMO_PUSHDOWN_DENSE_DB=1: fall back to the OLD slow path (pre-materialize
            #   db=take_along_dim(P,Dc) ~2.4GB/layer, capture in score_mod; backward
            #   zeros_and_scatters into 2.4GB). Only for debugging/parity.
            #
            # Profiling probes (override the default; each needs find_unused_params:true
            # in the config because they drop the grad to depth_emb):
            #   OLMO_PUSHDOWN_NO_BIAS=1  -> drop the bias entirely (pure causal flex).
            #   OLMO_PUSHDOWN_DETACH=1   -> gather-by-depth, P detached, NO manual grad
            #     (depth_emb gets no grad). Isolates the forward cost.
            #   OLMO_PUSHDOWN_GATHER_BY_DEPTH=1 -> gather-by-depth, P requires grad
            #     (CRASHES on GPU — inductor's scatter drops grad; kept for reference).
            _no_bias = bool(os.environ.get("OLMO_PUSHDOWN_NO_BIAS"))
            _detach = bool(os.environ.get("OLMO_PUSHDOWN_DETACH"))
            _gather = bool(os.environ.get("OLMO_PUSHDOWN_GATHER_BY_DEPTH"))
            _dense_db = bool(os.environ.get("OLMO_PUSHDOWN_DENSE_DB"))
            _fix2 = not (_dense_db or _no_bias or _detach or _gather)  # default production path
            if _no_bias:
                del P, D, Dc  # no bias materialized
            elif _dense_db:
                db = torch.take_along_dim(P, Dc.unsqueeze(1), dim=3)  # (B, n_h, n, n) fp32 ~2.4GB
                _db = db
                del D, Dc  # P not needed by score_mod; freed on return
            else:  # fix2 (default), detach, or gather: gather-by-depth
                _P = P.detach() if (_fix2 or _detach) else P  # (B, n_h, n, Dmax) fp32 — 77 MB
                _Dc = Dc  # (B, n, n) int64 depth tape (no grad)
                del D
            # score_mod signature is (score, batch, head, q_idx, k_idx) — score FIRST.
            # flex calls it positionally; putting score last swaps score<->batch,
            # which (a) gathers the bias at wrong indices and (b) makes the backward
            # treat the float `score` as an integer batch index -> no grad flows to
            # score -> inductor asserts "joint_subgraph_buffer is None".
            #
            # flex traces/evaluates the score_mod over a padded block grid, so the
            # index scalars can exceed the real tensor dims at the boundary (those
            # cells are in masked-out blocks -> their gathered value is unused).
            # Clamp in-bounds before indexing to avoid OOB. Cast to long: flex
            # passes the batch index as a float scalar in some paths, and fake-
            # tensor tracing rejects non-integer indices ("tensors used as indices
            # must be long, int, byte or bool").
            if _no_bias:
                def _depth_score_mod(score, b, h, q_idx, kv_idx):
                    return score
            elif _dense_db:
                def _depth_score_mod(score, b, h, q_idx, kv_idx):
                    b = b.long().clamp(0, _B - 1)
                    h = h.long().clamp(0, _nh - 1)
                    q_idx = q_idx.long().clamp(0, _n - 1)
                    kv_idx = kv_idx.long().clamp(0, _key_n - 1)
                    return score + _db[b, h, q_idx, kv_idx] * inv_sqrt_hs
            else:  # fix2 (default), detach, or gather
                def _depth_score_mod(score, b, h, q_idx, kv_idx):
                    b = b.long().clamp(0, _B - 1)
                    h = h.long().clamp(0, _nh - 1)
                    q_idx = q_idx.long().clamp(0, _n - 1)
                    kv_idx = kv_idx.long().clamp(0, _key_n - 1)
                    # Gather P by the depth tape at (b,q,kv). P is (B,n_h,n,Dmax);
                    # the gather is one indexed load per cell, fused into the flex
                    # kernel. (FIX2/DETACH: P detached -> no backward scatter.
                    # GATHER: P requires grad -> backward drops grad on GPU.)
                    d = _Dc[b, q_idx, kv_idx]            # scalar depth at (b,q,kv)
                    return score + _P[b, h, q_idx, d] * inv_sqrt_hs
            # flex_attention expects (B, n_h, n, hs); expand k,v for GQA.
            if n_kv != nh:
                rep = nh // n_kv
                k = k.repeat_interleave(rep, dim=1, output_size=nh)
                v = v.repeat_interleave(rep, dim=1, output_size=nh)
            if _fix2:
                out = self.flex_attention(
                    q, k, v, score_mod=_depth_score_mod,
                    block_mask=block_mask,
                    kernel_options=self.flex_attention_kernel_options,
                )
                # Add the manual grad-P path. Forward: out + 0 == out (unchanged).
                # Backward: _DepthBiasGradP computes grad_P from grad_out (manual
                # recompute of pT + scatter_add into 77MB P); flex's own backward
                # (P was detached) supplies grad_q/grad_k/grad_v. The exact dense
                # FlexAttention support is stashed once on the shared cache by
                # OLMo.forward and reused by every layer.
                from olmo.pushdown import _DepthBiasGradP
                _gi = cache.get("pushdown_grad_invalid") if cache is not None else None
                _eq = cache.get("pushdown_empty_query") if cache is not None else None
                _am = cache.get("pushdown_attn_mask") if cache is not None else None
                out = out + _DepthBiasGradP.apply(
                    q, k, v, P, Dc, _gi, _eq, _am,
                    inv_sqrt_hs, out.detach(),
                )
            else:
                out = self.flex_attention(
                    q,
                    k,
                    v,
                    score_mod=_depth_score_mod,
                    block_mask=block_mask,
                    kernel_options=self.flex_attention_kernel_options,
                )
            return out
        # else: fall through to the SDPA additive-mask path (flex disabled, no
        # block_mask, or generation with a KV cache).

        # depth_bias[b,h,q,k] = P[b,h,q, D[b,q,k]] — gather P's last dim by the
        # depth tape D (shared across heads). Use take_along_dim with a (B,1,n,n)
        # VIEW of D (no materialization): this avoids the old path's
        # `idx = D.expand(B,nh,n,n).contiguous()` which allocated a (B,nh,n,n) int64
        # tensor (~4.8 GB at B=12) purely to feed torch.gather. take_along_dim
        # broadcasts the index over the head dim directly. ~2.6x faster than the
        # contiguous-gather path on the same hardware, and peak alloc drops by 4.8 GB/layer.
        depth_bias = torch.take_along_dim(
            P, Dc.unsqueeze(1), dim=3
        )                                            # (B, n_h, q, k) fp32
        depth_bias = depth_bias / (hs ** 0.5)       # match SDPA's 1/sqrt(hs) scaling
        # Free the big intermediates before SDPA: P is (B, nh, n, D) fp32; it is not
        # needed once depth_bias is formed — keeping it alive through SDPA raised the
        # per-layer peak. (Dc is a (B,n,n) view, negligible.)
        del P, D, Dc

        # Merge with the causal+pad attention_bias prepared by OLMo.forward.
        if attention_bias is not None:
            ab = attention_bias
            # Reshape to (B, n_h, q, k). Cached decoding receives the
            # bottom rows of a square causal bias from ``attention``.
            if ab.dim() == 3:
                ab = ab.unsqueeze(1)
            ab = ab[..., -query_len:, :key_len]
            if ab.shape[1] == 1:
                ab = ab.expand(B, nh, query_len, key_len)
            ab = ab.to(dtype=torch.float32)
            # Replace any -inf (masked) kept as -inf; add depth bias elsewhere.
            mask_neg = ab.isneginf()
            attn_mask = ab + depth_bias
            attn_mask = attn_mask.masked_fill(mask_neg, float("-inf"))
            del ab, mask_neg, depth_bias
        else:
            # No prebuilt bias: build a causal mask + depth bias (fp32).
            q_positions = torch.arange(
                key_len - query_len, key_len, device=q.device
            )[:, None]
            k_positions = torch.arange(key_len, device=q.device)[None, :]
            causal = torch.zeros(
                (query_len, key_len), device=q.device, dtype=torch.float32
            ).masked_fill(k_positions > q_positions, float("-inf"))
            attn_mask = depth_bias + causal.unsqueeze(0).unsqueeze(0)
            del depth_bias, causal

        # SDPA expects (B, n_h, n, hs) and attn_mask (B, n_h, n, n) additive float.
        # Expand k,v from n_kv to nh heads if GQA.
        if n_kv != nh:
            rep = nh // n_kv
            k = k.repeat_interleave(rep, dim=1, output_size=nh)
            v = v.repeat_interleave(rep, dim=1, output_size=nh)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False,
        )

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        tree_spans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            return OLMoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return OLMoLlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class OLMoSequentialBlock(OLMoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). To compute it as ``LN(MLP(x + LN(Attention(x))))``,
    use the flag `norm_after`.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Attention input projection. Projects x -> (q, k, v)

        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        # Layer norms.
        self.attn_norm = LayerNorm.build(config, size=config.d_model)
        self.ff_norm = LayerNorm.build(config, size=config.d_model)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.att_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        tree_spans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # apply norm before
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                h = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                h = self.attn_norm(x)
        else:
            h = x

        qkv = self.att_proj(h)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
                block_mask=block_mask,
                tree_spans=tree_spans,
            )
        else:
            att, cache = self.attention(
                q,
                k,
                v,
                attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
                max_doc_len=max_doc_len,
                cu_doc_lens=cu_doc_lens,
                block_mask=block_mask,
                tree_spans=tree_spans,
            )

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.ff_proj(x)

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.dropout(x)
        x = og_x + x

        return x, cache


class OLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model // config.n_heads
            v_proj_out_dim = config.d_model // config.n_heads
        else:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model
            v_proj_out_dim = config.d_model
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.q_proj, std, cutoff_factor)
        init_normal(self.k_proj, std, cutoff_factor)
        init_normal(self.v_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if max_doc_len is not None or cu_doc_lens is not None:
            raise NotImplementedError(
                f"attention document masking is not implemented for {self.__class__.__name__}"
            )

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if is_causal:
            assert attn_mask is None

            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
        elif attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype)
        else:
            attn_bias = torch.zeros_like(attn_weights)

        attn_weights += attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        tree_spans: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
            block_mask=block_mask,
            tree_spans=tree_spans,
        )

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class OLMoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    """
    Hidden states from each block.
    """

    treereg_hidden: Optional[torch.Tensor] = None
    """
    Post-block residual hidden state at ``treereg_layer`` (for the TreeReg auxiliary
    loss, Nandi et al. 2025). ``None`` unless ``transformer_grammar_type == 'treereg'``.
    """

    attachment_logits: Optional[torch.Tensor] = None
    """
    Pushdown attachment-head logits ``(B, n, n)`` fp32 (Murty et al. 2023, Eq. 5):
    ``logits[b, k, j]`` is the score for reduce target ``r_k = j``. ``None`` unless
    ``transformer_grammar_type == 'pushdown'`` and ``compute_attachment_logits``.
    """

    final_hidden: Optional[torch.Tensor] = None
    """
    Final-layer residual hidden state ``(B, n, d)`` (captured before ``ln_f``).
    Exposed for the pushdown attachment-head loss. ``None`` unless pushdown.
    """


class OLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class OLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                    block_mask=block_mask,
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                    block_mask=block_mask,
                )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)


class OLMo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi:
            raise OLMoConfigurationError("ALiBi is currently not supported with FlashAttention")
    
        if self.config.flex_attention and self.config.attention_dropout != 0.0:
            raise OLMoConfigurationError("Flex attention is currently not supported with nonzero attention dropout")

        for field_name in (
            "flex_attention_train_min_sequence_length",
            "flex_attention_eval_min_sequence_length",
        ):
            if getattr(self.config, field_name) < 0:
                raise OLMoConfigurationError(f"{field_name} must be non-negative")
        flex_pad_multiple = self.config.flex_attention_pad_to_multiple
        if flex_pad_multiple is not None and (
            flex_pad_multiple <= 0 or flex_pad_multiple % 128 != 0
        ):
            raise OLMoConfigurationError(
                "flex_attention_pad_to_multiple must be a positive multiple of 128 or null"
            )

        if self.config.alibi and self.config.rope:
            raise OLMoConfigurationError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        if self.config.transformer_grammar_type == "treereg":
            if not 1 <= self.config.treereg_layer <= self.config.n_layers:
                raise OLMoConfigurationError(
                    "treereg_layer is 1-based and must be in [1, n_layers]"
                )
            if not 1 <= self.config.treereg_n_heads <= self.config.n_heads:
                raise OLMoConfigurationError(
                    "treereg_n_heads must be in [1, n_heads]"
                )
            if self.config.treereg_every_k < 0:
                raise OLMoConfigurationError("treereg_every_k must be non-negative")
            if self.config.block_group_size != 1:
                raise OLMoConfigurationError(
                    "TreeReg currently requires block_group_size=1 so the selected "
                    "intermediate post-block residual is available"
                )

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [OLMoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if config.embedding_layer_norm:
            self.transformer.update({"emb_norm": LayerNorm.build(config)})

        # Pushdown attachment head (Murty et al. 2023, Eq. 5): predicts each
        # token's reduce target r_k from the final-layer hidden states. Train-only
        # at the loss site (olmo/train.py); does not alter the depth-bias forward.
        # Single head at the model level (the paper uses W+MLP once, at layer L).
        if config.transformer_grammar_type == "pushdown":
            from olmo.attachment import PushdownAttachmentHead
            self.pushdown_attachment_head = PushdownAttachmentHead(
                d_model=config.d_model,
                vocab_size=config.embedding_size or config.vocab_size,
            )

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None
        self.__num_bck_flops: Optional[int] = None

        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.

        if self.config.init_fn == InitFnType.normal:
            # Note: We may potentially want to multiply the std by a factor of sqrt(d) in case of `scale_logits`
            # and `weight_tying`. However, we are currently not using either, and may need to rethink the init logic
            # if/when we do want it.
            wte_std = self.config.emb_init_std or self.config.init_std
            wte_cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            wte_std = self.config.emb_init_std or 1.0 / math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            wte_std = self.config.init_std
            if self.config.emb_init_std is not None:
                wte_std = self.config.emb_init_std
            elif self.config.scale_emb_init:
                wte_std *= math.sqrt(self.config.d_model)
            wte_cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.transformer.wte, std=wte_std, init_cutoff_factor=wte_cutoff_factor)

        if hasattr(self.transformer, "wpe"):
            if self.config.init_fn == InitFnType.normal:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                wpe_std = 1 / math.sqrt(self.config.d_model)
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                wpe_std = self.config.init_std
                wpe_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.wpe, std=wpe_std, init_cutoff_factor=wpe_cutoff_factor)

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            if self.config.init_fn == InitFnType.normal:
                ff_out_std = self.config.init_std
                ff_out_cutoff_factor = self.config.init_cutoff_factor
            elif self.config.init_fn == InitFnType.mitchell:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            elif self.config.init_fn == InitFnType.full_megatron:
                ff_out_std = 1 / math.sqrt(self.config.d_model)
                ff_out_cutoff_factor = self.config.init_cutoff_factor or 3.0
            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.transformer.ff_out, ff_out_std, ff_out_cutoff_factor)

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        past_final_hidden: Optional[torch.Tensor] = None,
        past_input_ids: Optional[torch.Tensor] = None,
        past_sentence_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        doc_lens: Optional[torch.Tensor] = None,
        max_doc_lens: Optional[Sequence[int]] = None,
        tree_spans: Optional[torch.Tensor] = None,
        pushdown_sentence_ids: Optional[torch.Tensor] = None,
        compute_attachment_logits: bool = False,
        return_final_hidden: bool = False,
        logits_range: Optional[Tuple[int, int]] = None,
        attachment_query_range: Optional[Tuple[int, int]] = None,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        :param doc_lens: Document lengths to use in attention for intra-document masking.
            Shape `(batch_size, max_docs)`.
        :param max_doc_lens: Maximum document length for each instance in the batch.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)
        if past_final_hidden is not None and past_final_hidden.shape[1] != past_length:
            raise ValueError("past_final_hidden length must match past_key_values")
        if past_input_ids is not None and past_input_ids.shape[1] != past_length:
            raise ValueError("past_input_ids length must match past_key_values")
        if past_sentence_ids is not None and past_sentence_ids.shape[1] != past_length:
            raise ValueError("past_sentence_ids length must match past_key_values")

        max_doc_len: Optional[int] = None
        cu_doc_lens: Optional[torch.Tensor] = None
        if doc_lens is not None and max_doc_lens is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        # Apply embedding layer norm.
        if self.config.embedding_layer_norm:
            x = self.transformer.emb_norm(x)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Route TG-style structured masks by workload and original sequence
        # length. Ordinary causal attention has no attention_bias here and keeps
        # using FlashAttention (or SDPA when flash-attn is unavailable). Pushdown
        # has its own score-mod route below.
        original_seq_len = seq_len
        original_has_attention_bias = attention_bias is not None
        structured_flex = _use_flex_for_structured_attention(
            self.config,
            original_seq_len,
            training=self.training,
            has_attention_bias=attention_bias is not None,
            has_past_key_values=past_key_values is not None,
            use_cache=use_cache,
        )
        structured_route_reason = "selected"
        flex_pad_multiple: Optional[int] = None
        if structured_flex:
            if attention_bias.shape[-2:] != (original_seq_len, original_seq_len):
                # A fixed square BlockMask cannot represent cached/rectangular
                # decoding. Preserve the exact additive mask through SDPA.
                structured_flex = False
                structured_route_reason = "non_square_bias"
            else:
                flex_pad_multiple = _flex_attention_pad_multiple(self.config)
                if (
                    flex_pad_multiple is None
                    and torch.is_grad_enabled()
                    and original_seq_len % 128 != 0
                ):
                    # Never enter the known non-divisible Flex backward path
                    # without padding. ``None`` therefore means safe SDPA
                    # fallback, not an unsafe exact-length Flex call.
                    structured_flex = False
                    structured_route_reason = "unsafe_unpadded_backward"
        elif attention_bias is not None:
            if past_key_values is not None or use_cache:
                structured_route_reason = "kv_cache"
            else:
                threshold = (
                    self.config.flex_attention_train_min_sequence_length
                    if self.training
                    else self.config.flex_attention_eval_min_sequence_length
                )
                structured_route_reason = f"below_threshold_{threshold}"

        flex_unpadded_seq_len: Optional[int] = None
        block_mask: Optional[BlockMask] = None
        if structured_flex:
            if attention_bias.dtype == torch.bool:
                flex_mask = attention_bias
            else:
                flex_mask = attention_bias == 1.0
            if flex_mask.dim() == 2:
                flex_mask = flex_mask[None, None, :, :]
            elif flex_mask.dim() == 3:
                flex_mask = flex_mask[:, None, :, :]
            elif flex_mask.dim() != 4:
                raise OLMoConfigurationError(
                    "structured attention_bias must have 2, 3, or 4 dimensions"
                )
            if flex_mask.shape[0] == 1 and batch_size != 1:
                flex_mask = flex_mask.expand(batch_size, -1, -1, -1)
            elif flex_mask.shape[0] != batch_size:
                raise OLMoConfigurationError(
                    "structured attention_bias batch dimension must be 1 or match input_ids"
                )

            # Merge optional key padding into the structured mask before vmap.
            # This replaces the old Python ``and`` closure (not vmap-safe) and
            # correctly indexes a separate key mask for every batch element.
            if attention_mask is not None:
                if attention_mask.dim() == 1:
                    if attention_mask.numel() != original_seq_len:
                        raise OLMoConfigurationError(
                            "1-D attention_mask length must match the structured sequence length"
                        )
                    valid_keys = attention_mask.unsqueeze(0).expand(batch_size, -1)
                else:
                    if attention_mask.numel() != batch_size * original_seq_len:
                        raise OLMoConfigurationError(
                            "attention_mask must contain one key-validity value per batch token"
                        )
                    valid_keys = attention_mask.reshape(batch_size, original_seq_len)
                valid_keys = valid_keys.to(device=flex_mask.device, dtype=torch.bool)
                flex_mask = flex_mask & valid_keys[:, None, None, :]

            # Padding happens after embedding and is removed before logits, so
            # the original loss shape is unchanged. Padded queries attend only
            # to themselves; original queries cannot attend to padded keys.
            if flex_pad_multiple is not None:
                padded_seq_len = math.ceil(original_seq_len / flex_pad_multiple) * flex_pad_multiple
                if padded_seq_len != original_seq_len:
                    flex_unpadded_seq_len = original_seq_len
                    x = F.pad(x, (0, 0, 0, padded_seq_len - original_seq_len))
                    padded_flex_mask = torch.zeros(
                        (*flex_mask.shape[:-2], padded_seq_len, padded_seq_len),
                        dtype=torch.bool,
                        device=flex_mask.device,
                    )
                    padded_flex_mask[..., :original_seq_len, :original_seq_len] = flex_mask
                    padded_positions = torch.arange(
                        original_seq_len, padded_seq_len, device=flex_mask.device
                    )
                    padded_flex_mask[..., padded_positions, padded_positions] = True
                    flex_mask = padded_flex_mask
                    seq_len = padded_seq_len
                    log.warning(
                        "FLEX_DIAG padded_sequence original=%d padded=%d multiple=%d",
                        original_seq_len,
                        seq_len,
                        flex_pad_multiple,
                    )

            _, mask_heads, q_len, kv_len = flex_mask.shape
            if mask_heads == 1:
                squeezed_flex_mask = flex_mask.squeeze(1)

                def tg_mask(batch, _head, q_idx, kv_idx):
                    return squeezed_flex_mask[batch, q_idx, kv_idx]

                block_mask = create_block_mask(
                    mask_mod=tg_mask,
                    B=batch_size,
                    H=None,
                    Q_LEN=q_len,
                    KV_LEN=kv_len,
                    device=x.device,
                )
            else:
                def tg_mask_per_head(batch, head, q_idx, kv_idx):
                    return flex_mask[batch, head, q_idx, kv_idx]

                block_mask = create_block_mask(
                    mask_mod=tg_mask_per_head,
                    B=batch_size,
                    H=mask_heads,
                    Q_LEN=q_len,
                    KV_LEN=kv_len,
                    device=x.device,
                )
            attention_bias = None

        if (
            os.environ.get("OLMO_ATTENTION_ROUTER_DIAGNOSTICS") == "1"
            and self.config.flex_attention
        ):
            pushdown_flex = (
                self.config.transformer_grammar_type == "pushdown"
                and self.config.pushdown_use_flex
                and attention_mask is not None
                and past_key_values is None
            )
            if structured_flex:
                backend = "flex_structured"
            elif pushdown_flex:
                backend = "flex_pushdown"
                structured_route_reason = "pushdown_score_mod"
            elif not original_has_attention_bias:
                backend = "flash_or_sdpa_causal"
                structured_route_reason = "no_structured_bias"
            else:
                backend = "sdpa_structured"
            log.warning(
                "ATTENTION_ROUTE grammar=%s training=%s original_seq=%d effective_seq=%d "
                "backend=%s reason=%s has_bias=%s has_attention_mask=%s",
                self.config.transformer_grammar_type,
                self.training,
                original_seq_len,
                seq_len,
                backend,
                structured_route_reason,
                original_has_attention_bias,
                attention_mask is not None,
            )

        if structured_flex:
            pass
        elif (self.config.transformer_grammar_type == "pushdown"
              and self.config.flex_attention and self.config.pushdown_use_flex
              and attention_mask is not None and past_key_values is None):
            # Pushdown fast path: fuse causality + padding into a flex block_mask
            # (built once per forward, reused by all 12 blocks) and put the per-(q,kv)
            # depth bias into the score_mod in OLMoBlock._pushdown_attention. This replaces
            # the (B, n_h, n, n) fp32 additive-mask + math-backend SDPA path (the ~100x
            # slowdown) with a single fused flash-class kernel. The 1-D bool attention_mask
            # is captured here BEFORE the else-branch reshapes it into a float bias; if it
            # is 2-D (B, seq) the kv_idx index below works directly, otherwise we squeeze.
            # `past_key_values is None` guards generation: flex block_masks are fixed-length
            # and don't extend with a KV cache, so decoding falls back to the SDPA path.
            _am = attention_mask
            if _am.dim() > 1:
                _am = _am.view(batch_size, seq_len)
            am = _am
            # Document masking (eval-only, generate_doc_lengths=True): block
            # cross-document attention so PPL reflects per-doc conditioning. The
            # depth bias is already doc-local (spans never cross an EOS doc boundary
            # — see olmo/data/parse_align.py tree_spans), so only the causal+pad
            # mask needs the doc constraint. doc_id[b, idx] = index of the document
            # owning position idx; built from the per-batch cumulative doc lengths.
            # Padded positions get an out-of-range id but are gated by `am[b, kv_idx]`.
            doc_id = None
            if doc_lens is not None and max_doc_lens is not None:
                # doc_lens: (B, max_docs) with 0-pad. Exclusive doc ends per batch.
                dl = doc_lens.to(dtype=torch.long, device=x.device)
                ends = torch.cumsum(dl, dim=-1)           # (B, max_docs)
                idxs = torch.arange(seq_len, device=x.device)
                # doc_id[b, idx] = #doc-ends <= idx  (positions before the 1st end
                # are doc 0, etc.). searchsorted(side=right) on each row.
                doc_id = torch.stack([
                    torch.searchsorted(ends[b], idxs, right=True) for b in range(batch_size)
                ], dim=0)                                  # (B, seq_len) long
            def _pushdown_mask_mod(b, h, q_idx, kv_idx):
                # Bitwise `&`, NOT Python `and`: `and` short-circuits on the *value*
                # of `am[b, kv_idx]`, which is data-dependent control flow -> vmap
                # (used by create_block_mask to infer the block sparsity) rejects it
                # with "attempting to use a Tensor in some data-dependent control
                # flow". `&` and `==` are plain elementwise tensor ops vmap can lower.
                m = am[b, kv_idx] & (q_idx >= kv_idx)
                if doc_id is not None:
                    m = m & (doc_id[b, q_idx] == doc_id[b, kv_idx])
                return m
            block_mask = create_block_mask(
                mask_mod=_pushdown_mask_mod, B=batch_size, H=None,
                Q_LEN=seq_len, KV_LEN=seq_len, device=x.device,
            )
            # Retain the 1-D key mask only for the explicit legacy-gradient
            # diagnostic. Production uses the exact dense support below. Both
            # entries are overwritten each forward, so there is no stale-batch risk.
            self.__cache["pushdown_attn_mask"] = am
            # The manual depth-bias gradient must use the exact same support as
            # FlexAttention. Cache its dense head-independent form ONCE per
            # forward instead of rebuilding a causal matrix in every layer's
            # backward. This is only B*n*n bool (16 MiB at B=4,n=2048), shared
            # by all 12 custom-autograd nodes. In particular, include doc_id:
            # the old manual backward used causal+pad only and therefore sent
            # spurious depth gradients through cross-document cells that the
            # forward FlexAttention kernel had masked out.
            positions = torch.arange(seq_len, device=x.device)
            pushdown_grad_mask = (
                am[:, None, :]
                & (positions[:, None] >= positions[None, :]).unsqueeze(0)
            )
            if doc_id is not None:
                pushdown_grad_mask &= doc_id[:, :, None] == doc_id[:, None, :]
            self.__cache["pushdown_empty_query"] = ~pushdown_grad_mask.any(dim=-1)
            self.__cache["pushdown_grad_invalid"] = ~pushdown_grad_mask
            attention_bias = None
        else:
            # Document masking (eval-only, generate_doc_lengths=True): when doc
            # boundaries are provided, leave attention_bias=None and the bool
            # attention_mask untouched, so OLMoBlock._scaled_dot_product_attention
            # takes the flash_attn_varlen_func doc-mask branch (which asserts
            # attn_mask is None). causality + doc boundary + pad are all handled
            # by varlen over cu_doc_lens. get_labels still masks pad for the CE.
            _doc_mask_active = (doc_lens is not None and max_doc_lens is not None)
            # Transform the attention mask into what the blocks expect.
            if attention_mask is not None and not _doc_mask_active:
                # shape: (batch_size, 1, 1, seq_len)
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
                attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

            # Merge attention mask with attention bias.
            if (
                attention_bias is not None
                or attention_mask is not None
                or self.config.alibi
                # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
                # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
                # scores correctly. But Flash_attn compute correctly
                or (past_key_values is not None and not self.config.flash_attention)
            ) and not _doc_mask_active:
                if attention_bias is None and self.config.alibi:
                    attention_bias = get_causal_attention_bias(
                        self.__cache, past_length + seq_len, x.device
                    ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
                elif attention_bias is None:
                    attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
                elif attention_bias.dtype in (torch.int8, torch.bool):
                    attention_bias = attention_bias.to(dtype=torch.float)
                    attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

                # Transform to the right shape and data type.
                mask_len = seq_len
                if attention_mask is not None:
                    mask_len = attention_mask.shape[-1]
                elif past_key_values is not None:
                    mask_len = past_key_values[0][0].shape[-2] + seq_len
                attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

                # Add in the masking bias.
                if attention_mask is not None:
                    attention_bias = attention_bias + attention_mask
                    # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                    # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                    # it can produce NaNs.
                    ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

                # All transformer blocks use the same autocast precision. Cast
                # a reusable SDPA mask once here instead of allocating the same
                # float32 -> bf16/fp16 conversion independently in every layer.
                # Pushdown's SDPA fallback deliberately keeps its merged depth
                # bias in fp32 and is therefore excluded.
                if (
                    attention_bias is not None
                    and self.config.transformer_grammar_type != "pushdown"
                ):
                    attention_bias = OLMoBlock._cast_attn_bias(attention_bias, x.dtype)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []
        treereg_hidden: Optional[torch.Tensor] = None
        _is_treereg = self.config.transformer_grammar_type == "treereg"
        _treereg_layer = self.config.treereg_layer
        # Invalidate the per-forward pushdown depth-tape memo: tree_spans changes
        # every step, so a cached entry from the previous forward must not survive.
        # (_pushdown_attention repopulates it on the first block that needs it, then
        # the remaining 11 blocks reuse the int8 S instead of recomputing it.)
        if self.config.transformer_grammar_type == "pushdown":
            self.__cache.pop("pushdown_depth_matrix", None)

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block,
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                        block_mask=block_mask,
                        tree_spans=tree_spans,
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(
                        x,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                        block_mask=block_mask,
                        tree_spans=tree_spans,
                    )

                if _is_treereg and block_idx + 1 == _treereg_layer:
                    # Capture post-block residual for the TreeReg SCIN loss.
                    treereg_hidden = x

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                    block_mask=block_mask,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if flex_unpadded_seq_len is not None:
            x = x[:, :flex_unpadded_seq_len]
            if output_hidden_states:
                all_hidden_states = [state[:, :flex_unpadded_seq_len] for state in all_hidden_states]

        # Pushdown attachment head: compute the reduce-target logits from the
        # final-layer residual, captured BEFORE ln_f and BEFORE the last_logits_only
        # slice (the head needs the full (B, n, d) sequence: keys h_j^L over all
        # prefix tokens, query h̃_k = MLP(emb(x_k), h_{k-1}^L)).  Cached document
        # PPL supplies the retained candidate-0 prefix residual explicitly.
        attachment_logits = None
        final_hidden = x if return_final_hidden else None
        if (self.config.transformer_grammar_type == "pushdown"
                and (compute_attachment_logits or return_final_hidden)):
            _profile_attachment = bool(os.environ.get("OLMO_PUSHDOWN_PHASE_PROFILE"))
            if _profile_attachment:
                _attachment_start = torch.cuda.Event(enable_timing=True)
                _attachment_end = torch.cuda.Event(enable_timing=True)
                _attachment_start.record()
            # Keep the *new* residual rows as the cache payload.  Attachment
            # scoring below may concatenate them with candidate-0 prefix rows.
            final_hidden = x  # (B, current_length, d), pre-ln_f residual
            if compute_attachment_logits:
                if past_final_hidden is not None:
                    if past_input_ids is None:
                        raise ValueError("cached attachment scoring requires past_input_ids")
                    attachment_hidden = torch.cat((past_final_hidden, x), dim=1)
                    attachment_input_ids = torch.cat((past_input_ids, input_ids), dim=1)
                    attachment_sentence_ids = (
                        None if past_sentence_ids is None or pushdown_sentence_ids is None
                        else torch.cat((past_sentence_ids, pushdown_sentence_ids), dim=1)
                    )
                    attachment_query_range = (
                        past_length,
                        past_length + x.shape[1],
                    )
                else:
                    attachment_hidden = x
                    attachment_input_ids = input_ids
                    attachment_sentence_ids = pushdown_sentence_ids
            if compute_attachment_logits:
                # Recover the 1-D valid-key bool mask from `attention_mask`. By this
                # point OLMo.forward's else-branch may have reshaped it to a float
                # (B,1,1,n) additive bias where 0.0 = valid and finfo.min = padded;
                # or it may still be the raw bool (B,n) if no branch ran. Normalize.
                am = attention_mask
                if am is not None:
                    if am.dtype == torch.bool:
                        am = am.view(am.shape[0], -1)
                    else:
                        # float additive bias: valid == 0.0, pad == finfo.min (<0).
                        am = (am.view(am.shape[0], -1) == 0.0)
                if past_final_hidden is not None:
                    am = torch.ones(
                        attachment_hidden.shape[:2], dtype=torch.bool, device=attachment_hidden.device
                    )
                attachment_logits = self.pushdown_attachment_head(
                    attachment_hidden,
                    attachment_input_ids,
                    self.transformer.wte.weight,
                    am,
                    root_token_id=self.config.bos_token_id,
                    eos_token_id=self.config.eos_token_id,
                    sentence_ids=attachment_sentence_ids,
                    query_range=attachment_query_range,
                )
                if _profile_attachment:
                    _attachment_end.record()
                    torch.cuda.synchronize()
                    _profile_rank = (
                        torch.distributed.get_rank()
                        if torch.distributed.is_available() and torch.distributed.is_initialized()
                        else 0
                    )
                    if _profile_rank == 0:
                        print(
                            f"[pushdown_attachment_forward] ms="
                            f"{_attachment_start.elapsed_time(_attachment_end):.1f}",
                            flush=True,
                        )

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        if logits_range is not None:
            start, end = map(int, logits_range)
            if not 0 <= start <= end <= x.shape[1]:
                raise ValueError(f"invalid logits_range={logits_range} for length {x.shape[1]}")
            x = x[:, start:end]

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OLMoOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            treereg_hidden=treereg_hidden,
            attachment_logits=attachment_logits,
            final_hidden=final_hidden,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False)
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * self.config.max_sequence_length)
        )
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params()
        params_flops_per_token = 4 * n_params
        attn_flops_per_token = self.config.n_layers * 8 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops

    def pause_generate(
        self,
        input_ids: torch.LongTensor,
        pause_spec: "tuple[int, int]",
        max_real_tokens: int,
        pause_token_id: Optional[int] = None,
        vocab: Optional[SentencepieceVocab] = None,
        attention_mask: Optional[torch.Tensor] = None,
        eos_token_id: Optional[int] = None,
        beam_size: int = 6,
        score_pause_tokens: bool = True,
    ) -> "OLMoGenerateOutput":
        """Constrained beam generation for pause-expanded causal LMs.

        The XSum prompt is already expanded by ``pause_input_ids``.  Generation
        must continue that *absolute* ``q real + p pause`` phase.  At a real
        position this method searches terminal tokens; at a pause position it
        forces either the checkpoint's dedicated pause token or, for legacy
        repeat-mode checkpoints, the most recent real token.  The returned token
        stream contains real positions only, so callers never have to guess the
        phase of a generated suffix with ``extract_real_tokens``.

        ``score_pause_tokens`` should be true for ordinary pause models, which
        learned the pause targets, and false for ``*_label`` models, whose pause
        targets were masked during training.

        XSum currently evaluates pause models with device batch size one.  Keep
        that contract explicit: variable-length left-padded prompts can have
        different pause phases and need a grouped decoder rather than silently
        sharing one global beam-search timestep.
        """
        p, q = pause_spec
        if p < 0:
            raise ValueError(f"pause numerator must be >= 0, got {p}")
        if q < 1:
            raise ValueError(f"pause denominator must be >= 1, got {q}")
        if max_real_tokens < 1:
            raise ValueError("max_real_tokens must be positive")
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "pause_generate currently requires input_ids shape (1, L); "
                "set device_eval_batch_size=1 for pause XSum"
            )
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        prompt_len = (
            int(attention_mask[0].sum().item())
            if attention_mask is not None
            else int(input_ids.shape[1])
        )
        period = q + p

        # Invert n + floor(n/q)*p.  Valid pause_input_ids outputs have exactly
        # one solution because the expansion length is strictly increasing.
        low, high = 0, prompt_len
        while low <= high:
            mid = (low + high) // 2
            expanded = mid + (mid // q) * p
            if expanded < prompt_len:
                low = mid + 1
            elif expanded > prompt_len:
                high = mid - 1
            else:
                prompt_real_tokens = mid
                break
        else:
            raise ValueError(
                f"prompt length {prompt_len} is not a valid expansion for pause{p}/{q}"
            )

        final_real_tokens = prompt_real_tokens + max_real_tokens
        final_expanded_len = final_real_tokens + (final_real_tokens // q) * p
        expanded_steps = final_expanded_len - prompt_len
        if input_ids.shape[1] + expanded_steps > self.config.max_sequence_length:
            raise ValueError(
                "pause generation exceeds model context: "
                f"prompt={input_ids.shape[1]} expanded_steps={expanded_steps} "
                f"max={self.config.max_sequence_length}"
            )

        beam_search = BeamSearch(
            eos,
            max_steps=expanded_steps,
            beam_size=beam_size,
        )
        tokens_generated = 0

        def flatten_past_key_values(
            values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            state = {}
            for layer, (key, value) in enumerate(values):
                state[f"past_key_{layer}"] = key
                state[f"past_value_{layer}"] = value
            return state

        def unflatten_past_key_values(
            state: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            return [
                (state[f"past_key_{layer}"], state[f"past_value_{layer}"])
                for layer in range(self.config.n_layers)
            ]

        def step(
            last_predictions: torch.Tensor,
            state: Dict[str, torch.Tensor],
            time_step: int,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            nonlocal tokens_generated
            current_attention_mask = state.get("attention_mask")
            if tokens_generated == 0:
                step_input_ids = state["input_ids"]
                past_key_values = None
            else:
                step_input_ids = last_predictions.unsqueeze(1)
                past_key_values = unflatten_past_key_values(state)
                if current_attention_mask is not None:
                    current_attention_mask = torch.cat(
                        (
                            current_attention_mask,
                            current_attention_mask.new_ones(
                                (step_input_ids.shape[0], 1)
                            ),
                        ),
                        dim=-1,
                    )

            output = self(
                step_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)
            tokens_generated += 1

            absolute_position = prompt_len + time_step
            is_pause_position = absolute_position % period >= q
            if is_pause_position:
                forced = (
                    torch.full_like(last_predictions, int(pause_token_id))
                    if pause_token_id is not None
                    else last_predictions
                )
                forced_scores = log_probs.gather(1, forced.unsqueeze(1)).squeeze(1)
                if not score_pause_tokens:
                    forced_scores = torch.zeros_like(forced_scores)
                constrained = torch.full_like(
                    log_probs, torch.finfo(log_probs.dtype).min
                )
                constrained.scatter_(1, forced.unsqueeze(1), forced_scores.unsqueeze(1))
                log_probs = constrained
            else:
                # Dedicated pause symbols and grammar non-terminals are format
                # tokens, not summary words.  They are only legal when forced at
                # a pause position.
                if pause_token_id is not None:
                    log_probs[:, int(pause_token_id)] = torch.finfo(log_probs.dtype).min
                if vocab is not None:
                    nt_start = int(vocab.opening_non_terminals[0])
                    nt_end = int(vocab.closing_non_terminals[1])
                    log_probs[:, nt_start:nt_end] = torch.finfo(log_probs.dtype).min

            new_state = flatten_past_key_values(output.attn_key_values)
            if current_attention_mask is not None:
                new_state["attention_mask"] = current_attention_mask
            return log_probs, new_state

        initial_predictions = input_ids.new_zeros((1,))
        initial_state: Dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            initial_state["attention_mask"] = attention_mask
        with torch.no_grad():
            expanded_ids, scores = beam_search.search(
                initial_predictions, initial_state, step
            )

        real_positions = torch.tensor(
            [
                (prompt_len + offset) % period < q
                for offset in range(expanded_ids.shape[-1])
            ],
            dtype=torch.bool,
            device=expanded_ids.device,
        )
        real_ids = expanded_ids[..., real_positions]
        return OLMoGenerateOutput(token_ids=real_ids, scores=scores)

    def pause_label_generate(
        self,
        input_ids: torch.LongTensor,
        pause_spec: "tuple[int, int]",
        max_real_tokens: int,
        eos_token_id: Optional[int] = None,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """Compatibility wrapper for legacy repeat-mode ``*_label`` callers."""
        generated = self.pause_generate(
            input_ids=input_ids,
            pause_spec=pause_spec,
            max_real_tokens=max_real_tokens,
            pause_token_id=None,
            eos_token_id=eos_token_id,
            beam_size=beam_size,
            score_pause_tokens=False,
        )
        return generated.token_ids[:, 0, :]

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see :class:`BeamSearch`.
        """
        beam_search = BeamSearch(
            self.config.eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= self.config.max_sequence_length
            )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
            else:
                past_key_values = None
                input_ids = state["input_ids"]

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.no_grad():
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: Optional[CheckpointType] = None
    ) -> OLMo:
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = OLMo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            load_result = model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            # Keep backward-compatible loading for older checkpoints, but expose
            # the fact that their train-only attachment head was absent so formal
            # joint-PPL callers can reject random initialization.
            model._pushdown_attachment_weights_loaded = not any(
                key.startswith("pushdown_attachment_head.")
                for key in load_result.missing_keys
            )
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            if train_config.sharded_checkpointer == ShardedCheckpointerType.olmo_core:
                from olmo_core.distributed.checkpoint import (  # type: ignore
                    load_model_and_optim_state,
                )

                model_config.init_device = device
                model = OLMo(model_config)
                load_model_and_optim_state(checkpoint_dir, model)
            else:
                # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
                from .checkpoint import load_model_state

                # Initialize model on target device. In this case the state dict is loaded in-place
                # so it's not necessary to start on CPU if the target device is a GPU.
                model_config.init_device = device
                model = OLMo(model_config)

                # Load state dict in place.
                load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new

    def word_sync_beam_search(self,
        vocab : SentencepieceVocab,
        eval_input_ids : Optional[torch.Tensor] = None,
        max_word_steps : Optional[int] = None,
        max_length : Optional[int] = None,
        beam_size : int = 300,
        nc : Optional[int] = None,
        pc : int = 4,
        past_input : Optional[torch.Tensor] = None,
        generate_TG_bias : Optional[TG_attention_bias] = None,
        tag_start : Optional[int] = None,
        tag_end : Optional[int] = None,
        strategy : BeamSearchType = BeamSearchType.default,
        transformer_grammar_type: str = "",
        tree_eval_type: str = "default",
        beam_dump: Optional[Dict] = None,
    ) -> OLMoOutput | float:
        """
        Word sync beam search for the model.
        Hyperparameter from Stern(2017): beam_size = k_word, sub_beam_size = kn = 10*k_word
        k_fast_track = ks = k_word // 10
        """
        # from tokenizers import Tokenizer
        # tmptokenizer:Tokenizer = Tokenizer.from_file("./dataset/bbc-news/TG_GPT2_tokenizer.json")

        is_TG_input = (generate_TG_bias is not None) or transformer_grammar_type=="tgtree"
        # Pause models are plain causal LMs whose input has deterministic pause
        # repeats interleaved; they do NOT generate tree-grammar non-terminal
        # structure. The tree-grammar decoder below samples the main beam
        # candidates over the FULL vocab (including the NT id range), which would
        # pollute a pause model's summary with bracket tokens it never learned to
        # emit. Mask the entire NT range out of the main candidates when
        # generating from a pause model (only the terminal fast-track top-k was
        # NT-masked before — see log_probs[:, NT_start:NT_end] below). No-op for
        # TG/tgtree (they emit NTs by design) and for scoring mode
        # (eval_input_ids is not None).
        is_pause = transformer_grammar_type[:5] == "pause"
        first_step = past_input is not None
        Genlength = eval_input_ids.shape[0] if eval_input_ids is not None else max_word_steps
        istart = 0
        initial_attention_bias = None
        if past_input is None:
            if eval_input_ids is not None:
                istart = 1  # if we have eval_input_ids, we start from 1, 
                past_input = torch.LongTensor([eval_input_ids[0].item()])
            else:
                past_input = torch.LongTensor([vocab.bos])
        elif generate_TG_bias is not None:
            generate_TG_bias.reset_state()
            initial_attention_bias, _ = generate_TG_bias(past_input, update_state=True)
        nc = int(1.2*Genlength) if nc is None else nc

        # print(f"past input shape is {past_input.shape}")
        # print(f"{self.device} past input is {tmptokenizer.decode(past_input.tolist(), skip_special_tokens=False)}")
        def collate_fn(data):
            max_input_len = 0
            for sample in data:
                if sample["input_ids"].shape[0] > max_input_len:
                    max_input_len = sample["input_ids"].shape[0]
            input_ids = []
            all_attention_bias = []
            log_probs = []
            last_token_index = []
            Stop_Add_NT = []
            bid = -1
            for sample in data:  # pad according to max_lengths
                bid += 1
                pad_shape = (   # right padding only
                    0, (max_input_len - sample["input_ids"].shape[0])
                )
                cur_input_id = F.pad(sample["input_ids"], pad_shape, value=vocab.pad)
                input_ids.append(cur_input_id)
                log_probs.append(sample["logprob"])
                last_token_index.append(sample["input_ids"].shape[0] - 1)

                attention_bias = sample.get("attention_bias")
                if generate_TG_bias is not None:
                    if attention_bias is None:
                        attention_bias, _ = generate_TG_bias(cur_input_id)
                    if not isinstance(attention_bias, torch.Tensor):
                        attention_bias = torch.tensor(attention_bias)
                    # Reshape to `(1, seq_len, seq_len)`
                    while len(attention_bias.shape) < 3:
                        attention_bias = attention_bias.unsqueeze(0)
                    all_attention_bias.append(attention_bias)

                if sample["number_of_consecutive_start_NT"] >= pc or sample["number_of_start_NT"] >= nc:
                    Stop_Add_NT.append(bid)

            batch = {
                "input_ids": torch.stack(input_ids).to(self.device),
                "last_token_index": torch.tensor(last_token_index, device=self.device),
                "log_probs": torch.tensor(log_probs).unsqueeze(1).to(self.device),
            }
            if all_attention_bias:
                batch["attention_bias"] = torch.stack(all_attention_bias).to(self.device)
            if len(Stop_Add_NT) > 0:
                batch["Stop_Add_NT"] = torch.LongTensor(Stop_Add_NT).to(self.device)

            return batch

        # print(f"eval input = {eval_input_ids} past input is {past_input} nc = {nc} pc = {pc}")
        start_beam = {
            "input_ids": past_input,
            "number_of_consecutive_start_NT": 0,
            "number_of_start_NT": 0,
            "logprob": 0,  # = -sum of loss
            "terminal_logprob": 0,
            "attention_bias": initial_attention_bias
        }
        kv_cache = None
        past_kv_cache = None
        kn = beam_size if eval_input_ids is not None else 10*beam_size       # sub-beam size kn
        ks = max(beam_size // 10, 1) # fast-shift ks terminal into next_beams
        NT_start = vocab.opening_non_terminals[0]
        NT_end = vocab.closing_non_terminals[1]
        start_surprisal = end_surprisal = 0
        beams = [start_beam]
        i = istart
        topk_threshold = False
        if strategy == BeamSearchType.default:
            StopConditions = lambda: j==0
        elif strategy == BeamSearchType.word_sync:
            StopConditions = lambda: len(next_beams) < kn
        elif strategy == BeamSearchType.word_sync_dfs:
            StopConditions = lambda: True
        while i<Genlength:
            next_beams = []
            j = 0
            while StopConditions():
                next_NT_beams = []
                if beams == []:
                    break
                data = collate_fn(beams)
                if kv_cache is not None:
                    past_kv_cache = [
                        ( k.expand(data["input_ids"].shape[0],-1,-1,-1),
                          v.expand(data["input_ids"].shape[0],-1,-1,-1) ) 
                        for k, v in kv_cache
                    ]
                out = self.forward(
                    input_ids=data["input_ids"],
                    attention_bias=data.get("attention_bias"),
                    past_key_values=past_kv_cache,
                    use_cache=first_step
                )
                if first_step:
                    logits, kv_cache = out.logits, out.attn_key_values
                else:
                    logits = out.logits
                log_probs = F.log_softmax(logits[torch.arange(len(beams)), data["last_token_index"], :], dim=-1) + data["log_probs"]

                # manage max_length and eos tokens
                if not first_step:
                    retain_indices = torch.nonzero(torch.bitwise_or(data["last_token_index"]>=max_length - 1 - is_TG_input, 
                                                    data["input_ids"][torch.arange(len(beams)), data["last_token_index"]]==self.config.eos_token_id))
                    if j==0 and len(retain_indices) == len(beams) and eval_input_ids is None:
                        return beams
                    log_probs[retain_indices, :] = torch.finfo(log_probs.dtype).min
                    if strategy==BeamSearchType.default or eval_input_ids is None:
                        for index in retain_indices:
                            next_beams.append(beams[index])
                flag_next_set = set()
                C = logits.shape[-1]
                # if data.get("Stop_Add_NT") is not None:
                #     log_probs[data["Stop_Add_NT"], vocab.opening_non_terminals[0] : vocab.opening_non_terminals[1]] = torch.finfo(log_probs.dtype).min
                # del data
                if eval_input_ids is None:
                    if is_pause:
                        # Pause models must NOT emit non-terminal tokens: mask the
                        # whole NT range out of the main candidate top-k BEFORE
                        # sampling (otherwise the tree-grammar decoder would insert
                        # bracket tokens the pause model never learned to produce).
                        log_probs[:, NT_start:NT_end] = torch.finfo(log_probs.dtype).min
                    topk_log_probs, topk_indices = torch.topk(log_probs.view(-1), kn, dim=-1)
                    log_probs[:, NT_start:NT_end] = torch.finfo(log_probs.dtype).min
                    topks_term_log_probs, topks_term_indices = torch.topk(log_probs.view(-1), ks, dim=-1)
                else:
                    token = eval_input_ids[i]
                    temp_log_probs = torch.cat([log_probs[:, NT_start:NT_end], log_probs[:, token].unsqueeze(1)], dim=1)
                    nonterm_size = min(kn, temp_log_probs.numel())
                    topk_log_probs, topk_indices = torch.topk(temp_log_probs.view(-1), nonterm_size, dim=-1)
                    row = NT_end - NT_start + 1
                    topk_indices = topk_indices//row * C + torch.where(topk_indices % row == (NT_end - NT_start),  token, topk_indices % row + NT_start)
                    term_size = min(ks, log_probs.shape[0])
                    topks_term_log_probs, topks_term_indices = torch.topk(log_probs[:, token], term_size, dim=-1)
                    topks_term_indices = topks_term_indices * C + token

                def add_next_beams(beam_index, token_index, log_prob):
                    if (beam_index, token_index) in flag_next_set or math.isnan(log_prob) or log_prob < -3e37:
                        return
                    if strategy == BeamSearchType.word_sync_dfs:
                        if topk_threshold and log_prob < next_beams[kn - 1]["logprob"]:
                            return

                    flag_next_set.add((beam_index, token_index))
                    beam = beams[beam_index]
                    if not is_TG_input:  # case txltree
                        input = torch.tensor([token_index], dtype=beam["input_ids"].dtype)
                    else: # case TG
                        if vocab.is_closing_non_terminal(token_index):
                            input = torch.tensor([token_index, token_index], dtype=beam["input_ids"].dtype)
                        else:
                            input = torch.tensor([token_index], dtype=beam["input_ids"].dtype)

                    next_beam = defaultdict(float)
                    if first_step:
                        next_beam["input_ids"] = input
                    else:
                        next_beam["input_ids"] = torch.cat([beam["input_ids"], input], dim=0)
                    next_beam["logprob"] = log_prob
                    # Terminal-only logprob: only accumulate logprob for terminal tokens
                    token_log_prob = log_prob - beam["logprob"]
                    if tree_eval_type == "terminal" and not vocab.is_non_terminal(token_index):
                        next_beam["terminal_logprob"] = beam["terminal_logprob"] + token_log_prob
                    else:
                        next_beam["terminal_logprob"] = beam["terminal_logprob"]
                    next_beam["number_of_consecutive_start_NT"] = beam["number_of_consecutive_start_NT"]
                    next_beam["number_of_start_NT"] = beam["number_of_start_NT"]
                    if vocab.is_opening_non_terminal(token_index):
                        next_beam["number_of_consecutive_start_NT"] += 1
                        next_beam["number_of_start_NT"] += 1
                    elif vocab.is_closing_non_terminal(token_index):
                        next_beam["number_of_consecutive_start_NT"] = 0
                    else:
                        next_beam["number_of_consecutive_start_NT"] = 0

                    if strategy == BeamSearchType.default:
                        next_beams.append(next_beam)
                    elif strategy == BeamSearchType.word_sync or strategy == BeamSearchType.word_sync_dfs:
                        if vocab.is_non_terminal(token_index):
                            next_NT_beams.append(next_beam)
                        else:
                            next_beams.append(next_beam)
                    return

                # prepare fast shift
                if strategy==BeamSearchType.word_sync or tag_start is not None:
                    topks_term_log_probs = topks_term_log_probs.tolist()
                    topks_term_indices = topks_term_indices.tolist()
                    for k in range(len(topks_term_indices)):
                        top_index = topks_term_indices[k]
                        beam_index = top_index // C
                        token_index = top_index % C
                        add_next_beams(beam_index, token_index, topks_term_log_probs[k])
                # prepare all next candidate
                topk_log_probs = topk_log_probs.tolist()
                topk_indices = topk_indices.tolist()
                for k in range(len(topk_indices)):
                    top_index = topk_indices[k]
                    beam_index = top_index // C
                    token_index = top_index % C
                    add_next_beams(beam_index, token_index, topk_log_probs[k])
                beams = next_NT_beams
                j = j + 1
                first_step = False
                if strategy == BeamSearchType.word_sync_dfs:
                    next_beams.sort(key=lambda x: x["logprob"], reverse=True)
                    next_beams = next_beams[:kn]
                    topk_threshold = len(next_beams)>=kn
                    
            
            next_beams.sort(key=lambda x: x["logprob"], reverse=True)
            beams = next_beams[:beam_size]
            # print("i={i}")
            # for kkk in range(len(beams)):
            #     print(f"{tmptokenizer.decode(beams[kkk]['input_ids'].tolist(), skip_special_tokens=False)}")
            topk_threshold = False
            if i==tag_start or i==tag_end:
                logprob_key = "terminal_logprob" if tree_eval_type == "terminal" else "logprob"
                logprob = [beam[logprob_key] for beam in beams]
                logprob = torch.tensor(logprob, device=self.device)
                if tree_eval_type == "terminal":
                    surprisal = -logprob.max().item()
                else:
                    surprisal = -torch.logsumexp(logprob, dim=0).item()
                if beam_dump is not None:
                    # Snapshot the beam at this scoring checkpoint so callers can
                    # inspect the surviving candidates (input_ids + scores) without
                    # rerunning the search. Store CPU ints/floats for easy pickling.
                    snap = []
                    for beam in beams:
                        snap.append({
                            "input_ids": beam["input_ids"].detach().to("cpu").tolist(),
                            "logprob": float(beam["logprob"]),
                            "terminal_logprob": float(beam["terminal_logprob"]),
                            "n_start_NT": int(beam["number_of_start_NT"]),
                        })
                    beam_dump["start" if i == tag_start else "end"] = snap
                if i==tag_start:
                    start_surprisal = surprisal
                elif i==tag_end:
                    end_surprisal = surprisal
                    return end_surprisal - start_surprisal
            i = i + 1

        return beams
    def _pushdown_beam_search_legacy(
        self,
        eval_input_ids: torch.Tensor,
        beam_size: int = 20,
        max_reduce: int = 4,
        bos_id: Optional[int] = None,
        tag: Optional[List[int]] = None,
        use_attachment_head: bool = False,
        return_spans: bool = False,
    ) -> float:
        # Kept as a private compatibility alias for out-of-tree callers.
        return self.pushdown_beam_search(
            eval_input_ids=eval_input_ids,
            beam_size=beam_size,
            max_reduce=max_reduce,
            bos_id=bos_id,
            tag=tag,
            use_attachment_head=use_attachment_head,
            return_spans=return_spans,
        )
        """Shift-reduce beam search that tracks span/stack state for pushdown models.

        Marginalizes over incremental parses y of the given terminal sequence x by
        maintaining, per beam hypothesis, the closed constituent spans (which become
        the pushdown depth-bias ``tree_spans`` input) and an open-constituent stack.
        Each step: enumerate 0..max_reduce REDUCE actions (each closes spans, changing
        the depth bias but not the token sequence), then SHIFT the next eval token,
        scoring with the model's depth-biased forward.

        This lets pushdown run its trained ``_pushdown_attention`` depth-bias path at
        inference (which ``tree_spans=None`` degenerates). When ``use_attachment_head``
        is False (default), reduce sequences use a uniform prior (score 0) and only
        SHIFT log-probs score. When True, the trained attachment head (Murty et al. 2023,
        Eq. 5) scores each candidate's reduce target ``r_k`` via ``log p(r_k | x_<k)``,
        added to the SHIFT log-prob (faithful to Eq. 7's joint ``p(x,y)=prod p(x_k)p(r_k)``).

        Args:
            eval_input_ids: 1-D long tensor of terminal token ids (the sequence x).
            beam_size: number of hypotheses retained after each step's prune.
            max_reduce: max consecutive reduces enumerated per hypothesis per step.
            bos_id: BOS token id; if None, uses ``self.config.eos_token_id`` fallback
                (caller should pass the tokenizer's BOS).
            tag: optional 0/1 list (len == len(eval_input_ids)) marking positions whose
                per-token CE should be summed (SG scoring). If None, returns full
                sequence surprisal -log p(x).

        Returns:
            If ``tag`` is None: surprisal = -logsumexp_y logprob_y (marginalized -log p(x)).
            If ``tag`` given: sum of per-token CE at tagged positions for the best beam
            (matches SG's ``sum(per_tok_ce * tag_tensor)`` scoring).
            If ``return_spans`` is True: returns a ``(score, spans)`` tuple where
            ``spans`` is the best beam's closed-constituent list as an ``(M, 3)`` long
            tensor (``[l, split, r]``; -1-padded), for callers (e.g. the boolq ICL path)
            that need the inferred parse to drive a depth-biased teacher-forced forward.
        """
        # closures are stored as (l, r) pairs (the depth tape ignores the split —
        # compute_depth_matrix_gpu only reads spans[:, 0] and spans[:, 2]). We keep a
        # consistent split = r for pad compatibility, but it is never consumed.
        from olmo.pushdown import compute_depth_matrix_gpu  # noqa: F401 (kept for parity)

        device = self.device
        if eval_input_ids.dim() > 1:
            eval_input_ids = eval_input_ids[0]
        tokens = eval_input_ids.tolist()
        n = len(tokens)

        # Initial beam: BOS only, empty closed spans, empty open stack.
        start_tok = bos_id if bos_id is not None else tokens[0]
        beams = [{
            "input_ids": [start_tok],   # terminals emitted so far (prefix)
            "closed": [],               # list of (l, r) closed constituent spans
            "stack": [],                # list of (l, r) open constituents (r = current right-end)
            "logprob": 0.0,
            "per_tok_lp": [],           # shift log-prob per emitted terminal (for tag scoring)
        }]

        def enumerate_reduce_seqs(stack_len: int, max_r: int) -> List[int]:
            # Number of consecutive reduces to apply: 0..min(max_r, stack_len).
            # Each reduce pops one open constituent and closes a span. Reducing more
            # than the stack allows is invalid.
            upper = min(max_r, stack_len)
            return list(range(upper + 1))

        def apply_reduces(beam: dict, n_reduces: int) -> Tuple[dict, int]:
            # Close n_reduces spans from the open stack. Each reduce pops the top open
            # constituent (l, r) and closes span (l, last_pos) where last_pos is the
            # current last token position. This forms a right-branching close chain:
            # the innermost open constituent closes first, the outermost last (largest
            # r), approximating the BBC binarized-tree depth profile as closely as the
            # shift-reduce action space allows. (The depth tape only uses (l, r); the
            # split is irrelevant — see compute_depth_matrix_gpu.)
            #
            # Returns (new_state, r_k) where r_k is the attachment head's reduce target
            # for the just-shifted token: the right-end of the OUTERMOST popped
            # constituent (the one x_{last_pos} reduces onto). For n_reduces==0
            # (shift-only), r_k = last_pos (self-attachment, Eq. 5 j==k branch).
            last_pos = len(beam["input_ids"]) - 1  # position of the last emitted token
            if n_reduces == 0:
                return beam, last_pos  # shift-only: r_k = k (self-score)
            closed = list(beam["closed"])
            stack = list(beam["stack"])
            r_k = last_pos  # default (shift-only fallback)
            for i in range(n_reduces):
                if not stack:
                    break
                l, r = stack.pop()
                closed.append((l, last_pos))  # (l, r); split unused by depth tape
                # The outermost popped constituent (last iteration) is the reduce
                # target; its right-end r is r_k. (Inner pops are sub-reduces in the
                # right-branching chain; the head scores only the outermost attach.)
                if i == n_reduces - 1:
                    r_k = r
            return ({"input_ids": beam["input_ids"], "closed": closed, "stack": stack,
                     "logprob": beam["logprob"], "per_tok_lp": beam["per_tok_lp"]}, r_k)

        # Process each eval token (SHIFT). The BOS is already in input_ids; shift
        # tokens[0..n-1]. Position k = len(input_ids)-1 after each shift.
        #
        # EFFICIENCY: batch ALL (beam x reduce-prefix) hypotheses into ONE forward per
        # SHIFT step. The model already supports batched tree_spans (B, M, 3) — see
        # olmo/data/collator.py:132 — so we collate the per-hypothesis closed spans into
        # a single (N, M, 3) -1-padded tensor and run self.forward once with
        # last_logits_only=True. This replaces the old ~100 serial batch-1 forwards per
        # token (beam_size * (max_reduce+1)) with a single batched forward.
        for t in tokens:
            # 1. Expand every beam by its reduce-prefixes -> candidate states.
            cand_states = []  # list of state dicts
            cand_rks = []     # attachment reduce target r_k per candidate (for head scoring)
            for beam in beams:
                for n_red in enumerate_reduce_seqs(len(beam["stack"]), max_reduce):
                    state, rk = apply_reduces(beam, n_red) if n_red > 0 else (beam, len(beam["input_ids"]) - 1)
                    cand_states.append(state)
                    cand_rks.append(rk)
            if not cand_states:
                break

            # 2. Collate into one batched forward. All states share the same input_ids
            # length K = len(cand_states[0]["input_ids"]) (every beam in `beams` is at
            # the same prefix length; reduce-only does not grow input_ids).
            K = len(cand_states[0]["input_ids"])
            inp_ids = torch.tensor(
                [s["input_ids"] for s in cand_states], dtype=torch.long, device=device
            )  # (N, K)
            attn_mask = torch.ones(inp_ids.shape[0], K, dtype=torch.bool, device=device)
            # Collate tree_spans: (N, M, 3) -1-padded, M = max closed count across batch.
            max_closed = max(len(s["closed"]) for s in cand_states)
            max_closed = max(max_closed, 1)
            ts = torch.full((inp_ids.shape[0], max_closed, 3), -1,
                            dtype=torch.long, device=device)
            for i, s in enumerate(cand_states):
                for j, (l, r) in enumerate(s["closed"]):
                    ts[i, j, 0] = l
                    ts[i, j, 1] = r   # split (unused by depth tape; set = r for safety)
                    ts[i, j, 2] = r

            # 3. One batched forward. last_logits_only -> (N, 1, vocab). When the
            # attachment head is enabled, also request attachment_logits (N, K, K) to
            # score each candidate's reduce target r_k (Murty et al. Eq. 7: add
            # log p(r_k | x_<k) to the SHIFT log-prob).
            with torch.no_grad():
                out = self.forward(
                    input_ids=inp_ids, attention_mask=attn_mask,
                    tree_spans=ts, last_logits_only=True,
                    compute_attachment_logits=use_attachment_head,
                )
            # log_softmax over vocab, gather the eval-token log-prob per candidate.
            log_probs = torch.log_softmax(out.logits[:, 0, :].float(), dim=-1)  # (N, vocab)
            tok_lps = log_probs[:, t]  # (N,)

            # Attachment-head structural prior (Murty et al. Eq. 7): add log p(r_k)
            # to each candidate's SHIFT log-prob. r_k is the candidate's reduce target
            # (the right-end of the outermost reduced constituent, or last_pos for
            # shift-only). The head's logits[b, q, j] score p(r_q = j); query q is the
            # last prefix position (last_pos = K-1), key j is r_k. Clamp r_k to [0, K-1].
            if use_attachment_head and out.attachment_logits is not None:
                att = out.attachment_logits                   # (N, K, K) fp32
                q_idx = K - 1                                  # the just-shifted query position
                rk = torch.tensor(cand_rks, dtype=torch.long, device=att.device)
                rk = rk.clamp(0, K - 1)                        # defensive
                att_logp = torch.log_softmax(att[:, q_idx, :].float(), dim=-1)  # (N, K)
                attach_lps = att_logp.gather(1, rk.unsqueeze(1)).squeeze(1)    # (N,)
                attach_lps = attach_lps.tolist()
            else:
                attach_lps = [0.0] * len(cand_states)

            # 4. Build next beams: append t, open a new constituent at position K.
            next_beams = []
            for state, shift_lp, attach_lp in zip(cand_states, tok_lps.tolist(), attach_lps):
                if math.isnan(shift_lp) or shift_lp < -3e37:
                    continue
                next_beams.append({
                    "input_ids": state["input_ids"] + [t],
                    "closed": state["closed"],
                    "stack": state["stack"] + [(K, K)],  # new constituent opens at K (l, r)
                    "logprob": state["logprob"] + shift_lp + attach_lp,
                    "per_tok_lp": state["per_tok_lp"] + [shift_lp],
                })
            if not next_beams:
                break
            # Prune to beam_size by logprob (keep best).
            next_beams.sort(key=lambda b: b["logprob"], reverse=True)
            beams = next_beams[:beam_size]

        if not beams:
            if return_spans:
                return float("inf"), torch.zeros((0, 3), dtype=torch.long)
            return float("inf")
        if tag is None:
            # Marginalize over parses (Murty et al.): p(x) = sum_y p(x,y) =>
            # log p(x) = logsumexp_y logprob_y. Surprisal = -log p(x).
            logprobs = torch.tensor([b["logprob"] for b in beams], dtype=torch.float64)
            score = -torch.logsumexp(logprobs, dim=0).item()
        else:
            # Tag scoring (SG): best beam's sum of per-token CE at tagged positions.
            # per_tok_lp aligns with the shifted terminals (tokens[0..n-1]); tag marks
            # which of those positions to sum (1 = include).
            best = max(beams, key=lambda b: b["logprob"])
            per_tok_ce = [-lp for lp in best["per_tok_lp"]]  # CE = -log p
            tag = list(tag)
            # Clamp length; tag may be one shorter (ce_loss is one shorter than input).
            m = min(len(per_tok_ce), len(tag))
            score = sum(per_tok_ce[i] for i in range(m) if tag[i])
        if not return_spans:
            return score
        # Best beam's closed spans as (M, 3) [l, split, r] long tensor (-1-padded).
        best = max(beams, key=lambda b: b["logprob"])
        closed = best["closed"]
        M = max(len(closed), 0)
        spans = torch.full((M, 3), -1, dtype=torch.long)
        for i, (l, r) in enumerate(closed):
            spans[i, 0] = l
            spans[i, 1] = r   # split (unused by depth tape; set = r for safety)
            spans[i, 2] = r
        return score, spans

    def pushdown_beam_search(
        self,
        eval_input_ids: torch.Tensor,
        beam_size: int = 20,
        max_reduce: Optional[int] = None,
        bos_id: Optional[int] = None,
        tag: Optional[List[int]] = None,
        use_attachment_head: bool = False,
        return_spans: bool = False,
        attachment_normalization: str = "stack_legal",
        sentence_local_stack: bool = False,
        require_complete_parse: bool = False,
    ):
        """Approximate ``p(x)`` with a normalized attachment-action beam.

        A beam is always a parse state *after* attaching its last token. At the
        next step the model first predicts the next word from that state, then
        predicts/normalizes that word's attachment decision, and finally updates
        the stack. This is the order in Eq. 7 of Murty et al. and avoids the
        former one-token attachment lag.

        ``attachment_normalization="stack_legal"`` (v1) renormalizes the head
        over the actions reachable from the current stack.
        ``"sentence_causal"`` (v2) first normalizes the complete causal row and
        then retains only legal expansions without renormalizing them.

        When no trained attachment head is requested, valid attachment actions
        receive a normalized uniform prior. They are never free zero-score
        branches, so marginal likelihood cannot grow merely because more parses
        were enumerated. Tagged SG scores use incremental, parse-marginalized
        word surprisal at each prefix rather than the word scores of one final
        best parse.

        ``sentence_local_stack`` is an opt-in evaluation mode for comparing beam
        support with supplied sentence parses. BOS remains LM context but is not
        a structural stack item or attachment target. With
        ``require_complete_parse=True``, final incomplete stacks are discarded
        before the last beam prune. Existing downstream callers keep the legacy
        root-containing behavior by default.
        """
        from olmo.attachment import (
            attachment_log_probs_for_targets,
            canonical_attachment_normalization,
        )

        attachment_normalization = canonical_attachment_normalization(
            attachment_normalization
        )
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if require_complete_parse and not sentence_local_stack:
            raise ValueError(
                "require_complete_parse currently requires sentence_local_stack"
            )
        device = self.device
        if eval_input_ids.dim() > 1:
            eval_input_ids = eval_input_ids[0]
        observed = [int(t) for t in eval_input_ids.tolist()]
        if not observed:
            empty = torch.zeros((0, 3), dtype=torch.long)
            return (0.0, empty) if return_spans else 0.0

        # Do not score an already-present BOS twice. If no explicit BOS is
        # present, use the requested BOS as context and score every observed
        # token. With bos_id=None the first observed token is the context token,
        # which is the correct convention for tokenizers trained without BOS.
        inserted_bos = bos_id is not None and observed[0] != int(bos_id)
        if inserted_bos:
            seed = int(bos_id)
            targets = observed
            target_positions = list(range(len(observed)))
        else:
            seed = observed[0]
            targets = observed[1:]
            target_positions = list(range(1, len(observed)))

        beams: List[dict] = [{
            "input_ids": [seed],
            "closed": [],
            "stack": [] if sentence_local_stack else [(0, 0)],
            "logprob": 0.0,
        }]
        incremental_word_lps: List[Tuple[int, float]] = []

        def collate_spans(states: List[dict], seq_len: int) -> torch.Tensor:
            max_closed = max(max((len(s["closed"]) for s in states), default=0), 1)
            spans = torch.full(
                (len(states), max_closed, 3), -1, dtype=torch.long, device=device
            )
            for bi, state in enumerate(states):
                for si, (left, right) in enumerate(state["closed"]):
                    if 0 <= left <= right < seq_len:
                        spans[bi, si] = torch.tensor(
                            [left, right, right], dtype=torch.long, device=device
                        )
            return spans

        def action_counts(state: dict, token: int, is_last: bool) -> List[int]:
            old_stack_len = len(state["stack"])
            # The sentence-final EOS attaches to the oldest stack item, closing
            # the root, as in the reference beam search. A single attachment
            # decision can collapse an arbitrary stack suffix.
            if (
                not sentence_local_stack
                and is_last
                and token == self.config.eos_token_id
            ):
                return [old_stack_len]
            # The default decoder reserves its BOS/root item until final EOS.
            # Sentence-local comparison mode has no structural root item, so
            # every existing stack constituent is a possible reduce target.
            reserved = 0 if sentence_local_stack else 1
            upper = max(old_stack_len - reserved, 0)
            if max_reduce is not None:
                upper = min(upper, max(max_reduce, 0))
            return list(range(upper + 1))

        def attach_shifted(state: dict, n_reduces: int) -> Tuple[dict, int]:
            stack = list(state["stack"])
            closed = list(state["closed"])
            current = stack.pop()  # the newly shifted singleton
            reduce_target = current[1]  # self = shift only
            for _ in range(n_reduces):
                if not stack:
                    break
                left_constituent = stack.pop()
                reduce_target = left_constituent[1]
                current = (left_constituent[0], current[1])
                closed.append(current)
            stack.append(current)
            return {
                "input_ids": state["input_ids"],
                "closed": closed,
                "stack": stack,
                "logprob": state["logprob"],
            }, reduce_target

        for step, (token, original_pos) in enumerate(zip(targets, target_positions)):
            if not beams:
                break
            prefix_len = len(beams[0]["input_ids"])
            inp = torch.tensor(
                [b["input_ids"] for b in beams], dtype=torch.long, device=device
            )
            mask = torch.ones_like(inp, dtype=torch.bool)
            spans = collate_spans(beams, prefix_len)
            with torch.no_grad():
                word_out = self.forward(
                    input_ids=inp,
                    attention_mask=mask,
                    tree_spans=spans,
                    last_logits_only=True,
                )
            word_lps = torch.log_softmax(
                word_out.logits[:, 0, :].float(), dim=-1
            )[:, token]

            previous_mass = torch.logsumexp(
                torch.tensor(
                    [b["logprob"] for b in beams],
                    dtype=torch.float64,
                    device=device,
                ),
                dim=0,
            )
            word_mass = torch.logsumexp(
                torch.tensor(
                    [b["logprob"] for b in beams],
                    dtype=torch.float64,
                    device=device,
                )
                + word_lps.to(torch.float64),
                dim=0,
            )
            incremental_word_lps.append(
                (original_pos, float((word_mass - previous_mass).item()))
            )

            shifted: List[dict] = []
            choices: List[List[int]] = []
            for beam, word_lp in zip(beams, word_lps.tolist()):
                if math.isnan(word_lp) or word_lp < -3e37:
                    continue
                shifted.append({
                    "input_ids": beam["input_ids"] + [token],
                    "closed": list(beam["closed"]),
                    "stack": list(beam["stack"]) + [(prefix_len, prefix_len)],
                    "logprob": beam["logprob"] + word_lp,
                })
                choices.append(action_counts(beam, token, step == len(targets) - 1))
            if not shifted:
                beams = []
                break

            # Score r_k only after x_k has been supplied to the attachment MLP.
            # The prior stack tape is retained until the decision is applied.
            attachment_rows: Optional[torch.Tensor] = None
            if use_attachment_head:
                shifted_inp = torch.tensor(
                    [s["input_ids"] for s in shifted], dtype=torch.long, device=device
                )
                shifted_mask = torch.ones_like(shifted_inp, dtype=torch.bool)
                shifted_spans = collate_spans(shifted, prefix_len + 1)
                shifted_sentence_ids = None
                if sentence_local_stack:
                    shifted_sentence_ids = torch.zeros_like(shifted_inp)
                    # Position zero is the LM-only BOS context. This matches
                    # terminal-only training, where document BOS has no tree ID.
                    shifted_sentence_ids[:, 0] = -1
                with torch.no_grad():
                    attachment_out = self.forward(
                        input_ids=shifted_inp,
                        attention_mask=shifted_mask,
                        tree_spans=shifted_spans,
                        pushdown_sentence_ids=shifted_sentence_ids,
                        last_logits_only=True,
                        compute_attachment_logits=True,
                    )
                if attachment_out.attachment_logits is not None:
                    attachment_rows = attachment_out.attachment_logits[:, prefix_len, :]

            next_beams: List[dict] = []
            for bi, (state, counts) in enumerate(zip(shifted, choices)):
                attached: List[Tuple[dict, int]] = [
                    attach_shifted(state, count) for count in counts
                ]
                targets_j = [target_j for _, target_j in attached]
                if attachment_rows is None:
                    structural_lps = [-math.log(len(attached))] * len(attached)
                else:
                    target_tensor = torch.tensor(
                        targets_j, dtype=torch.long, device=device
                    )
                    structural_lps = attachment_log_probs_for_targets(
                        attachment_rows[bi],
                        target_tensor,
                        attachment_normalization,
                    ).tolist()
                for (new_state, _), structural_lp in zip(attached, structural_lps):
                    new_state["logprob"] += structural_lp
                    next_beams.append(new_state)

            if sentence_local_stack and require_complete_parse and step == len(targets) - 1:
                # BOS is position zero and is outside the structural stack. A
                # complete non-empty sentence therefore has one frontier item
                # spanning positions 1..prefix_len after the final shift.
                next_beams = [
                    state
                    for state in next_beams
                    if len(state["stack"]) == 1
                    and state["stack"][0] == (1, prefix_len)
                ]
            next_beams.sort(key=lambda state: state["logprob"], reverse=True)
            beams = next_beams[:beam_size]

        if not beams:
            empty = torch.zeros((0, 3), dtype=torch.long)
            return (float("inf"), empty) if return_spans else float("inf")

        if tag is None:
            final_logprobs = torch.tensor(
                [b["logprob"] for b in beams], dtype=torch.float64
            )
            score = float(-torch.logsumexp(final_logprobs, dim=0).item())
        else:
            tag_values = list(tag)
            score = 0.0
            for original_pos, logp in incremental_word_lps:
                if original_pos < len(tag_values) and tag_values[original_pos]:
                    score -= logp

        if not return_spans:
            return score
        best = max(beams, key=lambda state: state["logprob"])
        closed = list(best["closed"])
        # If BOS was inserted outside eval_input_ids, translate spans back into
        # the caller's coordinates and discard root spans containing that BOS.
        if inserted_bos:
            closed = [(l - 1, r - 1) for l, r in closed if l > 0]
        spans_out = torch.full((len(closed), 3), -1, dtype=torch.long)
        for i, (left, right) in enumerate(closed):
            spans_out[i] = torch.tensor([left, right, right], dtype=torch.long)
        return score, spans_out

    def _pushdown_generate_legacy(
        self,
        input_ids: torch.LongTensor,
        max_steps: int = 10,
        beam_size: int = 6,
        max_reduce: int = 4,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> "OLMoGenerateOutput":
        # Kept as a private compatibility alias for out-of-tree callers.
        return self.pushdown_generate(
            input_ids=input_ids,
            max_steps=max_steps,
            beam_size=beam_size,
            max_reduce=max_reduce,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        """Shift-reduce beam-search GENERATION for pushdown models.

        Mirrors :meth:`pushdown_beam_search`, but instead of force-shifting a given
        eval token, each step samples the model's top-k next tokens. This keeps the
        trained ``_pushdown_attention`` depth-bias path active during open-ended
        generation (summarization), where the plain ``generate()`` path passes
        ``tree_spans=None`` and the depth bias vanishes -> degenerate
        "It is , and it is ..." output.

        The prompt's parse is built incrementally: each generated token opens a
        constituent (push) and reduces close constituents (closing spans), so the
        depth bias tracks the model's own generated tree structure. The prompt
        region itself contributes no spans (its tokens are not re-scored), which is
        correct — only generated-token attention is biased.

        Args:
            input_ids: ``(B, L)`` or ``(L,)`` prompt. Trailing pad is stripped.
            max_steps: max generated tokens (excludes prompt).
            beam_size: beams retained per step.
            max_reduce: max consecutive reduces per hypothesis per step.
            eos_token_id: stop token (defaults to ``config.eos_token_id``).
            pad_token_id: pad for output padding (defaults to ``config.pad_token_id``).

        Returns:
            ``OLMoGenerateOutput`` with ``token_ids`` ``(B, beam_size, max_steps)``
            (generated tokens only, pad-padded) and ``scores`` ``(B, beam_size)``,
            matching :meth:`generate`'s interface so callers are uniform.
        """
        device = self.device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B = input_ids.shape[0]
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        pad = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        prompt_lists = [input_ids[b].tolist() for b in range(B)]

        def apply_reduces(beam: dict, n_reduces: int) -> Tuple[dict, int]:
            last_pos = len(beam["input_ids"]) - 1
            if n_reduces == 0:
                return beam, last_pos
            closed = list(beam["closed"])
            stack = list(beam["stack"])
            r_k = last_pos
            for i in range(n_reduces):
                if not stack:
                    break
                l, r = stack.pop()
                closed.append((l, last_pos))
                if i == n_reduces - 1:
                    r_k = r
            return ({"input_ids": beam["input_ids"], "closed": closed, "stack": stack,
                     "logprob": beam["logprob"], "generated": beam["generated"],
                     "done": beam["done"]}, r_k)

        all_token_ids, all_scores = [], []
        for b in range(B):
            prompt = list(prompt_lists[b])
            while prompt and prompt[-1] == pad:
                prompt.pop()
            # Seed beam: prompt tokens, empty parse. Constituents open over
            # generated positions as the model emits tokens.
            beams = [{"input_ids": list(prompt), "closed": [], "stack": [],
                      "logprob": 0.0, "generated": [], "done": False}]
            finished: List[dict] = []
            for _step in range(max_steps):
                cand_states = []
                for beam in beams:
                    if beam["done"]:
                        continue
                    upper = min(max_reduce, len(beam["stack"]))
                    for n_red in range(upper + 1):
                        state, _rk = apply_reduces(beam, n_red)
                        cand_states.append(state)
                if not cand_states:
                    break
                K = len(cand_states[0]["input_ids"])
                inp_ids = torch.tensor(
                    [s["input_ids"] for s in cand_states], dtype=torch.long, device=device)
                attn_mask = torch.ones(inp_ids.shape, dtype=torch.bool, device=device)
                max_closed = max(max(len(s["closed"]) for s in cand_states), 1)
                ts = torch.full((inp_ids.shape[0], max_closed, 3), -1,
                                dtype=torch.long, device=device)
                for i, s in enumerate(cand_states):
                    for j, (l, r) in enumerate(s["closed"]):
                        ts[i, j, 0] = l
                        ts[i, j, 1] = r
                        ts[i, j, 2] = r
                with torch.no_grad():
                    out = self.forward(input_ids=inp_ids, attention_mask=attn_mask,
                                       tree_spans=ts, last_logits_only=True)
                log_probs = torch.log_softmax(out.logits[:, 0, :].float(), dim=-1)  # (N, vocab)
                # For each candidate, expand its top-`beam_size` next tokens.
                next_beams: List[dict] = []
                for ci, state in enumerate(cand_states):
                    topk_lp, topk_tok = torch.topk(log_probs[ci], beam_size)
                    for tl, tk in zip(topk_lp.tolist(), topk_tok.tolist()):
                        if math.isnan(tl) or tl < -3e37:
                            continue
                        tok = int(tk)
                        next_beams.append({
                            "input_ids": state["input_ids"] + [tok],
                            "closed": state["closed"],
                            "stack": state["stack"] + [(K, K)],
                            "logprob": state["logprob"] + tl,
                            "generated": state["generated"] + [tok],
                            "done": tok == eos,
                        })
                still = [nb for nb in next_beams if not nb["done"]]
                finished.extend(nb for nb in next_beams if nb["done"])
                still.sort(key=lambda x: x["logprob"], reverse=True)
                beams = still[:beam_size]
                if not beams:
                    break
            all_beams = finished + beams
            all_beams.sort(key=lambda x: x["logprob"], reverse=True)
            top = all_beams[:beam_size]
            gen_lists, sc = [], []
            for b_ in top:
                g = b_["generated"]
                if len(g) < max_steps:
                    g = g + [pad] * (max_steps - len(g))
                else:
                    g = g[:max_steps]
                gen_lists.append(g)
                sc.append(b_["logprob"])
            while len(gen_lists) < beam_size:
                gen_lists.append([pad] * max_steps)
                sc.append(-float("inf"))
            all_token_ids.append(gen_lists)
            all_scores.append(sc)

        token_ids = torch.tensor(all_token_ids, dtype=torch.long, device=device)  # (B, beam, max_steps)
        scores = torch.tensor(all_scores, dtype=torch.float, device=device)      # (B, beam)
        return OLMoGenerateOutput(token_ids=token_ids, scores=scores)

    def pushdown_generate(
        self,
        input_ids: torch.LongTensor,
        max_steps: int = 10,
        beam_size: int = 6,
        max_reduce: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        use_attachment_head: bool = False,
        prompt_spans: Optional[torch.Tensor] = None,
    ) -> "OLMoGenerateOutput":
        """Generate words and normalized Pushdown attachments jointly.

        When ``prompt_spans`` is supplied, it is the gold terminal-coordinate
        parse of the prompt.  It directly initializes the closed spans, stack,
        and depth tape, avoiding an extremely expensive latent beam parse of a
        long source article.  Otherwise the prompt parse is inferred with
        :meth:`pushdown_beam_search` for backwards compatibility. At every
        *generated* decode step, word probabilities and attachment actions are
        jointly beam searched.
        """
        device = self.device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        pad = self.config.pad_token_id if pad_token_id is None else pad_token_id

        def collate_spans(states: List[dict]) -> torch.Tensor:
            max_closed = max(max((len(s["closed"]) for s in states), default=0), 1)
            # Build once on the host and transfer once. The old implementation
            # performed one CUDA indexed assignment per span (thousands of tiny
            # kernel launches at beam=6 and a long parsed XSum prompt).
            rows: List[List[List[int]]] = []
            for state in states:
                seq_len = len(state["input_ids"])
                valid = [
                    [left, right, right]
                    for left, right in state["closed"]
                    if 0 <= left <= right < seq_len
                ]
                valid.extend([[-1, -1, -1]] * (max_closed - len(valid)))
                rows.append(valid)
            return torch.tensor(rows, dtype=torch.long, device=device)

        def apply_action(state: dict, n_reduces: int) -> Tuple[dict, int]:
            stack = list(state["stack"])
            closed = list(state["closed"])
            current = stack.pop()
            reduce_target = current[1]
            for _ in range(n_reduces):
                if not stack:
                    break
                left_constituent = stack.pop()
                reduce_target = left_constituent[1]
                current = (left_constituent[0], current[1])
                closed.append(current)
            stack.append(current)
            out = dict(state)
            out["closed"] = closed
            out["stack"] = stack
            return out, reduce_target

        all_tokens: List[List[List[int]]] = []
        all_scores: List[List[float]] = []
        for batch_idx in range(input_ids.shape[0]):
            prompt = [int(t) for t in input_ids[batch_idx].tolist()]
            left_pad = 0
            while prompt and prompt[0] == pad:
                prompt.pop(0)
                left_pad += 1
            while prompt and prompt[-1] == pad:
                prompt.pop()
            if not prompt:
                prompt = [int(eos)]

            if prompt_spans is not None:
                # DataCollator pads unused span rows with -1 and shifts valid
                # coordinates when it left-pads a batch.  Keep only spans fully
                # contained in this prompt and translate any leading padding.
                gold_spans = prompt_spans[batch_idx]
                closed = [
                    (int(row[0]) - left_pad, int(row[2]) - left_pad)
                    for row in gold_spans.tolist()
                    if (
                        int(row[0]) >= left_pad
                        and left_pad <= int(row[1]) <= int(row[2])
                        and int(row[2]) - left_pad < len(prompt)
                        and int(row[0]) < int(row[2])
                    )
                ]
            else:
                # Backwards-compatible latent prompt parsing for callers that
                # do not have a gold parse. XSum supplies ``prompt_spans`` and
                # therefore never takes this O(L) full-forward beam path.
                _, inferred_spans = self.pushdown_beam_search(
                    torch.tensor(prompt, dtype=torch.long, device=device),
                    beam_size=beam_size,
                    max_reduce=max_reduce,
                    bos_id=prompt[0],
                    tag=None,
                    use_attachment_head=use_attachment_head,
                    return_spans=True,
                )
                closed = [
                    (int(row[0]), int(row[2])) for row in inferred_spans.tolist()
                ]

            # Fast XSum path: prefill the gold prompt once, then carry per-layer
            # KV caches and final-layer residual states through the joint beam.
            # The fallback below is retained for lightweight test doubles and
            # callers that must infer the prompt parse.
            if (
                prompt_spans is not None
                and hasattr(self, "transformer")
                and hasattr(self, "pushdown_attachment_head")
            ):
                initial = {
                    "input_ids": prompt,
                    "closed": closed,
                    # Prompt spans remain available to the depth-biased attention
                    # path, but they are not legal attachment targets for the new
                    # summary sentence. XSum training resets the oracle stack at
                    # every sentence id and does not inject a synthetic ROOT, so
                    # generation must start with an empty, ROOT-free sentence stack.
                    "stack": [],
                    "generated": [],
                    "logprob": 0.0,
                    "done": False,
                }
                prompt_input = torch.tensor([prompt], dtype=torch.long, device=device)
                with torch.no_grad():
                    prefill = self.forward(
                        input_ids=prompt_input,
                        attention_mask=torch.ones_like(prompt_input, dtype=torch.bool),
                        tree_spans=collate_spans([initial]),
                        use_cache=True,
                        last_logits_only=True,
                        return_final_hidden=True,
                    )
                if prefill.attn_key_values is None or prefill.final_hidden is None:
                    raise RuntimeError("Pushdown cached prefill did not return KV/hidden state")

                beams = [initial]
                beam_cache = prefill.attn_key_values
                beam_hidden = prefill.final_hidden
                beam_logits = prefill.logits[:, 0, :]
                finished = []

                for _ in range(max_steps):
                    if not beams:
                        break
                    word_logprobs = torch.log_softmax(beam_logits.float(), dim=-1)
                    top_k = min(beam_size, word_logprobs.shape[-1])
                    # Select words for every parent in one launch and synchronize
                    # once. Calling ``topk(...).tolist()`` separately for each
                    # beam used to serialize the GPU many times per decode step.
                    top_lps_tensor, top_tokens_tensor = torch.topk(
                        word_logprobs, top_k, dim=-1
                    )
                    top_lps_rows = top_lps_tensor.tolist()
                    top_token_rows = top_tokens_tensor.tolist()
                    shifted: List[dict] = []
                    choice_counts: List[List[int]] = []
                    for parent_idx, beam in enumerate(beams):
                        for word_lp, token in zip(
                            top_lps_rows[parent_idx], top_token_rows[parent_idx]
                        ):
                            token = int(token)
                            if math.isnan(word_lp) or word_lp < -3e37:
                                continue
                            position = len(beam["input_ids"])
                            shifted.append({
                                "input_ids": beam["input_ids"] + [token],
                                "closed": list(beam["closed"]),
                                "stack": list(beam["stack"]) + [(position, position)],
                                "generated": beam["generated"] + [token],
                                "logprob": beam["logprob"] + word_lp,
                                "done": token == eos,
                                "parent_idx": parent_idx,
                            })
                            if token == eos:
                                # EOS has sentence id -1 in the XSum training data
                                # and therefore no attachment target.
                                choice_counts.append([0])
                            else:
                                # ROOT-free sentence stack: every existing summary
                                # constituent is a legal reduce target. Prompt
                                # constituents never enter this stack.
                                upper = len(beam["stack"])
                                if max_reduce is not None:
                                    upper = min(upper, max(max_reduce, 0))
                                choice_counts.append(list(range(upper + 1)))

                    attachment_rows = None
                    if shifted and use_attachment_head:
                        parent_indices = torch.tensor(
                            [state["parent_idx"] for state in shifted],
                            dtype=torch.long,
                            device=device,
                        )
                        candidate_tokens = torch.tensor(
                            [state["input_ids"][-1] for state in shifted],
                            dtype=torch.long,
                            device=device,
                        )
                        with torch.no_grad():
                            attachment_rows = self.pushdown_attachment_head.score_next(
                                beam_hidden.index_select(0, parent_indices),
                                candidate_tokens,
                                self.transformer.wte.weight,
                            )

                    # Score all legal attachment actions as a padded batch. The
                    # prior loop synchronized once per shifted word and eagerly
                    # copied every Python beam state, even though only the global
                    # top ``beam_size`` unfinished states can survive.
                    max_choices = max(
                        (len(counts) for counts in choice_counts), default=0
                    )
                    if max_choices == 0:
                        beams = []
                        break
                    if attachment_rows is None:
                        action_lps_cpu = torch.full(
                            (len(shifted), max_choices),
                            -float("inf"),
                            dtype=torch.float64,
                        )
                        for shifted_idx, counts in enumerate(choice_counts):
                            action_lps_cpu[shifted_idx, : len(counts)] = -math.log(
                                len(counts)
                            )
                    else:
                        target_rows: List[List[int]] = []
                        valid_rows: List[List[bool]] = []
                        for state, counts in zip(shifted, choice_counts):
                            targets = [
                                state["stack"][-(count + 1)][1] for count in counts
                            ]
                            target_rows.append(targets + [0] * (max_choices - len(targets)))
                            valid_rows.append(
                                [True] * len(targets)
                                + [False] * (max_choices - len(targets))
                            )
                        target_tensor = torch.tensor(
                            target_rows, dtype=torch.long, device=device
                        )
                        valid_tensor = torch.tensor(
                            valid_rows, dtype=torch.bool, device=device
                        )
                        action_logits = attachment_rows.gather(1, target_tensor).float()
                        action_logits.masked_fill_(~valid_tensor, -float("inf"))
                        # One device-to-host transfer replaces one transfer per
                        # candidate word. float64 on the host preserves the old
                        # Python-float accumulation and stable ranking behavior.
                        action_lps_cpu = torch.log_softmax(action_logits, dim=1).to(
                            device="cpu", dtype=torch.float64
                        )

                    base_scores = torch.tensor(
                        [state["logprob"] for state in shifted], dtype=torch.float64
                    )
                    candidate_scores = base_scores[:, None] + action_lps_cpu

                    # EOS admits only action zero. Preserve every completed beam,
                    # just as the eager expansion did, but materialize it lazily.
                    for shifted_idx, state in enumerate(shifted):
                        if state["done"]:
                            new_state, _ = apply_action(state, 0)
                            new_state["logprob"] = float(candidate_scores[shifted_idx, 0])
                            finished.append(new_state)

                    # Padded and completed rows cannot enter the unfinished beam.
                    unfinished_scores = candidate_scores.clone()
                    for shifted_idx, (state, counts) in enumerate(
                        zip(shifted, choice_counts)
                    ):
                        if state["done"]:
                            unfinished_scores[shifted_idx].fill_(-float("inf"))
                        elif len(counts) < max_choices:
                            unfinished_scores[shifted_idx, len(counts) :] = -float("inf")

                    # Row-major flattening has the same candidate order as the
                    # old nested loops. Stable sorting therefore also preserves
                    # tie behavior while avoiding thousands of Python objects.
                    flat_scores = unfinished_scores.flatten()
                    ranked = torch.argsort(flat_scores, descending=True, stable=True)
                    beams = []
                    for flat_idx in ranked[:beam_size].tolist():
                        score = float(flat_scores[flat_idx])
                        if not math.isfinite(score):
                            break
                        shifted_idx, action_idx = divmod(flat_idx, max_choices)
                        new_state, _ = apply_action(
                            shifted[shifted_idx], choice_counts[shifted_idx][action_idx]
                        )
                        new_state["logprob"] = score
                        beams.append(new_state)
                    if not beams:
                        break

                    # Every future word and attachment log-probability is <= 0.
                    # Once the sixth-best completed hypothesis is strictly above
                    # the best unfinished upper bound, no continuation can alter
                    # the final top beam and the remaining decode steps are dead
                    # work. Use a strict comparison to preserve tie ordering.
                    if len(finished) >= beam_size:
                        finished_cutoff = sorted(
                            (state["logprob"] for state in finished), reverse=True
                        )[beam_size - 1]
                        if finished_cutoff > beams[0]["logprob"]:
                            beams = []
                            break

                    parent_indices = torch.tensor(
                        [state["parent_idx"] for state in beams],
                        dtype=torch.long,
                        device=device,
                    )
                    next_tokens = torch.tensor(
                        [[state["input_ids"][-1]] for state in beams],
                        dtype=torch.long,
                        device=device,
                    )
                    selected_cache = [
                        (
                            key.index_select(0, parent_indices),
                            value.index_select(0, parent_indices),
                        )
                        for key, value in beam_cache
                    ]
                    with torch.no_grad():
                        decoded = self.forward(
                            input_ids=next_tokens,
                            past_key_values=selected_cache,
                            use_cache=True,
                            tree_spans=collate_spans(beams),
                            last_logits_only=True,
                            return_final_hidden=True,
                        )
                    if decoded.attn_key_values is None or decoded.final_hidden is None:
                        raise RuntimeError("Pushdown cached decode did not return KV/hidden state")
                    beam_hidden = torch.cat(
                        [beam_hidden.index_select(0, parent_indices), decoded.final_hidden],
                        dim=1,
                    )
                    beam_cache = decoded.attn_key_values
                    beam_logits = decoded.logits[:, 0, :]

                candidates = finished + beams
                candidates.sort(key=lambda state: state["logprob"], reverse=True)
                selected = candidates[:beam_size]
                token_rows = []
                score_rows = []
                for state in selected:
                    generated = state["generated"][:max_steps]
                    generated += [pad] * (max_steps - len(generated))
                    token_rows.append(generated)
                    score_rows.append(float(state["logprob"]))
                while len(token_rows) < beam_size:
                    token_rows.append([pad] * max_steps)
                    score_rows.append(-float("inf"))
                all_tokens.append(token_rows)
                all_scores.append(score_rows)
                continue

            beams: List[dict] = [{
                "input_ids": prompt,
                "closed": closed,
                # Keep prompt spans only as attention history. Attachments for the
                # generated sentence begin from an empty, ROOT-free stack.
                "stack": [],
                "generated": [],
                "logprob": 0.0,
                "done": False,
            }]
            finished: List[dict] = []

            for _ in range(max_steps):
                if not beams:
                    break
                inp = torch.tensor(
                    [state["input_ids"] for state in beams],
                    dtype=torch.long,
                    device=device,
                )
                mask = torch.ones_like(inp, dtype=torch.bool)
                with torch.no_grad():
                    out = self.forward(
                        input_ids=inp,
                        attention_mask=mask,
                        tree_spans=collate_spans(beams),
                        last_logits_only=True,
                    )
                word_logprobs = torch.log_softmax(out.logits[:, 0, :].float(), dim=-1)

                shifted: List[dict] = []
                choice_counts: List[List[int]] = []
                top_k = min(beam_size, word_logprobs.shape[-1])
                for bi, beam in enumerate(beams):
                    top_lps, top_tokens = torch.topk(word_logprobs[bi], top_k)
                    for word_lp, token in zip(top_lps.tolist(), top_tokens.tolist()):
                        token = int(token)
                        if math.isnan(word_lp) or word_lp < -3e37:
                            continue
                        position = len(beam["input_ids"])
                        state = {
                            "input_ids": beam["input_ids"] + [token],
                            "closed": list(beam["closed"]),
                            "stack": list(beam["stack"]) + [(position, position)],
                            "generated": beam["generated"] + [token],
                            "logprob": beam["logprob"] + word_lp,
                            "done": token == eos,
                        }
                        shifted.append(state)
                        if token == eos:
                            choice_counts.append([0])
                        else:
                            upper = len(beam["stack"])
                            if max_reduce is not None:
                                upper = min(upper, max(max_reduce, 0))
                            choice_counts.append(list(range(upper + 1)))

                attachment_rows: Optional[torch.Tensor] = None
                if shifted and use_attachment_head:
                    shifted_inp = torch.tensor(
                        [state["input_ids"] for state in shifted],
                        dtype=torch.long,
                        device=device,
                    )
                    shifted_mask = torch.ones_like(shifted_inp, dtype=torch.bool)
                    with torch.no_grad():
                        attachment_out = self.forward(
                            input_ids=shifted_inp,
                            attention_mask=shifted_mask,
                            tree_spans=collate_spans(shifted),
                            last_logits_only=True,
                            compute_attachment_logits=True,
                        )
                    if attachment_out.attachment_logits is not None:
                        attachment_rows = attachment_out.attachment_logits[:, -1, :]

                expanded: List[dict] = []
                for si, (state, counts) in enumerate(zip(shifted, choice_counts)):
                    actions = [apply_action(state, count) for count in counts]
                    target_ids = [target for _, target in actions]
                    if attachment_rows is None:
                        action_lps = [-math.log(len(actions))] * len(actions)
                    else:
                        logits = attachment_rows[
                            si, torch.tensor(target_ids, dtype=torch.long, device=device)
                        ].float()
                        action_lps = torch.log_softmax(logits, dim=0).tolist()
                    for (new_state, _), action_lp in zip(actions, action_lps):
                        new_state["logprob"] += action_lp
                        expanded.append(new_state)

                finished.extend(state for state in expanded if state["done"])
                unfinished = [state for state in expanded if not state["done"]]
                unfinished.sort(key=lambda state: state["logprob"], reverse=True)
                beams = unfinished[:beam_size]

            candidates = finished + beams
            candidates.sort(key=lambda state: state["logprob"], reverse=True)
            selected = candidates[:beam_size]
            token_rows: List[List[int]] = []
            score_rows: List[float] = []
            for state in selected:
                generated = state["generated"][:max_steps]
                generated += [pad] * (max_steps - len(generated))
                token_rows.append(generated)
                score_rows.append(float(state["logprob"]))
            while len(token_rows) < beam_size:
                token_rows.append([pad] * max_steps)
                score_rows.append(-float("inf"))
            all_tokens.append(token_rows)
            all_scores.append(score_rows)

        return OLMoGenerateOutput(
            token_ids=torch.tensor(all_tokens, dtype=torch.long, device=device),
            scores=torch.tensor(all_scores, dtype=torch.float, device=device),
        )
