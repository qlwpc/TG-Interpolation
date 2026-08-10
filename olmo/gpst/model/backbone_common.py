"""Shared types for GPST transformer backbones.

Both the HF GPT2 wrapper (``gpt2_flash_attn.py``) and the OLMo block stack
(``olmo_stack.py``) return the same lightweight output container so that
``FastGenerativeR2D2`` can consume either backbone interchangeably.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutput:
    """Minimal output of a backbone ``forward``.

    Mirrors the subset of HuggingFace's ``ModelOutput`` that GPST consumes:
    the final hidden states and, when caching is requested, the per-layer
    ``(key, value)`` tuples for incremental decoding.
    """

    last_hidden_state: torch.Tensor
    past_key_values: Optional[tuple] = None
