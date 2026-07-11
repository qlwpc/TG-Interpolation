"""Pushdown Layers (Murty et al., EMNLP 2023; arXiv 2310.19089).

A drop-in self-attention replacement that augments a transformer LM with a **stack
tape**: for each query position ``k``, every prefix key ``j`` is assigned a *depth*
(the number of closed constituents containing ``j`` as of prefix ``k`` — the "stale"
tape). Per layer, a depth embedding ``d_{kj}^l`` is added to the **key** before
computing attention scores::

    score_{kj} = q_k^T (k_j + W_key d_{kj}^l) = q_k^T k_j + q_k^T (W_key d_{kj}^l)

so the depth bias is a per-(query,key) additive term ``q_k . E_l[S[k,j]]`` where
``E_l[d] = W_key d_l^d`` and ``S[k,j]`` is the precomputed depth.

Implementation (fast, faithful):
* The depth matrix ``S[b,k,j]`` (int8, lower-triangular) is computed **on the GPU**
  from the compact constituent spans (a 2D difference + two ``torch.cumsum``s,
  <1 ms on GPU) — never stored to disk (the train corpus is 10G tokens; a
  materialized ``(n,n)`` tape would be ~20 TB).
* The bias is applied via FlexAttention ``score_mod``: ``score + Q[b,h,q] . E_l[h,
  S[b,q,kv]]``. FlexAttention fuses this into a FlashAttention-class kernel (the
  only path that takes a per-element bias without materializing a 1.2 GB tensor or
  disabling flash). Causality is the block_mask.
* When no parse is available (e.g. BLiMP/SG minimal pairs), ``S`` is all-zero and
  the depth bias vanishes -> plain causal flash attention, full speed, zero overhead.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def compute_depth_matrix_gpu(spans: torch.Tensor, n: int) -> torch.Tensor:
    """Compute the Pushdown stale stack tape ``S[b,k,j]`` on GPU from spans.

    Args:
        spans: ``(B, M, 3)`` long tensor of ``(left, split, right)`` constituent
            spans (terminal indices in ``[0, n)``), padded arbitrarily (invalid
            spans are masked by ``right < left`` -> no contribution).
        n: sequence length.

    Returns:
        ``S`` of shape ``(B, n, n)`` int8, lower-triangular, where
        ``S[b,k,j] = #{spans (l,r): l<=j<=r and r<=k}``.

    NOTE: uses ``index_put_``, which graph-breaks under ``torch.compile``/dynamo
    (and the resumed graph miscompiles the gather -> CUDA device-side assert).
    Therefore the model config MUST leave ``compile`` unset (so
    ``cfg.compile is None`` and ``scripts/train.py`` skips ``block.compile()``).
    With torch.compile disabled this is correct and fast (two cumsums, O(n^2)).
    """
    B, M, _ = spans.shape
    device = spans.device
    l0 = spans[..., 0]
    r0 = spans[..., 2]
    # Valid spans only (padding uses -1): l>=0, r>=0, l<=r, and within range.
    valid_span = (l0 >= 0) & (r0 >= 0) & (l0 <= r0) & (r0 < n)
    l = l0.clamp(0, n - 1)
    r = r0.clamp(0, n - 1)
    l = l.clamp(max=r)
    # 2D difference array D[b, r, l] += 1, D[b, r, r+1] -= 1 (per span), then two
    # cumulative sums propagate the +1 over the rectangle [r:n, l:r+1]. Invalid
    # (padded) spans contribute zero.
    D = torch.zeros(B, n, n, device=device, dtype=torch.float32)
    bidx = torch.arange(B, device=device).unsqueeze(1).expand(B, M)
    contrib = valid_span.to(torch.float32)  # 1 for valid, 0 for padded
    D.index_put_((bidx, r, l), contrib, accumulate=True)
    rr = r + 1
    valid_close = valid_span & (rr < n)
    if valid_close.any():
        D.index_put_((bidx[valid_close], r[valid_close], rr[valid_close]),
                     -torch.ones(int(valid_close.sum()), device=device, dtype=torch.float32),
                     accumulate=True)
    S = torch.cumsum(torch.cumsum(D, dim=1), dim=2)
    S = torch.tril(S)
    return S.to(torch.int8)


class PushdownDepthBias(nn.Module):
    """Per-layer depth-embedding for Pushdown Layers.

    Holds a depth embedding ``D_l`` of shape ``(max_depth+1, d_model)`` (in
    pre-projection key space, as in the paper). ``forward`` projects it through the
    key projection to produce per-head ``E_l`` of shape ``(n_heads, max_depth+1,
    d_head)``: ``E_l[h, d] = D_l[d] @ W_key_h^T``.
    """

    def __init__(self, max_depth: int, d_model: int, n_heads: int):
        super().__init__()
        self.max_depth = max_depth
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.depth_emb = nn.Embedding(max_depth + 1, d_model)

    def forward(self, key_weight: torch.Tensor) -> torch.Tensor:
        """Project the depth embedding through the key projection.

        Args:
            key_weight: the block's ``att_proj`` weight rows for the key, shape
                ``(n_kv_heads * d_head, d_model)`` (i.e. ``att_proj.weight[d_model :
                d_model + n_kv_heads*d_head]`` for the fused QKV projection).

        Returns:
            ``E_l`` of shape ``(n_heads, max_depth+1, d_head)``.
        """
        # D: (max_depth+1, d_model); key_weight: (n_kv_h*d_head, d_model).
        # E = D @ key_weight^T -> (max_depth+1, n_kv_h*d_head), then reshape to per-head.
        D = self.depth_emb.weight  # (max_depth+1, d_model)
        E = D @ key_weight.t()  # (max_depth+1, n_kv_h*d_head)
        n_kv = key_weight.shape[0] // self.d_head
        E = E.view(self.max_depth + 1, n_kv, self.d_head)  # (D, n_kv, d_head)
        # Expand/repeat to n_heads for GQA.
        if n_kv != self.n_heads:
            rep = self.n_heads // n_kv
            E = E.repeat_interleave(rep, dim=1)  # (D, n_heads, d_head)
        return E.permute(1, 0, 2).contiguous()  # (n_heads, D, d_head)
