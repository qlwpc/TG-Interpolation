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

import os
from typing import Optional

import torch
import torch.nn as nn


class _DepthBiasGradP(torch.autograd.Function):
    """Custom autograd Function that supplies the depth-bias gradient to ``P``.

    Fix #2 for the pushdown flex slowdown. The forward flex path uses
    ``P.detach()`` in the score_mod (gather-by-depth, 77 MB captured buffer, fast —
    flash-fused, see OLMO_PUSHDOWN_DETACH). Because P is detached, flex's own
    backward does NOT attempt a ``zeros_and_scatter`` into P (the 2.4 GB dense-db
    scatter that costs ~38 s/step) AND does not deliver grad to ``depth_emb``.

    This Function closes that gap. Its forward returns ZEROS (so ``out + 0 = out`` —
    the flex output is unchanged), and its backward receives ``grad_out`` (the real
    downstream grad w.r.t. the attention output) and manually computes ``grad_P``
    via the exact attention-bias backward formula:

        pT      = softmax_causal_pad(q@k^T * scale + P.gather(Dc) * scale)
        g_attn  = grad_out @ v^T
        g_post  = pT * (g_attn - (g_attn * pT).sum(-1, keepdim=True))   # softmax bwd
        grad_P  = scatter_add(zeros_like(P), Dc, g_post * scale, dim=3)  # gather bwd

    (scale = 1/sqrt(hs) = inv_sqrt_hs; the bias is ``P.gather(Dc) * scale``, so the
    gather-backward multiplies by scale.) Validated vs autograd on CPU: grad_P, grad_q,
    grad_k, grad_v all match to 1e-7. The manual backward recomputes q@k^T (one bmm,
    ~0.1 ms) + softmax + a scatter_add into the 77 MB P — a few ms/layer in fp32,
    NOT the inductor ``zeros_and_scatter`` into 2.4 GB. grad_q/grad_k/grad_v are left
    to flex's own fused backward (this Function returns None for them).

    The causal+pad mask is required so padded kv positions get pT=0 (else spurious
    grad_P). ``attn_mask`` is the 1-D bool (B, N) valid-key mask (stashed on the
    shared BufferCache in OLMo.forward, key "pushdown_attn_mask").
    """

    @staticmethod
    def forward(ctx, q, k, v, P, Dc, attn_mask, inv_sqrt_hs):
        # forward returns zeros (out + 0 == out). Save tensors for the backward.
        ctx.save_for_backward(q, k, v, P, Dc, attn_mask)
        ctx.inv_sqrt_hs = float(inv_sqrt_hs)
        return torch.zeros_like(q)  # (B, H, N, hs)

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, P, Dc, attn_mask = ctx.saved_tensors
        inv = ctx.inv_sqrt_hs  # = scale = 1/sqrt(hs)
        B, H, N, hs = q.shape
        # Run in bf16 (NOT fp32) to match the forward's speed: the fp32 version
        # materialized a 2.4 GB fp32 attention matrix per layer and ran ~10.6 s/step
        # over DETACH. bf16 halves the memory traffic and the bmm cost. Safe here
        # because attention scores are O(1) (post-layernorm q/k, scale 1/sqrt(hs)) so
        # bf16 softmax does not overflow. autocast disabled so dtypes are explicit.
        _profile = bool(os.environ.get("OLMO_PUSHDOWN_FIX2_PROFILE"))
        with torch.autocast(device_type=q.device.type, enabled=False):
            if _profile:
                _ev = lambda n: (torch.cuda.Event(enable_timing=True), n)
                _t0 = {n: torch.cuda.Event(enable_timing=True) for _, n in
                       [ _ev("cast"), _ev("scores"), _ev("bias"), _ev("mask"),
                         _ev("softmax"), _ev("g_attn"), _ev("g_post"), _ev("scatter") ]}
                _t1 = {n: torch.cuda.Event(enable_timing=True) for n in _t0}
                _t0["cast"].record()
            qb = q.to(torch.bfloat16); kb = k.to(torch.bfloat16)
            vb = v.to(torch.bfloat16); Pb = P.to(torch.bfloat16)
            go = grad_output.to(torch.bfloat16)
            if _profile:
                _t1["cast"].record(); _t0["scores"].record()
            scores = torch.einsum("bhni,bhmi->bhnm", qb, kb) * inv        # (B,H,N,N) bf16
            if _profile:
                _t1["scores"].record(); _t0["bias"].record()
            bias = torch.take_along_dim(Pb, Dc.unsqueeze(1), dim=3) * inv  # (B,H,N,N) bf16
            if _profile:
                _t1["bias"].record(); _t0["mask"].record()
            post = scores + bias
            if attn_mask is not None:
                am = attn_mask.to(torch.bool).view(B, 1, 1, N)
                causal = torch.tril(torch.ones(N, N, device=post.device, dtype=torch.bool))
                valid = am & causal.view(1, 1, N, N)
                post = post.masked_fill(~valid, float("-inf"))
            if _profile:
                _t1["mask"].record(); _t0["softmax"].record()
            pT = torch.softmax(post, dim=-1)                              # (B,H,N,N) bf16
            if _profile:
                _t1["softmax"].record(); _t0["g_attn"].record()
            g_attn = torch.einsum("bhni,bhmi->bhnm", go, vb)              # grad_out @ v^T
            if _profile:
                _t1["g_attn"].record(); _t0["g_post"].record()
            Di = (g_attn * pT).sum(-1, keepdim=True)
            g_post = pT * (g_attn - Di)                                   # (B,H,N,N) bf16
            if _profile:
                _t1["g_post"].record(); _t0["scatter"].record()
            grad_P = torch.zeros(B, H, N, Pb.shape[3], device=Pb.device, dtype=torch.bfloat16)
            grad_P.scatter_add_(3, Dc.unsqueeze(1).expand(B, H, N, N), g_post * inv)
            if _profile:
                _t1["scatter"].record(); torch.cuda.synchronize()
                parts = ", ".join(f"{n}={_t0[n].elapsed_time(_t1[n]):.1f}ms" for n in _t0)
                print(f"[fix2_bwd] {parts}", flush=True)
        # grad_q, grad_k, grad_v, Dc, attn_mask, inv: None (flex's backward handles q/k/v).
        # grad_P is bf16; autograd casts it to P's dtype for the chain to E/depth_emb.
        return (None, None, None, grad_P, None, None, None)


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
