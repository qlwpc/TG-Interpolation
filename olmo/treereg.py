"""TreeReg auxiliary loss (Nandi et al., NAACL 2025; arXiv 2411.18885).

TreeReg softly injects a syntactic inductive bias into a transformer LM by adding
an auxiliary loss ``L_TR`` to the LM objective (``L_LM + alpha * L_TR``). It converts
constituency-parse bracketing decisions into differentiable **orthogonality
constraints** on hidden states: a constituent's representation should be maximally
orthogonal to its surrounding context.

This module implements the **B1 variant** (per the project plan): the loss acts on
the post-block residual hidden state at a chosen layer (``treereg_layer``), using a
slice of ``|A| * d_head`` dims as the "circuit" vector. No architectural change at
inference (the loss is train-only).

Core quantity — **Span Contextual Independence Score (SCIN)** for span ``S_{i;j}``::

    SCIN(i, j) = -cos(h_{i-1}, h_j) - cos(h_j, h_{j+1})

(left-context orthogonality + right-context orthogonality; ``h_{-1}``/``h_{n}`` are
treated as zero, i.e. the term is dropped at boundaries). For a gold constituent
spanning ``[i, j]`` that bifurcates at ``p`` (left child ``[i, p]``, right child
``[p+1, j]``), the split score is ``s(q) = SCIN(i, q) + SCIN(q+1, j)`` for
``i <= q < j``, and the span-level loss is a cross-entropy favoring the gold split
``q = p``. ``L_TR`` is the sum over all gold constituents.

The SCIN chart is built with one ``(B, n, n)`` matmul (``G = H_norm @ H_norm^T``);
the per-span CE is a small gather + softmax. Applied every ``k`` LM steps on ~25%
of heads at the middle layer, overhead is ~2-3% (train-only; zero inference cost).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def compute_treereg_loss(
    hidden: torch.Tensor,
    spans: torch.Tensor,
    span_mask: torch.Tensor,
    n_heads_subset: int = 0,
    d_head: Optional[int] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the TreeReg auxiliary loss ``L_TR``.

    Args:
        hidden: ``(B, n, d_model)`` post-block residual hidden states at
            ``treereg_layer``.
        spans: ``(B, M, 3)`` long tensor of gold constituent spans
            ``(left, split, right)`` (terminal indices in ``[0, n)``), padded with
            ``-1`` (or any value, masked out by ``span_mask``).
        span_mask: ``(B, M)`` bool tensor, ``True`` for valid spans.
        n_heads_subset: number of attention heads in the circuit ``A``. ``0`` uses
            the full ``d_model``.
        d_head: head dim (``d_model // n_heads``); required when ``n_heads_subset > 0``.

    Returns:
        Scalar loss (mean over valid spans across the batch). Zero if no spans.
    """
    B, n, d_model = hidden.shape
    if n_heads_subset and n_heads_subset > 0 and d_head is not None:
        circuit = hidden[..., : n_heads_subset * d_head]
    else:
        circuit = hidden
    Hn = _l2_normalize(circuit, eps=eps)  # (B, n, d)
    # Cosine-similarity chart G[b,i,j] = cos(h_i, h_j).
    G = torch.bmm(Hn, Hn.transpose(1, 2))  # (B, n, n)

    # Pad G with a zero row/col so that G[:, -1, j] and G[:, j, n] are 0 (boundaries).
    # We use index -1 for "no left context" (i=0) and index n for "no right context"
    # (j=n-1): implement via clamped indices with a zero gather.
    zero = torch.zeros(B, n, device=hidden.device, dtype=G.dtype)
    # G_left[b, i, j] = G[b, i-1, j] if i>0 else 0.
    G_left = torch.cat([zero.unsqueeze(1), G[:, :-1, :]], dim=1)  # (B, n, n), row i = G[i-1]
    # G_right[b, i, j] = G[b, i, j+1] if j<n-1 else 0.
    G_right = torch.cat([G[:, :, 1:], zero.unsqueeze(2)], dim=2)  # (B, n, n), col j = G[j+1]

    # SCIN(i, j) = -G[i-1, j] - G[i, j+1].
    SCIN = -(G_left + G_right)  # (B, n, n); SCIN[b, i, j]

    # Flatten spans across the batch for a vectorized CE.
    # Valid spans: (b, left, split, right) with left < right (a binary split exists;
    # unary spans left==right have an empty split range and are skipped).
    valid = span_mask.bool() & (spans[..., 0] < spans[..., 2])
    if not valid.any():
        # Graph-connected zero so backward never errors on empty batches.
        return hidden.sum() * 0.0
    b_idx = torch.arange(B, device=hidden.device).unsqueeze(1).expand(B, spans.shape[1])[valid]  # (V,)
    left = spans[..., 0][valid].long()   # (V,)
    split = spans[..., 1][valid].long()  # (V,)
    right = spans[..., 2][valid].long()  # (V,)
    # Clamp to valid range (defensive against padding -1).
    left = left.clamp(0, n - 1)
    right = right.clamp(0, n - 1)
    split = split.clamp(left, right.clamp(min=1) - 1)

    # For each valid span (i, p, j): split scores s(q) = SCIN(i, q) + SCIN(q+1, j)
    # for q in [i, j-1]. Spans have varying length, so loop-free via a padded
    # max-span gather. Use a per-span arange up to max_len = (j - i).
    max_len = int((right - left).clamp(min=1).max().item())
    V = left.shape[0]
    # q = i + r, r in [0, max_len); shape (V, R).
    r = torch.arange(max_len, device=hidden.device)  # (R,)
    q = left.unsqueeze(1) + r.unsqueeze(0)  # (V, R)
    valid_q = q < right.unsqueeze(1)  # q <= j-1
    q_clamped = q.clamp(0, n - 1)
    # SCIN(i, q) and SCIN(q+1, j). Broadcast (V,1) batch/row indices with (V,R) cols.
    b_idx_v = b_idx.unsqueeze(1)  # (V, 1)
    left_v = left.unsqueeze(1)  # (V, 1)
    right_v = right.unsqueeze(1)  # (V, 1)
    scin_iq = SCIN[b_idx_v, left_v, q_clamped]  # (V, R)
    q1 = (q + 1).clamp(0, n - 1)
    scin_q1j = SCIN[b_idx_v, q1, right_v]  # (V, R)
    s = scin_iq + scin_q1j  # (V, R)
    s = s.masked_fill(~valid_q, float("-inf"))
    # CE favoring the gold split q = p (= split). gold position r_gold = p - i.
    r_gold = (split - left).clamp(0, max_len - 1)
    loss = F.cross_entropy(s, r_gold, reduction="mean")
    return loss
