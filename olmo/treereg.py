"""TreeReg auxiliary loss (Nandi et al., NAACL 2025; arXiv 2411.18885).

TreeReg softly injects a syntactic inductive bias into a transformer LM by adding
an auxiliary loss ``L_TR`` to the LM objective (``L_LM + alpha * L_TR``). It converts
constituency-parse bracketing decisions into differentiable **orthogonality
constraints** on hidden states: a constituent's representation should be maximally
orthogonal to its surrounding context.

This module is a faithful port of the reference implementation
(https://github.com/ananjan-nandi-9/tree_regularization, ``src/regularizer/``).
The loss acts on the post-block residual hidden state at a chosen layer
(``treereg_layer``), using a slice of ``|A| * d_head`` dims as the "circuit" vector.
No architectural change at inference (the loss is train-only).

Core quantity — **Span Contextual Independence Score (SCIN)** for span ``S_{a;b}``::

    SCIN(a, b) = ||orth(h_{a-1}, h_b)|| + ||orth(h_b, h_{b+1})||

where ``||orth(x, y)|| = ||x - proj_y(x)|| = ||x|| * sin(angle(x, y))`` is the norm of
the component of ``x`` orthogonal to ``y`` (matches ``scin_computer.get_all_orthogonal_scores``).
``h_{-1}`` / ``h_{n}`` are treated as zero (the term is dropped at boundaries).
The hidden states are **not** L2-normalized, so the score carries a ``||h||`` magnitude
weight, exactly as in the reference.

For a gold constituent spanning ``[st, en]`` that bifurcates at ``p`` (left child
``[st, p]``, right child ``[p+1, en]``), the split score is
``s(q) = SCIN(st, q) + SCIN(q+1, en)`` for ``st <= q < en``, which expands to the four
orthogonality terms of ``regularizer_main.get_span_score``:

    s(q) = ||orth(h_{st-1}, h_q)|| + ||orth(h_q,  h_{q+1})||
         + ||orth(h_q,    h_en)|| + ||orth(h_en, h_{en+1})||

Candidate splits ``q`` range over **all** token positions in ``[st, en-1]`` (the
reference restricts to word-boundary positions; that restriction is dropped here
because local BPE data has ~7% of gold splits landing on non-word-start subwords —
see plan). Spans with fewer than 2 candidate splits (``en - st < 2``) are skipped.

The span-level loss is a cross-entropy favoring the gold split ``q = p``. ``L_TR`` is a
**macro** average: per-sentence mean over that sentence's spans, then mean over
sentences in the batch (matches ``regularizer_main.get_score`` /
``trainer_main.sci_loss`` up to the reference's double-negation, which collapses to
minimizing ``mean(CE)``).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


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
        Scalar loss (macro mean: per-sentence span-CE mean, then mean over
        sentences). Zero (graph-connected) if no valid spans.
    """
    B, n, _ = hidden.shape
    if n_heads_subset and n_heads_subset > 0 and d_head is not None:
        circuit = hidden[..., : n_heads_subset * d_head]
    else:
        circuit = hidden
    # Orthogonality norms are small differences of near-aligned projections; run in
    # fp32 for numerical fidelity (the reference computes them in fp32 implicitly).
    circuit = circuit.float()

    # Orthogonality chart matching scin_computer.get_all_orthogonal_scores:
    #   O[b, i, j] = ||H[j] - proj_{Hn[i]}(H[j])||  (component of H[j] orthogonal to H[i])
    # Note the direction: the OUTER index i is the projection direction, the INNER
    # index j is the vector being projected (the reference uses
    # `orthogonal_magnitudes[st-1][en]`, so chart[(a,b)] = O[a-1, b]). H is NOT
    # normalized (keeps the ||H[j]|| magnitude weight, as in the reference).
    Hn = F.normalize(circuit, dim=-1, eps=eps)                  # (B, n, d)
    # proj[b, i, j] = H[j] · Hn[i]  = bmm(Hn, H^T)[b, i, j]
    proj = torch.bmm(Hn, circuit.transpose(1, 2))               # (B, n, n)
    # orth[b, i, j] = H[j] - proj[b,i,j] * Hn[i]; broadcast H[j] over axis 1, Hn[i] over axis 2.
    orth = circuit.unsqueeze(1) - proj.unsqueeze(-1) * Hn.unsqueeze(2)  # (B, n, n, d)
    O = orth.norm(dim=-1)                                       # (B, n, n)

    # Pad with a zero row above (index 0 = "no left context") and a zero column to
    # the right (index n = "no right context"), so O_pad[a, b] = O[a-1, b] with
    # O_pad[0, *] = 0 and O_pad[*, n] = 0. This lets SCIN(a, b) = O[a-1, b] + O[b, b+1]
    # be written as O_pad[a, b] + O_pad[b+1, b+1] with clamped indices.
    O_pad = F.pad(O, (0, 1, 1, 0))                              # (B, n+1, n+1)

    # Valid spans: binary split with >= 2 candidates (en - st >= 2). Unary spans
    # (en == st) and length-2 spans (only 1 candidate) are skipped, matching the
    # reference's `if en - st <= 1: return 0,0,0` and `if len(scores) < 2: return`.
    left_all = spans[..., 0]
    right_all = spans[..., 2]
    valid = span_mask.bool() & (left_all < right_all) & ((right_all - left_all) >= 2)
    if not valid.any():
        # Graph-connected zero so backward never errors on empty batches.
        return hidden.sum() * 0.0

    b_idx = torch.arange(B, device=hidden.device).unsqueeze(1).expand_as(valid)[valid]  # (V,)
    left = left_all[valid].long()
    split = spans[..., 1][valid].long()
    right = right_all[valid].long()
    # Clamp to valid range (defensive against padding -1).
    left = left.clamp(0, n - 1)
    right = right.clamp(0, n - 1)
    split = split.clamp(left, right - 1)  # gold split in [left, right-1]

    # Candidate splits q = left + r for r in [0, max_len); valid where q < right.
    max_len = int((right - left).clamp(min=2).max().item())
    r = torch.arange(max_len, device=hidden.device)            # (R,)
    q = left.unsqueeze(1) + r.unsqueeze(0)                     # (V, R)
    valid_q = q < right.unsqueeze(1)                           # q <= right-1

    # s(q) = SCIN(st, q) + SCIN(q+1, en) expands to the four terms of get_span_score:
    #   ||orth(h_{st-1}, h_q)|| + ||orth(h_q, h_{q+1})||
    # + ||orth(h_q,    h_en)|| + ||orth(h_en, h_{en+1})||
    # Indexing into O_pad (size n+1): O_pad[a, b] = O[a-1, b]; O_pad[0,*]=0; O_pad[*,n]=0.
    b_idx_v = b_idx.unsqueeze(1)                               # (V, 1)
    left_v = left.unsqueeze(1)                                 # (V, 1)
    right_v = right.unsqueeze(1)                               # (V, 1)
    q_clamped = q.clamp(0, n - 1)
    q1 = (q + 1).clamp(0, n)                                   # O_pad index in [0, n]
    term_st_q = O_pad[b_idx_v, left_v, q_clamped]              # ||orth(h_{st-1}, h_q)||
    term_q_q1 = O_pad[b_idx_v, q1, q1]                         # ||orth(h_q, h_{q+1})||
    term_q_en = O_pad[b_idx_v, q1, right_v]                    # ||orth(h_q, h_en)||
    term_en_en1 = O_pad[b_idx_v, right_v + 1, right_v + 1]     # ||orth(h_en, h_{en+1})||
    s = term_st_q + term_q_q1 + term_q_en + term_en_en1        # (V, R)
    s = s.masked_fill(~valid_q, float("-inf"))

    # Per-span CE favoring the gold split q = split; gold position r_gold = split - left.
    r_gold = (split - left).clamp(0, max_len - 1)              # (V,)
    logp = F.log_softmax(s, dim=-1)                            # (V, R)
    span_ce = -logp.gather(1, r_gold.unsqueeze(1)).squeeze(1)  # (V,)

    # Macro reduction: mean over spans within each sentence, then mean over sentences.
    b_idx_flat = b_idx  # (V,)
    sent_sum = torch.zeros(B, device=hidden.device, dtype=span_ce.dtype)
    sent_cnt = torch.zeros(B, device=hidden.device, dtype=span_ce.dtype)
    sent_sum.index_add_(0, b_idx_flat, span_ce)
    sent_cnt.index_add_(0, b_idx_flat, torch.ones_like(span_ce))
    sent_mean = sent_sum / sent_cnt.clamp(min=1)
    valid_sent = sent_cnt > 0
    loss = sent_mean[valid_sent].mean()
    return loss
