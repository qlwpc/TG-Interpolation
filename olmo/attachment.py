"""Pushdown attachment head (Murty et al., EMNLP 2023; arXiv 2310.19089, Eq. 5).

The depth-key bias (``olmo.pushdown``) modulates attention with a precomputed
stack tape. The **attachment head** is the complementary *predictive* part: at
each prefix ``k`` it predicts which earlier constituent's rightmost token ``j``
the new token ``x_k`` should reduce with (or shift-only, ``j=k``), via a bilinear
attention over the final-layer hidden states::

    h̃_k^L = MLP(emb(x_k), h_{k-1}^L)                 # "just predicted x_k" repr
    score_{k,j} = (h_j^L)^T W h̃_k^L     (j != k : shift + reduce onto j's constituent)
                = (h̃_k^L)^T W h̃_k^L     (j == k : shift only)

Attachment scores are computed only against the **rightmost token** of each
constituent (Fig. 2). Training supervises this head with a cross-entropy against
the oracle reduce target ``r_k^*`` derived from the gold parse by simulating the
shift-reduce stack (Algorithm 1). The head is train-only at the loss site; it
does not alter the depth-bias forward path (the stack tape stays gold-precomputed
during training).

This module is pure torch (no flex/CUDA), so it tests on CPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PushdownAttachmentHead(nn.Module):
    """Bilinear attachment head with an MLP-generated query ``h̃_k`` (Eq. 5).

    Args:
        d_model: hidden size ``d``.
        vocab_size: vocabulary size (for the embedding lookup of ``x_k``). The
            head does NOT own the embedding table — it receives ``wte_weight``
            (the model's tied token embedding) in ``forward`` so it shares the
            LM's embeddings, matching the paper's ``MLP(x_k, h_{k-1})``.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # W ∈ R^{d×d}: bilinear (h_j)^T W h̃_k.  nn.Linear stores W^T, so
        # `self.W(h̃)` = W h̃, and (h_j)^T (W h̃) = h_j · self.W(h̃).
        self.W = nn.Linear(d_model, d_model, bias=False)
        # MLP: concat(emb(x_k), h_{k-1}) [2d] -> d -> d.  Paper gives no detail;
        # a standard 2-layer MLP is the faithful minimal choice.
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        final_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        wte_weight: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attachment logits ``p(r_k = j | x_<k)`` (unnormalized).

        Args:
            final_hidden: ``(B, n, d)`` final-layer residual states ``h^L``
                (captured before ``ln_f`` in ``OLMo.forward``).
            input_ids: ``(B, n)`` long — to look up ``emb(x_k)``.
            wte_weight: ``(vocab, d)`` the model's token embedding table
                (weight tying; passed in so the head adds no embedding params).
            attention_mask: ``(B, n)`` bool — True for valid (non-pad) tokens.
                Padded key positions get ``-inf`` so they never win reduce.

        Returns:
            ``logits`` of shape ``(B, n, n)`` fp32, where ``logits[b, k, j]`` is
            the score for ``r_k = j``. Strictly upper-triangular (``j > k``) is
            ``-inf`` (a query can only reduce onto a prefix token); the diagonal
            ``j == k`` is the shift-only self-score and is kept finite.
        """
        B, n, d = final_hidden.shape
        h = final_hidden.float()                                   # (B,n,d) keys h_j^L
        emb = F.embedding(input_ids, wte_weight).float()          # (B,n,d) emb(x_k)
        # h_{k-1}^L: shift h right by one along time; position 0 gets a zero vector
        # (no "previous" hidden state for the first token).
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))                # (B,n,d)
        h_tilde = self.mlp(torch.cat([emb, h_prev], dim=-1))      # (B,n,d) query
        Wh_tilde = self.W(h_tilde)                                # (B,n,d)
        # logits[b,k,j] = (W h̃_k) · h_j  ==  (h_j)^T W h̃_k  (Eq. 5, j != k branch).
        logits = torch.bmm(Wh_tilde, h.transpose(1, 2))           # (B,n,n)
        # Causal mask: query k can only attend to keys j <= k. The diagonal is
        # KEPT (it is the shift-only self-score (h̃_k)^T W h̃_k, which the paper
        # folds into the same softmax). Strict upper triangle -> -inf.
        causal = torch.triu(
            torch.full((n, n), float("-inf"), device=logits.device, dtype=logits.dtype),
            diagonal=1,
        )
        logits = logits + causal
        # Pad mask: invalid key positions (j) must never be a reduce target.
        if attention_mask is not None:
            am = attention_mask.to(torch.bool).view(B, 1, n)
            logits = logits.masked_fill(~am, float("-inf"))
        return logits


def derive_oracle_reduce_targets(
    spans: torch.Tensor,
    n: int,
    span_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Derive each token's oracle reduce target ``r_k^*`` from gold spans.

    Simulates the shift-reduce stack of Algorithm 1 / Fig. 2 using only the
    ``(l, r)`` of each gold constituent (the ``split`` column is ignored — no
    binarization needed). For each token ``k``:

      * if no gold constituent ends at ``k``: ``r_k = k`` (shift-only);
      * else: ``r_k = <rightmost token of the constituent on top of the stack
        after popping until the stack top's left == the outermost closing
        constituent's left>``.

    See the plan's "Design Decision B" for the worked Fig. 2 example.

    Args:
        spans: ``(B, M, 3)`` long tensor of ``(left, split, right)`` constituent
            spans (terminal indices in ``[0, n)``), padded with ``-1``.
        n: sequence length.
        span_mask: ``(B, M)`` bool, True for valid spans. If None, derived as
            ``spans[..., 0] >= 0``.

    Returns:
        ``targets`` of shape ``(B, n)`` long, where ``targets[b, k] == k`` means
        shift-only and ``targets[b, k] == j`` (``j < k``) means reduce onto the
        constituent whose rightmost token is ``j``.
    """
    if spans.dim() == 2:
        spans = spans.unsqueeze(0)
    B, M, _ = spans.shape
    device = spans.device
    if span_mask is None:
        span_mask = spans[..., 0] >= 0
    else:
        span_mask = span_mask.to(torch.bool)

    # Move to CPU Python lists for the stack simulation (B and n are small enough
    # that this is not a bottleneck; it runs once per batch in model_forward).
    spans_cpu = spans.detach().to("cpu", dtype=torch.long)
    mask_cpu = span_mask.detach().to("cpu")
    out = torch.zeros(B, n, dtype=torch.long)

    for b in range(B):
        # Collect valid (l, r) pairs, grouped by right endpoint.
        closes_by_right: dict[int, list[int]] = {}
        for m in range(M):
            if not bool(mask_cpu[b, m]):
                continue
            l = int(spans_cpu[b, m, 0])
            r = int(spans_cpu[b, m, 2])
            if l < 0 or r < 0 or r >= n:
                continue
            closes_by_right.setdefault(r, []).append(l)

        stack: list[Tuple[int, int]] = []  # list of (left, right) closed constituents
        for k in range(n):
            closes = closes_by_right.get(k)
            if not closes:
                out[b, k] = k  # shift-only
                stack.append((k, k))
                continue
            outer_left = min(closes)  # outermost closing constituent's left
            # Pop until the stack top's left == outer_left (its right is r_k).
            while stack and stack[-1][0] != outer_left:
                stack.pop()
            # By construction of a well-formed parse, the stack top's left must
            # equal outer_left here (the outermost constituent's left child is on
            # the stack). r_k = that constituent's right.
            if stack:
                out[b, k] = stack[-1][1]
                stack.pop()
            else:
                # Malformed/gold-inconsistent spans: fall back to shift-only.
                out[b, k] = k
            stack.append((outer_left, k))  # new merged constituent
    return out.to(device)


def compute_attachment_loss(
    logits: torch.Tensor,
    oracle: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy of the attachment head against oracle reduce targets.

    ``L_attach = mean over valid (b, k) of CE(logits[b, k, :], oracle[b, k])``.
    The shift-only case (``oracle == k``) is a normal class — the diagonal
    self-score is its logit. Returns a graph-connected zero when no positions are
    valid (so backward never errors on empty batches).

    Args:
        logits: ``(B, n, n)`` from :meth:`PushdownAttachmentHead.forward`.
        oracle: ``(B, n)`` long from :func:`derive_oracle_reduce_targets`.
            ``oracle[b, k]`` is the target ``j`` (may equal ``k`` for shift-only).
        attention_mask: ``(B, n)`` bool — only valid query positions contribute.
    """
    B, n, _ = logits.shape
    # Flatten (B, n) -> (B*n, n) for cross_entropy; oracle -> (B*n,).
    flat_logits = logits.reshape(B * n, n)
    flat_target = oracle.reshape(B * n).to(torch.long).clamp(0, n - 1)
    if attention_mask is not None:
        flat_mask = attention_mask.to(torch.bool).reshape(B * n)
    else:
        flat_mask = torch.ones(B * n, dtype=torch.bool, device=logits.device)
    if not flat_mask.any():
        return logits.sum() * 0.0
    # F.cross_entropy with reduction='none' ignores the -inf-padded key positions
    # correctly (they contribute -inf to the denominator only if targeted, which
    # the clamp prevents). reduction='sum' / count gives a masked mean.
    ce = F.cross_entropy(flat_logits, flat_target, reduction="none")  # (B*n,)
    ce = ce[flat_mask]
    return ce.sum() / flat_mask.sum().clamp(min=1)
