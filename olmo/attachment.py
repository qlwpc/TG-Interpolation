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

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_attachment_query_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    bos_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Return positions with a Pushdown attachment target.

    Words and EOS have targets; BOS/ROOT and padding do not. With distinct
    BOS/EOS IDs every BOS is removed directly. If a tokenizer shares the two
    IDs, packed boundaries alternate ``... EOS, BOS ...``; position zero and a
    boundary immediately following another boundary are the BOS occurrences.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape (B,n), got {tuple(input_ids.shape)}"
        )
    if attention_mask is None:
        valid = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, got "
                f"{tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
            )
        valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)

    if bos_token_id != eos_token_id:
        is_root = input_ids == int(bos_token_id)
    else:
        boundary = input_ids == int(bos_token_id)
        previous_boundary = torch.cat(
            [
                torch.ones(
                    input_ids.shape[0], 1, dtype=torch.bool, device=input_ids.device
                ),
                boundary[:, :-1],
            ],
            dim=1,
        )
        is_root = boundary & previous_boundary
    return valid & ~is_root


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
        root_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        # The surrounding trainer runs under AMP. Bilinear scores and the
        # separately constructed diagonal must use one stable dtype; otherwise
        # autocast can produce a bf16 bmm destination and fp32 diagonal source,
        # which makes the indexed assignment fail on CUDA. The attachment logits
        # are modest compared with the transformer activations and are consumed
        # by cross-entropy, so construct them explicitly in fp32.
        with torch.autocast(device_type=final_hidden.device.type, enabled=False):
            return self._forward_fp32(
                final_hidden,
                input_ids,
                wte_weight,
                attention_mask,
                root_token_id,
                eos_token_id,
            )

    def _forward_fp32(
        self,
        final_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        wte_weight: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        root_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
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
            root_token_id: optional sentence ROOT/SOS ID. When supplied, packed
                sentences are isolated so a query cannot attach to a key before
                its own ROOT.
            eos_token_id: sentence EOS ID. Required only when ROOT and EOS share
                an ID, so alternating packed boundaries can be disambiguated.

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
        # Eq. 5 has a distinct shift-only branch on the diagonal:
        #   score[k,k] = h̃_k^T W h̃_k,
        # not h_k^T W h̃_k.  Compute it separately and scatter it over the
        # reduce-key matrix, matching the reference implementation's
        # ``next_word_key``/``logit_self`` insertion.
        self_logits = (Wh_tilde * h_tilde).sum(dim=-1)            # (B,n)
        diag = torch.arange(n, device=logits.device)
        logits[:, diag, diag] = self_logits
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
        # Packed preprocessing concatenates many independent
        # [ROOT, words..., EOS] sentences. The reference attachment softmax is
        # sentence-local, so keys from earlier sentences must not be distractors.
        if root_token_id is not None:
            boundary = input_ids == int(root_token_id)
            if eos_token_id is not None and root_token_id == eos_token_id:
                previous_boundary = torch.cat(
                    [
                        torch.ones(B, 1, dtype=torch.bool, device=input_ids.device),
                        boundary[:, :-1],
                    ],
                    dim=1,
                )
                is_root = boundary & previous_boundary
            else:
                is_root = boundary
            sentence_id = torch.cumsum(is_root.to(torch.long), dim=1)
            same_sentence = sentence_id[:, :, None] == sentence_id[:, None, :]
            logits = logits.masked_fill(~same_sentence, float("-inf"))
        return logits


def _derive_oracle_reduce_targets_reference(
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

    # Independent reference implementation. The production implementation below
    # is vectorized; keep this literal stack simulation for regression tests.
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
            # Algorithm 1 shifts x_k before applying the closes at k. This is
            # observable for preterminal/single-token spans (k,k): the new leaf
            # must sit on top of the previous stack, not cause that stack to be
            # cleared while searching for left endpoint k.
            stack.append((k, k))
            closes = closes_by_right.get(k)
            if not closes:
                out[b, k] = k  # shift-only
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


def derive_oracle_reduce_targets(
    spans: torch.Tensor,
    n: int,
    span_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Derive Pushdown oracle reduce targets without host transfers or Python loops.

    For a token ``k`` that closes at least one constituent, let ``l`` be the
    smallest left endpoint among spans ending at ``k`` (the outermost close).
    Immediately before that close, the stack item starting at ``l`` ends at the
    largest earlier right endpoint of a span with the same ``l``; when none
    exists, it is the shifted leaf ``l`` itself. Therefore the shift-then-reduce
    stack simulator is
    equivalent to sorting spans by ``(batch, left, right)``, taking the previous
    right endpoint within each ``(batch, left)`` group, and selecting the row with
    minimum ``left`` for each ``(batch, right)`` close event.

    All tensors retain the fixed ``B*M`` shape. Invalid/padded spans receive
    sentinel keys and sentinel scatter values, avoiding boolean-indexed dynamic
    outputs and GPU-to-CPU synchronization in the training hot path.
    """
    if os.environ.get("OLMO_PUSHDOWN_LEGACY_ORACLE"):
        return _derive_oracle_reduce_targets_legacy(spans, n, span_mask)
    if spans.dim() == 2:
        spans = spans.unsqueeze(0)
    B, M, _ = spans.shape
    device = spans.device
    if span_mask is None:
        span_mask = spans[..., 0] >= 0
    else:
        span_mask = span_mask.to(device=device, dtype=torch.bool)

    l0 = spans[..., 0].to(torch.long)
    r0 = spans[..., 2].to(torch.long)
    valid = span_mask & (l0 >= 0) & (r0 >= 0) & (l0 <= r0) & (r0 < n)
    l = l0.clamp(0, n - 1)
    r = r0.clamp(0, n - 1)
    batch = torch.arange(B, device=device, dtype=torch.long)[:, None].expand(B, M)

    flat_l = l.reshape(-1)
    flat_r = r.reshape(-1)
    flat_b = batch.reshape(-1)
    flat_valid = valid.reshape(-1)
    flat_position = torch.arange(B * M, device=device, dtype=torch.long)

    # Valid keys sort lexicographically by (batch, left, right). Give every
    # invalid slot a unique key after the valid range so it cannot form a group.
    sentinel_base = B * n * (n + 1)
    sort_key = (flat_b * n + flat_l) * (n + 1) + flat_r
    sort_key = torch.where(flat_valid, sort_key, sentinel_base + flat_position)
    order = torch.argsort(sort_key)
    sb = flat_b[order]
    sl = flat_l[order]
    sr = flat_r[order]
    sv = flat_valid[order]

    previous_b = torch.roll(sb, 1)
    previous_l = torch.roll(sl, 1)
    previous_r = torch.roll(sr, 1)
    same_left_group = sv & torch.roll(sv, 1) & (sb == previous_b) & (sl == previous_l)
    same_left_group[0] = False
    predecessor = torch.where(same_left_group, previous_r, sl)

    end_index = sb * n + sr
    outer_left = torch.full((B * n,), n, dtype=torch.long, device=device)
    outer_left.scatter_reduce_(
        0,
        end_index,
        torch.where(sv, sl, torch.full_like(sl, n)),
        reduce="amin",
        include_self=True,
    )
    is_outermost = sv & (sl == outer_left[end_index])
    chosen = torch.full((B * n,), n, dtype=torch.long, device=device)
    chosen.scatter_reduce_(
        0,
        end_index,
        torch.where(is_outermost, predecessor, torch.full_like(predecessor, n)),
        reduce="amin",
        include_self=True,
    )

    shifts = torch.arange(n, device=device, dtype=torch.long).repeat(B)
    return torch.where(chosen < n, chosen, shifts).view(B, n)


def _derive_oracle_reduce_targets_legacy(
    spans: torch.Tensor,
    n: int,
    span_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pre-optimization oracle, retained only for matched performance controls.

    This intentionally closes constituents before shifting ``k`` and therefore
    reproduces the old singleton-span stack-reset bug. Never use it for training
    except with the explicit diagnostic environment switch above.
    """
    if spans.dim() == 2:
        spans = spans.unsqueeze(0)
    B, M, _ = spans.shape
    device = spans.device
    if span_mask is None:
        span_mask = spans[..., 0] >= 0
    else:
        span_mask = span_mask.to(torch.bool)
    spans_cpu = spans.detach().to("cpu", dtype=torch.long)
    mask_cpu = span_mask.detach().to("cpu")
    out = torch.zeros(B, n, dtype=torch.long)
    for b in range(B):
        closes_by_right: dict[int, list[int]] = {}
        for m in range(M):
            if not bool(mask_cpu[b, m]):
                continue
            left = int(spans_cpu[b, m, 0])
            right = int(spans_cpu[b, m, 2])
            if left < 0 or right < 0 or right >= n:
                continue
            closes_by_right.setdefault(right, []).append(left)
        stack: list[Tuple[int, int]] = []
        for k in range(n):
            closes = closes_by_right.get(k)
            if not closes:
                out[b, k] = k
                stack.append((k, k))
                continue
            outer_left = min(closes)
            while stack and stack[-1][0] != outer_left:
                stack.pop()
            if stack:
                out[b, k] = stack[-1][1]
                stack.pop()
            else:
                out[b, k] = k
            stack.append((outer_left, k))
    return out.to(device)


def compute_attachment_loss(
    logits: torch.Tensor,
    oracle: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy of the attachment head against oracle reduce targets.

    ``L_attach`` is CE over valid ``(b, k)`` positions. ``reduction`` accepts
    ``"mean"``, ``"sum"``, or ``"none"`` and follows
    :func:`torch.nn.functional.cross_entropy` semantics (with invalid queries
    excluded).
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
        if reduction == "none":
            return logits.sum(dim=-1) * 0.0
        return logits.sum() * 0.0
    # F.cross_entropy with reduction='none' ignores the -inf-padded key positions
    # correctly (they contribute -inf to the denominator only if targeted, which
    # the clamp prevents). reduction='sum' / count gives a masked mean.
    ce = F.cross_entropy(flat_logits, flat_target, reduction="none")  # (B*n,)
    if reduction == "none":
        return ce.masked_fill(~flat_mask, 0.0).view(B, n)
    ce = ce[flat_mask]
    if reduction == "sum":
        return ce.sum()
    if reduction == "mean":
        return ce.sum() / flat_mask.sum().clamp(min=1)
    raise ValueError(f"unsupported attachment-loss reduction: {reduction!r}")
