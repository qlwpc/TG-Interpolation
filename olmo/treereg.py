"""Sentence-local TreeReg loss (Nandi et al., NAACL 2025).

This is a behavioral port of
``tree_regularization/src/regularizer/{scin_computer,regularizer_main}.py``.
TreeReg is evaluated independently for every *complete top-level parse tree*.
Document BOS/EOS tokens, whitespace between trees, and nested ``S`` nodes are
not sentence boundaries.

For a span ``[st, en]`` split after ``q``, the reference score is::

    ||orth(h_q,    h_{st-1})|| + ||orth(h_{q+1}, h_q)||
  + ||orth(h_en,   h_q)||      + ||orth(h_{en+1}, h_en)||

where the first/last term is zero at a top-level-tree boundary. Candidate
splits are restricted to positions for which ``q + 1`` starts a parser word.
Unary decisions, one-choice decisions, and internal BPE-word decisions are not
supervised.

The implementation intentionally computes only required span candidates.  It
never materializes the production-infeasible ``(B, n, n, d)`` orthogonal tensor.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def _orthogonal_norm(
    vectors: torch.Tensor,
    contexts: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference ``||v - proj_normalize(c)(v)||`` for matching leading shapes."""
    context_unit = F.normalize(contexts, dim=-1, eps=eps)
    projection = (vectors * context_unit).sum(dim=-1, keepdim=True)
    return (vectors - projection * context_unit).norm(dim=-1)


def _graph_zero(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.sum() * 0.0


def count_treereg_sentences(sentence_ids: torch.Tensor) -> torch.Tensor:
    """Count contiguous nonnegative top-level-tree runs in a padded batch."""
    if sentence_ids.ndim != 2:
        raise ValueError(
            f"sentence_ids must have shape (B,n), got {tuple(sentence_ids.shape)}"
        )
    valid = sentence_ids >= 0
    starts = valid.clone()
    if sentence_ids.shape[1] > 1:
        starts[:, 1:] &= sentence_ids[:, 1:] != sentence_ids[:, :-1]
    return starts.sum()


def compute_treereg_loss(
    hidden: torch.Tensor,
    spans: torch.Tensor,
    span_mask: torch.Tensor,
    n_heads_subset: int = 0,
    d_head: Optional[int] = None,
    eps: float = 1e-8,
    *,
    sentence_ids: Optional[torch.Tensor] = None,
    word_boundaries: Optional[torch.Tensor] = None,
    return_sentence_count: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute the upstream-faithful TreeReg auxiliary loss.

    Args:
        hidden: ``(B, n, d_model)`` post-layer residual states.
        spans: ``(B, M, 3)`` global-token ``(left, split, right)`` spans.
        span_mask: ``(B, M)`` validity mask.
        n_heads_subset: number of residual-head-width slices in circuit ``A``;
            ``0`` uses the full model dimension, matching the reference option.
        d_head: one attention-head width. Required when ``n_heads_subset > 0``.
        sentence_ids: ``(B, n)`` top-level-tree ids. ``-1`` means that the token
            is BOS/EOS/whitespace/padding and is excluded from TreeReg.
        word_boundaries: ``(B, n)`` boolean first-BPE-of-parser-word mask.
        return_sentence_count: also return the number of complete top-level
            trees included in the macro average.

    If both metadata tensors are omitted, each batch row is treated as one
    sentence and every token as a word start. This compatibility mode is useful
    for small mathematical unit tests; production training supplies exact
    metadata from :mod:`olmo.data.parse_align`.
    """
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape (B,n,d), got {tuple(hidden.shape)}")
    batch_size, seq_len, d_model = hidden.shape
    if spans.ndim != 3 or spans.shape[0] != batch_size or spans.shape[-1] != 3:
        raise ValueError(f"spans must have shape (B,M,3), got {tuple(spans.shape)}")
    if span_mask.shape != spans.shape[:2]:
        raise ValueError(
            f"span_mask shape {tuple(span_mask.shape)} does not match spans {tuple(spans.shape)}"
        )
    if n_heads_subset < 0:
        raise ValueError("n_heads_subset must be non-negative")
    if n_heads_subset:
        if d_head is None or d_head <= 0:
            raise ValueError("positive d_head is required when n_heads_subset > 0")
        circuit_width = n_heads_subset * d_head
        if circuit_width > d_model:
            raise ValueError(
                f"TreeReg circuit width {circuit_width} exceeds d_model={d_model}"
            )
        circuit = hidden[..., :circuit_width]
    else:
        circuit = hidden
    # Projection residuals are numerically fragile in bf16; the upstream code
    # effectively evaluates them in fp32.
    circuit = circuit.float()

    if (sentence_ids is None) != (word_boundaries is None):
        raise ValueError("sentence_ids and word_boundaries must be provided together")
    if sentence_ids is None:
        sentence_ids = torch.zeros(
            (batch_size, seq_len), dtype=torch.int32, device=hidden.device
        )
        word_boundaries = torch.ones(
            (batch_size, seq_len), dtype=torch.bool, device=hidden.device
        )
    else:
        if sentence_ids.shape != (batch_size, seq_len):
            raise ValueError(
                f"sentence_ids must have shape {(batch_size, seq_len)}, "
                f"got {tuple(sentence_ids.shape)}"
            )
        if word_boundaries is None or word_boundaries.shape != (batch_size, seq_len):
            raise ValueError(
                f"word_boundaries must have shape {(batch_size, seq_len)}"
            )
        sentence_ids = sentence_ids.to(device=hidden.device)
        word_boundaries = word_boundaries.to(device=hidden.device, dtype=torch.bool)

    # The original faithful port iterated over every sentence, span, and split
    # candidate in Python.  A production microbatch contains O(10^4) spans, so
    # that path issued tens of thousands of tiny kernels and synchronized on
    # every ``.item()``/``.cpu()``.  The code below represents the same ragged
    # candidate set with flat indices and performs each mathematical phase in a
    # bounded number of GPU operations.
    sentence_valid = sentence_ids >= 0
    sentence_starts = sentence_valid.clone()
    if seq_len > 1:
        sentence_starts[:, 1:] &= sentence_ids[:, 1:] != sentence_ids[:, :-1]
    sentence_count = sentence_starts.sum()

    token_positions = torch.arange(seq_len, device=hidden.device, dtype=torch.long)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
    sentence_start_positions = torch.where(
        sentence_starts, token_positions, torch.zeros_like(token_positions)
    ).cummax(dim=1).values
    batch_offsets = (
        torch.arange(batch_size, device=hidden.device, dtype=torch.long) * seq_len
    ).unsqueeze(1)
    sentence_keys = sentence_start_positions + batch_offsets

    spans = spans.to(device=hidden.device, dtype=torch.long)
    span_mask = span_mask.to(device=hidden.device, dtype=torch.bool)
    left_all, split_all, right_all = spans.unbind(dim=-1)
    in_bounds = (
        span_mask
        & (left_all >= 0)
        & (left_all <= split_all)
        & (split_all < right_all)
        & (right_all < seq_len)
    )

    # Clamp only for safe metadata gathers. ``in_bounds`` excludes the clamped
    # entries before they can contribute to the result.
    safe_left = left_all.clamp(0, seq_len - 1)
    safe_split = split_all.clamp(0, seq_len - 1)
    safe_right = right_all.clamp(0, seq_len - 1)
    left_sentence = sentence_ids.gather(1, safe_left)
    split_sentence = sentence_ids.gather(1, safe_split)
    right_sentence = sentence_ids.gather(1, safe_right)
    contained = (
        in_bounds
        & (left_sentence >= 0)
        & (left_sentence == split_sentence)
        & (left_sentence == right_sentence)
    )

    span_batch_all = torch.arange(
        batch_size, device=hidden.device, dtype=torch.long
    ).unsqueeze(1).expand_as(left_all)
    span_batch = span_batch_all[contained]
    left = left_all[contained]
    split = split_all[contained]
    right = right_all[contained]
    span_sentence_keys = sentence_keys.gather(1, safe_left)[contained]

    # Expand each span [left, right] into the possible split positions
    # q=left,...,right-1, then retain q whose q+1 token begins a parser word.
    widths = right - left
    span_index = torch.arange(widths.numel(), device=hidden.device)
    candidate_span = torch.repeat_interleave(span_index, widths)
    repeated_offsets = torch.repeat_interleave(
        widths.cumsum(0) - widths, widths
    )
    candidate_position = (
        left[candidate_span]
        + torch.arange(candidate_span.numel(), device=hidden.device)
        - repeated_offsets
    )
    candidate_batch = span_batch[candidate_span]
    word_candidate = word_boundaries[
        candidate_batch, candidate_position + 1
    ]
    candidate_span = candidate_span[word_candidate]
    candidate_position = candidate_position[word_candidate]
    candidate_batch = candidate_batch[word_candidate]

    candidate_count = torch.zeros(
        widths.numel(), device=hidden.device, dtype=torch.long
    )
    candidate_count.scatter_add_(
        0, candidate_span, torch.ones_like(candidate_span)
    )
    candidate_is_gold = candidate_position == split[candidate_span]
    gold_count = torch.zeros_like(candidate_count)
    gold_count.scatter_add_(0, candidate_span, candidate_is_gold.long())
    eligible_span = (candidate_count >= 2) & (gold_count == 1)

    # Avoid scoring unary, one-choice, and internal-BPE decisions. Candidate
    # groups remain sorted by span, but scatter reductions below do not depend
    # on that ordering.
    candidate_keep = eligible_span[candidate_span]
    candidate_span = candidate_span[candidate_keep]
    candidate_position = candidate_position[candidate_keep]
    candidate_batch = candidate_batch[candidate_keep]
    candidate_is_gold = candidate_is_gold[candidate_keep]

    flat_circuit = circuit.reshape(batch_size * seq_len, -1)
    flat_candidate = candidate_batch * seq_len + candidate_position
    h_q = flat_circuit[flat_candidate]
    score = _orthogonal_norm(
        flat_circuit[flat_candidate + 1], h_q, eps
    )

    span_for_candidate = candidate_span
    candidate_left = left[span_for_candidate]
    candidate_right = right[span_for_candidate]
    candidate_span_batch = span_batch[span_for_candidate]
    span_sentence = left_sentence[contained]

    previous_position = (candidate_left - 1).clamp_min(0)
    previous_sentence = sentence_ids[candidate_span_batch, previous_position]
    has_previous = (candidate_left > 0) & (
        previous_sentence == span_sentence[span_for_candidate]
    )
    previous_hidden = flat_circuit[
        candidate_span_batch * seq_len + previous_position
    ]
    score = score + _orthogonal_norm(h_q, previous_hidden, eps) * has_previous

    right_hidden = flat_circuit[
        candidate_span_batch * seq_len + candidate_right
    ]
    score = score + _orthogonal_norm(right_hidden, h_q, eps)

    # This final term is constant across a span's candidates (and therefore
    # cancels algebraically in CE), but retaining it keeps the computed score
    # faithful to the upstream expression and the reference implementation.
    next_right = (right + 1).clamp_max(seq_len - 1)
    next_right_sentence = sentence_ids[span_batch, next_right]
    has_future = (right + 1 < seq_len) & (
        next_right_sentence == span_sentence
    )
    span_right_hidden = flat_circuit[span_batch * seq_len + right]
    future_score = _orthogonal_norm(
        flat_circuit[span_batch * seq_len + next_right],
        span_right_hidden,
        eps,
    ) * has_future
    score = score + future_score[span_for_candidate]

    # Cross entropy for each ragged span: logsumexp(scores) - gold_score.
    # The detached group maximum is only a numerical-stability shift; detaching
    # avoids retaining the scatter-max backward graph without changing gradients.
    score_max = torch.full(
        (widths.numel(),),
        -torch.inf,
        device=hidden.device,
        dtype=score.dtype,
    )
    score_max.scatter_reduce_(
        0, span_for_candidate, score.detach(), reduce="amax", include_self=True
    )
    partition = torch.zeros_like(score_max)
    partition.scatter_add_(
        0,
        span_for_candidate,
        torch.exp(score - score_max[span_for_candidate]),
    )
    gold_score = torch.zeros_like(score_max)
    gold_score.scatter_add_(
        0, span_for_candidate, score * candidate_is_gold.to(score.dtype)
    )
    span_loss = (
        score_max[eligible_span]
        + partition[eligible_span].log()
        - gold_score[eligible_span]
    )

    # First macro-average decisions within a sentence, then average all complete
    # top-level sentences. Sentences with no supervised decision contribute zero.
    eligible_sentence_keys = span_sentence_keys[eligible_span]
    sentence_loss_sum = torch.zeros(
        batch_size * seq_len, device=hidden.device, dtype=score.dtype
    )
    sentence_loss_sum.scatter_add_(0, eligible_sentence_keys, span_loss)
    sentence_span_count = torch.zeros_like(sentence_loss_sum)
    sentence_span_count.scatter_add_(
        0, eligible_sentence_keys, torch.ones_like(span_loss)
    )
    sentence_loss = sentence_loss_sum / sentence_span_count.clamp_min(1)
    loss = (
        sentence_loss.sum() / sentence_count.clamp_min(1).to(score.dtype)
        + _graph_zero(hidden)
    )
    if return_sentence_count:
        return loss, sentence_count
    return loss
