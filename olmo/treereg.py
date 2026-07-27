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
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
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

    sentence_losses = []
    for batch_idx in range(batch_size):
        row_sentence_ids = sentence_ids[batch_idx]
        ids = torch.unique(row_sentence_ids[row_sentence_ids >= 0]).detach().cpu().tolist()
        row_spans = spans[batch_idx]
        row_span_mask = span_mask[batch_idx].bool()

        for sentence_id in ids:
            positions = torch.nonzero(
                row_sentence_ids == int(sentence_id), as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                continue
            start = int(positions[0].item())
            end = int(positions[-1].item()) + 1
            if end - start != int(positions.numel()):
                raise ValueError(
                    f"sentence id {sentence_id} in batch row {batch_idx} is not contiguous"
                )
            sentence_hidden = circuit[batch_idx, start:end]
            sentence_word_starts = word_boundaries[batch_idx, start:end]
            if not bool(sentence_word_starts[0].item()):
                raise ValueError(
                    f"top-level tree {sentence_id} in batch row {batch_idx} "
                    "does not begin at a parser word boundary"
                )

            # Only spans wholly contained in this complete top-level tree can
            # contribute. Coordinates remain global until the final subtraction.
            left_global = row_spans[:, 0]
            right_global = row_spans[:, 2]
            contained = (
                row_span_mask
                & (left_global >= start)
                & (right_global < end)
                & (left_global <= right_global)
            )
            sentence_spans = row_spans[contained].long() - start
            span_losses = []

            for left, split, right in sentence_spans.detach().cpu().tolist():
                # Unary/n-ary degenerate spans use split==right. The upstream
                # target extractor skips them; never clamp them into a label.
                if split == right:
                    continue
                if not (0 <= left <= split < right < len(sentence_hidden)):
                    raise ValueError(
                        f"invalid TreeReg span {(left, split, right)} for "
                        f"top-level tree length {len(sentence_hidden)}"
                    )

                candidates = torch.arange(
                    left, right, device=hidden.device, dtype=torch.long
                )
                candidates = candidates[
                    sentence_word_starts[candidates + 1]
                ]
                # With fewer than two word-level choices the CE is identically
                # zero and the reference skips this constituent.
                if candidates.numel() < 2:
                    continue
                gold_matches = torch.nonzero(
                    candidates == split, as_tuple=False
                ).flatten()
                # This is an internal BPE-word constituent/split. The upstream
                # ``tree_to_parse_decisions`` deliberately omits it.
                if gold_matches.numel() == 0:
                    continue

                h_q = sentence_hidden[candidates]
                score = torch.zeros(
                    candidates.shape, device=hidden.device, dtype=sentence_hidden.dtype
                )
                if left > 0:
                    score = score + _orthogonal_norm(
                        h_q,
                        sentence_hidden[left - 1].expand_as(h_q),
                        eps,
                    )
                score = score + _orthogonal_norm(
                    sentence_hidden[candidates + 1],
                    h_q,
                    eps,
                )
                score = score + _orthogonal_norm(
                    sentence_hidden[right].expand_as(h_q),
                    h_q,
                    eps,
                )
                if right + 1 < len(sentence_hidden):
                    future = _orthogonal_norm(
                        sentence_hidden[right + 1].unsqueeze(0),
                        sentence_hidden[right].unsqueeze(0),
                        eps,
                    ).squeeze(0)
                    score = score + future

                span_losses.append(
                    F.cross_entropy(
                        score.unsqueeze(0),
                        gold_matches[:1].to(dtype=torch.long),
                        reduction="mean",
                    )
                )

            if span_losses:
                sentence_losses.append(torch.stack(span_losses).mean())
            else:
                # The reference includes short/no-decision sentences as zero in
                # its outer per-sentence macro average.
                sentence_losses.append(_graph_zero(sentence_hidden))

    sentence_count = len(sentence_losses)
    loss = (
        torch.stack(sentence_losses).mean()
        if sentence_losses
        else _graph_zero(hidden)
    )
    if return_sentence_count:
        return loss, sentence_count
    return loss
