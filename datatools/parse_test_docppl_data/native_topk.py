"""Label-marginalized K-best decoding for native binary and n-ary trees.

The released test-PPL parser spends its K-best budget in Benepar's labeled,
unary-collapsed, possibly n-ary tree space.  GPST and Pushdown discard those
labels and unary chains and consume a binary topology, so many labeled
candidates can project to the same native tree.

This module defines a direct proposal over *strict, unlabeled binary trees*:

1. Aggregate all non-null Benepar labels at each non-trivial word span using
   logsumexp (label marginalization) or max (a Viterbi ablation).
2. Run a one-state TreeCRF with a K-max semiring.  With one state, every CKY
   node is present, so each derivation is one full binary topology.
3. Return each topology as post-order ``(left, split, right)`` spans.  The split
   sequence is exactly the GPST merge order; the spans are also the common
   structural input from which Pushdown supervision can be derived.

For a fixed strict-binary topology, logsumexp is an exact marginal over its
non-null label assignments because Benepar's chart score is additive by span.
This is intentionally *not* claimed to be the exact pushforward of Benepar's
labeled/n-ary distribution: marginalizing the NULL nodes that encode n-ary
branching under a later right-binarization requires a richer quotient DP.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple, Union

import numpy as np
import torch
import torch_struct


Span = Tuple[int, int, int]
NarySpan = Tuple[int, int]
ScoreMode = Literal["logsumexp", "max"]
DecodeBackend = Literal["lazy", "torch_struct"]
ArrayLike = Union[np.ndarray, torch.Tensor]


class _TreeCRFNoValidationWarning(torch_struct.TreeCRF):
    """TreeCRF with the empty constraint declaration required by torch."""

    arg_constraints = {}


@dataclass(frozen=True)
class _PackedKBestCell:
    """Compact scores and backpointers for one CKY cell."""

    scores: np.ndarray
    splits: np.ndarray
    left_ranks: np.ndarray
    right_ranks: np.ndarray


@dataclass(frozen=True)
class _PackedNaryCell:
    """K-best forest/tree states for the canonical n-ary CKY derivation."""

    scores: np.ndarray
    splits: np.ndarray
    left_ranks: np.ndarray
    right_ranks: np.ndarray
    root_is_real: np.ndarray


@dataclass(frozen=True)
class NativeBinaryCandidate:
    """One unlabeled full-binary topology over an ordered terminal sequence."""

    score: float
    spans: Tuple[Span, ...]

    @property
    def merge_orders(self) -> Tuple[int, ...]:
        """Bottom-up gap indices consumed by GPST."""

        return tuple(split for _left, split, _right in self.spans)

    @property
    def constituent_spans(self) -> Tuple[Tuple[int, int], ...]:
        """Post-order inclusive constituent boundaries used by Pushdown."""

        return tuple((left, right) for left, _split, right in self.spans)


@dataclass(frozen=True)
class NativeNaryCandidate:
    """One unlabeled n-ary topology as post-order non-trivial constituents."""

    score: float
    spans: Tuple[NarySpan, ...]


@dataclass(frozen=True)
class AdaptedNativeNaryCandidates:
    """Model-specific views of one shared native n-ary candidate set.

    ``pushdown_spans`` contains only real n-ary constituents.  ``gpst_candidates``
    is the deterministic right-binarized BPE view.  Multiple n-ary candidates
    may map to one GPST trajectory; ``candidate_to_gpst`` and the grouped fields
    preserve that quotient without repeated model forwards or lost probability
    mass.
    """

    tokens: Tuple[int, ...]
    pushdown_spans: Tuple[Tuple[Span, ...], ...]
    gpst_candidates: Tuple[NativeBinaryCandidate, ...]
    candidate_to_gpst: Tuple[int, ...]
    gpst_source_slots: Tuple[int, ...]
    gpst_multiplicities: Tuple[int, ...]
    gpst_log_masses: Tuple[float, ...]


@dataclass(frozen=True)
class AdaptedNativeBinaryCandidates:
    """GPST BPE merge trajectories for strict-binary word candidates."""

    tokens: Tuple[int, ...]
    candidates: Tuple[NativeBinaryCandidate, ...]


@dataclass(frozen=True)
class PackedNativeBinarySentence:
    """Batch-ready fixed-slot representation for one tokenized sentence.

    The candidate axis always has ``slots`` rows, while the structural width is
    sentence-local (``num_tokens - 1``).  This avoids global max-length padding
    on disk.  Rows after ``valid_count`` repeat candidate zero as a structurally
    safe placeholder but carry ``-inf`` proposal scores and must be excluded
    from model scoring/marginalization via :attr:`candidate_mask`.
    """

    tokens: np.ndarray
    merge_orders: np.ndarray
    spans: np.ndarray
    proposal_scores: np.ndarray
    valid_count: int

    @property
    def slots(self) -> int:
        return int(self.merge_orders.shape[0])

    @property
    def candidate_mask(self) -> np.ndarray:
        return np.arange(self.slots) < self.valid_count

    @property
    def valid_merge_orders(self) -> np.ndarray:
        return self.merge_orders[: self.valid_count]

    @property
    def valid_spans(self) -> np.ndarray:
        return self.spans[: self.valid_count]


def capped_catalan(num_leaves: int, cap: int) -> int:
    """Return ``min(Catalan(num_leaves - 1), cap)`` without huge integers."""

    if num_leaves < 1:
        raise ValueError(f"num_leaves must be positive, got {num_leaves}")
    if cap < 1:
        raise ValueError(f"cap must be positive, got {cap}")

    # C_0 = 1 and C_n = C_(n-1) * 2(2n-1)/(n+1).
    value = 1
    for n in range(1, num_leaves):
        value = value * 2 * (2 * n - 1) // (n + 1)
        if value >= cap:
            return cap
    return value


def capped_little_schroeder(num_leaves: int, cap: int) -> int:
    """Return the capped number of ordered unlabeled n-ary trees.

    Internal nodes have at least two children and the root is retained. Counts
    for one through seven leaves are ``1, 1, 3, 11, 45, 197, 903``.
    """

    if num_leaves < 1:
        raise ValueError(f"num_leaves must be positive, got {num_leaves}")
    if cap < 1:
        raise ValueError(f"cap must be positive, got {cap}")
    if num_leaves <= 2:
        return 1
    previous_previous = 1  # s_0
    previous = 1  # s_1
    for n in range(2, num_leaves):
        value = ((6 * n - 3) * previous - (n - 2) * previous_previous) // (n + 1)
        if value >= cap:
            return cap
        previous_previous, previous = previous, value
    return previous


def aggregate_label_scores(
    labeled_scores: ArrayLike,
    *,
    mode: ScoreMode = "logsumexp",
    normalize_null: bool = True,
) -> ArrayLike:
    """Collapse Benepar's label axis into one real-constituent span score.

    Args:
        labeled_scores: ``(L, L, NT)`` Benepar chart.  Label index 0 is NULL.
        mode: ``"logsumexp"`` exactly marginalizes non-null labels for a fixed
            topology.  ``"max"`` retains only its best labeled realization.
        normalize_null: subtract the NULL logit before aggregating.  Stock
            Benepar fixes that logit to zero, but normalization makes the
            intended real-vs-null score explicit and supports compatible
            score charts whose NULL logit is not zero.

    Returns:
        An ``(L, L)`` score chart using the same array backend as the input.
        Diagonal values are returned but are ignored by native binary decode:
        leaf unary labels do not change the topology.
    """

    input_is_numpy = isinstance(labeled_scores, np.ndarray)
    scores = torch.as_tensor(labeled_scores)
    if scores.ndim != 3:
        raise ValueError(
            "labeled_scores must have shape (length, length, labels), "
            f"got {tuple(scores.shape)}"
        )
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(f"span chart must be square, got {tuple(scores.shape[:2])}")
    if scores.shape[-1] < 2:
        raise ValueError("labeled_scores must contain NULL plus at least one real label")
    if not scores.is_floating_point():
        scores = scores.to(torch.float32)

    real_scores = scores[..., 1:]
    if normalize_null:
        real_scores = real_scores - scores[..., :1]

    if mode == "logsumexp":
        native_scores = torch.logsumexp(real_scores, dim=-1)
    elif mode == "max":
        native_scores = real_scores.max(dim=-1).values
    else:
        raise ValueError(f"mode must be 'logsumexp' or 'max', got {mode!r}")

    if input_is_numpy:
        return native_scores.detach().cpu().numpy()
    return native_scores


def _postorder_spans_from_indicator(indicator: torch.Tensor) -> Tuple[Span, ...]:
    """Recover the unique post-order tree from a full-binary span indicator."""

    length = indicator.shape[0]
    used = {
        (left, right)
        for left in range(length)
        for right in range(left, length)
        if float(indicator[left, right]) > 0.5
    }
    expected_nodes = 2 * length - 1
    if len(used) != expected_nodes:
        raise ValueError(
            "TreeCRF path is not a full binary tree: "
            f"got {len(used)} active spans, expected {expected_nodes}"
        )
    if any((i, i) not in used for i in range(length)) or (0, length - 1) not in used:
        raise ValueError("TreeCRF path is missing a leaf or the sentence root")

    spans: List[Span] = []

    def visit(left: int, right: int) -> None:
        if left == right:
            return
        valid_splits = [
            split
            for split in range(left, right)
            if (left, split) in used and (split + 1, right) in used
        ]
        if len(valid_splits) != 1:
            raise ValueError(
                f"span {(left, right)} has {len(valid_splits)} valid child splits: "
                f"{valid_splits}"
            )
        split = valid_splits[0]
        visit(left, split)
        visit(split + 1, right)
        spans.append((left, split, right))

    visit(0, length - 1)
    if len(spans) != length - 1:
        raise ValueError(f"decoded {len(spans)} internal nodes, expected {length - 1}")
    return tuple(spans)


def validate_candidate(candidate: NativeBinaryCandidate, num_leaves: int) -> None:
    """Raise if a candidate is not a legal full-binary ordered topology."""

    if num_leaves < 1:
        raise ValueError(f"num_leaves must be positive, got {num_leaves}")
    if len(candidate.spans) != num_leaves - 1:
        raise ValueError(
            f"candidate has {len(candidate.spans)} spans, expected {num_leaves - 1}"
        )
    available = {(i, i) for i in range(num_leaves)}
    for left, split, right in candidate.spans:
        if not 0 <= left <= split < right < num_leaves:
            raise ValueError(f"invalid binary span {(left, split, right)}")
        left_child = (left, split)
        right_child = (split + 1, right)
        if left_child not in available or right_child not in available:
            raise ValueError(
                f"span {(left, split, right)} is not post-order or lacks a child"
            )
        if (left, right) in available:
            raise ValueError(f"duplicate constituent span {(left, right)}")
        available.add((left, right))
    expected_gaps = tuple(range(max(num_leaves - 1, 0)))
    if tuple(sorted(candidate.merge_orders)) != expected_gaps:
        raise ValueError(
            "merge orders are not a permutation of terminal gaps: "
            f"got {candidate.merge_orders}, expected {expected_gaps}"
        )
    if num_leaves > 1 and candidate.spans[-1][::2] != (0, num_leaves - 1):
        raise ValueError(f"last post-order span is not the root: {candidate.spans[-1]}")


def _validate_and_sort_candidates(
    candidates: List[NativeBinaryCandidate], num_candidates: int, length: int
) -> List[NativeBinaryCandidate]:
    seen = set()
    for candidate in candidates:
        spans = candidate.spans
        if spans in seen:
            raise ValueError("K-best decoder returned a duplicate binary topology")
        seen.add(spans)
        validate_candidate(candidate, length)

    # torch.topk does not define tie ordering.  A canonical signature makes the
    # persisted proposal deterministic without altering any non-tied ranking.
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.spans))
    if len(candidates) != num_candidates:
        raise ValueError(f"decoded {len(candidates)} candidates, expected {num_candidates}")
    return candidates


def _decode_with_torch_struct(
    active: torch.Tensor, num_candidates: int
) -> List[NativeBinaryCandidate]:
    length = active.shape[0]
    potentials = active.clone()
    diagonal = torch.arange(length, device=potentials.device)
    potentials[diagonal, diagonal] = 0
    potentials = potentials.unsqueeze(0).unsqueeze(-1)  # (1, L, L, 1 state)
    lengths = torch.tensor([length], dtype=torch.long, device=potentials.device)

    dist = _TreeCRFNoValidationWarning(potentials, lengths=lengths)
    paths = dist.topk(num_candidates)[:, 0, :, :, 0]
    candidates = []
    for path in paths:
        spans = _postorder_spans_from_indicator(path)
        score = sum(float(active[left, right]) for left, _split, right in spans)
        candidates.append(NativeBinaryCandidate(score=score, spans=spans))
    return candidates


def _decode_with_lazy_kbest(
    active: torch.Tensor, num_candidates: int
) -> List[NativeBinaryCandidate]:
    """Exact K-best CKY using lazy Cartesian-product enumeration.

    For each split, left and right candidate scores are already sorted.  A heap
    starts at pair ``(0, 0)`` and exposes only the next two neighboring pairs
    after a pop.  This produces the exact top-K pair sums without materializing
    K squared combinations, which is substantially faster for the one-state
    native grammar than the generic autograd-based KMax semiring on CPU.
    """

    scores_np = active.detach().cpu().numpy()
    length = scores_np.shape[0]
    chart: List[List[_PackedKBestCell | None]] = [
        [None for _ in range(length)] for _ in range(length)
    ]
    leaf_cell = _PackedKBestCell(
        scores=np.asarray([0.0], dtype=scores_np.dtype),
        splits=np.asarray([-1], dtype=np.int16),
        left_ranks=np.asarray([-1], dtype=np.int32),
        right_ranks=np.asarray([-1], dtype=np.int32),
    )
    for i in range(length):
        chart[i][i] = leaf_cell

    for width in range(2, length + 1):
        cell_k = capped_catalan(width, num_candidates)
        for left in range(length - width + 1):
            right = left + width - 1
            root_score = float(scores_np[left, right])
            heap = []
            visited = {}
            for split in range(left, right):
                left_cell = chart[left][split]
                right_cell = chart[split + 1][right]
                assert left_cell is not None and right_cell is not None
                visited[split] = {(0, 0)}
                score = root_score + float(left_cell.scores[0]) + float(right_cell.scores[0])
                heapq.heappush(heap, (-score, split, 0, 0))

            out_scores = []
            out_splits = []
            out_left_ranks = []
            out_right_ranks = []
            while heap and len(out_scores) < cell_k:
                neg_score, split, left_rank, right_rank = heapq.heappop(heap)
                out_scores.append(-neg_score)
                out_splits.append(split)
                out_left_ranks.append(left_rank)
                out_right_ranks.append(right_rank)

                left_cell = chart[left][split]
                right_cell = chart[split + 1][right]
                assert left_cell is not None and right_cell is not None
                neighbors = ((left_rank + 1, right_rank), (left_rank, right_rank + 1))
                for next_left_rank, next_right_rank in neighbors:
                    pair = (next_left_rank, next_right_rank)
                    if (
                        next_left_rank >= len(left_cell.scores)
                        or next_right_rank >= len(right_cell.scores)
                        or pair in visited[split]
                    ):
                        continue
                    visited[split].add(pair)
                    score = (
                        root_score
                        + float(left_cell.scores[next_left_rank])
                        + float(right_cell.scores[next_right_rank])
                    )
                    heapq.heappush(
                        heap,
                        (-score, split, next_left_rank, next_right_rank),
                    )

            if len(out_scores) != cell_k:
                raise ValueError(
                    f"lazy K-best produced {len(out_scores)} candidates for "
                    f"span {(left, right)}, expected {cell_k}"
                )
            chart[left][right] = _PackedKBestCell(
                scores=np.asarray(out_scores, dtype=scores_np.dtype),
                splits=np.asarray(out_splits, dtype=np.int16),
                left_ranks=np.asarray(out_left_ranks, dtype=np.int32),
                right_ranks=np.asarray(out_right_ranks, dtype=np.int32),
            )

    root_cell = chart[0][length - 1]
    assert root_cell is not None
    candidates = []
    for root_rank in range(num_candidates):
        spans: List[Span] = []

        def visit(left: int, right: int, rank: int) -> None:
            if left == right:
                return
            cell = chart[left][right]
            assert cell is not None
            split = int(cell.splits[rank])
            visit(left, split, int(cell.left_ranks[rank]))
            visit(split + 1, right, int(cell.right_ranks[rank]))
            spans.append((left, split, right))

        visit(0, length - 1, root_rank)
        candidates.append(
            NativeBinaryCandidate(score=float(root_cell.scores[root_rank]), spans=tuple(spans))
        )
    return candidates


def decode_native_binary_topk(
    span_scores: ArrayLike,
    *,
    k: int = 300,
    length: int | None = None,
    backend: DecodeBackend = "lazy",
) -> List[NativeBinaryCandidate]:
    """Decode the top-K unique strict-binary trees from unlabeled span scores.

    ``span_scores[i, j]`` is added exactly once when ``[i, j]`` is an internal
    node.  Diagonal scores are replaced by zero, because leaf unary choices are
    topology-independent.  The returned count is automatically capped by the
    number of ordered full-binary trees, ``Catalan(length - 1)``.

    The default ``lazy`` backend is an exact specialized K-best CKY that avoids
    K-squared Cartesian products.  ``torch_struct`` retains the generic KMax
    implementation as an independent correctness reference and optional GPU
    path.
    """

    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    scores = torch.as_tensor(span_scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"span_scores must be a square matrix, got {tuple(scores.shape)}")
    if not scores.is_floating_point():
        scores = scores.to(torch.float32)

    chart_length = scores.shape[0]
    length = chart_length if length is None else length
    if not 1 <= length <= chart_length:
        raise ValueError(f"length must be in [1, {chart_length}], got {length}")
    if backend not in ("lazy", "torch_struct"):
        raise ValueError(f"backend must be 'lazy' or 'torch_struct', got {backend!r}")

    if length == 1:
        return [NativeBinaryCandidate(score=0.0, spans=())]

    active = scores[:length, :length]
    internal_values = torch.stack(
        [active[left, right] for left in range(length) for right in range(left + 1, length)]
    )
    if not bool(torch.isfinite(internal_values).all()):
        raise ValueError("all valid non-trivial span scores must be finite")

    num_candidates = capped_catalan(length, k)
    if backend == "lazy":
        candidates = _decode_with_lazy_kbest(active, num_candidates)
    else:
        candidates = _decode_with_torch_struct(active, num_candidates)
    return _validate_and_sort_candidates(candidates, num_candidates, length)


def validate_nary_candidate(candidate: NativeNaryCandidate, num_leaves: int) -> None:
    """Raise if spans do not define one ordered, unary-free n-ary tree."""

    if num_leaves < 1:
        raise ValueError(f"num_leaves must be positive, got {num_leaves}")
    if num_leaves == 1:
        if candidate.spans:
            raise ValueError("a one-leaf native n-ary tree has no non-trivial spans")
        return
    if not candidate.spans or candidate.spans[-1] != (0, num_leaves - 1):
        raise ValueError("native n-ary candidate must end with the sentence root span")
    seen = set()
    for left, right in candidate.spans:
        if not 0 <= left < right < num_leaves:
            raise ValueError(f"invalid n-ary constituent span {(left, right)}")
        if (left, right) in seen:
            raise ValueError(f"duplicate n-ary constituent span {(left, right)}")
        # Constituents must be laminar: disjoint or nested, never crossing.
        for other_left, other_right in seen:
            crosses = (
                other_left < left <= other_right < right
                or left < other_left <= right < other_right
            )
            if crosses:
                raise ValueError(
                    f"crossing n-ary spans {(other_left, other_right)} and {(left, right)}"
                )
        seen.add((left, right))


def _combine_nary_forests(
    chart_a: List[List[_PackedNaryCell | None]],
    chart_b: List[List[_PackedNaryCell | None]],
    left: int,
    right: int,
    k: int,
    dtype,
) -> _PackedNaryCell:
    """Top-K canonical ``A(left) + B(right)`` forest concatenations."""

    heap = []
    visited = {}
    for split in range(left, right):
        left_cell = chart_a[left][split]
        right_cell = chart_b[split + 1][right]
        assert left_cell is not None and right_cell is not None
        visited[split] = {(0, 0)}
        score = float(left_cell.scores[0]) + float(right_cell.scores[0])
        heapq.heappush(heap, (-score, split, 0, 0))

    out_scores = []
    out_splits = []
    out_left_ranks = []
    out_right_ranks = []
    while heap and len(out_scores) < k:
        neg_score, split, left_rank, right_rank = heapq.heappop(heap)
        out_scores.append(-neg_score)
        out_splits.append(split)
        out_left_ranks.append(left_rank)
        out_right_ranks.append(right_rank)
        left_cell = chart_a[left][split]
        right_cell = chart_b[split + 1][right]
        assert left_cell is not None and right_cell is not None
        for next_left_rank, next_right_rank in (
            (left_rank + 1, right_rank),
            (left_rank, right_rank + 1),
        ):
            pair = (next_left_rank, next_right_rank)
            if (
                next_left_rank >= len(left_cell.scores)
                or next_right_rank >= len(right_cell.scores)
                or pair in visited[split]
            ):
                continue
            visited[split].add(pair)
            score = (
                float(left_cell.scores[next_left_rank])
                + float(right_cell.scores[next_right_rank])
            )
            heapq.heappush(heap, (-score, split, next_left_rank, next_right_rank))

    return _PackedNaryCell(
        scores=np.asarray(out_scores, dtype=dtype),
        splits=np.asarray(out_splits, dtype=np.int16),
        left_ranks=np.asarray(out_left_ranks, dtype=np.int32),
        right_ranks=np.asarray(out_right_ranks, dtype=np.int32),
        root_is_real=np.zeros(len(out_scores), dtype=np.bool_),
    )


def decode_native_nary_topk(
    span_scores: ArrayLike,
    *,
    k: int = 300,
    length: int | None = None,
) -> List[NativeNaryCandidate]:
    """Decode top-K unique unlabeled n-ary trees from real-span scores.

    The DP is Benepar's canonical binary encoding of arbitrary branching: state
    ``A`` may have a NULL root, while a non-trivial ``B`` root is real; every
    binary split combines ``A`` on the left with ``B`` on the right.  Removing
    NULL roots yields one unique n-ary tree per derivation.  Diagonal real labels
    are forbidden, which removes preterminal/unary alternatives.
    """

    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    scores = torch.as_tensor(span_scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"span_scores must be a square matrix, got {tuple(scores.shape)}")
    if not scores.is_floating_point():
        scores = scores.to(torch.float32)
    chart_length = scores.shape[0]
    length = chart_length if length is None else length
    if not 1 <= length <= chart_length:
        raise ValueError(f"length must be in [1, {chart_length}], got {length}")
    if length == 1:
        return [NativeNaryCandidate(score=0.0, spans=())]

    active = scores[:length, :length]
    internal_values = torch.stack(
        [active[left, right] for left in range(length) for right in range(left + 1, length)]
    )
    if not bool(torch.isfinite(internal_values).all()):
        raise ValueError("all valid non-trivial span scores must be finite")
    scores_np = active.detach().cpu().numpy()
    num_candidates = capped_little_schroeder(length, k)

    chart_a: List[List[_PackedNaryCell | None]] = [
        [None for _ in range(length)] for _ in range(length)
    ]
    chart_b: List[List[_PackedNaryCell | None]] = [
        [None for _ in range(length)] for _ in range(length)
    ]
    leaf = _PackedNaryCell(
        scores=np.asarray([0.0], dtype=scores_np.dtype),
        splits=np.asarray([-1], dtype=np.int16),
        left_ranks=np.asarray([-1], dtype=np.int32),
        right_ranks=np.asarray([-1], dtype=np.int32),
        root_is_real=np.asarray([False]),
    )
    for index in range(length):
        chart_a[index][index] = leaf
        chart_b[index][index] = leaf

    for width in range(2, length + 1):
        for left in range(length - width + 1):
            right = left + width - 1
            combined = _combine_nary_forests(
                chart_a, chart_b, left, right, num_candidates, scores_np.dtype
            )
            real_scores = combined.scores + scores_np[left, right]
            chart_b[left][right] = _PackedNaryCell(
                scores=real_scores,
                splits=combined.splits,
                left_ranks=combined.left_ranks,
                right_ranks=combined.right_ranks,
                root_is_real=np.ones(len(combined.scores), dtype=np.bool_),
            )

            # A may retain the combined forest as a NULL-root derivation or wrap
            # it in one real constituent. Merge the two sorted shifted lists.
            choices = []
            for rank, score in enumerate(combined.scores):
                choices.append((float(score), False, rank))
                choices.append((float(real_scores[rank]), True, rank))
            choices.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    int(combined.splits[item[2]]),
                    int(combined.left_ranks[item[2]]),
                    int(combined.right_ranks[item[2]]),
                )
            )
            choices = choices[:num_candidates]
            source_ranks = np.asarray([rank for _score, _real, rank in choices], dtype=np.int32)
            chart_a[left][right] = _PackedNaryCell(
                scores=np.asarray([score for score, _real, _rank in choices], dtype=scores_np.dtype),
                splits=combined.splits[source_ranks],
                left_ranks=combined.left_ranks[source_ranks],
                right_ranks=combined.right_ranks[source_ranks],
                root_is_real=np.asarray([real for _score, real, _rank in choices]),
            )

    root_cell = chart_b[0][length - 1]
    assert root_cell is not None
    if len(root_cell.scores) != num_candidates:
        raise ValueError(
            f"decoded {len(root_cell.scores)} n-ary roots, expected {num_candidates}"
        )
    candidates = []
    for root_rank in range(num_candidates):
        spans: List[NarySpan] = []

        def visit(state: str, left: int, right: int, rank: int) -> None:
            if left == right:
                return
            cell = chart_a[left][right] if state == "A" else chart_b[left][right]
            assert cell is not None
            split = int(cell.splits[rank])
            visit("A", left, split, int(cell.left_ranks[rank]))
            visit("B", split + 1, right, int(cell.right_ranks[rank]))
            if bool(cell.root_is_real[rank]):
                spans.append((left, right))

        visit("B", 0, length - 1, root_rank)
        candidate = NativeNaryCandidate(
            score=float(root_cell.scores[root_rank]), spans=tuple(spans)
        )
        validate_nary_candidate(candidate, length)
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: (-candidate.score, candidate.spans))
    if len({candidate.spans for candidate in candidates}) != num_candidates:
        raise ValueError("n-ary K-best decoder returned duplicate projected topologies")
    return candidates


def decode_labeled_scores_nary_topk(
    labeled_scores: ArrayLike,
    *,
    k: int = 300,
    length: int | None = None,
    mode: ScoreMode = "logsumexp",
    normalize_null: bool = True,
) -> List[NativeNaryCandidate]:
    """Aggregate Benepar labels and decode native n-ary top-K in one call."""

    native_scores = aggregate_label_scores(
        labeled_scores,
        mode=mode,
        normalize_null=normalize_null,
    )
    return decode_native_nary_topk(native_scores, k=k, length=length)


def decode_labeled_scores_topk(
    labeled_scores: ArrayLike,
    *,
    k: int = 300,
    length: int | None = None,
    mode: ScoreMode = "logsumexp",
    normalize_null: bool = True,
    backend: DecodeBackend = "lazy",
) -> List[NativeBinaryCandidate]:
    """Aggregate Benepar labels and decode native binary top-K in one call."""

    native_scores = aggregate_label_scores(
        labeled_scores,
        mode=mode,
        normalize_null=normalize_null,
    )
    return decode_native_binary_topk(native_scores, k=k, length=length, backend=backend)


def _word_piece_boundaries(
    word_piece_ids: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    if not word_piece_ids or any(not pieces for pieces in word_piece_ids):
        raise ValueError("word_piece_ids must contain one non-empty row per word")
    tokens: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    for pieces in word_piece_ids:
        starts.append(len(tokens))
        tokens.extend(map(int, pieces))
        ends.append(len(tokens) - 1)
    return tuple(tokens), tuple(starts), tuple(ends)


def nary_candidate_to_pushdown_spans(
    candidate: NativeNaryCandidate,
    word_piece_ids: Sequence[Sequence[int]],
    *,
    token_offset: int = 0,
) -> Tuple[Span, ...]:
    """Map real word-level n-ary constituents to BPE Pushdown spans.

    The third coordinate is deliberately degenerate (``split == right``):
    Pushdown consumes only the real constituent interval and must not see the
    artificial nodes introduced by GPST binarization.
    """

    validate_nary_candidate(candidate, len(word_piece_ids))
    _tokens, starts, ends = _word_piece_boundaries(word_piece_ids)
    return tuple(
        (
            starts[left] + token_offset,
            ends[right] + token_offset,
            ends[right] + token_offset,
        )
        for left, right in candidate.spans
    )


def nary_candidate_to_gpst_binary(
    candidate: NativeNaryCandidate,
    word_piece_ids: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, ...], NativeBinaryCandidate]:
    """Right-binarize one n-ary tree after expanding words into BPE siblings.

    BPE pieces belonging to a word are spliced into the word's parent rather
    than wrapped in a scored word constituent.  Right binarization then supplies
    GPST's required full merge trajectory.  Its artificial nodes affect only the
    representation, never the shared n-ary proposal score.
    """

    num_words = len(word_piece_ids)
    validate_nary_candidate(candidate, num_words)
    tokens, _starts, _ends = _word_piece_boundaries(word_piece_ids)
    real_spans = set(candidate.spans)

    def immediate_children(left: int, right: int) -> List[Tuple[int, int]]:
        contained = [
            span
            for span in real_spans
            if span != (left, right)
            and left <= span[0]
            and span[1] <= right
        ]
        return sorted(
            span
            for span in contained
            if not any(
                other != span
                and other != (left, right)
                and other[0] <= span[0]
                and span[1] <= other[1]
                for other in contained
            )
        )

    def make_children(left: int, right: int):
        children = []
        child_spans = immediate_children(left, right)
        cursor = left
        for child_left, child_right in child_spans:
            while cursor < child_left:
                children.extend(map(int, word_piece_ids[cursor]))
                cursor += 1
            children.append(make_node(child_left, child_right))
            cursor = child_right + 1
        while cursor <= right:
            children.extend(map(int, word_piece_ids[cursor]))
            cursor += 1
        return children

    def right_binary(children):
        if not children:
            raise ValueError("n-ary node has no expanded children")
        if len(children) == 1:
            return children[0]
        tail = children[-1]
        for child in reversed(children[1:-1]):
            tail = ("X|<", [child, tail])
        return ("X", [children[0], tail])

    def make_node(left: int, right: int):
        return right_binary(make_children(left, right))

    if num_words == 1:
        tree = right_binary(list(map(int, word_piece_ids[0])))
    else:
        tree = make_node(0, num_words - 1)

    if isinstance(tree, int):
        spans: Tuple[Span, ...] = ()
        leaves = [tree]
    else:
        from olmo.data.parse_align import tree_spans

        leaves, raw_spans = tree_spans(tree)
        spans = tuple(raw_spans)
    if tuple(leaves) != tokens:
        raise ValueError("GPST binarization changed terminal order")
    binary = NativeBinaryCandidate(score=candidate.score, spans=spans)
    validate_candidate(binary, len(tokens))
    return tokens, binary


def adapt_native_nary_candidates(
    candidates: Sequence[NativeNaryCandidate],
    word_piece_ids: Sequence[Sequence[int]],
    *,
    pushdown_token_offset: int = 0,
) -> AdaptedNativeNaryCandidates:
    """Create Pushdown and mass-preserving deduplicated GPST views."""

    if not candidates:
        raise ValueError("at least one native n-ary candidate is required")
    if any(candidates[i].score < candidates[i + 1].score for i in range(len(candidates) - 1)):
        raise ValueError("candidates must be sorted by non-increasing proposal score")

    pushdown = []
    unique_gpst: List[NativeBinaryCandidate] = []
    source_slots: List[int] = []
    grouped_scores: List[List[float]] = []
    positions = {}
    candidate_to_gpst = []
    reference_tokens = None
    for slot, candidate in enumerate(candidates):
        pushdown.append(
            nary_candidate_to_pushdown_spans(
                candidate, word_piece_ids, token_offset=pushdown_token_offset
            )
        )
        tokens, binary = nary_candidate_to_gpst_binary(candidate, word_piece_ids)
        if reference_tokens is None:
            reference_tokens = tokens
        elif tokens != reference_tokens:
            raise ValueError("native candidate adapters produced different terminals")
        position = positions.get(binary.spans)
        if position is None:
            position = len(unique_gpst)
            positions[binary.spans] = position
            unique_gpst.append(binary)
            source_slots.append(slot)
            grouped_scores.append([])
        candidate_to_gpst.append(position)
        grouped_scores[position].append(float(candidate.score))

    log_masses = []
    for scores in grouped_scores:
        maximum = max(scores)
        log_masses.append(maximum + float(np.log(sum(np.exp(x - maximum) for x in scores))))
    return AdaptedNativeNaryCandidates(
        tokens=reference_tokens or (),
        pushdown_spans=tuple(pushdown),
        gpst_candidates=tuple(unique_gpst),
        candidate_to_gpst=tuple(candidate_to_gpst),
        gpst_source_slots=tuple(source_slots),
        gpst_multiplicities=tuple(map(len, grouped_scores)),
        gpst_log_masses=tuple(log_masses),
    )


def adapt_native_binary_candidates_to_gpst(
    candidates: Sequence[NativeBinaryCandidate],
    word_piece_ids: Sequence[Sequence[int]],
) -> AdaptedNativeBinaryCandidates:
    """Expand strict-binary word trees into unique GPST BPE trajectories.

    Each parser word first receives one fixed, unscored BPE subtree. The CKY
    word tree is then materialized over those subtrees, so every selected CKY
    split maps exactly to its parser-word BPE boundary. Unlike projection from
    arbitrary n-ary candidates, this model-native mapping must be injective.
    """

    if not candidates:
        raise ValueError("at least one native binary candidate is required")
    if any(candidates[i].score < candidates[i + 1].score for i in range(len(candidates) - 1)):
        raise ValueError("candidates must be sorted by non-increasing proposal score")
    output = []
    reference_tokens = None
    signatures = set()
    for candidate in candidates:
        validate_candidate(candidate, len(word_piece_ids))
        tokens, expanded = expand_candidate_to_subwords(candidate, word_piece_ids)
        if reference_tokens is None:
            reference_tokens = tokens
        elif tokens != reference_tokens:
            raise ValueError("GPST binary adapters produced different terminals")
        if expanded.spans in signatures:
            raise ValueError("distinct binary word candidates collided after BPE expansion")
        signatures.add(expanded.spans)
        output.append(expanded)
    return AdaptedNativeBinaryCandidates(reference_tokens or (), tuple(output))


def candidate_to_tree(
    candidate: NativeBinaryCandidate,
    leaves: Sequence[object],
    *,
    label: str = "X",
):
    """Materialize a native candidate in the tuple tree format used by OLMo.

    This helper is intended for validation and adapters.  Production storage
    should retain the compact spans rather than duplicate labeled trees.
    """

    if len(leaves) != len(candidate.spans) + 1:
        raise ValueError(
            f"candidate expects {len(candidate.spans) + 1} leaves, got {len(leaves)}"
        )
    # Integer-like leaves are normalized for OLMo's tuple-tree parser.  Nested
    # tuple trees are retained so callers can substitute a fixed subword tree
    # for each parser-word leaf before materializing the word-level topology.
    nodes = {
        (i, i): leaf if isinstance(leaf, tuple) else int(leaf)
        for i, leaf in enumerate(leaves)
    }
    for left, split, right in candidate.spans:
        left_child = nodes.get((left, split))
        right_child = nodes.get((split + 1, right))
        if left_child is None or right_child is None:
            raise ValueError(f"spans are not in post-order at {(left, split, right)}")
        nodes[(left, right)] = (label, [left_child, right_child])
    return nodes[(0, len(leaves) - 1)]


def expand_candidate_to_subwords(
    candidate: NativeBinaryCandidate,
    word_piece_ids: Sequence[Sequence[int]],
    *,
    label: str = "X",
) -> Tuple[Tuple[int, ...], NativeBinaryCandidate]:
    """Expand a word-level candidate into one strict-binary subword tree.

    Every parser word must already have a fixed, non-empty tokenization.  Its
    pieces are grouped with a deterministic right-recursive subtree, matching
    the repository's default right binarization.  This mapping is independent
    of the sentence-level candidate, so all candidates retain identical token
    IDs and word boundaries while differing only in their word topology.
    """

    if len(word_piece_ids) != len(candidate.spans) + 1:
        raise ValueError(
            f"candidate expects {len(candidate.spans) + 1} words, "
            f"got {len(word_piece_ids)} tokenized words"
        )
    if any(len(pieces) == 0 for pieces in word_piece_ids):
        raise ValueError("every parser word must map to at least one subword token")

    def right_recursive_piece_tree(pieces: Sequence[int]):
        tail = int(pieces[-1])
        for piece in reversed(pieces[:-1]):
            tail = (f"{label}|<", [int(piece), tail])
        return tail

    word_subtrees = [right_recursive_piece_tree(pieces) for pieces in word_piece_ids]
    expanded_tree = candidate_to_tree(candidate, word_subtrees, label=label)

    # Reuse the canonical OLMo traversal so adapter output is byte-for-byte
    # aligned with the GPST/Pushdown preprocessing representation.
    from olmo.data.parse_align import tree_spans

    leaves, spans = tree_spans(expanded_tree)
    expanded = NativeBinaryCandidate(score=candidate.score, spans=tuple(spans))
    validate_candidate(expanded, len(leaves))
    return tuple(leaves), expanded


def pack_candidate_slots(
    tokens: Sequence[int],
    candidates: Sequence[NativeBinaryCandidate],
    *,
    slots: int = 300,
) -> PackedNativeBinarySentence:
    """Pack unique candidates into a fixed candidate axis without mass padding.

    This is the in-memory contract for the disk layout used by fast evaluators:

    * GPST reads ``merge_orders`` directly as ``(slots, T-1)``;
    * Pushdown reads ``spans`` directly as ``(slots, T-1, 3)``;
    * both share one terminal row and one ``valid_count``;
    * evaluation slices ``[:valid_count]`` (or applies ``candidate_mask``), so
      physical padding is never interpreted as repeated probability mass.
    """

    if slots < 1:
        raise ValueError(f"slots must be positive, got {slots}")
    if not candidates:
        raise ValueError("at least one native candidate is required")
    if len(candidates) > slots:
        raise ValueError(f"got {len(candidates)} candidates for only {slots} slots")

    token_values = np.asarray(tokens)
    if token_values.ndim != 1 or token_values.size < 1:
        raise ValueError("tokens must be a non-empty one-dimensional sequence")
    if np.any(token_values < 0):
        raise ValueError("token ids must be non-negative")
    token_dtype = np.uint16 if int(token_values.max()) <= np.iinfo(np.uint16).max else np.uint32
    packed_tokens = token_values.astype(token_dtype, copy=False)
    num_tokens = len(packed_tokens)
    if num_tokens > np.iinfo(np.int16).max:
        raise ValueError("native structure indices require fewer than 32768 tokens")

    signatures = set()
    for candidate in candidates:
        validate_candidate(candidate, num_tokens)
        if candidate.spans in signatures:
            raise ValueError("candidates must be unique before fixed-slot packing")
        signatures.add(candidate.spans)
    if any(candidates[i].score < candidates[i + 1].score for i in range(len(candidates) - 1)):
        raise ValueError("candidates must be sorted by non-increasing proposal score")

    width = num_tokens - 1
    merge_orders = np.empty((slots, width), dtype=np.int16)
    spans = np.empty((slots, width, 3), dtype=np.int16)
    proposal_scores = np.full(slots, -np.inf, dtype=np.float32)

    # Safe physical padding: a valid row avoids undefined indices if a caller
    # accidentally materializes all slots, while -inf/mask preserves semantics.
    padding_candidate = candidates[0]
    for slot in range(slots):
        candidate = candidates[slot] if slot < len(candidates) else padding_candidate
        if width:
            merge_orders[slot] = candidate.merge_orders
            spans[slot] = candidate.spans
        if slot < len(candidates):
            proposal_scores[slot] = candidate.score

    return PackedNativeBinarySentence(
        tokens=packed_tokens,
        merge_orders=merge_orders,
        spans=spans,
        proposal_scores=proposal_scores,
        valid_count=len(candidates),
    )
