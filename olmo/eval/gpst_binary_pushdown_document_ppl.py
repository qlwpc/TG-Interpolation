"""Document PPL for v2 GPST strict-binary candidates scored by Pushdown.

This protocol deliberately uses the *independent GPST* candidate axis from the
native-model-topk v2 corpus.  Every valid merge-order row is converted to
terminal-coordinate binary spans, candidate 0 is committed as document
history, and the supplied support is marginalized with a truncated **sum**
(never a mean over candidate count).

The joint metric supports both the historical evaluator-v1 attachment
probabilities (illegal stack actions are removed before softmax) and the
training-objective evaluator-v2 probabilities (the complete sentence-causal
row is normalized).  The second metric scores only the terminal probabilities
along candidate 0's structured trajectory.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from olmo.attachment import (
    ATTACHMENT_NORMALIZATION_V1,
    canonical_attachment_normalization,
)
from olmo.data.parse_align import TreeVocab
from olmo.eval.native_model_topk_corpus import NativeModelTopKCorpus
from olmo.eval.pushdown_document_ppl import (
    PushdownCandidateScores,
    PushdownGoldCandidate,
    _compose,
    _drop_leading_bos,
    _trim_prefix,
)
from olmo.model import OLMo


PUSHDOWN_GPST_BINARY_PROTOCOL_VERSION = 1


def gpst_merge_orders_to_pushdown_spans(
    merge_orders: np.ndarray,
    content_start: int,
    content_length: int,
    *,
    validate: bool = True,
) -> np.ndarray:
    """Convert batched strict-binary gap merge orders to postorder spans.

    ``merge_orders[k]`` is a permutation of the ``T-1`` original leaf gaps.
    At each merge gap ``s``, the active constituent ending at ``s`` combines
    with the active constituent starting at ``s+1``.  The result is emitted as
    Pushdown's ``(left, split, right)`` triple in terminal-token coordinates.

    Runtime is ``O(K*T)`` and candidates are converted together; no tree object
    or bracket-string round trip is involved.
    """
    orders = np.asarray(merge_orders)
    if orders.ndim != 2:
        raise ValueError(f"merge_orders must have shape (K,T-1), got {orders.shape}")
    if content_length <= 0 or content_start < 0:
        raise ValueError(
            f"invalid content interval start={content_start}, length={content_length}"
        )
    expected_width = content_length - 1
    if orders.shape[1] != expected_width:
        raise ValueError(
            f"merge-order width {orders.shape[1]} does not match T-1={expected_width}"
        )
    if orders.shape[0] <= 0:
        raise ValueError("at least one valid GPST candidate is required")
    orders64 = orders.astype(np.int64, copy=False)
    if expected_width and bool(np.any((orders64 < 0) | (orders64 >= expected_width))):
        raise ValueError("GPST merge gap is outside 0..T-2")
    if validate and expected_width:
        expected = np.arange(expected_width, dtype=np.int64)
        if not bool(np.all(np.sort(orders64, axis=1) == expected[None, :])):
            raise ValueError(
                "each GPST merge-order row must be a permutation of 0..T-2"
            )
    spans = np.empty((orders.shape[0], expected_width, 3), dtype=np.int64)
    if expected_width == 0:
        return spans

    candidate_rows = np.arange(orders.shape[0], dtype=np.int64)
    leaf_positions = np.arange(content_length, dtype=np.int64)
    left_by_right = np.broadcast_to(
        leaf_positions, (orders.shape[0], content_length)
    ).copy()
    right_by_left = left_by_right.copy()
    for step in range(expected_width):
        split = orders64[:, step]
        left = left_by_right[candidate_rows, split]
        right = right_by_left[candidate_rows, split + 1]
        spans[:, step, 0] = left + content_start
        spans[:, step, 1] = split + content_start
        spans[:, step, 2] = right + content_start
        left_by_right[candidate_rows, right] = left
        right_by_left[candidate_rows, left] = right

    # ``validate=False`` skips only the expensive per-row permutation sort.
    # Coordinate safety and the final-root invariant are part of conversion,
    # not optional input auditing.
    if not bool(
        np.all(spans[:, -1, 0] == content_start)
        and np.all(spans[:, -1, 2] == content_start + content_length - 1)
    ):
        raise ValueError("merge-order row did not finish in one full-sentence root")
    return spans


def _binary_attachment_actions(
    spans: np.ndarray,
    token_length: int,
    content_left: int,
    content_right: int,
) -> Tuple[np.ndarray, List[List[Tuple[int, ...]]]]:
    """Derive stack targets/legal sets from strict-binary closing counts.

    At content token ``k``, let ``c_k`` be the number of binary constituents
    that close there.  Before reducing, the legal v1 actions are the new token
    followed by the open stack from top to bottom; the gold target is exactly
    action ``c_k``.  This is the literal transition in
    :func:`derive_gold_attachment_actions`, specialized to complete binary
    trees and avoids rebuilding closure dictionaries through Torch K times.
    """
    if spans.ndim != 3 or spans.shape[2] != 3:
        raise ValueError(f"spans must have shape (K,M,3), got {spans.shape}")
    content_length = content_right - content_left
    if content_length <= 0 or not 0 <= content_left < content_right <= token_length:
        raise ValueError("invalid content interval for attachment derivation")
    candidate_count = spans.shape[0]
    closing_counts = np.zeros((candidate_count, content_length), dtype=np.int16)
    if spans.shape[1]:
        right = spans[:, :, 2] - content_left
        rows = np.broadcast_to(
            np.arange(candidate_count, dtype=np.int64)[:, None], right.shape
        )
        np.add.at(closing_counts, (rows, right), 1)

    targets = np.full((candidate_count, token_length), -1, dtype=np.int64)
    all_legal: List[List[Tuple[int, ...]]] = []
    for candidate_index in range(candidate_count):
        stack: List[int] = []
        legal_row: List[Tuple[int, ...]] = [() for _ in range(token_length)]
        for local_token in range(content_length):
            global_token = content_left + local_token
            keys_local = (local_token, *reversed(stack))
            reduce_count = int(closing_counts[candidate_index, local_token])
            if reduce_count >= len(keys_local):
                raise ValueError(
                    "binary spans request more reductions than the open stack"
                )
            targets[candidate_index, global_token] = (
                content_left + keys_local[reduce_count]
            )
            legal_row[global_token] = tuple(content_left + key for key in keys_local)
            if reduce_count:
                del stack[-reduce_count:]
            stack.append(local_token)
        if stack != [content_length - 1]:
            raise ValueError("binary attachment transitions did not close one root")
        all_legal.append(legal_row)
    return targets, all_legal


class NativeGPSTBinaryPushdownCorpus:
    """Expose valid GPST rows in the checkpoint's fixed-word-atom binary form."""

    structure_source = "v2_gpst_strict_binary_to_pushdown"
    source_candidate_axis = "gpst"
    binarization = "direct_strict_binary_cky_with_fixed_word_bpe_atoms"
    deduplicated_binary_structures = True

    def __init__(
        self,
        native_path: str,
        tokenizer_path: str,
        max_sentences: Optional[int] = None,
        start_document: int = 0,
        end_document: Optional[int] = None,
        validate_merge_orders: bool = True,
    ) -> None:
        self.native = NativeModelTopKCorpus(native_path)
        self.vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
        self.samples_per_sentence = int(self.native.manifest["candidate_slots"])
        self.start_document = int(start_document)
        total_documents = int(self.native.manifest["document_count"])
        self.end_document = (
            total_documents if end_document is None else int(end_document)
        )
        self.start_sentence, end_sentence = self.native.document_sentence_range(
            self.start_document, self.end_document
        )
        self.num_sentences = end_sentence - self.start_sentence
        if max_sentences is not None:
            if max_sentences < 0:
                raise ValueError("max_sentences cannot be negative")
            self.num_sentences = min(self.num_sentences, int(max_sentences))
        self.validate_merge_orders = bool(validate_merge_orders)

    def __len__(self) -> int:
        return self.num_sentences

    def sentence_candidates(
        self, sentence_index: int
    ) -> Tuple[PushdownGoldCandidate, ...]:
        if not 0 <= sentence_index < len(self):
            raise IndexError(sentence_index)
        row = self.native.sentence(self.start_sentence + sentence_index)
        return self._candidates_from_row(row)

    def _candidates_from_row(self, row) -> Tuple[PushdownGoldCandidate, ...]:
        valid_count = int(row.gpst_valid_count)
        if valid_count <= 0 or row.gpst_merge_orders.shape[0] != valid_count:
            raise ValueError(
                f"sentence {row.global_sentence_id} has invalid GPST count {valid_count}"
            )
        content_left, content_right = map(int, row.content_bounds)
        tokens = tuple(map(int, row.tokens))
        if not 0 <= content_left < content_right <= len(tokens):
            raise ValueError(
                f"sentence {row.global_sentence_id} has invalid content bounds "
                f"{row.content_bounds} for {len(tokens)} tokens"
            )
        spans = gpst_merge_orders_to_pushdown_spans(
            row.gpst_merge_orders,
            content_left,
            content_right - content_left,
            validate=self.validate_merge_orders,
        )
        sentence_ids = tuple(
            0 if content_left <= index < content_right else -1
            for index in range(len(tokens))
        )
        targets, legal = _binary_attachment_actions(
            spans,
            len(tokens),
            content_left,
            content_right,
        )
        candidates: List[PushdownGoldCandidate] = []
        for candidate_index in range(valid_count):
            candidates.append(
                PushdownGoldCandidate(
                    tokens=tokens,
                    spans=tuple(
                        tuple(map(int, span)) for span in spans[candidate_index]
                    ),
                    sentence_ids=sentence_ids,
                    attachment_targets=tuple(
                        map(int, targets[candidate_index].tolist())
                    ),
                    legal_attachment_targets=tuple(
                        tuple(map(int, keys)) for keys in legal[candidate_index]
                    ),
                )
            )
        return tuple(candidates)

    def _load_sentence(
        self, sentence_index: int
    ) -> Tuple[int, Tuple[PushdownGoldCandidate, ...]]:
        row = self.native.sentence(self.start_sentence + sentence_index)
        return row.document_id, self._candidates_from_row(row)

    def __iter__(
        self,
    ) -> Iterator[Tuple[int, Tuple[PushdownGoldCandidate, ...]]]:
        for index in range(len(self)):
            yield self._load_sentence(index)

    def iter_prefetched(
        self, queue_depth: int = 2
    ) -> Iterator[Tuple[int, Tuple[PushdownGoldCandidate, ...]]]:
        """Convert upcoming sentences while the GPU scores the current one."""
        if queue_depth <= 0:
            yield from self
            return
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpst-binary"
        ) as pool:
            futures = {}
            next_submit = 0
            while next_submit < min(queue_depth, len(self)):
                futures[next_submit] = pool.submit(self._load_sentence, next_submit)
                next_submit += 1
            for index in range(len(self)):
                future = futures.pop(index)
                if next_submit < len(self):
                    futures[next_submit] = pool.submit(self._load_sentence, next_submit)
                    next_submit += 1
                yield future.result()


def right_binarize_native_nary_spans(
    nary_spans: np.ndarray,
    span_counts: np.ndarray,
    content_left: int,
    content_right: int,
    *,
    deduplicate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Right-binarize native n-ary spans after splicing their BPE leaves.

    For every terminal gap, the smallest real n-ary constituent crossing that
    gap is its immediate parent.  Right binarization contributes one binary
    node for each such parent/child boundary, proceeding from the parent's
    rightmost boundary to its leftmost boundary.  The resulting *set* of
    ``(left, split, right)`` spans is independent of post-order serialization;
    rows here are stored canonically by ascending split so candidate equality
    and deduplication are deterministic.

    Multiple distinct n-ary candidates can map to the same binary topology.
    When ``deduplicate`` is true, only the first (highest proposal-ranked) row
    is retained because the Pushdown latent variable is the binary topology,
    not its n-ary provenance.
    """
    spans = np.asarray(nary_spans)
    counts = np.asarray(span_counts)
    if spans.ndim != 3 or spans.shape[2] != 3:
        raise ValueError(f"nary_spans must have shape (K,M,3), got {spans.shape}")
    if counts.shape != (spans.shape[0],):
        raise ValueError("span_counts must contain one count per candidate")
    if spans.shape[0] <= 0:
        raise ValueError("at least one n-ary candidate is required")
    if not 0 <= content_left < content_right:
        raise ValueError("invalid content interval")
    if bool(np.any((counts < 0) | (counts > spans.shape[1]))):
        raise ValueError("n-ary span count is outside the padded row width")

    candidate_count = spans.shape[0]
    gap_count = content_right - content_left - 1
    if gap_count == 0:
        binary = np.empty((candidate_count, 0, 3), dtype=np.int64)
    else:
        left = spans[:, :, 0].astype(np.int64, copy=False)
        right = spans[:, :, 2].astype(np.int64, copy=False)
        slots = np.arange(spans.shape[1], dtype=np.int64)[None, :]
        valid = slots < counts.astype(np.int64, copy=False)[:, None]
        if bool(
            np.any(
                valid
                & ((left < content_left) | (right >= content_right) | (left >= right))
            )
        ):
            raise ValueError("native n-ary span is outside the content interval")

        absolute_gaps = content_left + np.arange(gap_count, dtype=np.int64)
        if spans.shape[1] == 0:
            # One parser word can still contain several BPE terminals.  This
            # deliberately treats its pieces as siblings under an implicit
            # root.  It is a useful spliced-BPE diagnostic, but differs from
            # checkpoint pretraining, which retained the parser preterminal as
            # a fixed per-word BPE subtree.
            parent_left = np.full(
                (candidate_count, gap_count), content_left, dtype=np.int64
            )
            parent_right = np.full(
                (candidate_count, gap_count), content_right - 1, dtype=np.int64
            )
        else:
            gap_grid = absolute_gaps[None, None, :]
            crosses = (
                valid[:, :, None]
                & (left[:, :, None] <= gap_grid)
                & (right[:, :, None] > gap_grid)
            )
            # The narrowest crossing real constituent is the immediate n-ary
            # parent. An implicit full-content root handles a missing-root row
            # and is never selected when a narrower real parent exists.
            widths = np.where(
                crosses,
                right[:, :, None] - left[:, :, None],
                content_right - content_left + 1,
            )
            parent_slot = widths.argmin(axis=1)
            has_real_parent = crosses.any(axis=1)
            parent_left = np.take_along_axis(left, parent_slot, axis=1)
            parent_right = np.take_along_axis(right, parent_slot, axis=1)
            parent_left = np.where(has_real_parent, parent_left, content_left)
            parent_right = np.where(has_real_parent, parent_right, content_right - 1)

        # Gaps assigned to one parent are its immediate child boundaries.  The
        # left edge of the child ending at gap g is the preceding boundary + 1,
        # or the parent left edge for its first child.
        parent_signature = parent_left * (content_right + 1) + parent_right
        previous_gap_grid = np.arange(gap_count, dtype=np.int64)[None, None, :]
        current_gap_grid = np.arange(gap_count, dtype=np.int64)[None, :, None]
        same_parent = parent_signature[:, :, None] == parent_signature[:, None, :]
        previous_gap = np.where(
            same_parent & (previous_gap_grid < current_gap_grid),
            previous_gap_grid,
            -1,
        ).max(axis=2)
        binary_left = np.where(
            previous_gap >= 0,
            content_left + previous_gap + 1,
            parent_left,
        )
        binary = np.stack(
            (
                binary_left,
                np.broadcast_to(absolute_gaps[None, :], (candidate_count, gap_count)),
                parent_right,
            ),
            axis=2,
        ).astype(np.int64, copy=False)

    source_indices = np.arange(candidate_count, dtype=np.int64)
    if deduplicate and candidate_count > 1:
        # np.unique sorts rows; re-sort first-occurrence indices to preserve the
        # source proposal ranking and therefore candidate-0 history semantics.
        _unique, first = np.unique(binary, axis=0, return_index=True)
        source_indices = np.sort(first.astype(np.int64, copy=False))
        binary = binary[source_indices]
    return binary, source_indices


def gpst_merge_orders_to_spliced_bpe_spans(
    merge_orders: np.ndarray,
    word_starts: np.ndarray,
    content_left: int,
    content_right: int,
    *,
    deduplicate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-expand GPST word topologies after splicing BPEs into their parent.

    This matches :func:`nary_candidate_to_gpst_binary`, not the checkpoint's
    precomputed Pushdown representation.  The latter retains each parser
    preterminal and is represented directly by the stored GPST trajectory.
    """
    orders = np.asarray(merge_orders)
    starts = np.asarray(word_starts, dtype=np.int64)
    if orders.ndim != 2:
        raise ValueError("merge_orders must have shape (K,T-1)")
    content_length = content_right - content_left
    if (
        starts.ndim != 1
        or starts.size == 0
        or int(starts[0]) != 0
        or bool(np.any(starts[1:] <= starts[:-1]))
        or int(starts[-1]) >= content_length
    ):
        raise ValueError("invalid recovered word starts")
    word_count = int(starts.size)
    ends = np.concatenate((starts[1:] - 1, np.asarray([content_length - 1])))
    word_gap_count = word_count - 1
    if orders.shape[1] != content_length - 1:
        raise ValueError("GPST merge width does not match the BPE content length")
    if word_gap_count:
        cross_word_gaps = ends[:-1]
        gap_to_word = np.full(content_length - 1, -1, dtype=np.int64)
        gap_to_word[cross_word_gaps] = np.arange(word_gap_count)
        mapped = gap_to_word[orders]
        is_word_gap = mapped >= 0
        if not bool(np.all(is_word_gap.sum(axis=1) == word_gap_count)):
            raise ValueError("stored GPST trajectory violates recovered word atoms")
        word_orders = mapped[is_word_gap].reshape(orders.shape[0], word_gap_count)
    else:
        word_orders = np.empty((orders.shape[0], 0), dtype=np.int64)

    word_spans = gpst_merge_orders_to_pushdown_spans(
        word_orders,
        content_start=0,
        content_length=word_count,
        validate=False,
    )
    mapped_spans = np.empty_like(word_spans)
    if word_gap_count:
        mapped_spans[:, :, 0] = content_left + starts[word_spans[:, :, 0]]
        mapped_spans[:, :, 1] = mapped_spans[:, :, 2] = (
            content_left + ends[word_spans[:, :, 2]]
        )
    return right_binarize_native_nary_spans(
        mapped_spans,
        np.full(orders.shape[0], word_gap_count, dtype=np.int64),
        content_left,
        content_right,
        deduplicate=deduplicate,
    )


def right_binarize_native_nary_spans_with_word_atoms(
    nary_spans: np.ndarray,
    span_counts: np.ndarray,
    word_starts: np.ndarray,
    content_left: int,
    content_right: int,
    *,
    deduplicate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert native word-level n-ary candidates to checkpoint-training CNF.

    The checkpoint's actual preprocessing keeps every parser preterminal.
    Consequently a multi-BPE word is first represented by one fixed,
    right-recursive subtree; deterministic right-CNF is then applied to the
    word-level constituency tree.  This is the same representation produced by
    ``expand_candidate_to_subwords`` for the direct GPST axis.

    Native Pushdown candidates store only real word-level constituents mapped
    to BPE intervals.  We recover those word intervals, right-binarize them in
    word coordinates, map the binary word nodes back to BPE coordinates, and
    add the candidate-independent fixed word-atom nodes.  Every final row has
    exactly one binary node per BPE gap and is stored in split-gap order.
    """
    spans = np.asarray(nary_spans)
    counts = np.asarray(span_counts)
    starts = np.asarray(word_starts, dtype=np.int64)
    if spans.ndim != 3 or spans.shape[2] != 3:
        raise ValueError(f"nary_spans must have shape (K,M,3), got {spans.shape}")
    if counts.shape != (spans.shape[0],):
        raise ValueError("span_counts must contain one count per candidate")
    content_length = content_right - content_left
    if (
        content_length <= 0
        or starts.ndim != 1
        or starts.size == 0
        or int(starts[0]) != 0
        or bool(np.any(starts[1:] <= starts[:-1]))
        or int(starts[-1]) >= content_length
    ):
        raise ValueError("invalid recovered word starts or content interval")
    if bool(np.any((counts < 0) | (counts > spans.shape[1]))):
        raise ValueError("n-ary span count is outside the padded row width")

    word_count = int(starts.size)
    ends = np.concatenate((starts[1:] - 1, np.asarray([content_length - 1])))
    absolute_starts = content_left + starts
    absolute_ends = content_left + ends
    word_nary = np.full_like(spans, -1, dtype=np.int64)
    if spans.shape[1]:
        slots = np.arange(spans.shape[1], dtype=np.int64)[None, :]
        valid = slots < counts.astype(np.int64, copy=False)[:, None]
        span_left = spans[:, :, 0].astype(np.int64, copy=False)
        span_right = spans[:, :, 2].astype(np.int64, copy=False)
        word_left = np.searchsorted(absolute_starts, span_left)
        word_right = np.searchsorted(absolute_ends, span_right)
        safe_left = np.minimum(word_left, word_count - 1)
        safe_right = np.minimum(word_right, word_count - 1)
        aligned = (
            (word_left < word_count)
            & (word_right < word_count)
            & (absolute_starts[safe_left] == span_left)
            & (absolute_ends[safe_right] == span_right)
            & (word_left < word_right)
        )
        if bool(np.any(valid & ~aligned)):
            raise ValueError(
                "native n-ary span is not aligned to recovered word boundaries"
            )
        word_nary[:, :, 0] = np.where(valid, word_left, -1)
        word_nary[:, :, 1] = np.where(valid, word_right, -1)
        word_nary[:, :, 2] = np.where(valid, word_right, -1)

    word_binary, source_indices = right_binarize_native_nary_spans(
        word_nary,
        counts,
        0,
        word_count,
        deduplicate=deduplicate,
    )
    candidate_count = word_binary.shape[0]
    gap_count = content_length - 1
    binary = np.full((candidate_count, gap_count, 3), -1, dtype=np.int64)

    # A retained preterminal containing pieces [start..end] is binarized as
    # (start, (start+1, (..., end))).  Thus every internal gap g contributes
    # the node (g, g, end).
    for word_start, word_end in zip(absolute_starts, absolute_ends):
        gaps = np.arange(int(word_start), int(word_end), dtype=np.int64)
        if gaps.size:
            word_nodes = np.stack(
                (gaps, gaps, np.full_like(gaps, int(word_end))), axis=1
            )
            binary[:, gaps - content_left] = word_nodes[None, :, :]

    # Word-level binary nodes split at the final BPE of their left child.
    if word_binary.shape[1]:
        binary_left = absolute_starts[word_binary[:, :, 0]]
        binary_split = absolute_ends[word_binary[:, :, 1]]
        binary_right = absolute_ends[word_binary[:, :, 2]]
        candidate_rows = np.broadcast_to(
            np.arange(candidate_count, dtype=np.int64)[:, None],
            binary_split.shape,
        )
        binary[candidate_rows, binary_split - content_left] = np.stack(
            (binary_left, binary_split, binary_right), axis=2
        )
    if gap_count and bool(np.any(binary < 0)):
        raise AssertionError("training-CNF conversion did not fill every BPE gap")
    return binary, source_indices


class NativeNaryRightBinarizedPushdownCorpus(NativeGPSTBinaryPushdownCorpus):
    """Pushdown n-ary top-K support with diagnostic spliced-BPE right-CNF."""

    structure_source = "v2_pushdown_nary_topk_spliced_bpe_right_binarized"
    source_candidate_axis = "pushdown"
    binarization = "deterministic_right_cnf_after_bpe_splicing"
    deduplicated_binary_structures = True

    def _candidates_from_row(self, row) -> Tuple[PushdownGoldCandidate, ...]:
        valid_count = int(row.pushdown_valid_count)
        if valid_count <= 0 or row.pushdown_spans.shape[0] != valid_count:
            raise ValueError(
                f"sentence {row.global_sentence_id} has invalid Pushdown count "
                f"{valid_count}"
            )
        content_left, content_right = map(int, row.content_bounds)
        tokens = tuple(map(int, row.tokens))
        if not 0 <= content_left < content_right <= len(tokens):
            raise ValueError(
                f"sentence {row.global_sentence_id} has invalid content bounds "
                f"{row.content_bounds} for {len(tokens)} tokens"
            )
        binary_spans, _source_indices = right_binarize_native_nary_spans(
            row.pushdown_spans,
            row.pushdown_span_counts,
            content_left,
            content_right,
            deduplicate=True,
        )
        sentence_ids = tuple(
            0 if content_left <= index < content_right else -1
            for index in range(len(tokens))
        )
        targets, legal = _binary_attachment_actions(
            binary_spans,
            len(tokens),
            content_left,
            content_right,
        )
        return tuple(
            PushdownGoldCandidate(
                tokens=tokens,
                spans=tuple(tuple(map(int, span)) for span in candidate_spans),
                sentence_ids=sentence_ids,
                attachment_targets=tuple(map(int, targets[index].tolist())),
                legal_attachment_targets=tuple(
                    tuple(map(int, keys)) for keys in legal[index]
                ),
            )
            for index, candidate_spans in enumerate(binary_spans)
        )


class NativeGPSTSplicedBPEPushdownCorpus(NativeGPSTBinaryPushdownCorpus):
    """Direct strict-binary word proposals with diagnostic BPE splicing.

    The stored GPST axis wraps every recovered parser word in a fixed BPE atom
    before applying its word-level CKY topology.  This corpus deliberately
    removes those atoms and right-binarizes the BPE-spliced word tree.  It is a
    controlled topology diagnostic, not the checkpoint training representation.
    """

    structure_source = "v2_gpst_strict_binary_cky_spliced_bpe_right_binarized"
    source_candidate_axis = "gpst"
    binarization = "gpst_word_topology_spliced_bpe_deterministic_right_cnf"
    deduplicated_binary_structures = True

    def _candidates_from_row(self, row) -> Tuple[PushdownGoldCandidate, ...]:
        valid_count = int(row.gpst_valid_count)
        if valid_count <= 0 or row.gpst_merge_orders.shape[0] != valid_count:
            raise ValueError(
                f"sentence {row.global_sentence_id} has invalid GPST count "
                f"{valid_count}"
            )
        content_left, content_right = map(int, row.content_bounds)
        tokens = tuple(map(int, row.tokens))
        binary_spans, _source_indices = gpst_merge_orders_to_spliced_bpe_spans(
            row.gpst_merge_orders,
            row.word_starts,
            content_left,
            content_right,
            deduplicate=True,
        )
        sentence_ids = tuple(
            0 if content_left <= index < content_right else -1
            for index in range(len(tokens))
        )
        targets, legal = _binary_attachment_actions(
            binary_spans,
            len(tokens),
            content_left,
            content_right,
        )
        return tuple(
            PushdownGoldCandidate(
                tokens=tokens,
                spans=tuple(tuple(map(int, span)) for span in candidate_spans),
                sentence_ids=sentence_ids,
                attachment_targets=tuple(map(int, targets[index].tolist())),
                legal_attachment_targets=tuple(
                    tuple(map(int, keys)) for keys in legal[index]
                ),
            )
            for index, candidate_spans in enumerate(binary_spans)
        )


class NativeNaryWordAtomRightBinarizedPushdownCorpus(
    NativeGPSTBinaryPushdownCorpus
):
    """Native n-ary support converted to the checkpoint's actual binary tree."""

    structure_source = "v2_pushdown_nary_topk_word_atom_right_binarized"
    source_candidate_axis = "pushdown"
    binarization = "deterministic_right_cnf_with_fixed_word_bpe_atoms"
    deduplicated_binary_structures = True

    def _candidates_from_row(self, row) -> Tuple[PushdownGoldCandidate, ...]:
        valid_count = int(row.pushdown_valid_count)
        if valid_count <= 0 or row.pushdown_spans.shape[0] != valid_count:
            raise ValueError(
                f"sentence {row.global_sentence_id} has invalid Pushdown count "
                f"{valid_count}"
            )
        content_left, content_right = map(int, row.content_bounds)
        tokens = tuple(map(int, row.tokens))
        binary_spans, _source_indices = (
            right_binarize_native_nary_spans_with_word_atoms(
                row.pushdown_spans,
                row.pushdown_span_counts,
                row.word_starts,
                content_left,
                content_right,
                deduplicate=True,
            )
        )
        sentence_ids = tuple(
            0 if content_left <= index < content_right else -1
            for index in range(len(tokens))
        )
        targets, legal = _binary_attachment_actions(
            binary_spans,
            len(tokens),
            content_left,
            content_right,
        )
        return tuple(
            PushdownGoldCandidate(
                tokens=tokens,
                spans=tuple(tuple(map(int, span)) for span in candidate_spans),
                sentence_ids=sentence_ids,
                attachment_targets=tuple(map(int, targets[index].tolist())),
                legal_attachment_targets=tuple(
                    tuple(map(int, keys)) for keys in legal[index]
                ),
            )
            for index, candidate_spans in enumerate(binary_spans)
        )


@dataclass(frozen=True)
class PushdownPrefixKVCache:
    """Transformer and final-hidden state for candidate-0 history."""

    context: Tuple[PushdownGoldCandidate, ...]
    key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
    final_hidden: torch.Tensor
    input_ids: torch.Tensor
    sentence_ids: torch.Tensor


def _pack_binary_candidates(
    prefix: Sequence[PushdownGoldCandidate],
    candidates: Sequence[PushdownGoldCandidate],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Pack shared terminals, spans, gold actions, and ragged legal indices."""
    if not candidates:
        raise ValueError("candidates cannot be empty")
    token_row, span_row, sid_row, _targets, _legal, prefix_length = _compose(
        prefix, candidates[0]
    )
    total_length = int(token_row.numel())
    current_length = total_length - prefix_length
    reference = candidates[0]
    if any(
        candidate.tokens != reference.tokens
        or candidate.sentence_ids != reference.sentence_ids
        for candidate in candidates[1:]
    ):
        raise ValueError("binary candidates do not share terminals/sentence IDs")

    prefix_spans = [
        tuple(map(int, span))
        for span in span_row.tolist()
        if int(span[0]) < prefix_length
    ]
    max_spans = max(
        len(prefix_spans) + len(candidate.spans) for candidate in candidates
    )
    max_legal = max(
        1,
        max(
            (
                len(keys)
                for candidate in candidates
                for keys in candidate.legal_attachment_targets
            ),
            default=0,
        ),
    )
    spans = np.full((len(candidates), max(max_spans, 1), 3), -1, dtype=np.int64)
    targets = np.full((len(candidates), current_length), -100, dtype=np.int64)
    legal_indices = np.full(
        (len(candidates), current_length, max_legal), -1, dtype=np.int64
    )
    for batch, candidate in enumerate(candidates):
        row_spans = prefix_spans + [
            tuple(value + prefix_length for value in span) for span in candidate.spans
        ]
        if row_spans:
            spans[batch, : len(row_spans)] = row_spans
        if (
            len(candidate.attachment_targets) != current_length
            or len(candidate.legal_attachment_targets) != current_length
        ):
            raise ValueError("attachment actions do not match current token length")
        for query, (target, keys) in enumerate(
            zip(
                candidate.attachment_targets,
                candidate.legal_attachment_targets,
            )
        ):
            if not keys:
                continue
            if int(target) not in keys:
                raise ValueError("gold attachment target is outside its legal set")
            targets[batch, query] = int(target) + prefix_length
            legal_indices[batch, query, : len(keys)] = (
                np.asarray(keys, dtype=np.int64) + prefix_length
            )
    return (
        token_row,
        sid_row,
        torch.from_numpy(spans),
        torch.from_numpy(targets),
        torch.from_numpy(legal_indices),
        prefix_length,
    )


def _v1_attachment_nll_from_hidden(
    model: OLMo,
    full_hidden: torch.Tensor,
    current_input_ids: torch.Tensor,
    targets: torch.Tensor,
    legal_indices: torch.Tensor,
    prefix_length: int,
) -> torch.Tensor:
    """Score v1 attachments directly on legal positions.

    Selecting legal logits and then applying ``logsumexp`` is algebraically
    identical to masking every illegal dense-logit position to ``-inf`` before
    softmax.  Computing only these values avoids both the dense attachment row
    and a dense boolean legal mask.
    """
    if full_hidden.ndim != 3 or current_input_ids.ndim != 2:
        raise ValueError("expected full_hidden (B,N,D) and current_input_ids (B,Q)")
    batch_size, total_length, hidden_size = full_hidden.shape
    if current_input_ids.shape[0] != batch_size:
        raise ValueError("hidden and token batch sizes differ")
    current_length = current_input_ids.shape[1]
    if total_length != prefix_length + current_length:
        raise ValueError("hidden length does not match prefix plus current tokens")
    if targets.shape != (batch_size, current_length):
        raise ValueError("attachment target shape does not match current tokens")
    if legal_indices.shape[:2] != targets.shape:
        raise ValueError("legal attachment rows do not match targets")

    head = model.pushdown_attachment_head
    device = full_hidden.device
    with torch.autocast(device_type=device.type, enabled=False):
        hidden = full_hidden.float()
        embeddings = F.embedding(
            current_input_ids, model.transformer.wte.weight
        ).float()
        if prefix_length:
            previous_hidden = hidden[:, prefix_length - 1 : total_length - 1]
        else:
            previous_hidden = F.pad(hidden[:, : total_length - 1], (0, 0, 1, 0))
        h_tilde = head.mlp(torch.cat((embeddings, previous_hidden), dim=-1))
        projected = head.W(h_tilde)

        legal = legal_indices.to(device=device, dtype=torch.long)
        legal_valid = legal >= 0
        safe_legal = legal.clamp(0, max(total_length - 1, 0))
        batch_index = torch.arange(batch_size, device=device)[:, None, None]
        key_hidden = hidden[batch_index, safe_legal]
        legal_scores = torch.einsum("bqd,bqad->bqa", projected, key_hidden)
        global_queries = prefix_length + torch.arange(current_length, device=device)
        self_scores = (projected * h_tilde).sum(dim=-1)
        legal_scores = torch.where(
            safe_legal == global_queries[None, :, None],
            self_scores[:, :, None],
            legal_scores,
        )
        legal_scores = legal_scores.masked_fill(~legal_valid, float("-inf"))

        gold = targets.to(device=device, dtype=torch.long)
        valid_query = gold != -100
        gold_match = legal_valid & (legal == gold[:, :, None])
        if not bool(gold_match[valid_query].any(dim=-1).all()):
            raise ValueError("gold attachment target is outside its legal set")
        gold_scores = legal_scores.masked_fill(~gold_match, float("-inf")).amax(dim=-1)
        losses = torch.logsumexp(legal_scores, dim=-1) - gold_scores
        losses = torch.where(valid_query, losses, torch.zeros_like(losses))
    return losses.to(torch.float64).sum(dim=1).cpu()


def _v2_attachment_nll_from_hidden(
    model: OLMo,
    full_hidden: torch.Tensor,
    current_input_ids: torch.Tensor,
    current_sentence_ids: torch.Tensor,
    targets: torch.Tensor,
    legal_indices: torch.Tensor,
    prefix_length: int,
) -> torch.Tensor:
    """Score training-objective attachments over full sentence-causal rows.

    History sentences are never attachment keys for the current sentence, so
    only the current hidden-state suffix is materialized.  This is exactly the
    dense head's causal + same-sentence mask, while avoiding a ``Q x N`` row
    whose history columns are all ``-inf``.
    """
    if full_hidden.ndim != 3 or current_input_ids.ndim != 2:
        raise ValueError("expected full_hidden (B,N,D) and current_input_ids (B,Q)")
    batch_size, total_length, _hidden_size = full_hidden.shape
    if current_input_ids.shape[0] != batch_size:
        raise ValueError("hidden and token batch sizes differ")
    current_length = current_input_ids.shape[1]
    if total_length != prefix_length + current_length:
        raise ValueError("hidden length does not match prefix plus current tokens")
    if targets.shape != (batch_size, current_length):
        raise ValueError("attachment target shape does not match current tokens")
    if legal_indices.shape[:2] != targets.shape:
        raise ValueError("legal attachment rows do not match targets")
    if current_sentence_ids.shape != (current_length,):
        raise ValueError("current sentence IDs must have shape (Q,)")

    head = model.pushdown_attachment_head
    device = full_hidden.device
    with torch.autocast(device_type=device.type, enabled=False):
        hidden = full_hidden.float()
        embeddings = F.embedding(
            current_input_ids, model.transformer.wte.weight
        ).float()
        if prefix_length:
            previous_hidden = hidden[:, prefix_length - 1 : total_length - 1]
        else:
            previous_hidden = F.pad(hidden[:, : total_length - 1], (0, 0, 1, 0))
        h_tilde = head.mlp(torch.cat((embeddings, previous_hidden), dim=-1))
        projected = head.W(h_tilde)
        current_hidden = hidden[:, prefix_length:]
        scores = torch.bmm(projected, current_hidden.transpose(1, 2))
        diagonal = (projected * h_tilde).sum(dim=-1)
        positions = torch.arange(current_length, device=device)
        scores[:, positions, positions] = diagonal

        sid = current_sentence_ids.to(device=device, dtype=torch.long)
        same_sentence = sid[:, None] == sid[None, :]
        same_sentence &= (sid[:, None] >= 0) & (sid[None, :] >= 0)
        causal = positions[None, :] <= positions[:, None]
        support = same_sentence & causal
        scores = scores.masked_fill(~support[None, :, :], float("-inf"))

        gold = targets.to(device=device, dtype=torch.long)
        valid_query = gold != -100
        legal = legal_indices.to(device=device, dtype=torch.long)
        legal_valid = legal >= 0
        gold_match = legal_valid & (legal == gold[:, :, None])
        if not bool(gold_match[valid_query].any(dim=-1).all()):
            raise ValueError("gold attachment target is outside its legal set")
        local_gold = (gold - prefix_length).clamp(0, current_length - 1)
        gold_supported = support[positions[None, :], local_gold]
        if not bool(gold_supported[valid_query].all()):
            raise ValueError(
                "gold attachment target is outside sentence-causal support"
            )
        gold_scores = scores.gather(-1, local_gold.unsqueeze(-1)).squeeze(-1)
        losses = torch.logsumexp(scores, dim=-1) - gold_scores
        losses = torch.where(valid_query, losses, torch.zeros_like(losses))
    return losses.to(torch.float64).sum(dim=1).cpu()


@torch.no_grad()
def _build_prefix_cache(
    model: OLMo,
    context: Sequence[PushdownGoldCandidate],
    device: torch.device | str,
) -> PushdownPrefixKVCache:
    """Rebuild one bounded candidate-0 prefix after a context-window slide."""
    if not context:
        raise ValueError("cannot build an empty prefix cache")
    device = torch.device(device)
    token_row, spans, sentence_ids, _targets, _legal, _prefix = _compose(
        context[:-1], context[-1]
    )
    input_ids = token_row.unsqueeze(0).to(device)
    sid = sentence_ids.unsqueeze(0).to(device)
    tree_spans = spans.unsqueeze(0).to(device)
    length = int(token_row.numel())
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        tree_spans=tree_spans,
        pushdown_sentence_ids=sid,
        compute_attachment_logits=False,
        return_final_hidden=True,
        use_cache=True,
        logits_range=(length - 1, length),
    )
    if out.attn_key_values is None or out.final_hidden is None:
        raise RuntimeError("prefix prefill did not return KV/final-hidden state")
    return PushdownPrefixKVCache(
        context=tuple(context),
        key_values=tuple(
            (key[:1].detach(), value[:1].detach()) for key, value in out.attn_key_values
        ),
        final_hidden=out.final_hidden[:1].detach(),
        input_ids=input_ids[:1].detach(),
        sentence_ids=sid[:1].detach(),
    )


@torch.no_grad()
def score_gpst_binary_pushdown_candidates(
    model: OLMo,
    prefix: Sequence[PushdownGoldCandidate],
    candidates: Sequence[PushdownGoldCandidate],
    device: torch.device | str,
    prefix_cache: Optional[PushdownPrefixKVCache] = None,
    return_candidate0_cache: bool = False,
    attachment_normalization: str = ATTACHMENT_NORMALIZATION_V1,
) -> Tuple[PushdownCandidateScores, Optional[PushdownPrefixKVCache]]:
    """Score one candidate microbatch under joint token + attachment loss."""
    attachment_normalization = canonical_attachment_normalization(
        attachment_normalization
    )
    device = torch.device(device)
    token_row, sid_row, spans, targets, legal, prefix_length = _pack_binary_candidates(
        prefix, candidates
    )
    batch_size = len(candidates)
    use_cache = prefix_cache is not None
    if use_cache:
        if prefix_cache.context != tuple(prefix):
            raise ValueError("KV cache context does not match candidate-0 prefix")
        input_ids = (
            token_row[prefix_length:].unsqueeze(0).expand(batch_size, -1).to(device)
        )
        sentence_ids = (
            sid_row[prefix_length:].unsqueeze(0).expand(batch_size, -1).to(device)
        )
        past_key_values = tuple(
            (
                key.expand(batch_size, -1, -1, -1),
                value.expand(batch_size, -1, -1, -1),
            )
            for key, value in prefix_cache.key_values
        )
        attention_mask = None
        logits_range = None
    else:
        input_ids = token_row.unsqueeze(0).expand(batch_size, -1).to(device)
        sentence_ids = sid_row.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = None
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        target_start = max(prefix_length, 1)
        logits_range = (target_start - 1, int(token_row.numel()) - 1)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tree_spans=spans.to(device, non_blocking=True),
        pushdown_sentence_ids=sentence_ids,
        compute_attachment_logits=False,
        return_final_hidden=True,
        use_cache=return_candidate0_cache,
        logits_range=logits_range,
        past_key_values=past_key_values,
    )
    if out.final_hidden is None:
        raise RuntimeError("Pushdown scoring forward did not return final hidden state")

    if use_cache:
        prefix_last = model.transformer.ln_f(prefix_cache.final_hidden[:, -1:])
        prefix_logits = F.linear(prefix_last, model.transformer.wte.weight)
        if model.config.scale_logits:
            prefix_logits = prefix_logits * (1 / math.sqrt(model.config.d_model))
        token_logits = torch.cat(
            (prefix_logits.expand(batch_size, -1, -1), out.logits[:, :-1]), dim=1
        )
        labels = input_ids
        full_hidden = torch.cat(
            (prefix_cache.final_hidden.expand(batch_size, -1, -1), out.final_hidden),
            dim=1,
        )
    else:
        token_logits = out.logits
        target_start = max(prefix_length, 1)
        labels = input_ids[:, target_start:]
        full_hidden = out.final_hidden
    token_nll = (
        F.cross_entropy(token_logits.float().transpose(1, 2), labels, reduction="none")
        .sum(dim=1)
        .to(torch.float64)
        .cpu()
    )
    current_ids = (
        token_row[prefix_length:].unsqueeze(0).expand(batch_size, -1).to(device)
    )
    if attachment_normalization == ATTACHMENT_NORMALIZATION_V1:
        attachment_nll = _v1_attachment_nll_from_hidden(
            model,
            full_hidden,
            current_ids,
            targets.to(device, non_blocking=True),
            legal.to(device, non_blocking=True),
            prefix_length,
        )
    else:
        attachment_nll = _v2_attachment_nll_from_hidden(
            model,
            full_hidden,
            current_ids,
            sid_row[prefix_length:],
            targets.to(device, non_blocking=True),
            legal.to(device, non_blocking=True),
            prefix_length,
        )
    scores = PushdownCandidateScores(
        token_nll + attachment_nll, token_nll, attachment_nll
    )

    next_cache: Optional[PushdownPrefixKVCache] = None
    if return_candidate0_cache:
        if out.attn_key_values is None:
            raise RuntimeError("candidate-0 forward did not return KV state")
        next_cache = PushdownPrefixKVCache(
            context=tuple(prefix) + (candidates[0],),
            key_values=tuple(
                (key[:1].detach(), value[:1].detach())
                for key, value in out.attn_key_values
            ),
            final_hidden=full_hidden[:1].detach(),
            input_ids=token_row.unsqueeze(0).to(device).detach(),
            sentence_ids=sid_row.unsqueeze(0).to(device).detach(),
        )
    return scores, next_cache


@dataclass(frozen=True)
class GPSTBinaryPushdownDocumentPPLResult:
    joint_document_perplexity: float
    joint_log_likelihood: float
    candidate0_structured_terminal_perplexity: float
    candidate0_terminal_log_likelihood: float
    terminal_count: int
    sentence_count: int
    document_count: int
    valid_candidate_count: int
    candidate_slots: int
    model_candidate_forwards: int
    kv_cache_hits: int
    kv_cache_rebuilds: int
    oom_retries: int
    nonfinite_retries: int
    max_sequence_length: int
    max_candidates_per_sentence: int
    protocol_version: int = PUSHDOWN_GPST_BINARY_PROTOCOL_VERSION
    structure_source: str = "v2_gpst_strict_binary_to_pushdown"
    source_candidate_axis: str = "gpst"
    binarization: str = "direct_strict_binary_cky_with_fixed_word_bpe_atoms"
    deduplicated_binary_structures: bool = True
    prefix_policy: str = "candidate0"
    context_truncation: str = "left_drop_complete_sentences"
    attachment_normalization: str = ATTACHMENT_NORMALIZATION_V1
    candidate_aggregation: str = "valid_unique_truncated_joint_sum"
    divide_by_candidate_count: bool = False
    ppl_denominator: str = "terminal_count"
    beam_search: bool = False

    def as_dict(self) -> dict:
        result = dict(self.__dict__)
        suffix = (
            "v1"
            if self.attachment_normalization == ATTACHMENT_NORMALIZATION_V1
            else "v2"
        )
        result[f"joint_document_perplexity_{suffix}"] = result.pop(
            "joint_document_perplexity"
        )
        result[f"joint_log_likelihood_{suffix}"] = result.pop("joint_log_likelihood")
        return result

    @property
    def joint_document_perplexity_v1(self) -> float:
        """Backward-compatible accessor for historical v1 callers/tests."""
        return self.joint_document_perplexity

    @property
    def joint_log_likelihood_v1(self) -> float:
        """Backward-compatible accessor for historical v1 callers/tests."""
        return self.joint_log_likelihood


def evaluate_gpst_binary_pushdown_document_ppl(
    model: OLMo,
    corpus: NativeGPSTBinaryPushdownCorpus,
    device: torch.device | str,
    eval_batch_size: int = 64,
    max_sequence_length: int = 2048,
    max_batch_tokens: int = 65536,
    max_batch_attention_elements: int = 16777216,
    use_kv_cache: bool = True,
    prefetch_sentences: int = 2,
    progress: Optional[Callable[[int, int, int], None]] = None,
    document_complete: Optional[Callable[[int, dict], None]] = None,
    attachment_normalization: str = ATTACHMENT_NORMALIZATION_V1,
) -> GPSTBinaryPushdownDocumentPPLResult:
    """Evaluate the two requested metrics over exactly the valid GPST rows."""
    if eval_batch_size <= 0 or max_batch_tokens <= 0:
        raise ValueError("batch-size and token limits must be positive")
    if max_batch_attention_elements <= 0 or max_sequence_length <= 0:
        raise ValueError("attention and context limits must be positive")
    if prefetch_sentences < 0:
        raise ValueError("prefetch_sentences cannot be negative")
    attachment_normalization = canonical_attachment_normalization(
        attachment_normalization
    )
    protocol_version = (
        1 if attachment_normalization == ATTACHMENT_NORMALIZATION_V1 else 2
    )
    if not hasattr(model, "pushdown_attachment_head"):
        raise RuntimeError("joint Pushdown PPL requires attachment-head weights")
    model.eval()
    structure_source = getattr(
        corpus, "structure_source", "v2_gpst_strict_binary_to_pushdown"
    )
    source_candidate_axis = getattr(corpus, "source_candidate_axis", "gpst")
    binarization = getattr(
        corpus,
        "binarization",
        "direct_strict_binary_cky_with_fixed_word_bpe_atoms",
    )
    deduplicated_binary_structures = bool(
        getattr(corpus, "deduplicated_binary_structures", True)
    )
    prefix: Tuple[PushdownGoldCandidate, ...] = ()
    prefix_cache: Optional[PushdownPrefixKVCache] = None
    previous_doc: Optional[int] = None
    joint_ll = candidate0_token_ll = 0.0
    terminal_count = document_count = valid_candidate_count = 0
    candidate_slots = model_candidate_forwards = 0
    cache_hits = cache_rebuilds = oom_retries = nonfinite_retries = 0
    doc_joint_ll = doc_token_ll = 0.0
    doc_terminals = doc_sentences = doc_candidates = 0

    def emit_document(doc_id: int) -> None:
        if document_complete is None or doc_sentences == 0:
            return
        document_complete(
            doc_id,
            {
                "protocol_version": protocol_version,
                "structure_source": structure_source,
                "source_candidate_axis": source_candidate_axis,
                "binarization": binarization,
                "deduplicated_binary_structures": deduplicated_binary_structures,
                "prefix_policy": "candidate0",
                "context_truncation": "left_drop_complete_sentences",
                "attachment_normalization": attachment_normalization,
                "candidate_aggregation": "valid_unique_truncated_joint_sum",
                "divide_by_candidate_count": False,
                "document_id": doc_id,
                f"joint_log_likelihood_{'v1' if protocol_version == 1 else 'v2'}": doc_joint_ll,
                "candidate0_terminal_log_likelihood": doc_token_ll,
                f"joint_document_perplexity_{'v1' if protocol_version == 1 else 'v2'}": math.exp(
                    -doc_joint_ll / doc_terminals
                ),
                "candidate0_structured_terminal_perplexity": math.exp(
                    -doc_token_ll / doc_terminals
                ),
                "terminal_count": doc_terminals,
                "sentence_count": doc_sentences,
                "document_count": 1,
                "valid_candidate_count": doc_candidates,
            },
        )

    rows = (
        corpus.iter_prefetched(prefetch_sentences)
        if prefetch_sentences
        else iter(corpus)
    )
    for index, (doc_id, original) in enumerate(rows):
        first = doc_id != previous_doc
        if first:
            if previous_doc is not None:
                emit_document(previous_doc)
            doc_joint_ll = doc_token_ll = 0.0
            doc_terminals = doc_sentences = doc_candidates = 0
            prefix = ()
            prefix_cache = None
            previous_doc = doc_id
            document_count += 1
        candidates = (
            original
            if first
            else tuple(
                _drop_leading_bos(candidate, corpus.vocab.bos) for candidate in original
            )
        )
        if not candidates:
            raise ValueError(f"sentence {index} has no valid GPST candidates")
        current = candidates[0]
        if first and (not current.tokens or current.tokens[0] != corpus.vocab.bos):
            raise ValueError(
                f"document {doc_id} does not begin with the tokenizer BOS; "
                "the first terminal would otherwise have no LM context"
            )
        context = _trim_prefix(prefix, current, max_sequence_length)
        active_cache: Optional[PushdownPrefixKVCache] = None
        if use_kv_cache and context:
            if prefix_cache is not None and prefix_cache.context == context:
                active_cache = prefix_cache
                cache_hits += 1
            else:
                active_cache = _build_prefix_cache(model, context, device)
                cache_rebuilds += 1

        total_length = sum(len(sentence.tokens) for sentence in context) + len(
            current.tokens
        )
        model_input_length = (
            len(current.tokens) if active_cache is not None else total_length
        )
        attention_cells = model_input_length * total_length
        batch_size = min(
            eval_batch_size,
            len(candidates),
            max(1, max_batch_tokens // max(model_input_length, 1)),
            max(
                1,
                max_batch_attention_elements // max(attention_cells, 1),
            ),
        )
        while True:
            parts: List[PushdownCandidateScores] = []
            next_cache: Optional[PushdownPrefixKVCache] = None
            candidate0_cache: Optional[PushdownPrefixKVCache] = None
            try:
                for start in range(0, len(candidates), batch_size):
                    part, candidate0_cache = score_gpst_binary_pushdown_candidates(
                        model,
                        context,
                        candidates[start : start + batch_size],
                        device,
                        active_cache,
                        return_candidate0_cache=(start == 0 and use_kv_cache),
                        attachment_normalization=attachment_normalization,
                    )
                    nonfinite = {}
                    for field in ("joint_nll", "token_nll", "attachment_nll"):
                        values = getattr(part, field)
                        bad = torch.nonzero(~torch.isfinite(values), as_tuple=False)
                        if bad.numel():
                            local_indices = bad.flatten().tolist()
                            nonfinite[field] = [
                                {
                                    "candidate_index": start + local_index,
                                    "value": float(values[local_index].item()),
                                }
                                for local_index in local_indices
                            ]
                    if nonfinite:
                        raise FloatingPointError(
                            "non-finite GPST Pushdown candidate score: "
                            f"document_id={doc_id} corpus_sentence_index={index} "
                            f"document_sentence_index={doc_sentences} "
                            f"candidate_range=[{start},{start + len(part.joint_nll)}) "
                            f"microbatch_size={batch_size} "
                            f"context_tokens={total_length - len(current.tokens)} "
                            f"current_tokens={len(current.tokens)} "
                            f"kv_cache_active={active_cache is not None} "
                            f"nonfinite={nonfinite}"
                        )
                    parts.append(part)
                    if start == 0:
                        next_cache = candidate0_cache
                break
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                parts.clear()
                next_cache = None
                candidate0_cache = None
                batch_size = max(1, batch_size // 2)
                oom_retries += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except FloatingPointError:
                # FlexAttention/Inductor has occasionally produced a transient
                # non-finite microbatch on RTX 3090 while the same candidates
                # are finite at a smaller batch or under full-prefix scoring.
                # Retry the whole sentence so candidate ordering and sums stay
                # exact; never serialize or silently skip a non-finite value.
                parts.clear()
                next_cache = None
                candidate0_cache = None
                nonfinite_retries += 1
                if batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                elif active_cache is not None:
                    active_cache = None
                else:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        scores = PushdownCandidateScores(
            *(
                torch.cat([getattr(part, field) for part in parts])
                for field in ("joint_nll", "token_nll", "attachment_nll")
            )
        )
        if scores.joint_nll.numel() != len(candidates):
            raise RuntimeError(
                "candidate microbatching changed the valid candidate count"
            )
        sentence_joint_ll = torch.logsumexp(
            -scores.joint_nll.to(torch.float64), dim=0
        ).item()
        sentence_candidate0_token_ll = -float(scores.token_nll[0].item())
        joint_ll += sentence_joint_ll
        candidate0_token_ll += sentence_candidate0_token_ll
        doc_joint_ll += sentence_joint_ll
        doc_token_ll += sentence_candidate0_token_ll

        sentence_terminals = len(current.tokens) - (1 if first else 0)
        terminal_count += sentence_terminals
        doc_terminals += sentence_terminals
        doc_sentences += 1
        valid_candidate_count += len(candidates)
        doc_candidates += len(candidates)
        candidate_slots += corpus.samples_per_sentence
        model_candidate_forwards += len(candidates)
        prefix = context + (current,)
        prefix_cache = next_cache if use_kv_cache else None
        if progress is not None:
            progress(index + 1, len(corpus), doc_id)

    if previous_doc is not None:
        emit_document(previous_doc)

    def perplexity(log_likelihood: float) -> float:
        return (
            math.exp(-log_likelihood / terminal_count) if terminal_count else math.nan
        )

    return GPSTBinaryPushdownDocumentPPLResult(
        joint_document_perplexity=perplexity(joint_ll),
        joint_log_likelihood=joint_ll,
        candidate0_structured_terminal_perplexity=perplexity(candidate0_token_ll),
        candidate0_terminal_log_likelihood=candidate0_token_ll,
        terminal_count=terminal_count,
        sentence_count=len(corpus),
        document_count=document_count,
        valid_candidate_count=valid_candidate_count,
        candidate_slots=candidate_slots,
        model_candidate_forwards=model_candidate_forwards,
        kv_cache_hits=cache_hits,
        kv_cache_rebuilds=cache_rebuilds,
        oom_retries=oom_retries,
        nonfinite_retries=nonfinite_retries,
        max_sequence_length=max_sequence_length,
        max_candidates_per_sentence=corpus.samples_per_sentence,
        structure_source=structure_source,
        source_candidate_axis=source_candidate_axis,
        binarization=binarization,
        deduplicated_binary_structures=deduplicated_binary_structures,
        protocol_version=protocol_version,
        attachment_normalization=attachment_normalization,
    )
