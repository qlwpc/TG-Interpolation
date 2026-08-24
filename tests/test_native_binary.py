"""Exactness and representation tests for native-binary K-best decoding."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
import torch

from datatools.native_binary import (
    NativeBinaryCandidate,
    NativeNaryCandidate,
    adapt_native_binary_candidates_to_gpst,
    adapt_native_nary_candidates,
    aggregate_label_scores,
    candidate_to_tree,
    capped_catalan,
    capped_little_schroeder,
    decode_labeled_scores_topk,
    decode_native_binary_topk,
    decode_native_nary_topk,
    expand_candidate_to_subwords,
    pack_candidate_slots,
)
from olmo.data.parse_align import tree_spans
from olmo.gpst.reader.dataset_gold import tree_to_merge_orders


@lru_cache(maxsize=None)
def _enumerate_spans(left: int, right: int):
    if left == right:
        return ((),)
    trees = []
    for split in range(left, right):
        for left_spans in _enumerate_spans(left, split):
            for right_spans in _enumerate_spans(split + 1, right):
                trees.append(left_spans + right_spans + ((left, split, right),))
    return tuple(trees)


def _brute_force_topk(scores: np.ndarray, k: int):
    length = scores.shape[0]
    ranked = []
    for spans in _enumerate_spans(0, length - 1):
        score = sum(float(scores[left, right]) for left, _split, right in spans)
        ranked.append((score, spans))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked[:k]


def _enumerate_nary_spans(length: int):
    """Enumerate n-ary trees by contracting non-root edges of binary trees."""

    unique = set()
    for binary_spans in _enumerate_spans(0, length - 1):
        root = (binary_spans[-1][0], binary_spans[-1][2])
        optional = binary_spans[:-1]
        for mask in range(1 << len(optional)):
            kept = tuple(
                (left, right)
                for index, (left, _split, right) in enumerate(optional)
                if mask & (1 << index)
            )
            unique.add(kept + (root,))
    return tuple(unique)


@pytest.mark.parametrize(
    ("num_leaves", "expected"),
    [(1, 1), (2, 1), (3, 2), (4, 5), (5, 14), (6, 42), (7, 132), (8, 300)],
)
def test_capped_catalan(num_leaves, expected):
    assert capped_catalan(num_leaves, 300) == expected


@pytest.mark.parametrize(
    ("num_leaves", "expected"),
    [(1, 1), (2, 1), (3, 3), (4, 11), (5, 45), (6, 197), (7, 300)],
)
def test_capped_little_schroeder(num_leaves, expected):
    assert capped_little_schroeder(num_leaves, 300) == expected


def test_aggregate_label_scores_exact_logsumexp_and_max():
    scores = torch.tensor(
        [
            [[2.0, 3.0, 4.0], [1.0, 2.0, 5.0]],
            [[0.0, 0.0, 0.0], [-1.0, 1.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    marginalized = aggregate_label_scores(scores, mode="logsumexp")
    expected = torch.logsumexp(scores[..., 1:] - scores[..., :1], dim=-1)
    torch.testing.assert_close(marginalized, expected)

    viterbi = aggregate_label_scores(scores.numpy(), mode="max")
    np.testing.assert_allclose(viterbi, (scores[..., 1:] - scores[..., :1]).max(-1).values)


@pytest.mark.parametrize("length", range(2, 8))
def test_kbest_matches_catalan_brute_force(length):
    # Continuous deterministic scores avoid ties, letting us compare complete
    # rankings rather than only the top-K score multiset.
    rng = np.random.default_rng(6198 + length)
    scores = rng.normal(size=(length, length))
    k = min(37, capped_catalan(length, 300))

    decoded = decode_native_binary_topk(scores, k=k)
    brute = _brute_force_topk(scores, k)

    assert [candidate.spans for candidate in decoded] == [spans for _, spans in brute]
    np.testing.assert_allclose(
        [candidate.score for candidate in decoded],
        [score for score, _ in brute],
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("length", range(2, 7))
def test_nary_kbest_matches_edge_contraction_brute_force(length):
    rng = np.random.default_rng(14901 + length)
    scores = rng.normal(size=(length, length))
    all_spans = _enumerate_nary_spans(length)
    assert len(all_spans) == capped_little_schroeder(length, 10_000)
    brute = sorted(
        (
            sum(float(scores[left, right]) for left, right in spans),
            spans,
        )
        for spans in all_spans
    )
    brute.sort(key=lambda item: (-item[0], item[1]))
    k = min(37, len(brute))
    decoded = decode_native_nary_topk(scores, k=k)
    assert [candidate.spans for candidate in decoded] == [spans for _, spans in brute[:k]]
    np.testing.assert_allclose(
        [candidate.score for candidate in decoded],
        [score for score, _ in brute[:k]],
        rtol=1e-6,
        atol=1e-6,
    )


def test_shared_nary_adapters_keep_pushdown_spans_and_group_gpst_collision():
    candidates = [
        NativeNaryCandidate(score=3.0, spans=((0, 2),)),
        NativeNaryCandidate(score=2.0, spans=((1, 2), (0, 2))),
    ]
    adapted = adapt_native_nary_candidates(candidates, [[10], [11], [12]])

    assert adapted.tokens == (10, 11, 12)
    assert adapted.pushdown_spans == (
        ((0, 2, 2),),
        ((1, 2, 2), (0, 2, 2)),
    )
    assert len(adapted.gpst_candidates) == 1
    assert adapted.gpst_candidates[0].merge_orders == (1, 0)
    assert adapted.candidate_to_gpst == (0, 0)
    assert adapted.gpst_source_slots == (0,)
    assert adapted.gpst_multiplicities == (2,)
    assert adapted.gpst_log_masses[0] == pytest.approx(np.logaddexp(3.0, 2.0))


def test_nary_gpst_adapter_splices_word_pieces_before_right_binarizing():
    candidate = NativeNaryCandidate(score=1.0, spans=((0, 1),))
    adapted = adapt_native_nary_candidates([candidate], [[10, 11], [12, 13]])

    assert adapted.tokens == (10, 11, 12, 13)
    assert adapted.pushdown_spans == (((0, 3, 3),),)
    binary = adapted.gpst_candidates[0]
    assert len(binary.spans) == 3
    assert binary.merge_orders == (2, 1, 0)


def test_direct_gpst_adapter_preserves_distinct_binary_cky_candidates():
    candidates = [
        NativeBinaryCandidate(score=4.0, spans=((0, 0, 1), (0, 1, 2))),
        NativeBinaryCandidate(score=3.0, spans=((1, 1, 2), (0, 0, 2))),
    ]
    adapted = adapt_native_binary_candidates_to_gpst(
        candidates, [[10, 11], [12], [13, 14]]
    )

    assert adapted.tokens == (10, 11, 12, 13, 14)
    assert [candidate.score for candidate in adapted.candidates] == [4.0, 3.0]
    assert len({candidate.spans for candidate in adapted.candidates}) == 2
    assert len({candidate.merge_orders for candidate in adapted.candidates}) == 2
    assert all(len(candidate.spans) == 4 for candidate in adapted.candidates)
    # CKY word splits map to exact BPE word boundaries. Shared local word
    # subtrees are unscored representation nodes and do not move these splits.
    assert {(0, 1, 2), (0, 2, 4)} <= set(adapted.candidates[0].spans)
    assert {(2, 2, 4), (0, 1, 4)} <= set(adapted.candidates[1].spans)


def test_direct_gpst_adapter_maps_every_cky_split_to_word_boundary():
    scores = np.random.default_rng(20260822).normal(size=(6, 6))
    word_candidates = decode_native_binary_topk(scores, k=42)
    pieces = [[10, 11], [12], [13, 14, 15], [16], [17, 18], [19]]
    adapted = adapt_native_binary_candidates_to_gpst(word_candidates, pieces)
    starts = np.cumsum([0] + [len(row) for row in pieces[:-1]])
    ends = np.cumsum([len(row) for row in pieces]) - 1

    for word_candidate, bpe_candidate in zip(word_candidates, adapted.candidates):
        mapped_cky_spans = {
            (int(starts[left]), int(ends[split]), int(ends[right]))
            for left, split, right in word_candidate.spans
        }
        assert mapped_cky_spans <= set(bpe_candidate.spans)


def test_short_sentence_returns_all_unique_topologies_without_padding():
    scores = np.arange(16, dtype=np.float64).reshape(4, 4)
    decoded = decode_native_binary_topk(scores, k=300)
    assert len(decoded) == 5
    assert len({candidate.spans for candidate in decoded}) == 5


def test_lazy_and_torch_struct_backends_agree():
    rng = np.random.default_rng(20260822)
    scores = rng.normal(size=(6, 6))
    lazy = decode_native_binary_topk(scores, k=31, backend="lazy")
    reference = decode_native_binary_topk(scores, k=31, backend="torch_struct")
    assert [candidate.spans for candidate in lazy] == [candidate.spans for candidate in reference]
    np.testing.assert_allclose(
        [candidate.score for candidate in lazy],
        [candidate.score for candidate in reference],
        rtol=1e-6,
        atol=1e-6,
    )


def test_label_variants_are_marginalized_before_topology_search():
    rng = np.random.default_rng(7)
    labeled = rng.normal(size=(5, 5, 6))
    direct = decode_labeled_scores_topk(labeled, k=14, mode="logsumexp")
    aggregated = aggregate_label_scores(labeled, mode="logsumexp")
    separate = decode_native_binary_topk(aggregated, k=14)
    assert direct == separate


def test_candidate_is_shared_by_gpst_and_pushdown_representations():
    candidate = NativeBinaryCandidate(
        score=0.0,
        spans=((0, 0, 1), (3, 3, 4), (2, 2, 4), (0, 1, 4)),
    )
    leaves = [10, 11, 12, 13, 14]
    tree = candidate_to_tree(candidate, leaves)

    pushdown_leaves, pushdown_spans = tree_spans(tree)
    gpst_leaves, merge_orders = tree_to_merge_orders(tree, direction="right")

    assert pushdown_leaves == gpst_leaves == leaves
    assert tuple(pushdown_spans) == candidate.spans
    assert tuple(merge_orders) == candidate.merge_orders
    assert candidate.constituent_spans == ((0, 1), (3, 4), (2, 4), (0, 4))


def test_word_candidate_expands_deterministically_to_shared_subword_tree():
    word_candidate = NativeBinaryCandidate(
        score=3.5,
        spans=((0, 0, 1), (0, 1, 2)),  # ((word0 word1) word2)
    )
    tokens, expanded = expand_candidate_to_subwords(
        word_candidate,
        [[10, 11], [12], [13, 14, 15]],
    )
    assert tokens == (10, 11, 12, 13, 14, 15)
    assert expanded.score == word_candidate.score
    assert len(expanded.spans) == len(tokens) - 1
    assert sorted(expanded.merge_orders) == list(range(len(tokens) - 1))

    tree = candidate_to_tree(expanded, tokens)
    parsed_tokens, pushdown_spans = tree_spans(tree)
    gpst_tokens, gpst_orders = tree_to_merge_orders(tree)
    assert tuple(parsed_tokens) == tuple(gpst_tokens) == tokens
    assert tuple(pushdown_spans) == expanded.spans
    assert tuple(gpst_orders) == expanded.merge_orders


def test_fixed_slots_are_batch_ready_but_padding_has_no_probability_mass():
    scores = np.random.default_rng(11).normal(size=(4, 4))
    candidates = decode_native_binary_topk(scores, k=300)
    assert len(candidates) == 5  # Catalan(3), despite 300 physical slots below.

    packed = pack_candidate_slots([10, 11, 12, 13], candidates, slots=300)
    assert packed.tokens.dtype == np.uint16
    assert packed.merge_orders.shape == (300, 3)
    assert packed.spans.shape == (300, 3, 3)
    assert packed.proposal_scores.shape == (300,)
    assert packed.valid_count == 5
    assert packed.candidate_mask.sum() == 5
    assert np.isfinite(packed.proposal_scores[:5]).all()
    assert np.isneginf(packed.proposal_scores[5:]).all()
    # Padded rows are structurally safe copies, but evaluators slice them out.
    np.testing.assert_array_equal(packed.merge_orders[5], packed.merge_orders[0])
    np.testing.assert_array_equal(packed.spans[299], packed.spans[0])
    assert packed.valid_merge_orders.shape == (5, 3)
    assert packed.valid_spans.shape == (5, 3, 3)


def test_fixed_slot_packer_rejects_duplicate_native_mass():
    candidate = NativeBinaryCandidate(score=0.0, spans=((0, 0, 1),))
    with pytest.raises(ValueError, match="unique"):
        pack_candidate_slots([10, 11], [candidate, candidate])


def test_invalid_arguments_fail_loudly():
    with pytest.raises(ValueError, match="positive"):
        capped_catalan(3, 0)
    with pytest.raises(ValueError, match="NULL"):
        aggregate_label_scores(np.zeros((3, 3, 1)))
    with pytest.raises(ValueError, match="finite"):
        bad = np.zeros((3, 3))
        bad[0, 2] = np.nan
        decode_native_binary_topk(bad)
