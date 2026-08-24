"""Gold-tree document-PPL regression tests (no beam search)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import olmo.gpst  # noqa: F401
from olmo.gpst.eval.document_ppl import (
    GoldTree300Corpus,
    GoldSegment,
    _candidate_signature,
    _compress_candidates,
    _retain_prefix_for_any_future_sentence,
    _trim_prefix,
    aggregate_candidate_nll,
)
from olmo.gpst.reader.dataset_gold import GoldTreeCollator, tree_to_merge_orders


def test_unary_chains_are_collapsed_before_merge_conversion():
    tree = ("S", [("A", [("B", [10])]), ("C", [("D", [20, 30])])])
    leaves, orders = tree_to_merge_orders(tree)
    assert leaves == [10, 20, 30]
    assert len(orders) == len(leaves) - 1


def test_collator_flattens_one_merge_row_per_segment():
    item = {
        "text": np.array([10, 20, 30, 40, 50], dtype=np.int32),
        "sentence_splits": [3, 5],
        "merge_orders": [
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
        ],
    }
    batch = GoldTreeCollator()([item])
    assert batch["merge_orders"].shape == (1, 4)
    # local gaps 0,1 and 3, followed by the ignored inter-segment boundary 2
    assert batch["merge_orders"].tolist() == [[0, 1, 3, 2]]
    # The collator must not mutate reusable dataset records.
    assert item["sentence_splits"] == [3, 5]


def test_candidate_aggregation_matches_olmo_legacy_and_uniform_mixture():
    nll = torch.zeros(300, dtype=torch.float64)
    assert aggregate_candidate_nll(nll, normalize_mixture=False) == pytest.approx(
        np.log(300)
    )
    assert aggregate_candidate_nll(nll, normalize_mixture=True) == pytest.approx(0.0)


def test_candidate_compression_preserves_repeated_slot_probability():
    left = (GoldSegment((10, 20, 30), (0, 1)),)
    right = (GoldSegment((10, 20, 30), (1, 0)),)
    unique, counts = _compress_candidates((left, left, right, left, right))
    assert unique == (left, right)
    assert counts.tolist() == [3, 2]
    unique_nll = torch.tensor([1.25, 2.5], dtype=torch.float64)
    expanded_nll = torch.tensor([1.25, 1.25, 2.5, 1.25, 2.5], dtype=torch.float64)
    assert aggregate_candidate_nll(unique_nll, multiplicities=counts) == pytest.approx(
        aggregate_candidate_nll(expanded_nll)
    )
    assert aggregate_candidate_nll(
        unique_nll, normalize_mixture=True, multiplicities=counts
    ) == pytest.approx(aggregate_candidate_nll(expanded_nll, normalize_mixture=True))


def test_bounded_prefix_retention_preserves_every_future_context():
    # Document 3930 has hundreds of short sentences.  The evaluator must not
    # retain all earlier Python segments merely because the scoring context is
    # bounded.  Test both a tiny future sentence and a multi-segment one: the
    # retained suffix has exactly the same context as the full history.
    prefix = tuple(GoldSegment((index, index + 1000), (0,)) for index in range(80))
    retained = _retain_prefix_for_any_future_sentence(
        prefix, max_action_nodes=32, max_terminals=32
    )
    assert len(retained) < len(prefix)
    for current in (
        (GoldSegment((9,), ()),),
        (GoldSegment((9, 10, 11), (0, 1)),),
        (GoldSegment((9, 10), (0,)), GoldSegment((11, 12), (0,))),
    ):
        assert _trim_prefix(prefix, current, 32, 32) == _trim_prefix(
            retained, current, 32, 32
        )


def test_force_gold_tree_emits_prescribed_action_sequences():
    from olmo.gpst.model.model_factory import create_model

    model = create_model(
        "r2d2-gen-fast",
        "olmo/gpst/data/en_config/r2d2_256_4_1.json",
        "olmo/gpst/data/gpt2-small/config.json",
        backbone="olmo",
    ).eval()
    items = [
        {
            "text": np.array([10, 20, 30, 40], dtype=np.int32),
            "sentence_splits": [4],
            "merge_orders": [np.array([0, 1, 2], dtype=np.int32)],
        },
        {
            "text": np.array([10, 20, 30, 40], dtype=np.int32),
            "sentence_splits": [4],
            "merge_orders": [np.array([2, 1, 0], dtype=np.int32)],
        },
    ]
    batch = GoldTreeCollator()(items)
    with torch.no_grad():
        output = model(
            **batch,
            force_gold_tree=True,
            score_token_range=(0, 4),
            score_action_range=(0, 7),
        )
    # SHIFT=0, REDUCE=1.  These are the exact post-order traversals of the
    # left- and right-recursive prescribed trees, respectively.
    assert output.action_targets.tolist() == [
        [0, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1, 1],
    ]
    assert output.logits.shape[:2] == output.token_targets.shape == (2, 4)
    assert output.action_logits.shape[:2] == output.action_targets.shape == (2, 7)


def test_real_tree300_first_sentence_all_candidates_convert():
    paths = (
        "dataset/testppl_tree/tree_300.npy",
        "dataset/testppl_tree/tree_sent_index.npy",
        "dataset/testppl_tree/tree_doc_index.npy",
        "dataset/bbc-news/TG_GPT2_tokenizer.json",
    )
    if not all(os.path.exists(path) for path in paths):
        pytest.skip("tree_300 evaluation corpus not present")
    corpus = GoldTree300Corpus(*paths, max_sentences=1)
    candidates = corpus.sentence_candidates(0)
    assert len(candidates) == 300
    assert all(sum(len(seg.tokens) for seg in candidate) == 11 for candidate in candidates)
    # Labels and unary chains disappear in GPST's unlabeled binary y; the 300
    # serialized parses map to five distinct merge trajectories in this record.
    assert len({_candidate_signature(candidate) for candidate in candidates}) == 5
