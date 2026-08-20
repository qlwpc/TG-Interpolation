"""Phase 3 test: GoldTreeDataset + tree_to_merge_orders.

Verifies that a gold constituency parse converts to the merge-order format
``CPPChartTableManager`` expects, and that the merge orders reconstruct the
original (binarized) tree via ``get_tree_from_merge_trajectory``.
"""
from __future__ import annotations

import numpy as np

import olmo.gpst  # noqa: F401
from olmo.gpst.reader.dataset_gold import tree_to_merge_orders
from olmo.gpst.utils.tree_utils import get_tree_from_merge_trajectory


def _mk_tree():
    """Build a hand-coded n-ary tree: (S (NP A B) (VP C D E))."""
    A, B, C, D, E = 10, 20, 30, 40, 50
    np_node = ("NP", [A, B])
    vp_node = ("VP", [C, D, E])
    root = ("S", [np_node, vp_node])
    return root


def test_tree_to_merge_orders_binary():
    """((A B) C): leaves [10,20,30] -> merge_orders [0, 1]."""
    tree = ("S", [("X", [10, 20]), ("Y", [30])])
    leaves, mo = tree_to_merge_orders(tree, direction="right")
    assert leaves == [10, 20, 30]
    assert mo == [0, 1], mo


def test_tree_to_merge_orders_repeated_interned_leaf_is_permutation():
    tree = ("S", [82, ("B", [1, 82])])
    leaves, mo = tree_to_merge_orders(tree, direction="right")
    assert leaves == [82, 1, 82]
    assert sorted(mo) == list(range(len(leaves) - 1))


def test_tree_to_merge_orders_nary():
    """An n-ary tree binarizes; merge_orders must have len == leaves-1 and
    reconstruct the same leaf coverage via the merge-trajectory builder."""
    tree = _mk_tree()
    leaves, mo = tree_to_merge_orders(tree, direction="right")
    assert len(leaves) == 5
    assert len(mo) == 4
    root = get_tree_from_merge_trajectory(np.array(mo), len(leaves))
    assert root.i == 0 and root.j == len(leaves) - 1
    order = []
    def _collect(node, out):
        if node.left is None and node.right is None:
            out.append(node.i)
            return
        if node.left:
            _collect(node.left, out)
        if node.right:
            _collect(node.right, out)
    _collect(root, order)
    assert order == list(range(5)), order


def test_gold_tree_dataset_real_corpus():
    """End-to-end on the real BBC tree stream: a few samples parse cleanly."""
    import os
    import pytest
    tree_npy = "dataset/bbc-news/tree/dev.npy"
    tok = "dataset/bbc-news/TG_GPT2_tokenizer.json"
    if not (os.path.exists(tree_npy) and os.path.exists(tok)):
        pytest.skip("BBC tree corpus not present")
    from olmo.gpst.reader.dataset_gold import GoldTreeDataset
    ds = GoldTreeDataset(tree_npy, tok, max_seq_len=128, num_samples=20)
    assert len(ds) > 0
    checked = 0
    for i in range(min(20, len(ds))):
        item = ds[i]
        leaves = item["text"]
        mo = item["merge_orders"]
        assert len(leaves) >= 2
        assert len(mo) == len(leaves) - 1, (len(leaves), len(mo))
        assert mo.min() >= 0 and mo.max() <= len(leaves) - 2
        checked += 1
    assert checked >= 1
