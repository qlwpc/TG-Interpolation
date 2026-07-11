"""Tests for olmo.data.parse_align (Pushdown/TreeReg parse-tree alignment)."""

import numpy as np
import pytest

from olmo.data.parse_align import (
    TreeVocab,
    parse_tree_block,
    binarize_tree,
    tree_spans,
    compute_depth_matrix,
    chunk_units,
    chunk_to_tensors,
)


# A minimal fake vocab: opening NTs 100..102 (labels A,B,S), closing 103..105.
def fake_vocab() -> TreeVocab:
    # label_of_opening reads id2tok; populate it.
    v = TreeVocab(op_lo=100, op_hi=102, cl_lo=103, cl_hi=105, bos=0, eos=1, pad=2,
                  id2tok={100: "<(A>", 101: "<(B>", 102: "<(S>", 103: "<A)>", 104: "<B)>", 105: "<S)>"})
    return v


def encode_tree(vocab: TreeVocab, label: str, *children) -> list:
    """Build a tree-token list: <(label> ...children... <label)>."""
    op = {"A": 100, "B": 101, "S": 102}[label]
    cl = {"A": 103, "B": 104, "S": 105}[label]
    out = [op]
    for c in children:
        if isinstance(c, list):
            out.extend(c)
        else:
            out.append(c)
    out.append(cl)
    return out


# --------------------------------------------------------------------------- #
# Tree parsing
# --------------------------------------------------------------------------- #
def test_parse_tree_block_basic():
    v = fake_vocab()
    # [BOS, <(S> <(A> 10 11 <A)> <(B> 12 <B)> <S)>, EOS]
    inner_a = encode_tree(v, "A", 10, 11)
    inner_b = encode_tree(v, "B", 12)
    s_tree = [102] + inner_a + inner_b + [105]
    block = [0] + s_tree + [1]
    prefix, tree, suffix = parse_tree_block(block, v)
    assert prefix == [0]
    assert suffix == [1]
    assert tree is not None
    label, children = tree
    assert label == "S"
    assert len(children) == 2
    # First child = (A, [10,11]); second = (B,[12])
    assert children[0] == ("A", [10, 11])
    assert children[1] == ("B", [12])


def test_parse_tree_block_no_tree():
    v = fake_vocab()
    block = [0, 10, 11, 1]  # no NTs
    prefix, tree, suffix = parse_tree_block(block, v)
    assert prefix == [0, 10, 11, 1]
    assert tree is None
    assert suffix == []


# --------------------------------------------------------------------------- #
# Binarization
# --------------------------------------------------------------------------- #
def test_binarize_left_and_right_preserve_leaves():
    # (A 1 2 3 4)
    v = fake_vocab()
    node = ("A", [1, 2, 3, 4])
    bl = binarize_tree(node, "left")
    # Leaves in order
    assert _leaves(bl) == [1, 2, 3, 4]
    # Left-binarized: root has 2 children, the 2nd is a chain of "|<" nodes.
    assert len(bl[1]) == 2
    assert bl[1][0] == 1
    assert isinstance(bl[1][1], tuple) and bl[1][1][0] == "A|<"
    br = binarize_tree(node, "right")
    assert _leaves(br) == [1, 2, 3, 4]
    assert len(br[1]) == 2
    assert br[1][1] == 4
    assert isinstance(br[1][0], tuple) and br[1][0][0] == "A|>"


def test_binarize_already_binary_unchanged():
    node = ("A", [1, 2])
    assert binarize_tree(node, "left") == ("A", [1, 2])
    assert binarize_tree(node, "right") == ("A", [1, 2])


def _leaves(node):
    if isinstance(node, int):
        return [node]
    out = []
    for c in node[1]:
        out.extend(_leaves(c))
    return out


# --------------------------------------------------------------------------- #
# Depth matrix: the paper's stale-tape example
# --------------------------------------------------------------------------- #
def test_depth_matrix_paper_example():
    """Murty 2023 example: parse [[The dog][is happy]].
    At prefix 3 ("[The dog] is") tape = [1,1,0]; at prefix 4 (after reduce)
    tape = [2,2,2,2]. S[k,j] = #constituents containing j that have closed by k.

    Constituents (leaf indices 0-based): [The dog]=span(0,1) closes at r=1;
    [is happy]=span(2,3) closes at r=3; [[The dog][is happy]]=span(0,3) closes at r=3.
    """
    spans = [(0, 1, 1), (2, 3, 3), (0, 3, 3)]  # (left, split, right)
    n = 4
    S = compute_depth_matrix(spans, n)
    S = np.asarray(S)
    # Row k=3 (prefix of all 4 tokens): depths [2,2,2,2]
    assert S[3].tolist() == [2, 2, 2, 2], f"row 3 = {S[3].tolist()}"
    # Row k=2 (prefix "The dog is", first 3 tokens): [The dog] closed (r=1<=2),
    # [is happy] not closed (r=3>2), outer not closed. So The=1,dog=1,is=0.
    assert S[2].tolist() == [1, 1, 0, 0], f"row 2 = {S[2].tolist()}"
    # Row k=1 (prefix "The dog"): [The dog] closed. The=1, dog=1.
    assert S[1].tolist() == [1, 1, 0, 0], f"row 1 = {S[1].tolist()}"
    # Lower triangular: S[k,j]=0 for j>k.
    assert S[0].tolist() == [0, 0, 0, 0]


def test_depth_matrix_lower_triangular_and_nondecreasing():
    # Random binary tree spans.
    spans = [(0, 2, 4), (0, 1, 2), (3, 3, 4), (0, 4, 4)]
    S = np.asarray(compute_depth_matrix(spans, 5))
    for k in range(5):
        for j in range(5):
            if j > k:
                assert S[k, j] == 0
    # Depth of a fixed j is non-decreasing in k.
    for j in range(5):
        col = S[:, j]
        assert all(col[k + 1] >= col[k] for k in range(4))


def test_depth_matrix_zero_for_no_spans():
    S = np.asarray(compute_depth_matrix([], 4))
    assert S.shape == (4, 4)
    assert S.sum() == 0


# --------------------------------------------------------------------------- #
# tree_spans
# --------------------------------------------------------------------------- #
def test_tree_spans_binary():
    # (S (A 10 11) (B 12 13))
    tree = ("S", [("A", [10, 11]), ("B", [12, 13])])
    leaves, spans = tree_spans(tree)
    assert leaves == [10, 11, 12, 13]
    # Root span (0,1,3): left child [0..1], right [2..3], split=1.
    assert (0, 1, 3) in spans
    assert (0, 0, 1) in spans  # A: (0,0,1)
    assert (2, 2, 3) in spans  # B: (2,2,3)


# --------------------------------------------------------------------------- #
# Chunking: whole-tree integrity
# --------------------------------------------------------------------------- #
def test_chunk_units_whole_trees():
    v = fake_vocab()
    # Two small blocks, each [BOS <(S> a b <S)> EOS].
    def mkblock(a, b):
        return [0] + [102, a, b, 105] + [1]
    blocks = [mkblock(10, 11), mkblock(12, 13), mkblock(14, 15)]
    chunks = chunk_units(blocks, v, "left", max_len=6)
    # Each block is 6 terminals; max_len=6 -> one block per chunk.
    assert len(chunks) == 3
    for ch in chunks:
        assert sum(u.n_terminals for u in ch) <= 6


def test_chunk_units_packs_small_blocks():
    v = fake_vocab()
    def mkblock(a, b):
        return [0] + [102, a, b, 105] + [1]  # 6 terminals
    blocks = [mkblock(10, 11), mkblock(12, 13)]
    chunks = chunk_units(blocks, v, "left", max_len=12)
    assert len(chunks) == 1
    assert len(chunks[0]) == 2


def test_chunk_to_tensors_alignment():
    v = fake_vocab()
    # block: [BOS <(S> <(A> 10 11 <A)> <(B> 12 13 <B)> <S)> EOS]
    inner_a = encode_tree(v, "A", 10, 11)
    inner_b = encode_tree(v, "B", 12, 13)
    block = [0] + [102] + inner_a + inner_b + [105] + [1]
    chunks = chunk_units([block], v, "left", max_len=2048)
    assert len(chunks) == 1
    out = chunk_to_tensors(chunks[0])
    # terminals: [BOS, 10,11,12,13, EOS]
    assert out["terminals"].tolist() == [0, 10, 11, 12, 13, 1]
    # Depth matrix is 6x6 (one row/col per terminal incl BOS/EOS).
    assert out["depth_matrix"].shape == (6, 6)
    # BOS (idx 0) and EOS (idx 5) are not in any constituent -> depth 0 always.
    assert out["depth_matrix"][:, 0].sum() == 0
    assert out["depth_matrix"][:, 5].sum() == 0
    # Token 10 (idx 1) is inside [A 10 11] (closes at idx 2) and [S ...] (closes idx 4).
    # At prefix k=4 (all content): depth of idx1 = 2.
    assert int(out["depth_matrix"][4, 1]) == 2
    # Spans cover the content leaves (idx 1..4).
    assert out["spans"].shape[1] == 3
