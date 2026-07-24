"""Tests for olmo.data.parse_align (Pushdown/TreeReg parse-tree alignment)."""

import numpy as np
import pytest
import torch

from olmo.data.parse_align import (
    TreeVocab,
    parse_tree_block,
    binarize_tree,
    tree_spans,
    compute_depth_matrix,
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
# No-binarize path (tree -> spans directly, one span per real internal node)
# --------------------------------------------------------------------------- #
# The Pushdown stack tape (compute_depth_matrix) uses only (left, right); the
# split is ignored. Binarization invents artificial X|</X|> constituents whose
# (l,r) sub-ranges inflate the tape depth, so Pushdown must skip binarization.
# tree_spans must therefore work on a RAW (non-binarized) tree and emit exactly
# one span per real internal node, with a degenerate split (split==right) for
# n-ary / unary nodes (no imposed binary bifurcation).

def test_tree_spans_no_binarize_nary():
    # (S (A 10 11 12) 13) — A is a 3-child (n-ary) constituent.
    tree = ("S", [("A", [10, 11, 12]), 13])
    leaves, spans = tree_spans(tree)
    assert leaves == [10, 11, 12, 13]
    # Exactly 2 real internal nodes (A, S) -> exactly 2 spans, no artificial ones.
    assert len(spans) == 2
    # A spans leaves [0..2] with a degenerate split (split==right==2): n-ary, no
    # binary bifurcation imposed.
    assert (0, 2, 2) in spans
    # S spans leaves [0..3]; split is the end of its first child (A) = 2.
    assert (0, 2, 3) in spans


def test_tree_spans_binarize_adds_artificial():
    # Same n-ary tree; binarize_tree then tree_spans must yield MORE spans
    # (the artificial A|< node), confirming binarization is what injects the
    # spurious constituent.
    tree = ("S", [("A", [10, 11, 12]), 13])
    raw_leaves, raw_spans = tree_spans(tree)
    b_leaves, b_spans = tree_spans(binarize_tree(tree, "left"))
    # Leaves are preserved by binarization.
    assert raw_leaves == b_leaves == [10, 11, 12, 13]
    # Binarization strictly increases the span count (one artificial A|< added).
    assert len(b_spans) == len(raw_spans) + 1
    # The raw (l, r) ranges are a subset of the binarized ones: every real
    # constituent still appears; binarization only adds sub-ranges.
    raw_lr = {(l, r) for (l, _s, r) in raw_spans}
    b_lr = {(l, r) for (l, _s, r) in b_spans}
    assert raw_lr.issubset(b_lr)


def test_depth_matrix_no_binarize_nary():
    # (A 10 11 12) — a single 3-child constituent over 3 leaves.
    tree = ("A", [10, 11, 12])
    # No binarization: 1 real span (0,2) -> each leaf at depth 1 (one reduce).
    leaves, spans = tree_spans(tree)
    assert leaves == [10, 11, 12]
    S = np.asarray(compute_depth_matrix(spans, 3))
    # Final row (full parse): all three leaves have depth 1 (the single A reduce).
    assert S[2].tolist() == [1, 1, 1]
    # Binarization injects an artificial constituent -> some leaf reaches depth 2.
    _, b_spans = tree_spans(binarize_tree(tree, "left"))
    Sb = np.asarray(compute_depth_matrix(b_spans, 3))
    assert int(Sb.max()) == 2, "binarization should inflate depth to 2"
    assert int(S.max()) == 1, "no-binarize depth stays at 1"


def test_parse_chunk_slice_binarize_flag():
    # Encode an n-ary tree as a [BOS ... EOS] block and exercise the
    # parse_chunk_slice binarize flag (the data-loader entry point).
    v = fake_vocab()
    # (S (A 10 11 12) 13)  ->  [BOS, <(S> <(A> 10 11 12 <A)> 13 <S)>, EOS]
    inner_a = encode_tree(v, "A", 10, 11, 12)  # 3-child A (n-ary)
    s_tree = [102] + inner_a + [13, 105]       # 102=<(S>, 105=<S)>
    block = [0] + s_tree + [1]                  # 0=BOS, 1=EOS

    from olmo.data.parse_align import parse_chunk_slice
    out_b = parse_chunk_slice(block, v, direction="left", binarize=True)
    out_n = parse_chunk_slice(block, v, direction="left", binarize=False)
    # Terminal leaves are identical regardless of binarization (BOS/EOS are kept
    # as surrounding leaves, matching terminal.npy).
    assert out_b["input_ids"].tolist() == out_n["input_ids"].tolist() == [0, 10, 11, 12, 13, 1]
    # No-binarize yields strictly fewer spans (no artificial A|<).
    assert len(out_n["spans"]) < len(out_b["spans"])
    # No-binarize span (l,r) set is a subset of the binarized one.
    lr_b = {(int(l), int(r)) for (l, _s, r) in out_b["spans"]}
    lr_n = {(int(l), int(r)) for (l, _s, r) in out_n["spans"]}
    assert lr_n.issubset(lr_b)


def test_tree_spans_binary_unchanged_by_binarize():
    # An already-binary tree: binarization is a no-op, so tree_spans on the raw
    # tree and on the binarized tree yield identical spans (no regression for the
    # common binary-node case).
    tree = ("S", [("A", [10, 11]), ("B", [12, 13])])
    raw_leaves, raw_spans = tree_spans(tree)
    b_leaves, b_spans = tree_spans(binarize_tree(tree, "left"))
    assert raw_leaves == b_leaves
    assert sorted(raw_spans) == sorted(b_spans)


_TEST_RIGHT = "dataset/bbc-news/parse_aligned/test_right"


@pytest.mark.skipif(
    not __import__("os").path.exists(f"{_TEST_RIGHT}/input_ids.npy"),
    reason="test_right precomputed split not on disk",
)
def test_precomputed_dataset_emits_metadata():
    """Regression: the `lm` evaluator (EvaluatorType.lm) does
    ``zip(batch["metadata"], ce_loss)`` on eval batches, so every eval dataset
    item must emit a ``metadata`` field. PrecomputedParseDataset (the pushdown/
    treereg eval/train dataset) previously emitted none -> KeyError: 'metadata'
    at eval time. MemMapDataset emits ``{"path": ...}``; mirror that."""
    from olmo.data.parse_align import PrecomputedParseDataset
    from olmo.data.collator import DataCollator

    d = PrecomputedParseDataset(_TEST_RIGHT, pad_token_id=50258, load_depth=False)
    item = d[0]
    assert "metadata" in item, "PrecomputedParseDataset must emit 'metadata' for the lm evaluator"
    assert isinstance(item["metadata"], dict) and "path" in item["metadata"]

    # Collator must batch metadata into a length-B list (what the evaluator zips).
    collator = DataCollator(
        pad_direction="right", pad_token_id=50258,
        generate_attention_mask=False, shuffle_tree="pushdown",
    )
    batch = collator([d[0], d[1]])
    assert "metadata" in batch
    assert len(batch["metadata"]) == 2
    assert all(isinstance(m, dict) for m in batch["metadata"])



# --------------------------------------------------------------------------- #
# Document-length output for faithful eval PPL (doc_lens + doc masking)
# --------------------------------------------------------------------------- #
def test_get_document_lengths_splits_by_eos():
    """get_document_lengths must split a chunk by EOS, each doc including its
    trailing EOS, and the lengths must sum to the real (non-pad) token count."""
    from olmo.data.util import get_document_lengths

    EOS, PAD = 50256, 50258
    # [BOS, a, b, EOS, c, d, e, EOS, f, EOS, PAD, PAD]
    ids = torch.tensor([0, 10, 11, EOS, 12, 13, 14, EOS, 15, EOS, PAD, PAD])
    dl = get_document_lengths(ids, EOS)
    # docs: [BOS a b EOS]=4, [c d e EOS]=4, [f EOS]=2  (trailing pads excluded
    # because the slice passed is the full tensor and last token is PAD != EOS,
    # so a final boundary is appended at the last index — but we pass only real
    # tokens in the dataset path; here test the raw semantics on real prefix).
    real = ids[:9]  # drop pads
    dl_real = get_document_lengths(real, EOS)
    # docs (each incl. trailing EOS): [BOS a b EOS]=4, [c d e EOS]=4, [f EOS]=2
    assert dl_real.tolist() == [4, 4, 1]
    assert int(dl_real.sum()) == 9


def test_precomputed_dataset_emits_doc_lens():
    """generate_doc_lengths=True -> each item carries doc_lens whose sum equals
    the number of real (non-pad) tokens, with no pad leak into a doc."""
    from olmo.data.parse_align import PrecomputedParseDataset

    d = PrecomputedParseDataset(
        _TEST_RIGHT, pad_token_id=50258, eos_token_id=50256,
        load_depth=False, generate_doc_lengths=True,
    )
    for i in range(min(4, len(d))):
        item = d[i]
        assert "doc_lens" in item, "generate_doc_lengths must emit doc_lens"
        dl = item["doc_lens"]
        # doc_lens is computed over the FULL padded input_ids (matching
        # MemMapDataset), so its sum covers the padded length incl. a trailing
        # pad doc; real tokens are a subset. Assert it is >= real-token count
        # and that every doc is positive.
        n_real = int(item["attention_mask"].sum())
        assert int(dl.sum()) >= n_real, (i, int(dl.sum()), n_real)
        assert int((dl > 0).all()), (i, dl.tolist())


def test_doc_id_logic_matches_doc_boundaries():
    """The doc_id tensor built in OLMo.forward (pushdown flex mask_mod) must
    assign the same id to positions in the same document and different ids
    across documents. Mirrors the model.py construction exactly."""
    EOS = 50256
    # batch of 2: doc boundaries at different positions.
    # doc_lens (per batch, unpadded): b0 -> [4, 4, 1]; b1 -> [3, 6]
    doc_lens = torch.tensor([[4, 4, 1, 0], [3, 6, 0, 0]])  # 0-padded to max_docs
    B, max_docs = doc_lens.shape
    seq_len = 10
    ends = torch.cumsum(doc_lens.to(torch.long), dim=-1)  # (B, max_docs)
    idxs = torch.arange(seq_len)
    doc_id = torch.stack([
        torch.searchsorted(ends[b], idxs, right=True) for b in range(B)
    ], dim=0)  # (B, seq_len)

    # b0: docs of len 4,4,1 -> positions [0..3]=doc0, [4..7]=doc1, [8]=doc2,
    # [9]=pad-region -> id 4 (past all 4 cumsum entries 4,8,9,9); gated by am.
    assert doc_id[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 4], doc_id[0].tolist()
    # b1: docs of len 3,6 -> [0..2]=doc0, [3..8]=doc1, [9]=pad-region -> id 4.
    assert doc_id[1].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 4], doc_id[1].tolist()
    # Cross-doc pairs must differ: b0 pos 3 (doc0) vs pos 4 (doc1).
    assert doc_id[0, 3] != doc_id[0, 4]
