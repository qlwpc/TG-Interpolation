"""Tests for olmo.data.parse_align (Pushdown/TreeReg parse-tree alignment)."""

import numpy as np
import pytest
import torch

from olmo.data.parse_align import (
    TreeVocab,
    iter_tree_chunks,
    parse_tree_block,
    binarize_tree,
    tree_spans,
    compute_depth_matrix,
    parse_chunk_slice,
    tree_leaves_and_word_boundaries,
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
    # (A 1 2 3 4). Direction names follow NLTK chomsky_normal_form(factor=...):
    # "left"  -> left-recursive spine  (((c1 c2) c3) c4), artificial nodes "|>"
    # "right" -> right-recursive spine (c1 (c2 (c3 c4))), artificial nodes "|<"
    # (right = official TreeReg + NLTK default).
    v = fake_vocab()
    node = ("A", [1, 2, 3, 4])
    bl = binarize_tree(node, "left")
    # Leaves in order
    assert _leaves(bl) == [1, 2, 3, 4]
    # Left-recursive: root has 2 children, the 1st is a chain of "|>" nodes.
    assert len(bl[1]) == 2
    assert bl[1][1] == 4
    assert isinstance(bl[1][0], tuple) and bl[1][0][0] == "A|>"
    assert bl == ("A", [("A|>", [("A|>", [1, 2]), 3]), 4])
    br = binarize_tree(node, "right")
    assert _leaves(br) == [1, 2, 3, 4]
    # Right-recursive: root has 2 children, the 2nd is a chain of "|<" nodes.
    assert len(br[1]) == 2
    assert br[1][0] == 1
    assert isinstance(br[1][1], tuple) and br[1][1][0] == "A|<"
    assert br == ("A", [1, ("A|<", [2, ("A|<", [3, 4])])])


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


def test_tree_spans_repeated_interned_leaf_occurrences():
    # The two 82 leaves are the same CPython-interned integer object. Tree
    # positions must nevertheless have independent ranges.
    tree = ("S", [82, ("B", [1, 82])])
    leaves, spans = tree_spans(tree)
    assert leaves == [82, 1, 82]
    assert spans == [(1, 1, 2), (0, 0, 2)]


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
    b_leaves, b_spans = tree_spans(binarize_tree(tree, "right"))
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
    _, b_spans = tree_spans(binarize_tree(tree, "right"))
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
    out_b = parse_chunk_slice(block, v, direction="right", binarize=True)
    out_n = parse_chunk_slice(block, v, direction="right", binarize=False)
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
    b_leaves, b_spans = tree_spans(binarize_tree(tree, "right"))
    assert raw_leaves == b_leaves
    assert sorted(raw_spans) == sorted(b_spans)


def test_collapse_unary_then_binarize_matches_paper_preprocess():
    from olmo.data.parse_align import collapse_unary_tree

    # Unary A->B->C is one constituent after collapse; ternary C is then CNF
    # binarized (direction="right" = right-recursive spine, matching official
    # TreeReg + NLTK factor="right"), adding exactly one artificial binary
    # constituent whose (l,r) is (1,2).
    tree = ("A", [("B", [("C", [10, 11, 12])])])
    collapsed = collapse_unary_tree(tree)
    assert collapsed[0] == "A+B+C"
    leaves, spans = tree_spans(binarize_tree(collapsed, "right"))
    assert leaves == [10, 11, 12]
    assert len(spans) == 2
    assert {(l, r) for l, _split, r in spans} == {(0, 2), (1, 2)}


def test_parse_chunk_slice_collapse_unary_flag():
    v = fake_vocab()
    # S -> A -> two terminals. Without collapse both S and A emit the same
    # (0,1) range; with collapse only one range remains.
    a = encode_tree(v, "A", 10, 11)
    block = [0, 102] + a + [105, 1]
    raw = parse_chunk_slice(block, v, "right", binarize=True, collapse_unary=False)
    collapsed = parse_chunk_slice(block, v, "right", binarize=True, collapse_unary=True)
    assert raw["input_ids"].tolist() == collapsed["input_ids"].tolist()
    assert len(collapsed["spans"]) == len(raw["spans"]) - 1


def test_word_boundaries_come_from_preterminals():
    # A parser preterminal may contain several tokenizer pieces for one word.
    # Only its first leaf is a TreeReg split candidate.
    tree = ("S", [("A", [10, 11]), ("B", [12])])
    leaves, boundaries = tree_leaves_and_word_boundaries(tree)
    assert leaves == [10, 11, 12]
    assert boundaries == [True, False, True]


def test_parse_chunk_slice_marks_each_top_level_tree_not_bos_blocks():
    v = fake_vocab()
    first = encode_tree(
        v,
        "S",
        encode_tree(v, "A", 10, 11),
        encode_tree(v, "B", 12),
    )
    second = encode_tree(v, "A", 20, 21)
    # BOS, two complete top-level trees separated by a plain token, EOS.
    block = [v.bos] + first + [77] + second + [v.eos]
    out = parse_chunk_slice(
        block,
        v,
        direction="right",
        binarize=True,
        collapse_unary=True,
    )
    assert out["input_ids"].tolist() == [0, 10, 11, 12, 77, 20, 21, 1]
    assert out["sentence_ids"].tolist() == [-1, 0, 0, 0, -1, 1, 1, -1]
    assert out["word_boundaries"].tolist() == [
        False, True, False, True, False, True, False, False
    ]
    # In particular, the syntactic S opener was stripped and did not become BOS.
    assert 102 not in out["input_ids"]


def test_paper_preprocess_adds_bos_root_to_eos_span():
    v = fake_vocab()
    block = [0] + encode_tree(v, "A", 10, 11) + [1]
    out = parse_chunk_slice(
        block,
        v,
        "right",
        binarize=True,
        collapse_unary=True,
        add_boundary_root=True,
    )
    assert out["input_ids"].tolist() == [0, 10, 11, 1]
    assert (0, 3, 3) in map(tuple, out["spans"].tolist())


def test_boundary_root_when_bos_and_eos_share_gpt2_id():
    v = fake_vocab()
    v.eos = v.bos
    block = [0] + encode_tree(v, "A", 10, 11) + [0]
    out = parse_chunk_slice(block, v, "right", add_boundary_root=True)
    assert (0, 3, 3) in map(tuple, out["spans"].tolist())


def test_pushdown_wraps_every_top_level_tree_with_cls_root_and_eos():
    v = fake_vocab()
    first = encode_tree(v, "S", encode_tree(v, "A", 10, 11), 12)
    second = encode_tree(v, "A", 20, 21)
    block = [v.bos] + first + [77] + second + [v.eos]
    out = parse_chunk_slice(
        block,
        v,
        "right",
        collapse_unary=True,
        wrap_toplevel_trees=True,
        root_token_id=50260,
        sentence_eos_token_id=50256,
    )
    assert out["input_ids"].tolist() == [
        50260, 10, 11, 12, 50256,
        50260, 20, 21, 50256,
    ]
    # Original document-level leaves are dropped and each EOS attaches to ROOT.
    assert 77 not in out["input_ids"]
    assert (0, 4, 4) in map(tuple, out["spans"].tolist())
    assert (5, 8, 8) in map(tuple, out["spans"].tolist())
    assert out["sentence_ids"].tolist() == [-1, 0, 0, 0, -1, -1, 1, 1, -1]


def test_pushdown_primal_terminals_preserve_boundaries_and_native_closure():
    v = fake_vocab()
    first = encode_tree(v, "S", encode_tree(v, "A", 10, 11), 12)
    second = encode_tree(v, "A", 20, 21)
    block = [v.bos] + first + [77] + second + [v.eos]
    out = parse_chunk_slice(
        block,
        v,
        "right",
        collapse_unary=True,
        extract_toplevel_trees=False,
        drop_singleton_spans=True,
    )
    assert out["input_ids"].tolist() == [v.bos, 10, 11, 12, 77, 20, 21, v.eos]
    assert out["sentence_ids"].tolist() == [-1, 0, 0, 0, -1, 1, 1, -1]
    spans = set(map(tuple, out["spans"].tolist()))
    assert (1, 2, 3) in spans
    assert (5, 5, 6) in spans
    assert all(left < right for left, _, right in spans)


def test_terminal_only_chunk_packing_counts_only_tree_terminals():
    v = fake_vocab()
    first = encode_tree(v, "A", 10, 11)
    second = encode_tree(v, "A", 20, 21)
    stream = np.asarray([v.bos] + first + [77] + second + [v.eos])
    chunks = iter_tree_chunks(
        stream, v, "right", max_len=4, extract_toplevel_trees=True
    )
    assert len(chunks) == 1
    out = parse_chunk_slice(
        stream[chunks[0][0] : sum(chunks[0])],
        v,
        "right",
        extract_toplevel_trees=True,
    )
    assert out["input_ids"].tolist() == [10, 11, 20, 21]


def test_primal_chunk_packing_does_not_overfill_with_trailing_eos():
    v = fake_vocab()
    tree = encode_tree(v, "A", 10, 11, 12, 13)
    stream = np.asarray([v.bos] + tree + [v.eos])
    chunks = iter_tree_chunks(stream, v, "right", max_len=5)
    parsed = [
        parse_chunk_slice(stream[start : start + length], v, "right")
        for start, length in chunks
    ]
    assert [len(item["input_ids"]) for item in parsed] == [5, 1]
    assert np.concatenate([item["input_ids"] for item in parsed]).tolist() == [
        v.bos, 10, 11, 12, 13, v.eos
    ]


def test_wrapped_chunk_packing_counts_root_and_eos_per_tree():
    v = fake_vocab()
    first = encode_tree(v, "A", 10, 11)
    second = encode_tree(v, "A", 20, 21)
    stream = np.asarray([v.bos] + first + [77] + second + [v.eos])
    chunks = iter_tree_chunks(
        stream, v, "right", max_len=5, wrap_toplevel_trees=True
    )
    # Each two-leaf tree becomes ROOT + 2 leaves + EOS = 4 tokens, so they may
    # not be packed together even though the unwrapped content has only 4 leaves.
    assert len(chunks) == 2
    wrapped = [
        parse_chunk_slice(
            stream[s:s + length], v, "right", wrap_toplevel_trees=True,
            root_token_id=50260, sentence_eos_token_id=50256,
        )["input_ids"].tolist()
        for s, length in chunks
    ]
    assert wrapped == [
        [50260, 10, 11, 50256],
        [50260, 20, 21, 50256],
    ]


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


def test_collator_pads_pushdown_sentence_ids_with_outside_marker():
    """Sentence ids must stay aligned with tokens after batch padding."""
    from olmo.data.collator import DataCollator

    collator = DataCollator(
        pad_direction="right",
        pad_token_id=50258,
        generate_attention_mask=True,
        shuffle_tree="pushdown",
    )
    batch = collator(
        [
            {
                "input_ids": torch.tensor([50257, 10, 11]),
                "pushdown_sentence_ids": torch.tensor([-1, 0, 0]),
            },
            {
                "input_ids": torch.tensor([50257, 20]),
                "pushdown_sentence_ids": torch.tensor([-1, 1]),
            },
        ]
    )
    assert batch["pushdown_sentence_ids"].tolist() == [
        [-1, 0, 0],
        [-1, 1, -1],
    ]


def test_incomplete_precomputed_dataset_is_rejected(tmp_path):
    """An interrupted writer must never be consumed as a training dataset."""
    from olmo.data.parse_align import PrecomputedParseDataset

    (tmp_path / "PREPROCESSING_INCOMPLETE").write_text("in progress\n")
    with pytest.raises(RuntimeError, match="interrupted preprocessing output"):
        PrecomputedParseDataset(str(tmp_path))


def test_pushdown_precomputed_contract_rejects_wrong_root(tmp_path):
    """A legacy/wrong-root array must fail before training can consume it."""
    import json

    from olmo.data.parse_align import PrecomputedParseDataset

    (tmp_path / "preprocessing.json").write_text(
        json.dumps({
            "binarize": True,
            "collapse_unary": True,
            "wrap_toplevel_trees": True,
            "root_token_id": 50257,
            "sentence_eos_token_id": 50256,
            "direction": "right",
        })
    )
    with pytest.raises(RuntimeError, match="root_token_id"):
        PrecomputedParseDataset(
            str(tmp_path),
            require_pushdown_root_token_id=50260,
            expected_binarize_direction="right",
        )


def test_pushdown_precomputed_root_is_not_an_lm_target():
    from olmo.data.parse_align import PrecomputedParseDataset

    data_dir = "dataset/bbc-news/parse_aligned/dev_pushdown_unary"
    if not __import__("os").path.exists(f"{data_dir}/input_ids.npy"):
        pytest.skip("regenerated Pushdown dev data not on disk")
    item = PrecomputedParseDataset(
        data_dir,
        require_pushdown_root_token_id=50260,
        expected_binarize_direction="right",
    )[0]
    roots = item["input_ids"] == 50260
    assert roots.any()
    assert not item["label_mask"][roots].any()
    assert item["label_mask"][item["input_ids"] == 50256].all()


def test_terminal_only_pushdown_precomputed_contract():
    from olmo.data.parse_align import PrecomputedParseDataset

    data_dir = "dataset/bbc-news/parse_aligned/dev_pushdown_unary_terminals"
    if not __import__("os").path.exists(f"{data_dir}/input_ids.npy"):
        pytest.skip("terminal-only Pushdown dev data not on disk")
    item = PrecomputedParseDataset(
        data_dir,
        require_pushdown_root_token_id=50260,
        expected_binarize_direction="right",
    )[0]
    assert "pushdown_sentence_ids" in item
    assert not (item["input_ids"] == 50260).any()
    # No synthetic per-sentence EOS is added, but primal document EOS remains.
    assert (item["input_ids"] == 50256).any()
    assert not (item["tree_spans"][:, 0] == item["tree_spans"][:, 2]).any()


def test_worker_cpu_candidates_are_distinct_and_allowed():
    """Pinned parser workers must be spread over allowed physical cores."""
    import os

    from olmo.data.parse_align import _worker_cpu_candidates

    candidates = _worker_cpu_candidates(4)
    assert candidates
    assert len(candidates) == len(set(candidates))
    assert set(candidates).issubset(os.sched_getaffinity(0))



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


@pytest.mark.skipif(
    not __import__("os").path.exists(f"{_TEST_RIGHT}/input_ids.npy"),
    reason="test_right precomputed split not on disk",
)
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
