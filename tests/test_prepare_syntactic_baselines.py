"""End-to-end alignment tests for datatools/prepare_syntactic_baselines.py.

These verify the docstring claim that ``ParseAlignedDataset``'s ``input_ids``
are bit-identical to ``terminal/*.npy`` — the foundation for cross-baseline
PPL comparison between Pushdown/TreeReg and the terminal baseline.

Three layers:
  (A) Synthetic unit test — no data dependency; covers oversize atomic tree.
  (B) Real-data integration test — bbc-news/tree/dev vs terminal/dev bit-identity.
  (C) Stats regression test — guards the ``chunks[:1]`` min bug in build_chunk_index.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from datatools.prepare_syntactic_baselines import build_chunk_index
from olmo.data.parse_align import (
    ParseAlignedDataset,
    TreeVocab,
    iter_tree_chunks,
    parse_chunk_slice,
)

# Paths to the real training/eval data (GPT-2 TG tokenizer + bbc-news tree stream).
_TOK = "dataset/bbc-news/TG_GPT2_tokenizer.json"
_TREE_DEV = "dataset/bbc-news/tree/dev.npy"
_TERM_DEV = "dataset/bbc-news/terminal/dev.npy"

_HAVE_REAL = os.path.exists(_TOK) and os.path.exists(_TREE_DEV) and os.path.exists(_TERM_DEV)


# --------------------------------------------------------------------------- #
# Synthetic tokenizer + tree-stream builder
# --------------------------------------------------------------------------- #
def _write_synth_tokenizer(path: str) -> str:
    """Write a minimal HF tokenizer JSON with <(LABEL>/<LABEL)> NT tokens.

    Token ids: 0=BOS, 1=EOS, 2=PAD, 100..199 plain leaves, 10..12 opening NTs
    (A/B/S), 13..15 closing NTs. Matches the ``TreeVocab.from_tokenizer_file``
    format (op range [10,12], cl range [13,15]). Wide leaf-id range so we can
    build an atomic tree with many distinct leaves.
    """
    added = [
        {"content": "<|beginoftext|>", "id": 0},
        {"content": "</s>", "id": 1},
        {"content": "<|pad|>", "id": 2},
        {"content": "<(A>", "id": 10}, {"content": "<(B>", "id": 11}, {"content": "<(S>", "id": 12},
        {"content": "<A)>", "id": 13}, {"content": "<B)>", "id": 14}, {"content": "<S)>", "id": 15},
    ]
    # Plain leaf tokens 100..199 in the base vocab.
    vocab = {f"leaf{i}": i for i in range(100, 200)}
    # NT tokens must also be in the combined map (added_tokens override vocab).
    for a in added:
        vocab[a["content"]] = a["id"]
    tok = {"model": {"vocab": vocab}, "added_tokens": added}
    with open(path, "w") as f:
        json.dump(tok, f)
    return path


def _encode_tree(label: str, *children) -> list:
    """Build a tree-token list: <(label> ...children... <label)>.

    Children are ints (leaves) or nested ``_encode_tree`` lists. Wide non-binary
    trees are fine — ``binarize_tree`` handles >2 children. Avoid deeply
    left/right-nested binary trees: ``tree_spans`` has an edge-case IndexError
    on very deep nesting (a separate known issue, not exercised here).
    """
    op = {"A": 10, "B": 11, "S": 12}[label]
    cl = {"A": 13, "B": 14, "S": 15}[label]
    out = [op]
    for c in children:
        if isinstance(c, list):
            out.extend(c)
        else:
            out.append(c)
    out.append(cl)
    return out


def _build_synth_stream() -> np.ndarray:
    """A 1D tree stream: BOS, several small trees with leaf-runs between them,
    a final EOS, plus one wide atomic tree whose leaves exceed max_len."""
    stream = [0]  # BOS
    stream += _encode_tree("S", 100, 101)        # small tree (2 leaves)
    stream += [102, 103]                         # between-tree leaf run
    stream += _encode_tree("S", _encode_tree("A", 104, 105), 106)  # nested (3 leaves)
    stream += [0, 1]                             # another doc's BOS/EOS run
    # Atomic wide tree: 60 leaves > max_len=16.
    stream += _encode_tree("S", *list(range(110, 170)))
    stream += [1]                                # trailing EOS
    return np.asarray(stream, dtype=np.int64)


# --------------------------------------------------------------------------- #
# (A) Synthetic unit test
# --------------------------------------------------------------------------- #
def test_chunk_index_alignment_synthetic(tmp_path):
    """iter_tree_chunks + parse_chunk_slice: leaves reconstruct the stream,
    slices cover the stream with no overlap, every chunk is bracket-balanced."""
    tok_path = _write_synth_tokenizer(str(tmp_path / "tok.json"))
    v = TreeVocab.from_tokenizer_file(tok_path)
    stream = _build_synth_stream()
    max_len = 16  # smaller than the 60-leaf atomic tree

    tree_path = str(tmp_path / "tree.npy")
    np.save(tree_path, stream.astype(np.uint16))

    out_dir = str(tmp_path / "out")
    build_chunk_index(tree_path, tok_path, max_len, out_dir)

    # Load the produced chunk_index and reconstruct leaves via parse_chunk_slice.
    idx = np.load(os.path.join(out_dir, "chunk_index.npy"))
    tree = np.load(tree_path, mmap_mode="r")
    recon = []
    prev_end = 0
    cover_ok = True
    for s, l in idx:
        s, l = int(s), int(l)
        if s != prev_end:
            cover_ok = False
        prev_end = s + l
        out = parse_chunk_slice(np.asarray(tree[s:s + l]), v, "right")
        recon.extend(int(x) for x in out["input_ids"])
    recon = np.asarray(recon, dtype=np.int64)

    # Ground-truth leaves = all non-NT tokens of the stream, in order.
    is_nt = (stream >= v.op_lo) & (stream <= v.cl_hi)
    true_leaves = stream[~is_nt]

    # Invariant 1: leaf identity (the docstring "bit-identical to terminal" claim).
    assert np.array_equal(recon, true_leaves), (
        f"recon != true_leaves; recon len {len(recon)} vs {len(true_leaves)}")
    # Invariant 2: full coverage, no overlap.
    assert cover_ok and prev_end == len(stream), "chunks do not tile the stream"
    # Invariant 3: every chunk is bracket-balanced (whole-tree integrity).
    for s, l in idx:
        seg = np.asarray(tree[int(s):int(s) + int(l)])
        n_op = int(((seg >= v.op_lo) & (seg <= v.op_hi)).sum())
        n_cl = int(((seg >= v.cl_lo) & (seg <= v.cl_hi)).sum())
        assert n_op == n_cl, f"chunk {s}:{l} unbalanced (op={n_op} cl={n_cl})"


def test_chunk_index_oversize_atomic_tree_is_own_chunk(tmp_path):
    """An atomic tree whose leaves exceed max_len is emitted as its own chunk
    (never cut mid-tree). iter_tree_chunks packs by in-tree leaves, so a chunk
    whose in-tree leaves exceed max_len must be a single atomic tree."""
    tok_path = _write_synth_tokenizer(str(tmp_path / "tok.json"))
    v = TreeVocab.from_tokenizer_file(tok_path)
    stream = _build_synth_stream()
    max_len = 16

    chunks = iter_tree_chunks(stream, v, direction="right", max_len=max_len)
    # In-tree leaf count per chunk (iter_tree_chunks' packing budget).
    def in_tree_leaves(seg):
        depth = 0
        n = 0
        for t in seg:
            t = int(t)
            if v.op_lo <= t <= v.op_hi:
                depth += 1
            elif v.cl_lo <= t <= v.cl_hi:
                depth -= 1
            elif depth > 0:
                n += 1
        return n

    oversize = [c for c in chunks if in_tree_leaves(np.asarray(stream[c[0]:c[0] + c[1]])) > max_len]
    assert len(oversize) == 1, f"expected exactly 1 oversize atomic chunk, got {len(oversize)}"
    # The oversize chunk must be bracket-balanced (a whole tree, not a fragment).
    s, l = oversize[0]
    seg = np.asarray(stream[s:s + l])
    assert int(((seg >= v.op_lo) & (seg <= v.op_hi)).sum()) == \
           int(((seg >= v.cl_lo) & (seg <= v.cl_hi)).sum())


# --------------------------------------------------------------------------- #
# (B) Real-data integration test
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAVE_REAL, reason="bbc-news tree/terminal dev data not on disk")
def test_real_dev_bit_identity_with_terminal(tmp_path):
    """bbc-news/tree/dev.npy -> build_chunk_index -> parse_chunk_slice must
    reconstruct terminal/dev.npy token-for-token. This is the real end-to-end
    check of the docstring's 'bit-identical to terminal/*.npy' claim, via the
    untruncated parse path (ParseAlignedDataset truncates oversize chunks — see
    test_real_dev_parsedataset_truncates_oversize_chunks)."""
    v = TreeVocab.from_tokenizer_file(_TOK)
    out_dir = str(tmp_path / "out")
    build_chunk_index(_TREE_DEV, _TOK, max_len=2048, out_dir=out_dir)

    idx = np.load(os.path.join(out_dir, "chunk_index.npy"))
    tree = np.load(_TREE_DEV, mmap_mode="r")
    recon = []
    for s, l in idx:
        s, l = int(s), int(l)
        out = parse_chunk_slice(np.asarray(tree[s:s + l]), v, "right")
        recon.extend(int(x) for x in out["input_ids"])
    recon = np.asarray(recon, dtype=np.int64)

    term = np.load(_TERM_DEV)
    assert np.array_equal(recon, np.asarray(term, dtype=np.int64)), (
        f"recon ({len(recon)}) != terminal/dev ({len(term)})")


@pytest.mark.skipif(not _HAVE_REAL, reason="bbc-news tree/terminal dev data not on disk")
def test_real_dev_parsedataset_bit_identity_with_terminal(tmp_path):
    """ParseAlignedDataset (the production consume path, with its [:max_len]
    truncation logic) must reconstruct terminal/dev.npy exactly. This guards
    the iter_tree_chunks packing fix: chunks must pack <= max_len leaves so the
    truncation path is never triggered (previously dropped ~0.7% of dev tokens)."""
    v = TreeVocab.from_tokenizer_file(_TOK)
    out_dir = str(tmp_path / "out")
    build_chunk_index(_TREE_DEV, _TOK, max_len=2048, out_dir=out_dir)

    ds = ParseAlignedDataset(
        tree_npy=_TREE_DEV,
        chunk_index_npy=os.path.join(out_dir, "chunk_index.npy"),
        tokenizer=_TOK,
        direction="right",
        max_len=2048,
    )
    recon = []
    for i in range(len(ds)):
        recon.extend(int(x) for x in ds[i]["input_ids"].tolist())
    recon = np.asarray(recon, dtype=np.int64)

    # Guard the packing fix directly: recompute per-chunk leaf counts from the
    # raw tree slices and assert none exceed max_len (so the [:max_len]
    # truncation path in __getitem__ never fires and drops tokens).
    idx = np.load(os.path.join(out_dir, "chunk_index.npy"))
    tree = np.load(_TREE_DEV, mmap_mode="r")
    for s, l in idx:
        seg = np.asarray(tree[int(s):int(s) + int(l)])
        is_nt = (seg >= v.op_lo) & (seg <= v.cl_hi)
        assert int((~is_nt).sum()) <= ds.max_len, (
            f"chunk {s}:{l} has {int((~is_nt).sum())} leaves > max_len {ds.max_len}; "
            "iter_tree_chunks packing regressed")

    term = np.load(_TERM_DEV)
    assert np.array_equal(recon, np.asarray(term, dtype=np.int64))


@pytest.mark.skipif(not _HAVE_REAL, reason="bbc-news tree/terminal dev data not on disk")
def test_real_dev_chunk_coverage(tmp_path):
    """Chunk slices must tile the dev stream: full coverage, no overlap."""
    v = TreeVocab.from_tokenizer_file(_TOK)
    out_dir = str(tmp_path / "out")
    build_chunk_index(_TREE_DEV, _TOK, max_len=2048, out_dir=out_dir)
    idx = np.load(os.path.join(out_dir, "chunk_index.npy"))
    tree = np.load(_TREE_DEV, mmap_mode="r")

    prev_end = 0
    for s, l in idx:
        s, l = int(s), int(l)
        assert s == prev_end, f"gap/overlap at {s} (prev_end {prev_end})"
        prev_end = s + l
    assert prev_end == int(tree.shape[0]), "chunks do not cover the full stream"


# --------------------------------------------------------------------------- #
# (C) Stats regression test
# --------------------------------------------------------------------------- #
def test_stats_min_covers_all_chunks(tmp_path, capsys):
    """Regression for the ``chunks[:1]`` bug: the printed min leaf count must
    equal the true min over ALL chunks, not just the first."""
    tok_path = _write_synth_tokenizer(str(tmp_path / "tok.json"))
    v = TreeVocab.from_tokenizer_file(tok_path)
    stream = _build_synth_stream()
    max_len = 16
    tree_path = str(tmp_path / "tree.npy")
    np.save(tree_path, stream.astype(np.uint16))

    build_chunk_index(tree_path, tok_path, max_len, str(tmp_path / "out"))
    out = capsys.readouterr().out

    # Independently recompute the true min over all chunks.
    chunks = iter_tree_chunks(np.load(tree_path, mmap_mode="r"), v, "right", max_len=max_len)
    true_mins = []
    for s, l in chunks:
        seg = np.asarray(np.load(tree_path, mmap_mode="r")[s:s + l])
        is_nt = (seg >= v.op_lo) & (seg <= v.cl_hi)
        true_mins.append(int((~is_nt).sum()))
    true_min = min(true_mins) if true_mins else 0

    assert f"min {true_min}" in out, (
        f"printed stats did not contain 'min {true_min}'; got:\n{out}")
