"""Phase 0 test: the C++ chart-table backend builds + produces correct
post-order generation indices.

We drive ``CPPChartTableManager`` directly (no model) on a tiny hand-checked
example: sentence ``A B C`` (3 tokens) merged as ``((A B) C)``.

Merge-order semantics (from cpp_extension/py_backend.cpp TableManager ctor):
``merge_orders`` is a per-sentence ``(L-1,)`` array of *gap indices* giving the
order adjacent cells merge. Gap 0 is between tokens 0-1, gap 1 between 1-2.
For ``((A B) C)`` we first merge gap 0 (A+B -> AB) then gap 1 (AB+C) -> [0, 1].

``prepare_generation`` returns (span_masks, split_targets, ldr_cache_ids,
position_ids, tgt_ids, token_indices, ext_vocab_ids) where:
- ``ldr_cache_ids[g, t]`` = cache id of the t-th node in post-order (the
  surrogate input sequence for the generative model);
- ``tgt_ids[g, t]`` = the action target: ``reduce_id`` for a COMP (non-terminal)
  or the input token id for a GEN (terminal);
- ``split_targets`` = the gold split position per non-terminal (for parser loss);
- ``token_indices`` = where each terminal's GEN sits in the post-order sequence.

Post-order of ``((A B) C)`` is: A, B, COMP(AB), C, COMP(ABC)  (5 nodes).
Action sequence (M-step Eq.): GEN(A), GEN(B), COMP, GEN(C), COMP.
"""
from __future__ import annotations

import numpy as np
import torch

import olmo.gpst  # noqa: F401  (ensures package importable)


def _make_mgr(seq_lens, merge_orders, window_size=2):
    from olmo.gpst.data_structure.py_backend import CPPChartTableManager
    group_ids = np.arange(len(seq_lens), dtype=np.int32)
    if not isinstance(merge_orders, np.ndarray):
        merge_orders = np.array(merge_orders, dtype=np.int32)
    return CPPChartTableManager(
        seq_lens=np.array(seq_lens, dtype=np.int32),
        window_size=window_size,
        merge_orders=merge_orders,
        cache_id_offset=3,           # SPECIAL_TOKEN_NUM
        detach_cache_id_offset=0,    # unused here
        group_ids=group_ids,
    )


def test_import_cppbackend():
    # Importing the wrapper module bootstraps sys.path to find cppbackend.so.
    import olmo.gpst.data_structure.py_backend as pb  # noqa: F401
    import cppbackend
    assert hasattr(cppbackend, "TableManager")


def test_construct_inside_groups_single_sentence():
    seq_lens = [3]                      # A B C
    merge_orders = [[0, 1]]            # ((A B) C)
    mgr = _make_mgr(seq_lens, merge_orders)
    device = torch.device("cpu")
    tgt_ids, span_ids, cache_ids, detach_ids = mgr.construct_inside_groups(device)
    # 2 inside steps for a 3-token sentence (levels 1 and 2).
    assert isinstance(tgt_ids, list)
    assert len(tgt_ids) == 2
    roots = mgr.root_ids
    assert roots.shape[0] == 1


def test_prepare_generation_postorder():
    """For ((A B) C) the post-order action targets must be:
    GEN(A), GEN(B), COMP, GEN(C), COMP  -> with shift-right the tgt seq has
    reduce at the two COMP positions and each terminal exactly once."""
    seq_lens = [3]
    merge_orders = [[0, 1]]
    mgr = _make_mgr(seq_lens, merge_orders)
    device = torch.device("cpu")

    tgt_cache, span_ids, cache_ids, detach_ids = mgr.construct_inside_groups(device)
    # score_orders / split_orders: per-step (num_cells, split_size) int arrays
    # giving the ranking of split candidates. induce_best_splits picks
    # best_splits[step][cell, 0] as the best split, so a zeros tensor (all pointing
    # to candidate 0) makes every cell's best split its candidate 0 — exactly the
    # canonical left split, which reconstructs ((A B) C) for merge_orders [0, 1].
    score_orders = [torch.zeros(tuple(c.shape[:2]), dtype=torch.long) for c in cache_ids]

    input_ids = np.array([[10, 20, 30]], dtype=np.int32)   # synthetic token ids
    group_ids = np.array([0], dtype=np.int32)
    eos_labels = np.array([50256], dtype=np.int32)
    reduce_id = 50257
    max_input_len = 3

    span_masks, split_targets, ldr_cache_ids, position_ids, tgt_ids, \
        token_indices, ext_vocab_ids = mgr.prepare_generation(
            score_orders, score_orders,
            atom_spans=None,
            input_ids=input_ids,
            groups_ids=group_ids,
            eos_id=50256,
            reduce_id=reduce_id,
            max_input_len=max_input_len)

    # tgt_ids shape: (group_size, max_seq_len + 1)
    assert tgt_ids.shape[0] == 1
    seq = tgt_ids[0].tolist()
    assert eos_labels[0] in seq, f"eos label {eos_labels[0]} not in {seq}"
    eos_pos = seq.index(int(eos_labels[0]))
    prefix = seq[:eos_pos + 1]
    assert reduce_id in prefix, f"reduce id {reduce_id} not in {prefix}"
    # The three terminals A,B,C must each appear exactly once as a GEN target.
    for tok in (10, 20, 30):
        assert prefix.count(tok) == 1, f"token {tok} count != 1 in {prefix}"
    # Exactly two COMP actions for a 3-token binary tree.
    assert prefix.count(reduce_id) == 2, f"expected 2 reduce, got {prefix.count(reduce_id)} in {prefix}"


def test_prepare_generation_two_sentences():
    """Batch of two sentences; group_ids 0,1 (no chunking)."""
    seq_lens = [3, 2]
    # merge_orders must be a rectangular (N, max_seq_len-1) array; pad shorter
    # sentences. The C++ TableManager reads only seq_len-1 entries per sentence.
    merge_orders = np.array([[0, 1], [0, -1]], dtype=np.int32)
    mgr = _make_mgr(seq_lens, merge_orders)
    device = torch.device("cpu")
    tgt_cache, span_ids, cache_ids, detach_ids = mgr.construct_inside_groups(device)
    score_orders = [torch.zeros(tuple(c.shape[:2]), dtype=torch.long) for c in cache_ids]
    input_ids = np.array([[10, 20, 30], [40, 50, 0]], dtype=np.int32)
    group_ids = np.array([0, 1], dtype=np.int32)
    eos_labels = np.array([50256, 50256], dtype=np.int32)
    span_masks, split_targets, ldr_cache_ids, position_ids, tgt_ids, \
        token_indices, ext_vocab_ids = mgr.prepare_generation(
            score_orders, score_orders, None, input_ids, group_ids,
            eos_id=50256, reduce_id=50257, max_input_len=3)
    assert tgt_ids.shape[0] == 2
    seq2 = tgt_ids[1].tolist()
    assert seq2.count(50257) == 1
