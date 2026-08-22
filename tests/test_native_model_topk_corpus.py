from __future__ import annotations

import json

import numpy as np

from olmo.eval.native_model_topk_corpus import NativeModelTopKShard
from scripts.generate_native_nary_test import (
    SentenceInput,
    _decode_and_adapt,
    _prepare_shard_arrays,
    _write_result,
)


def test_v2_loader_keeps_pushdown_and_gpst_candidate_axes_independent(tmp_path):
    row = SentenceInput(
        global_sentence_id=7,
        document_id=2,
        terminal_tokens=(1, 10, 11, 12, 13, 14, 2),
        content_start=1,
        content_end=6,
        words=("a", "b", "c"),
        word_piece_ids=((10, 11), (12,), (13, 14)),
    )
    chart = np.random.default_rng(7).normal(size=(3, 3, 5))
    _, pushdown, pushdown_spans, gpst = _decode_and_adapt(
        (0, chart, row.word_piece_ids, row.content_start, 300, "both")
    )
    arrays, _token_offsets, gpst_offsets, pushdown_offsets = _prepare_shard_arrays(
        tmp_path, [row]
    )
    _write_result(
        0, pushdown, pushdown_spans, gpst, [row], arrays,
        gpst_offsets, pushdown_offsets,
    )
    for array in arrays.values():
        array.flush()
    (tmp_path / "run.json").write_text(
        json.dumps({"status": "complete", "format_version": 2}), encoding="utf-8"
    )

    sentence = NativeModelTopKShard(tmp_path).sentence(0)
    assert sentence.pushdown_valid_count == 3
    assert sentence.gpst_valid_count == 2
    assert sentence.pushdown_proposal_scores.shape == (3,)
    assert sentence.gpst_proposal_scores.shape == (2,)
    assert sentence.pushdown_spans.shape == (3, 2, 3)
    assert sentence.gpst_merge_orders.shape == (2, 4)
    assert len({tuple(row) for row in sentence.gpst_merge_orders}) == 2
