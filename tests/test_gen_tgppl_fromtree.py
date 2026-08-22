from __future__ import annotations

import numpy as np

from datatools.gen_tgppl_fromtree import (
    adjusted_sentence_index,
    validate_existing_conversion,
    validate_inputs,
    write_tg_data,
)


def test_streaming_tree_to_tg_conversion_and_validation(tmp_path):
    tree = np.asarray([1, 4, 10, 5, 2, 4, 11, 5], dtype=np.uint16)
    tree_lengths = np.asarray([5, 3], dtype=np.uint16)
    documents = np.asarray([2], dtype=np.uint32)
    close_ids = np.asarray([5], dtype=np.int64)
    validate_inputs(tree, tree_lengths, documents, samples_per_sentence=1)

    tg_lengths_path = tmp_path / "tg_sent_index.npy"
    closes = adjusted_sentence_index(
        tree, tree_lengths, close_ids, tg_lengths_path, batch_size=1
    )
    tg_path = tmp_path / "tg_300.npy"
    digest = write_tg_data(tree, close_ids, tg_path, closes, chunk_tokens=3)
    tg = np.load(tg_path, mmap_mode="r")
    tg_lengths = np.load(tg_lengths_path, mmap_mode="r")

    np.testing.assert_array_equal(tg, [1, 4, 10, 5, 5, 2, 4, 11, 5, 5])
    np.testing.assert_array_equal(tg_lengths, [6, 4])
    report = validate_existing_conversion(
        tree, tree_lengths, documents, tg, tg_lengths, documents,
        close_ids, batch_size=1,
    )
    assert closes == 2
    assert report["exact"] is True
    assert report["duplicated_closing_tokens"] == 2
    assert report["tg_blake2b"] == digest
