"""Regression tests for document-level testppl BOS/EOS normalization."""

from __future__ import annotations

import numpy as np
import pytest

from olmo.eval.downstream import normalize_testppl_document_record


BOS = 1
EOS = 2
PAD = 0


def normalize(values, *, first=False, last=False):
    return normalize_testppl_document_record(
        np.asarray(values, dtype=np.uint16),
        first_in_document=first,
        last_in_document=last,
        bos_token_id=BOS,
        eos_token_id=EOS,
        pad_token_id=PAD,
    )


@pytest.mark.parametrize(
    ("values", "first", "last", "expected"),
    [
        ([BOS, 10], True, False, [BOS, 10]),
        ([10], True, False, [BOS, 10]),
        ([10, EOS], False, True, [10, EOS]),
        ([10], False, True, [10, EOS]),
        ([BOS, 10, EOS], True, True, [BOS, 10, EOS]),
        ([10], True, True, [BOS, 10, EOS]),
        ([10], False, False, [10]),
    ],
)
def test_normalize_testppl_document_record_is_idempotent_and_complete(
    values, first, last, expected
):
    normalized = normalize(values, first=first, last=last)
    assert normalized.tolist() == expected
    assert normalize(normalized, first=first, last=last).tolist() == expected


@pytest.mark.parametrize(
    ("values", "first", "last", "message"),
    [
        ([BOS, BOS, 10], True, False, "duplicated"),
        ([10, EOS, EOS], False, True, "duplicated"),
        ([10, BOS], True, False, "BOS occurs outside"),
        ([BOS, 10], False, False, "BOS occurs outside"),
        ([EOS, 10], False, True, "EOS occurs outside"),
        ([10, EOS], False, False, "EOS occurs outside"),
        ([10, PAD], False, False, "PAD"),
    ],
)
def test_normalize_testppl_document_record_rejects_corrupt_boundaries(
    values, first, last, message
):
    with pytest.raises(ValueError, match=message):
        normalize(values, first=first, last=last)


def test_normalize_testppl_document_record_rejects_non_vector_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_testppl_document_record(
            np.asarray([[10]], dtype=np.uint16),
            first_in_document=False,
            last_in_document=False,
            bos_token_id=BOS,
            eos_token_id=EOS,
            pad_token_id=PAD,
        )
