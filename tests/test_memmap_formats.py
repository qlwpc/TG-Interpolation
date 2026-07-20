from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from olmo.data.memmap_dataset import MemMapDataset
from olmo.memmap_utils import inspect_memmap_file
from scripts.compute_decay_data_files import get_file_example_count


def _dataset(*paths, chunk_size=4, **kwargs):
    return MemMapDataset(
        *paths,
        chunk_size=chunk_size,
        memmap_dtype=np.uint16,
        include_instance_metadata=False,
        **kwargs,
    )


def test_standard_npy_skips_header(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(10, dtype=np.uint16))

    dataset = _dataset(path)

    assert len(dataset) == 2
    assert torch.equal(dataset[0]["input_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(dataset[1]["input_ids"], torch.tensor([4, 5, 6, 7]))


def test_raw_and_npy_files_can_be_mixed(tmp_path):
    raw_path = tmp_path / "raw.bin"
    npy_path = tmp_path / "standard.npy"
    np.arange(8, dtype=np.uint16).tofile(raw_path)
    np.save(npy_path, np.arange(10, 18, dtype=np.uint16))

    dataset = _dataset(raw_path, npy_path)

    assert len(dataset) == 4
    assert dataset.offsets == [(0, 2), (2, 4)]
    assert dataset[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert dataset[2]["input_ids"].tolist() == [10, 11, 12, 13]
    assert dataset[-1]["input_ids"].tolist() == [14, 15, 16, 17]


def test_forced_raw_format_does_not_interpret_magic(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(8, dtype=np.uint16))

    dataset = _dataset(path, memmap_format="raw")

    assert dataset[0]["input_ids"][0].item() == 20115  # first uint16 of b"\x93NUMPY"


def test_forced_npy_rejects_raw_file(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(8, dtype=np.uint16).tofile(path)

    with pytest.raises(ValueError, match="not a NumPy"):
        len(_dataset(path, memmap_format="npy"))


def test_npy_dtype_mismatch_fails_fast(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(8, dtype=np.uint32))

    with pytest.raises(ValueError, match="dtype mismatch"):
        len(_dataset(path))


def test_structured_npy_is_rejected_by_generic_loader(tmp_path):
    path = tmp_path / "structured.npy"
    np.save(path, np.arange(8, dtype=np.uint16).reshape(2, 4))

    with pytest.raises(ValueError, match="requires a 1-D token stream"):
        len(_dataset(path))


def test_truncated_npy_payload_is_rejected(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(8, dtype=np.uint16))
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="payload size"):
        inspect_memmap_file(path, np.uint16)


def test_raw_size_must_be_divisible_by_dtype(tmp_path):
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"abc")

    with pytest.raises(ValueError, match="not divisible"):
        inspect_memmap_file(path, np.uint16)


def test_empty_raw_file_needs_no_range_read(tmp_path, monkeypatch):
    path = tmp_path / "empty.bin"
    path.touch()
    monkeypatch.setattr(
        "olmo.memmap_utils.get_bytes_range",
        lambda *args: pytest.fail("empty raw files should not issue range reads"),
    )

    info = inspect_memmap_file(path, np.uint16)

    assert info.file_format == "raw"
    assert info.element_count == 0


def test_non_token_dtype_is_rejected(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(8, dtype=np.float32))

    with pytest.raises(ValueError, match="unsigned integer"):
        inspect_memmap_file(path, np.float32)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_supported_numpy_header_versions(tmp_path, version):
    path = tmp_path / f"v{version[0]}.npy"
    with path.open("wb") as handle:
        np.lib.format.write_array(handle, np.arange(8, dtype=np.uint16), version=version)

    info = inspect_memmap_file(path, np.uint16)

    assert info.file_format == "npy"
    assert info.element_count == 8


def test_standard_npy_label_mask_uses_its_payload_offset(tmp_path):
    token_path = tmp_path / "tokens.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(token_path, np.arange(8, dtype=np.uint16))
    np.save(mask_path, np.array([True, False, True, False, False, True, False, True]))

    dataset = _dataset(token_path, label_mask_paths=[mask_path])

    assert len(dataset) == 2
    assert dataset[0]["label_mask"].tolist() == [True, False, True, False]


def test_label_mask_element_count_must_match_input(tmp_path):
    token_path = tmp_path / "tokens.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(token_path, np.arange(8, dtype=np.uint16))
    np.save(mask_path, np.ones(7, dtype=np.bool_))

    with pytest.raises(ValueError, match="same number of elements"):
        len(_dataset(token_path, label_mask_paths=[mask_path]))


def test_inspection_uses_bounded_range_reads(monkeypatch):
    buffer = io.BytesIO()
    np.save(buffer, np.arange(8, dtype=np.uint16))
    contents = buffer.getvalue()
    reads = []

    def fake_range(path, start, length):
        reads.append((start, length))
        return contents[start : start + length]

    monkeypatch.setattr("olmo.memmap_utils.file_size", lambda path: len(contents))
    monkeypatch.setattr("olmo.memmap_utils.get_bytes_range", fake_range)

    info = inspect_memmap_file("s3://bucket/tokens.npy", np.uint16)

    assert info.data_offset == 128
    assert max(start + length for start, length in reads) <= info.data_offset
    assert all(length <= 128 for _, length in reads)


def test_decay_counter_matches_dataset_length(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(10, dtype=np.uint16))

    assert get_file_example_count(path, 4, np.uint16) == len(_dataset(path)) == 2
