from __future__ import annotations

import ast
import math
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .aliases import PathOrStr
from .util import file_size, get_bytes_range


NUMPY_MAGIC = b"\x93NUMPY"
MAX_NUMPY_HEADER_SIZE = 10_000
MemMapFileFormat = Literal["auto", "npy", "raw"]


@dataclass(frozen=True)
class MemMapFileInfo:
    """Physical layout information for a flat array stored in a file."""

    file_format: Literal["npy", "raw"]
    dtype: np.dtype
    data_offset: int
    element_count: int


def _read_exact(path: PathOrStr, start: int, length: int) -> bytes:
    data = get_bytes_range(path, start, length)
    if len(data) != length:
        raise ValueError(
            f"Could not read {length} bytes at offset {start} from '{path}'; "
            f"received {len(data)} bytes"
        )
    return data


def _parse_npy_info(
    path: PathOrStr,
    total_size: int,
    expected_dtype: np.dtype,
    *,
    require_1d: bool,
) -> MemMapFileInfo:
    prefix = _read_exact(path, 0, 8)
    if prefix[:6] != NUMPY_MAGIC:
        raise ValueError(f"File '{path}' is not a NumPy .npy file")

    version = (prefix[6], prefix[7])
    if version == (1, 0):
        length_size = 2
        length_format = "<H"
        encoding = "latin1"
    elif version in ((2, 0), (3, 0)):
        length_size = 4
        length_format = "<I"
        encoding = "utf8" if version == (3, 0) else "latin1"
    else:
        raise ValueError(f"Unsupported NumPy format version {version} in '{path}'")

    header_length_bytes = _read_exact(path, 8, length_size)
    header_length = struct.unpack(length_format, header_length_bytes)[0]
    if header_length > MAX_NUMPY_HEADER_SIZE:
        raise ValueError(
            f"NumPy header in '{path}' is {header_length} bytes, exceeding the "
            f"{MAX_NUMPY_HEADER_SIZE}-byte safety limit"
        )

    header_start = 8 + length_size
    try:
        header = ast.literal_eval(_read_exact(path, header_start, header_length).decode(encoding))
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid NumPy header in '{path}'") from exc

    if not isinstance(header, dict) or not {"descr", "fortran_order", "shape"}.issubset(header):
        raise ValueError(f"Invalid NumPy header fields in '{path}'")
    if not isinstance(header["fortran_order"], bool):
        raise ValueError(f"Invalid 'fortran_order' value in NumPy header for '{path}'")

    shape = header["shape"]
    if not isinstance(shape, tuple) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError(f"Invalid array shape {shape!r} in NumPy header for '{path}'")
    if require_1d and len(shape) != 1:
        raise ValueError(
            f"MemMapDataset requires a 1-D token stream, but '{path}' has shape {shape}. "
            "Use the dataset's structured loader instead."
        )

    try:
        dtype = np.dtype(header["descr"])
    except TypeError as exc:
        raise ValueError(f"Invalid dtype descriptor in NumPy header for '{path}'") from exc
    if dtype.hasobject:
        raise ValueError(f"Object arrays are not supported for memory-mapped data: '{path}'")
    if dtype != expected_dtype:
        raise ValueError(
            f"NumPy dtype mismatch for '{path}': header declares {dtype}, "
            f"but memmap_dtype expects {expected_dtype}"
        )

    element_count = math.prod(shape)
    data_offset = header_start + header_length
    expected_size = data_offset + element_count * dtype.itemsize
    if total_size != expected_size:
        raise ValueError(
            f"Invalid NumPy payload size for '{path}': file has {total_size} bytes, "
            f"but its header requires {expected_size}"
        )
    return MemMapFileInfo("npy", dtype, data_offset, element_count)


def inspect_memmap_file(
    path: PathOrStr,
    expected_dtype,
    file_format: MemMapFileFormat = "auto",
    *,
    require_1d: bool = True,
) -> MemMapFileInfo:
    """Inspect a standard ``.npy`` or headerless raw array without loading it."""

    if file_format not in ("auto", "npy", "raw"):
        raise ValueError(
            f"memmap_format must be one of 'auto', 'npy', or 'raw', got {file_format!r}"
        )

    dtype = np.dtype(expected_dtype)
    if dtype.hasobject or dtype.itemsize <= 0:
        raise ValueError(f"Unsupported memmap dtype {dtype}")
    if dtype.kind not in ("u", "b"):
        raise ValueError(
            f"Unsupported memmap dtype {dtype}; token streams must use an unsigned integer "
            "dtype and label masks must use bool"
        )

    total_size = file_size(path)
    detected_format = file_format
    if file_format == "auto":
        prefix = get_bytes_range(path, 0, 6) if total_size >= 6 else b""
        detected_format = "npy" if prefix == NUMPY_MAGIC else "raw"

    if detected_format == "npy":
        return _parse_npy_info(path, total_size, dtype, require_1d=require_1d)

    if total_size % dtype.itemsize != 0:
        raise ValueError(
            f"Raw file '{path}' has {total_size} bytes, which is not divisible by "
            f"dtype {dtype}'s {dtype.itemsize}-byte item size"
        )
    return MemMapFileInfo("raw", dtype, 0, total_size // dtype.itemsize)
