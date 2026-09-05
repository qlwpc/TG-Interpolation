"""Assemble tokenized shards into aligned BBC train/dev/test NumPy streams.

Unlike the historical ``gen_final_train.py``, this implementation never loads
the full corpus or concatenates complete arrays in RAM.  It makes two passes:
first count selected document spans, then write them into ``open_memmap`` output
files.  Standard ``.npy`` headers are therefore preserved correctly.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from tokenizers import Tokenizer

from datatools.parse_pretrain_data.pipeline_io import atomic_json, load_index, sha256_file, validate_split_indices

Index = Mapping[str, Sequence[int] | set[int]]

FORMATS = ("terminal", "tree", "tg")
SPLITS = ("train", "dev", "test")


def document_bounds(tokens: np.ndarray, bos_token_id: int) -> np.ndarray:
    # A full-shard boolean comparison can itself consume gigabytes of RAM.
    chunk = 4 * 1024 * 1024
    positions = np.concatenate([
        np.flatnonzero(tokens[start:start + chunk] == bos_token_id) + start
        for start in range(0, tokens.size, chunk)
    ]) if tokens.size else np.empty(0, dtype=np.int64)
    if positions.size == 0:
        raise ValueError("token shard contains no document BOS tokens")
    if positions[0] != 0:
        raise ValueError(
            f"first document BOS is at token {int(positions[0])}, expected token 0"
        )
    return np.concatenate((positions, np.asarray([tokens.size], dtype=np.int64)))


def selected_documents(
    document_count: int, dev: Sequence[int] | set[int], test: Sequence[int] | set[int], split: str
) -> Iterator[int]:
    if any(type(i) is not int or i < 0 for i in (*dev, *test)):
        raise ValueError("split indices must be nonnegative integers")
    dev_set, test_set = set(dev), set(test)
    if len(dev_set) != len(dev) or len(test_set) != len(test):
        raise ValueError("duplicate split document indices")
    overlap = dev_set & test_set
    if overlap:
        raise ValueError(f"dev/test document indices overlap: {sorted(overlap)[:5]}")
    invalid = [index for index in dev_set | test_set if index >= document_count]
    if invalid:
        raise IndexError(
            f"split document index outside [0, {document_count}): {sorted(invalid)[:5]}"
        )
    if split == "dev":
        yield from sorted(dev) if isinstance(dev, set) else dev
    elif split == "test":
        yield from sorted(test) if isinstance(test, set) else test
    elif split == "train":
        held_out = dev_set | test_set
        yield from (index for index in range(document_count) if index not in held_out)
    else:
        raise ValueError(f"unknown split: {split}")


def coalesce_spans(
    bounds: np.ndarray, selected: Iterable[int]
) -> Iterator[tuple[int, int]]:
    """Merge adjacent selected documents into sequential copy spans."""
    start: int | None = None
    stop: int | None = None
    for index in selected:
        doc_start = int(bounds[index])
        doc_stop = int(bounds[index + 1])
        if stop == doc_start:
            stop = doc_stop
        else:
            if start is not None and stop is not None:
                yield start, stop
            start, stop = doc_start, doc_stop
    if start is not None and stop is not None:
        yield start, stop


def shard_paths(input_dir: Path, order: Sequence[str]) -> list[Path]:
    paths = [input_dir / f"{stem}.npy" for stem in order]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"missing {len(missing)} ordered token shards: {preview}")
    return paths


def plan_split(
    paths: Sequence[Path], dev: Index, test: Index,
    split: str, bos_token_id: int,
) -> tuple[int, int]:
    total_tokens = 0
    total_documents = 0
    for path in paths:
        tokens = np.load(path, mmap_mode="r")
        if tokens.ndim != 1:
            raise ValueError(f"{path} is not a 1-D token stream: {tokens.shape}")
        bounds = document_bounds(tokens, bos_token_id)
        selected = selected_documents(
            len(bounds) - 1, dev.get(path.stem, set()), test.get(path.stem, set()), split
        )
        for start, stop in coalesce_spans(bounds, selected):
            total_tokens += stop - start
        selected_again = selected_documents(
            len(bounds) - 1, dev.get(path.stem, set()), test.get(path.stem, set()), split
        )
        total_documents += sum(1 for _ in selected_again)
    return total_tokens, total_documents


def write_split(
    paths: Sequence[Path], output: Path, dev: Index,
    test: Index, split: str, bos_token_id: int,
    total_tokens: int, overwrite: bool = False,
) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.load(paths[0], mmap_mode="r").dtype
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".npy", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        destination = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=dtype, shape=(total_tokens,)
        )
        cursor = 0
        for path in paths:
            tokens = np.load(path, mmap_mode="r")
            if tokens.dtype != dtype:
                raise TypeError(f"dtype mismatch: {path} is {tokens.dtype}, expected {dtype}")
            bounds = document_bounds(tokens, bos_token_id)
            selected = selected_documents(
                len(bounds) - 1, dev.get(path.stem, set()), test.get(path.stem, set()), split
            )
            for start, stop in coalesce_spans(bounds, selected):
                length = stop - start
                destination[cursor : cursor + length] = tokens[start:stop]
                cursor += length
        destination.flush()
        del destination
        if cursor != total_tokens:
            raise AssertionError(f"planned {total_tokens} tokens but wrote {cursor}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def read_order(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def assemble(
    input_root: Path, output_root: Path, order: Sequence[str],
    dev_index: Index, test_index: Index,
    bos_token_id: int, formats: Sequence[str] = FORMATS, overwrite: bool = False,
) -> dict[str, dict[str, dict[str, int | str]]]:
    manifest: dict[str, dict[str, dict[str, int | str]]] = {}
    if not order or len(order) != len(set(order)):
        raise ValueError("shard order must be nonempty and unique")
    if not formats or len(formats) != len(set(formats)) or set(formats) - set(FORMATS):
        raise ValueError("formats must be nonempty, unique, and known")
    if any(Path(stem).name != stem or stem in (".", "..") for stem in order):
        raise ValueError("shard order must contain filename stems, not paths")
    unknown = (set(dev_index) | set(test_index)) - set(order)
    if unknown:
        raise ValueError(f"unknown split shard keys: {sorted(unknown)}")
    # Preflight EVERY source/target and per-shard alignment before any writes.
    # Total document counts alone cannot detect swapped or dropped documents.
    reference_counts = None
    reference_dtype = None
    plans = {}
    source_paths = {p.resolve() for fmt in formats for p in shard_paths(input_root / fmt, order)}
    for fmt in formats:
        paths = shard_paths(input_root / fmt, order)
        counts = []
        for path in paths:
            tokens = np.load(path, mmap_mode="r", allow_pickle=False)
            if tokens.ndim != 1 or tokens.dtype not in (np.dtype("uint16"), np.dtype("uint32")):
                raise ValueError(f"invalid token shard {path}: {tokens.shape} {tokens.dtype}")
            if reference_dtype is None:
                reference_dtype = tokens.dtype
            elif tokens.dtype != reference_dtype:
                raise ValueError(f"dtype mismatch: {path}")
            counts.append(len(document_bounds(tokens, bos_token_id)) - 1)
        if reference_counts is not None and counts != reference_counts:
            raise ValueError(f"per-shard document counts are not aligned: {fmt}")
        reference_counts = counts
        for split in SPLITS:
            output = output_root / fmt / f"{split}.npy"
            if output.resolve() in source_paths:
                raise ValueError(f"output aliases an input shard: {output}")
            if output.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
            plans[fmt, split] = plan_split(paths, dev_index, test_index, split, bos_token_id)
    for input_format in formats:
        paths = shard_paths(input_root / input_format, order)
        format_results: dict[str, dict[str, int | str]] = {}
        for split in SPLITS:
            total_tokens, total_documents = plans[input_format, split]
            output = output_root / input_format / f"{split}.npy"
            write_split(
                paths, output, dev_index, test_index, split, bos_token_id,
                total_tokens, overwrite=overwrite,
            )
            format_results[split] = {
                "path": str(output),
                "tokens": total_tokens,
                "documents": total_documents,
                "dtype": str(np.load(output, mmap_mode="r").dtype),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        manifest[input_format] = format_results
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dev-index", type=Path, required=True)
    parser.add_argument("--test-index", type=Path, required=True)
    parser.add_argument(
        "--shard-order",
        type=Path,
        default=Path(__file__).with_name("bbc_configs.txt"),
    )
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--released-indices", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    bos = tokenizer.token_to_id("<|beginoftext|>")
    if bos is None:
        raise ValueError(f"tokenizer {args.tokenizer} has no <|beginoftext|>")
    order = read_order(args.shard_order)
    split_metadata = validate_split_indices(args.dev_index, args.test_index, order, released=args.released_indices)
    manifest_path = args.manifest or args.output_root / "assembly_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --overwrite")
    result = assemble(
        args.input_root.resolve(), args.output_root.resolve(), order,
        load_index(args.dev_index), load_index(args.test_index), bos,
        formats=args.formats, overwrite=args.overwrite,
    )
    source_receipts = {}
    for stem in order:
        receipt = args.input_root / "manifests" / f"{stem}.json"
        if receipt.is_file():
            source_receipts[stem] = json.loads(receipt.read_text())
    atomic_json(manifest_path, {
        "schema_version": 2, "formats": result, "split_indices": split_metadata,
        "tokenizer_sha256": sha256_file(args.tokenizer), "shard_order": order,
        "historical_byte_identical": False,
        "source_receipts": source_receipts,
    })
    print(f"assembled {', '.join(args.formats)} -> {args.output_root}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
