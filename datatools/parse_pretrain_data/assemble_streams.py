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
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from tokenizers import Tokenizer

FORMATS = ("terminal", "tree", "tg")
SPLITS = ("train", "dev", "test")


def load_index(path: Path) -> dict[str, set[int]]:
    raw = json.loads(path.read_text())
    return {str(key): {int(index) for index in value} for key, value in raw.items()}


def document_bounds(tokens: np.ndarray, bos_token_id: int) -> np.ndarray:
    positions = np.flatnonzero(tokens == bos_token_id).astype(np.int64, copy=False)
    if positions.size == 0:
        raise ValueError("token shard contains no document BOS tokens")
    if positions[0] != 0:
        raise ValueError(
            f"first document BOS is at token {int(positions[0])}, expected token 0"
        )
    return np.concatenate((positions, np.asarray([tokens.size], dtype=np.int64)))


def selected_documents(
    document_count: int, dev: set[int], test: set[int], split: str
) -> Iterator[int]:
    overlap = dev & test
    if overlap:
        raise ValueError(f"dev/test document indices overlap: {sorted(overlap)[:5]}")
    invalid = [index for index in dev | test if index < 0 or index >= document_count]
    if invalid:
        raise IndexError(
            f"split document index outside [0, {document_count}): {sorted(invalid)[:5]}"
        )
    if split == "dev":
        yield from sorted(dev)
    elif split == "test":
        yield from sorted(test)
    elif split == "train":
        held_out = dev | test
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
    paths: Sequence[Path], dev: Mapping[str, set[int]], test: Mapping[str, set[int]],
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
    paths: Sequence[Path], output: Path, dev: Mapping[str, set[int]],
    test: Mapping[str, set[int]], split: str, bos_token_id: int,
    total_tokens: int, overwrite: bool = False,
) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.load(paths[0], mmap_mode="r").dtype
    temporary = output.with_name(output.name + ".tmp.npy")
    if temporary.exists():
        temporary.unlink()
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
        temporary.unlink(missing_ok=True)
        raise AssertionError(f"planned {total_tokens} tokens but wrote {cursor}")
    os.replace(temporary, output)
    return output


def read_order(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def assemble(
    input_root: Path, output_root: Path, order: Sequence[str],
    dev_index: Mapping[str, set[int]], test_index: Mapping[str, set[int]],
    bos_token_id: int, formats: Sequence[str] = FORMATS, overwrite: bool = False,
) -> dict[str, dict[str, dict[str, int | str]]]:
    manifest: dict[str, dict[str, dict[str, int | str]]] = {}
    reference_docs: dict[str, int] | None = None
    for input_format in formats:
        paths = shard_paths(input_root / input_format, order)
        format_results: dict[str, dict[str, int | str]] = {}
        docs_for_format: dict[str, int] = {}
        for split in SPLITS:
            total_tokens, total_documents = plan_split(
                paths, dev_index, test_index, split, bos_token_id
            )
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
            }
            docs_for_format[split] = total_documents
        if reference_docs is None:
            reference_docs = docs_for_format
        elif docs_for_format != reference_docs:
            raise AssertionError(
                f"document split counts are not aligned: {input_format}={docs_for_format}, "
                f"reference={reference_docs}"
            )
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
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    bos = tokenizer.token_to_id("<|beginoftext|>")
    if bos is None:
        raise ValueError(f"tokenizer {args.tokenizer} has no <|beginoftext|>")
    order = read_order(args.shard_order)
    result = assemble(
        args.input_root.resolve(), args.output_root.resolve(), order,
        load_index(args.dev_index), load_index(args.test_index), bos,
        formats=args.formats, overwrite=args.overwrite,
    )
    manifest_path = args.manifest or args.output_root / "assembly_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"assembled {', '.join(args.formats)} -> {args.output_root}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
