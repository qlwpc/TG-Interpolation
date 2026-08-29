#!/usr/bin/env python3
"""Convert document-PPL tree candidates to Transformer Grammar format.

TG serialization differs from tree serialization in exactly one respect: every
closing non-terminal token is repeated in place. The corpus is several GB, so
this script uses memory maps and never loads the full token stream into RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "dataset/bbc-news/testppl_tree"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dataset/bbc-news/testppl_tg"
DEFAULT_TOKENIZER = REPO_ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json"
CLOSE_RE = re.compile(r"^<[A-Za-z0-9]+\)>$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Duplicate every closing non-terminal in testppl_tree."
    )
    # Retain historical option names for compatibility with existing commands.
    parser.add_argument("--input_prefix", "--input-prefix", default="tree")
    parser.add_argument("--output_prefix", "--output-prefix", default="tg")
    parser.add_argument(
        "--directory", "--input-dir", type=Path, default=DEFAULT_INPUT_DIR
    )
    parser.add_argument(
        "--output_dir", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--record-batch-size", type=int, default=250_000)
    parser.add_argument("--chunk-tokens", type=int, default=16_000_000)
    parser.add_argument("--samples-per-sentence", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="verify existing outputs exactly and write manifest.json",
    )
    return parser.parse_args()


def closing_token_ids(tokenizer_path: Path) -> np.ndarray:
    with tokenizer_path.open(encoding="utf-8") as f:
        tokenizer = json.load(f)
    token_to_id: dict[str, int] = {}
    token_to_id.update(tokenizer.get("model", {}).get("vocab", {}))
    token_to_id.update(
        {item["content"]: item["id"] for item in tokenizer.get("added_tokens", [])}
    )
    ids = sorted(
        {int(i) for token, i in token_to_id.items() if CLOSE_RE.fullmatch(token)}
    )
    if not ids:
        raise ValueError(f"No closing non-terminal tokens found in {tokenizer_path}")
    return np.asarray(ids, dtype=np.int64)


def closing_mask(tokens: np.ndarray, close_ids: np.ndarray) -> np.ndarray:
    """Return a mask without assuming that tokenizer IDs are contiguous."""
    if close_ids.size == int(close_ids[-1] - close_ids[0] + 1):
        return (tokens >= close_ids[0]) & (tokens <= close_ids[-1])
    return np.isin(tokens, close_ids)


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    return (
        args.output_dir / f"{args.output_prefix}_300.npy",
        args.output_dir / f"{args.output_prefix}_sent_index.npy",
        args.output_dir / f"{args.output_prefix}_doc_index.npy",
    )


def validate_inputs(
    data: np.ndarray,
    sent_index: np.ndarray,
    doc_index: np.ndarray,
    samples_per_sentence: int,
) -> None:
    if data.ndim != 1 or sent_index.ndim != 1 or doc_index.ndim != 1:
        raise ValueError("Token data and both indexes must be one-dimensional")
    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError(f"Token data must be integer, got {data.dtype}")
    if not np.issubdtype(sent_index.dtype, np.integer):
        raise TypeError(f"Sentence index must be integer, got {sent_index.dtype}")
    if np.any(sent_index == 0):
        raise ValueError("Sentence index contains a zero-length candidate")

    indexed_tokens = int(np.sum(sent_index, dtype=np.uint64))
    if indexed_tokens != data.size:
        raise ValueError(
            f"Sentence lengths sum to {indexed_tokens:,}, but token data has "
            f"{data.size:,} entries (the transfer may be incomplete)"
        )
    expected_records = int(np.sum(doc_index, dtype=np.uint64)) * samples_per_sentence
    if expected_records != sent_index.size:
        raise ValueError(
            f"Document index implies {expected_records:,} candidate records, but "
            f"sentence index contains {sent_index.size:,}"
        )


def adjusted_sentence_index(
    data: np.ndarray,
    sent_index: np.ndarray,
    close_ids: np.ndarray,
    destination: Path,
    batch_size: int,
) -> int:
    """Write adjusted record lengths and return the total close-token count."""
    tmp = destination.with_name(f".{destination.name}.tmp")
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=sent_index.dtype, shape=sent_index.shape
    )
    dtype_limit = np.iinfo(sent_index.dtype).max
    source_offset = 0
    total_closes = 0
    started = time.monotonic()
    try:
        for first in range(0, sent_index.size, batch_size):
            last = min(first + batch_size, sent_index.size)
            lengths = np.asarray(sent_index[first:last], dtype=np.uint64)
            boundaries = np.empty(lengths.size + 1, dtype=np.uint64)
            boundaries[0] = 0
            np.cumsum(lengths, dtype=np.uint64, out=boundaries[1:])
            batch_tokens = int(boundaries[-1])
            segment = data[source_offset : source_offset + batch_tokens]
            mask = closing_mask(segment, close_ids)
            counts = np.add.reduceat(mask, boundaries[:-1].astype(np.intp))
            adjusted = lengths + counts.astype(np.uint64, copy=False)
            batch_max = int(adjusted.max(initial=0))
            if batch_max > dtype_limit:
                raise OverflowError(
                    f"Adjusted length {batch_max} does not fit {sent_index.dtype}"
                )
            out[first:last] = adjusted
            source_offset += batch_tokens
            total_closes += int(np.count_nonzero(mask))
            if first == 0 or last == sent_index.size or last % (batch_size * 20) == 0:
                print(
                    f"index pass: {last:,}/{sent_index.size:,} records, "
                    f"{source_offset:,} tokens ({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
        out.flush()
        del out
        os.replace(tmp, destination)
    except BaseException:
        del out
        tmp.unlink(missing_ok=True)
        raise
    return total_closes


def write_tg_data(
    data: np.ndarray,
    close_ids: np.ndarray,
    destination: Path,
    total_closes: int,
    chunk_tokens: int,
) -> str:
    tmp = destination.with_name(f".{destination.name}.tmp")
    output_size = data.size + total_closes
    out = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=data.dtype, shape=(output_size,)
    )
    output_offset = 0
    digest = hashlib.blake2b(digest_size=32)
    started = time.monotonic()
    try:
        for start in range(0, data.size, chunk_tokens):
            end = min(start + chunk_tokens, data.size)
            chunk = np.asarray(data[start:end])
            mask = closing_mask(chunk, close_ids)
            repeats = np.ones(chunk.size, dtype=np.uint8)
            repeats[mask] = 2
            expanded = np.repeat(chunk, repeats)
            out[output_offset : output_offset + expanded.size] = expanded
            digest.update(expanded.tobytes())
            output_offset += expanded.size
            if start == 0 or end == data.size or end % (chunk_tokens * 20) == 0:
                print(
                    f"data pass: {end:,}/{data.size:,} input tokens, "
                    f"{output_offset:,} output tokens "
                    f"({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
        if output_offset != output_size:
            raise RuntimeError(
                f"Allocated {output_size:,} output tokens but wrote {output_offset:,}"
            )
        out.flush()
        del out
        os.replace(tmp, destination)
    except BaseException:
        del out
        tmp.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def validate_existing_conversion(
    tree: np.ndarray,
    tree_lengths: np.ndarray,
    tree_documents: np.ndarray,
    tg: np.ndarray,
    tg_lengths: np.ndarray,
    tg_documents: np.ndarray,
    close_ids: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    """Prove token, candidate-record, and document equivalence in one pass."""
    if tree_lengths.shape != tg_lengths.shape:
        raise ValueError("Tree and TG sentence-index shapes differ")
    if not np.array_equal(tree_documents, tg_documents):
        raise ValueError("Tree and TG document indexes differ")
    source_offset = output_offset = total_closes = 0
    digest = hashlib.blake2b(digest_size=32)
    for first in range(0, tree_lengths.size, batch_size):
        last = min(first + batch_size, tree_lengths.size)
        lengths = np.asarray(tree_lengths[first:last], dtype=np.uint64)
        boundaries = np.empty(lengths.size + 1, dtype=np.uint64)
        boundaries[0] = 0
        np.cumsum(lengths, dtype=np.uint64, out=boundaries[1:])
        source_count = int(boundaries[-1])
        segment = np.asarray(tree[source_offset : source_offset + source_count])
        mask = closing_mask(segment, close_ids)
        close_counts = np.add.reduceat(mask, boundaries[:-1].astype(np.intp))
        expected_lengths = lengths + close_counts.astype(np.uint64, copy=False)
        if not np.array_equal(expected_lengths.astype(tg_lengths.dtype), tg_lengths[first:last]):
            raise ValueError(f"TG sentence-index mismatch in records [{first}, {last})")
        repeats = np.ones(segment.size, dtype=np.uint8)
        repeats[mask] = 2
        expected_tokens = np.repeat(segment, repeats)
        actual_tokens = np.asarray(tg[output_offset : output_offset + expected_tokens.size])
        if not np.array_equal(expected_tokens, actual_tokens):
            mismatch = int(np.flatnonzero(expected_tokens != actual_tokens)[0])
            raise ValueError(f"TG token mismatch at output offset {output_offset + mismatch}")
        digest.update(actual_tokens.tobytes())
        source_offset += source_count
        output_offset += expected_tokens.size
        total_closes += int(np.count_nonzero(mask))
    if source_offset != tree.size or output_offset != tg.size:
        raise ValueError("Tree/TG index coverage does not match token-array sizes")
    return {
        "format_version": 1,
        "status": "complete",
        "conversion": "duplicate each closing non-terminal token in place",
        "tree_tokens": int(tree.size),
        "tg_tokens": int(tg.size),
        "duplicated_closing_tokens": total_closes,
        "candidate_records": int(tree_lengths.size),
        "documents": int(tree_documents.size),
        "sentences": int(np.sum(tree_documents, dtype=np.uint64)),
        "tg_blake2b": digest.hexdigest(),
        "exact": True,
    }


def write_manifest(output_dir: Path, report: dict[str, object]) -> None:
    destination = output_dir / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    if args.record_batch_size <= 0 or args.chunk_tokens <= 0:
        raise ValueError("Batch and chunk sizes must be positive")
    if args.samples_per_sentence <= 0:
        raise ValueError("--samples-per-sentence must be positive")

    input_dir = args.directory.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    tree_path = input_dir / f"{args.input_prefix}_300.npy"
    sent_path = input_dir / f"{args.input_prefix}_sent_index.npy"
    doc_path = input_dir / f"{args.input_prefix}_doc_index.npy"
    destinations = output_paths(args)

    for path in (tree_path, sent_path, doc_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(tree_path, mmap_mode="r")
    sent_index = np.load(sent_path, mmap_mode="r")
    doc_index = np.load(doc_path, mmap_mode="r")
    validate_inputs(data, sent_index, doc_index, args.samples_per_sentence)
    close_ids = closing_token_ids(tokenizer_path)
    print(
        f"input: {data.size:,} {data.dtype} tokens, {sent_index.size:,} records; "
        f"closing NT IDs: {close_ids.size} ({close_ids[0]}..{close_ids[-1]})",
        flush=True,
    )

    tg_path, tg_sent_path, tg_doc_path = destinations
    if args.validate_only:
        for path in destinations:
            if not path.is_file():
                raise FileNotFoundError(path)
        report = validate_existing_conversion(
            data, sent_index, doc_index,
            np.load(tg_path, mmap_mode="r"),
            np.load(tg_sent_path, mmap_mode="r"),
            np.load(tg_doc_path, mmap_mode="r"),
            close_ids, args.record_batch_size,
        )
        report.update(
            input_tree=str(tree_path), output_tg=str(tg_path),
            tokenizer=str(tokenizer_path),
        )
        write_manifest(args.output_dir, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output exists (use --overwrite): {names}")
    total_closes = adjusted_sentence_index(
        data, sent_index, close_ids, tg_sent_path, args.record_batch_size
    )
    print(
        f"found {total_closes:,} closing non-terminals; "
        f"TG stream has {data.size + total_closes:,} tokens",
        flush=True,
    )
    tg_digest = write_tg_data(data, close_ids, tg_path, total_closes, args.chunk_tokens)

    doc_tmp = tg_doc_path.with_name(f".{tg_doc_path.name}.tmp")
    shutil.copyfile(doc_path, doc_tmp)
    os.replace(doc_tmp, tg_doc_path)
    report = {
        "format_version": 1,
        "status": "complete",
        "conversion": "duplicate each closing non-terminal token in place",
        "input_tree": str(tree_path),
        "output_tg": str(tg_path),
        "tokenizer": str(tokenizer_path),
        "tree_tokens": int(data.size),
        "tg_tokens": int(data.size + total_closes),
        "duplicated_closing_tokens": total_closes,
        "candidate_records": int(sent_index.size),
        "documents": int(doc_index.size),
        "sentences": int(np.sum(doc_index, dtype=np.uint64)),
        "tg_blake2b": tg_digest,
        "exact": True,
    }
    write_manifest(args.output_dir, report)
    print(f"wrote: {tg_path}\n       {tg_sent_path}\n       {tg_doc_path}")


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        sys.exit(130)
