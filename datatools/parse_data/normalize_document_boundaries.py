#!/usr/bin/env python3
"""Normalize all testppl Tree/TG candidates to document BOS ... EOS framing.

The historical 300-candidate arrays contain exactly one boundary special per
document/candidate: either BOS on the first sentence or EOS on the last.  This
script preserves every existing token and inserts only the missing boundary.
Sentence-record lengths are updated; document sentence counts are unchanged.

With ``--install``, the normalized arrays replace the canonical files
atomically after recoverable reflink backups have been created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from olmo.data.parse_align import TreeVocab  # noqa: E402


SAMPLES_PER_SENTENCE = 300
DEFAULT_TOKENIZER = REPO / "dataset/bbc-news/TG_GPT2_tokenizer.json"
DEFAULT_MANIFEST = REPO / "dataset/bbc-news/testppl_boundary_normalization.json"


@dataclass(frozen=True)
class FormatPaths:
    name: str
    directory: Path

    @property
    def data(self) -> Path:
        return self.directory / f"{self.name}_300.npy"

    @property
    def sentence_index(self) -> Path:
        return self.directory / f"{self.name}_sent_index.npy"

    @property
    def document_index(self) -> Path:
        return self.directory / f"{self.name}_doc_index.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--tree-dir", type=Path, default=REPO / "dataset/bbc-news/testppl_tree"
    )
    parser.add_argument(
        "--tg-dir", type=Path, default=REPO / "dataset/bbc-news/testppl_tg"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=32 * 1024 * 1024,
        help="source tokens copied per chunk",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="atomically replace canonical arrays after making reflink backups",
    )
    parser.add_argument(
        "--install-staged",
        action="store_true",
        help="validate and install existing hidden *.normalized.npy outputs",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def backup_path(path: Path) -> Path:
    return path.with_name(
        path.name.removesuffix(".npy") + ".pre_bos_eos_normalization.npy"
    )


def make_reflink_backup(path: Path) -> Path:
    backup = backup_path(path)
    if backup.exists():
        if backup.stat().st_size:
            raise FileExistsError(f"refusing to overwrite backup: {backup}")
        # GNU cp may leave an empty destination after a failed reflink attempt.
        backup.unlink()
    try:
        subprocess.run(
            [
                "cp",
                "--reflink=always",
                "--preserve=mode,timestamps",
                str(path),
                str(backup),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  reflink backup: {backup}", flush=True)
    except subprocess.CalledProcessError as error:
        if backup.exists():
            backup.unlink()
        # Atomic replacement leaves the old inode alive under this hard-link
        # name, so this is recoverable even on filesystems without reflinks.
        os.link(path, backup)
        print(
            f"  reflink unavailable; hard-link backup: {backup} "
            f"({error.stderr.strip()})",
            flush=True,
        )
    return backup


def load_layout(paths: FormatPaths):
    for path in (paths.data, paths.sentence_index, paths.document_index):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = np.load(paths.data, mmap_mode="r").reshape(-1)
    lengths = np.load(paths.sentence_index, mmap_mode="r").reshape(-1)
    documents = np.load(paths.document_index, mmap_mode="r").reshape(-1)
    if lengths.size % SAMPLES_PER_SENTENCE:
        raise ValueError(f"{paths.sentence_index} length is not divisible by 300")
    if np.any(documents <= 0):
        raise ValueError(f"{paths.document_index} contains a non-positive count")
    sentence_count = lengths.size // SAMPLES_PER_SENTENCE
    if int(documents.sum(dtype=np.uint64)) != sentence_count:
        raise ValueError(f"{paths.document_index} does not cover all sentences")
    offsets = np.empty(lengths.size + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(lengths, dtype=np.uint64, out=offsets[1:])
    if int(offsets[-1]) != data.size:
        raise ValueError(f"{paths.sentence_index} does not cover {paths.data}")
    return data, lengths, documents, offsets


def boundary_record_ids(documents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sentence_ends = np.cumsum(documents, dtype=np.uint64)
    sentence_starts = np.concatenate(
        (np.zeros(1, dtype=np.uint64), sentence_ends[:-1])
    )
    candidates = np.arange(SAMPLES_PER_SENTENCE, dtype=np.uint64)
    first = (sentence_starts[:, None] * SAMPLES_PER_SENTENCE + candidates).reshape(-1)
    last = ((sentence_ends[:, None] - 1) * SAMPLES_PER_SENTENCE + candidates).reshape(-1)
    return first, last


def build_events(
    data: np.ndarray,
    offsets: np.ndarray,
    first_records: np.ndarray,
    last_records: np.ndarray,
    vocab: TreeVocab,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    first_positions = offsets[first_records]
    last_positions = offsets[last_records + 1] - 1
    starts_with_bos = np.asarray(data[first_positions]) == vocab.bos
    ends_with_eos = np.asarray(data[last_positions]) == vocab.eos
    missing_bos_records = first_records[~starts_with_bos]
    missing_eos_records = last_records[~ends_with_eos]

    # An event position is a boundary in the original flat array.  EOS is
    # inserted first when it shares a position with the following record's BOS.
    bos_positions = offsets[missing_bos_records]
    eos_positions = offsets[missing_eos_records + 1]
    positions = np.concatenate((eos_positions, bos_positions))
    values = np.concatenate(
        (
            np.full(eos_positions.size, vocab.eos, dtype=data.dtype),
            np.full(bos_positions.size, vocab.bos, dtype=data.dtype),
        )
    )
    priorities = np.concatenate(
        (
            np.zeros(eos_positions.size, dtype=np.uint8),
            np.ones(bos_positions.size, dtype=np.uint8),
        )
    )
    order = np.lexsort((priorities, positions))
    stats = {
        "boundary_records": int(first_records.size),
        "source_first_records_with_bos": int(starts_with_bos.sum()),
        "source_last_records_with_eos": int(ends_with_eos.sum()),
        "inserted_bos": int(missing_bos_records.size),
        "inserted_eos": int(missing_eos_records.size),
    }
    return (
        positions[order],
        values[order],
        np.concatenate((missing_eos_records, missing_bos_records)),
        stats,
    )


def copy_with_insertions(
    source: np.ndarray,
    destination: Path,
    positions: np.ndarray,
    values: np.ndarray,
    vocab: TreeVocab,
    chunk_tokens: int,
) -> dict[str, int]:
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=source.dtype,
        shape=(source.size + positions.size,),
    )
    source_specials = {"bos": 0, "eos": 0, "pad": 0}
    output_cursor = 0
    event_cursor = 0
    started = time.monotonic()
    try:
        for chunk_id, start in enumerate(range(0, source.size, chunk_tokens)):
            end = min(start + chunk_tokens, source.size)
            block = np.asarray(source[start:end])
            source_specials["bos"] += int(np.count_nonzero(block == vocab.bos))
            source_specials["eos"] += int(np.count_nonzero(block == vocab.eos))
            source_specials["pad"] += int(np.count_nonzero(block == vocab.pad))
            event_end = int(np.searchsorted(positions, end, side="left"))
            local_positions = positions[event_cursor:event_end] - start
            local_values = values[event_cursor:event_end]
            expanded = np.insert(block, local_positions.astype(np.intp), local_values)
            output[output_cursor : output_cursor + expanded.size] = expanded
            output_cursor += expanded.size
            event_cursor = event_end
            if chunk_id % 16 == 0 or end == source.size:
                pct = 100.0 * end / source.size
                print(
                    f"  copied {pct:5.1f}% ({end:,}/{source.size:,} source tokens)",
                    flush=True,
                )

        # The final EOS may be inserted at position len(source).
        tail_values = values[event_cursor:]
        if len(tail_values):
            if not np.all(positions[event_cursor:] == source.size):
                raise AssertionError("unconsumed insertion event before end of source")
            output[output_cursor : output_cursor + len(tail_values)] = tail_values
            output_cursor += len(tail_values)
        if output_cursor != output.size:
            raise AssertionError(f"output cursor mismatch: {output_cursor} != {output.size}")
        output.flush()
        del output
        os.replace(temporary, destination)
    except BaseException:
        del output
        if temporary.exists():
            temporary.unlink()
        raise
    print(f"  copy completed in {time.monotonic() - started:.1f}s", flush=True)
    return source_specials


def normalize_format(
    paths: FormatPaths, vocab: TreeVocab, chunk_tokens: int
) -> tuple[Path, Path, dict[str, object]]:
    print(f"normalizing {paths.name}: {paths.data}", flush=True)
    data, lengths, documents, offsets = load_layout(paths)
    first_records, last_records = boundary_record_ids(documents)
    positions, values, changed_records, stats = build_events(
        data, offsets, first_records, last_records, vocab
    )
    output_data = paths.directory / f".{paths.name}_300.normalized.npy"
    output_index = paths.directory / f".{paths.name}_sent_index.normalized.npy"
    for output in (output_data, output_index):
        if output.exists():
            output.unlink()

    source_specials = copy_with_insertions(
        data, output_data, positions, values, vocab, chunk_tokens
    )
    stats["source_specials"] = source_specials
    expected_existing_bos = stats["source_first_records_with_bos"]
    expected_existing_eos = stats["source_last_records_with_eos"]
    if source_specials != {
        "bos": expected_existing_bos,
        "eos": expected_existing_eos,
        "pad": 0,
    }:
        raise AssertionError(
            "source has duplicated or misplaced BOS/EOS/PAD: "
            f"observed={source_specials}, expected bos/eos="
            f"{expected_existing_bos}/{expected_existing_eos}"
        )

    new_lengths = np.asarray(lengths).copy()
    if changed_records.size:
        increment_counts = np.bincount(
            changed_records.astype(np.int64), minlength=new_lengths.size
        )
        if np.any(
            new_lengths.astype(np.uint64) + increment_counts
            > np.iinfo(new_lengths.dtype).max
        ):
            raise OverflowError("sentence-record length cannot be incremented")
        new_lengths += increment_counts.astype(new_lengths.dtype)
    atomic_save(output_index, new_lengths)
    normalized = np.load(output_data, mmap_mode="r")
    new_offsets = np.empty(new_lengths.size + 1, dtype=np.uint64)
    new_offsets[0] = 0
    np.cumsum(new_lengths, dtype=np.uint64, out=new_offsets[1:])
    if int(new_offsets[-1]) != normalized.size:
        raise AssertionError("normalized sentence index does not cover token array")
    if not np.all(normalized[new_offsets[first_records]] == vocab.bos):
        raise AssertionError("a normalized first record does not begin with BOS")
    if not np.all(normalized[new_offsets[last_records + 1] - 1] == vocab.eos):
        raise AssertionError("a normalized last record does not end with EOS")
    expected_specials = first_records.size
    if source_specials["bos"] + stats["inserted_bos"] != expected_specials:
        raise AssertionError("normalized BOS count is not one per document/candidate")
    if source_specials["eos"] + stats["inserted_eos"] != expected_specials:
        raise AssertionError("normalized EOS count is not one per document/candidate")
    stats.update(
        {
            "source_shape": int(data.size),
            "normalized_shape": int(normalized.size),
            "sentence_records": int(lengths.size),
            "sentences": int(lengths.size // SAMPLES_PER_SENTENCE),
            "documents": int(documents.size),
            "normalized_specials": {
                "bos": int(expected_specials),
                "eos": int(expected_specials),
                "pad": 0,
            },
            "normalized_sha256": sha256_file(output_data),
        }
    )
    return output_data, output_index, stats


def validate_staged_format(
    paths: FormatPaths, vocab: TreeVocab, chunk_tokens: int
) -> tuple[Path, Path, dict[str, object]]:
    print(f"validating staged {paths.name} normalization", flush=True)
    data, lengths, documents, offsets = load_layout(paths)
    first_records, last_records = boundary_record_ids(documents)
    _, _, changed_records, stats = build_events(
        data, offsets, first_records, last_records, vocab
    )
    output_data = paths.directory / f".{paths.name}_300.normalized.npy"
    output_index = paths.directory / f".{paths.name}_sent_index.normalized.npy"
    if not output_data.is_file() or not output_index.is_file():
        raise FileNotFoundError(f"missing staged normalization for {paths.name}")
    normalized = np.load(output_data, mmap_mode="r").reshape(-1)
    new_lengths = np.load(output_index, mmap_mode="r").reshape(-1)
    expected_lengths = np.asarray(lengths).copy()
    increment_counts = np.bincount(
        changed_records.astype(np.int64), minlength=expected_lengths.size
    )
    expected_lengths += increment_counts.astype(expected_lengths.dtype)
    if not np.array_equal(new_lengths, expected_lengths):
        raise AssertionError(f"staged {paths.name} sentence index is incorrect")
    new_offsets = np.empty(new_lengths.size + 1, dtype=np.uint64)
    new_offsets[0] = 0
    np.cumsum(new_lengths, dtype=np.uint64, out=new_offsets[1:])
    if int(new_offsets[-1]) != normalized.size:
        raise AssertionError(f"staged {paths.name} index does not cover its data")
    if not np.all(normalized[new_offsets[first_records]] == vocab.bos):
        raise AssertionError(f"staged {paths.name} has a missing BOS")
    if not np.all(normalized[new_offsets[last_records + 1] - 1] == vocab.eos):
        raise AssertionError(f"staged {paths.name} has a missing EOS")

    source_specials = {"bos": 0, "eos": 0, "pad": 0}
    for start in range(0, data.size, chunk_tokens):
        block = np.asarray(data[start : min(start + chunk_tokens, data.size)])
        source_specials["bos"] += int(np.count_nonzero(block == vocab.bos))
        source_specials["eos"] += int(np.count_nonzero(block == vocab.eos))
        source_specials["pad"] += int(np.count_nonzero(block == vocab.pad))
    if source_specials != {
        "bos": stats["source_first_records_with_bos"],
        "eos": stats["source_last_records_with_eos"],
        "pad": 0,
    }:
        raise AssertionError(f"source {paths.name} has misplaced boundary specials")
    stats.update(
        {
            "source_specials": source_specials,
            "source_shape": int(data.size),
            "normalized_shape": int(normalized.size),
            "sentence_records": int(lengths.size),
            "sentences": int(lengths.size // SAMPLES_PER_SENTENCE),
            "documents": int(documents.size),
            "normalized_specials": {
                "bos": int(first_records.size),
                "eos": int(first_records.size),
                "pad": 0,
            },
            "normalized_sha256": sha256_file(output_data),
        }
    )
    return output_data, output_index, stats


def install_normalized(
    paths: FormatPaths, output_data: Path, output_index: Path
) -> dict[str, str]:
    data_backup = make_reflink_backup(paths.data)
    index_backup = make_reflink_backup(paths.sentence_index)
    os.replace(output_data, paths.data)
    os.replace(output_index, paths.sentence_index)
    return {
        "data": str(paths.data),
        "sentence_index": str(paths.sentence_index),
        "data_backup": str(data_backup),
        "sentence_index_backup": str(index_backup),
    }


def main() -> None:
    args = parse_args()
    if args.install_staged:
        args.install = True
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    vocab = TreeVocab.from_tokenizer_file(str(args.tokenizer.resolve()))
    formats = (
        FormatPaths("tree", args.tree_dir.resolve()),
        FormatPaths("tg", args.tg_dir.resolve()),
    )
    doc_hashes = {sha256_file(paths.document_index) for paths in formats}
    if len(doc_hashes) != 1:
        raise AssertionError("Tree and TG document indexes differ")

    outputs = {}
    manifest: dict[str, object] = {
        "contract": "every document/candidate begins with BOS and ends with EOS",
        "samples_per_sentence": SAMPLES_PER_SENTENCE,
        "token_ids": {"bos": vocab.bos, "eos": vocab.eos, "pad": vocab.pad},
        "document_index_sha256": next(iter(doc_hashes)),
        "formats": {},
    }
    for paths in formats:
        if args.install_staged:
            output_data, output_index, stats = validate_staged_format(
                paths, vocab, args.chunk_tokens
            )
        else:
            output_data, output_index, stats = normalize_format(
                paths, vocab, args.chunk_tokens
            )
        outputs[paths.name] = (paths, output_data, output_index)
        manifest["formats"][paths.name] = stats

    tree_lengths = np.load(outputs["tree"][2], mmap_mode="r")
    tg_lengths = np.load(outputs["tg"][2], mmap_mode="r")
    old_tree_lengths = np.load(formats[0].sentence_index, mmap_mode="r")
    old_tg_lengths = np.load(formats[1].sentence_index, mmap_mode="r")
    if not np.array_equal(tg_lengths - tree_lengths, old_tg_lengths - old_tree_lengths):
        raise AssertionError("Tree/TG structural length difference changed")
    manifest["tree_tg_length_delta_preserved"] = True

    if args.install:
        for name, (paths, output_data, output_index) in outputs.items():
            manifest["formats"][name]["installed"] = install_normalized(
                paths, output_data, output_index
            )
            manifest["formats"][name]["installed_data_sha256"] = sha256_file(paths.data)
    else:
        for name, (_, output_data, output_index) in outputs.items():
            manifest["formats"][name]["staged"] = {
                "data": str(output_data),
                "sentence_index": str(output_index),
            }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_save_json = args.manifest.with_name(f".{args.manifest.name}.tmp")
    atomic_save_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(atomic_save_json, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
