#!/usr/bin/env python3
"""Build terminal, Tree, and TG test streams from testppl_tree candidate 0.

The canonical document/sentence boundaries come from tree_doc_index.npy and
tree_sent_index.npy.  Historical testppl records contain only one document
boundary special token per document (BOS on the first document of each source
split, EOS otherwise), so this builder normalizes every document to
``[BOS, ..., EOS]``.

For each output format the script writes:

* ``test.npy``: flat uint16 token stream;
* ``test_sent_index.npy``: token count of every sentence record, including BOS
  on the first sentence and EOS on the last sentence of a document;
* ``test_doc_index.npy``: sentence count of every document.

Tree is candidate 0 unchanged apart from document-boundary normalization.  TG
duplicates every closing non-terminal in place.  Terminal removes every
non-terminal.  Outputs are written atomically and verified before publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from olmo.data.parse_align import TreeVocab  # noqa: E402


SAMPLES_PER_SENTENCE = 300
DEFAULT_INPUT = REPO / "dataset/bbc-news/testppl_tree"
DEFAULT_TOKENIZER = REPO / "dataset/bbc-news/TG_GPT2_tokenizer.json"
DEFAULT_OUTPUT = REPO / "dataset/bbc-news/testppl_aligned"
FORMATS = ("terminal", "tree", "tg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_zero_offsets(lengths: np.ndarray) -> np.ndarray:
    if lengths.ndim != 2 or lengths.shape[1] != SAMPLES_PER_SENTENCE:
        raise ValueError(
            f"expected sentence lengths shaped (N, 300), got {lengths.shape}"
        )
    totals = lengths.sum(axis=1, dtype=np.uint64)
    offsets = np.empty(len(totals) + 1, dtype=np.uint64)
    offsets[0] = 0
    np.cumsum(totals, dtype=np.uint64, out=offsets[1:])
    return offsets


def strip_document_specials(block: np.ndarray, vocab: TreeVocab) -> np.ndarray:
    mask = (block != vocab.bos) & (block != vocab.eos) & (block != vocab.pad)
    return np.asarray(block[mask])


def normalize_sentence_record(
    block: np.ndarray,
    vocab: TreeVocab,
    *,
    first_in_document: bool,
    last_in_document: bool,
) -> np.ndarray:
    content = strip_document_specials(block, vocab)
    extra = int(first_in_document) + int(last_in_document)
    out = np.empty(len(content) + extra, dtype=block.dtype)
    cursor = 0
    if first_in_document:
        out[cursor] = vocab.bos
        cursor += 1
    out[cursor : cursor + len(content)] = content
    cursor += len(content)
    if last_in_document:
        out[cursor] = vocab.eos
    return out


def validate_tree_record(record: Sequence[int], vocab: TreeVocab) -> None:
    depth = 0
    roots = 0
    for raw in record:
        token = int(raw)
        if vocab.is_opening(token):
            if depth == 0:
                roots += 1
            depth += 1
        elif vocab.is_closing(token):
            depth -= 1
            if depth < 0:
                raise ValueError("closing non-terminal precedes its opening token")
    if depth != 0 or roots != 1:
        raise ValueError(f"expected one balanced top-level tree, roots={roots}, depth={depth}")


def terminal_projection(record: np.ndarray, vocab: TreeVocab) -> np.ndarray:
    return np.asarray(
        record[
            ~((record >= vocab.op_lo) & (record <= vocab.cl_hi))
        ]
    )


def tg_projection(record: np.ndarray, vocab: TreeVocab) -> np.ndarray:
    close = (record >= vocab.cl_lo) & (record <= vocab.cl_hi)
    repeats = np.ones(len(record), dtype=np.uint8)
    repeats[close] = 2
    return np.repeat(record, repeats)


def atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def open_atomic_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype):
    temporary = path.with_name(f".{path.name}.tmp")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    return temporary, array


def verify_outputs(
    output_root: Path,
    sentence_lengths: dict[str, np.ndarray],
    document_counts: np.ndarray,
    vocab: TreeVocab,
) -> None:
    arrays = {
        fmt: np.load(output_root / fmt / "test.npy", mmap_mode="r")
        for fmt in FORMATS
    }
    offsets = {}
    for fmt in FORMATS:
        lengths = np.asarray(sentence_lengths[fmt], dtype=np.uint64)
        offsets[fmt] = np.empty(len(lengths) + 1, dtype=np.uint64)
        offsets[fmt][0] = 0
        np.cumsum(lengths, dtype=np.uint64, out=offsets[fmt][1:])
        if int(offsets[fmt][-1]) != len(arrays[fmt]):
            raise AssertionError(f"{fmt} sentence index does not cover test.npy")

    for sentence_id in range(len(sentence_lengths["tree"])):
        ts, te = map(int, offsets["tree"][sentence_id : sentence_id + 2])
        xs, xe = map(int, offsets["terminal"][sentence_id : sentence_id + 2])
        gs, ge = map(int, offsets["tg"][sentence_id : sentence_id + 2])
        tree_record = np.asarray(arrays["tree"][ts:te])
        if not np.array_equal(
            terminal_projection(tree_record, vocab), arrays["terminal"][xs:xe]
        ):
            raise AssertionError(f"Tree -> terminal mismatch at sentence {sentence_id}")
        if not np.array_equal(tg_projection(tree_record, vocab), arrays["tg"][gs:ge]):
            raise AssertionError(f"Tree -> TG mismatch at sentence {sentence_id}")

    sentence_ends = np.cumsum(document_counts, dtype=np.uint64)
    sentence_starts = np.concatenate((np.zeros(1, dtype=np.uint64), sentence_ends[:-1]))
    for fmt in FORMATS:
        array = arrays[fmt]
        if int(np.count_nonzero(array == vocab.bos)) != len(document_counts):
            raise AssertionError(f"{fmt} does not have exactly one BOS per document")
        if int(np.count_nonzero(array == vocab.eos)) != len(document_counts):
            raise AssertionError(f"{fmt} does not have exactly one EOS per document")
        if np.count_nonzero(array == vocab.pad):
            raise AssertionError(f"{fmt} unexpectedly contains PAD")
        for document_id, (first_sentence, last_sentence) in enumerate(
            zip(sentence_starts, sentence_ends)
        ):
            start = int(offsets[fmt][int(first_sentence)])
            end = int(offsets[fmt][int(last_sentence)])
            if int(array[start]) != vocab.bos or int(array[end - 1]) != vocab.eos:
                raise AssertionError(
                    f"{fmt} document {document_id} is not framed by BOS/EOS"
                )

    print(
        "verified: all sentence indexes cover their streams; Tree -> terminal "
        "and Tree -> TG agree exactly; every document has one BOS and one EOS",
        flush=True,
    )


def build(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    tree_path = input_dir / "tree_300.npy"
    sent_path = input_dir / "tree_sent_index.npy"
    doc_path = input_dir / "tree_doc_index.npy"
    for path in (tree_path, sent_path, doc_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    destinations = {
        fmt: output_root / fmt / "test.npy" for fmt in FORMATS
    }
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "outputs already exist (use --overwrite): "
            + ", ".join(str(path) for path in existing)
        )
    for fmt in FORMATS:
        (output_root / fmt).mkdir(parents=True, exist_ok=True)

    tree = np.load(tree_path, mmap_mode="r").reshape(-1)
    raw_lengths = np.load(sent_path, mmap_mode="r").reshape(-1)
    if raw_lengths.size % SAMPLES_PER_SENTENCE:
        raise ValueError("tree_sent_index.npy is not divisible by 300")
    lengths = raw_lengths.reshape(-1, SAMPLES_PER_SENTENCE)
    document_counts = np.asarray(np.load(doc_path, mmap_mode="r"), dtype=np.uint32)
    if int(document_counts.sum(dtype=np.uint64)) != len(lengths):
        raise ValueError("tree_doc_index.npy does not cover all sentences")
    sentence_offsets = candidate_zero_offsets(lengths)
    if int(sentence_offsets[-1]) != len(tree):
        raise ValueError("tree_sent_index.npy does not cover tree_300.npy")
    document_ends = np.cumsum(document_counts, dtype=np.uint64)
    document_starts = np.concatenate((np.zeros(1, dtype=np.uint64), document_ends[:-1]))
    vocab = TreeVocab.from_tokenizer_file(str(tokenizer_path))

    sentence_lengths = {
        fmt: np.empty(len(lengths), dtype=np.uint32) for fmt in FORMATS
    }
    old_special_counts = {"bos": 0, "eos": 0, "pad": 0}
    totals = {fmt: 0 for fmt in FORMATS}

    print("pass 1/2: validating candidate 0 and sizing outputs", flush=True)
    document_id = 0
    for sentence_id in range(len(lengths)):
        while sentence_id >= int(document_ends[document_id]):
            document_id += 1
        start = int(sentence_offsets[sentence_id])
        end = start + int(lengths[sentence_id, 0])
        block = np.asarray(tree[start:end])
        old_special_counts["bos"] += int(np.count_nonzero(block == vocab.bos))
        old_special_counts["eos"] += int(np.count_nonzero(block == vocab.eos))
        old_special_counts["pad"] += int(np.count_nonzero(block == vocab.pad))
        record = normalize_sentence_record(
            block,
            vocab,
            first_in_document=sentence_id == int(document_starts[document_id]),
            last_in_document=sentence_id + 1 == int(document_ends[document_id]),
        )
        validate_tree_record(record, vocab)
        terminal_length = int(
            np.count_nonzero(
                ~((record >= vocab.op_lo) & (record <= vocab.cl_hi))
            )
        )
        close_count = int(
            np.count_nonzero((record >= vocab.cl_lo) & (record <= vocab.cl_hi))
        )
        values = {
            "tree": len(record),
            "terminal": terminal_length,
            "tg": len(record) + close_count,
        }
        for fmt, value in values.items():
            sentence_lengths[fmt][sentence_id] = value
            totals[fmt] += value

    if old_special_counts != {"bos": 88, "eos": 4878, "pad": 0}:
        raise ValueError(f"unexpected historical boundary counts: {old_special_counts}")

    temporary_paths = {}
    outputs = {}
    for fmt in FORMATS:
        temporary_paths[fmt], outputs[fmt] = open_atomic_memmap(
            destinations[fmt], (totals[fmt],), tree.dtype
        )
    cursors = {fmt: 0 for fmt in FORMATS}

    print("pass 2/2: writing terminal, Tree, and TG streams", flush=True)
    try:
        document_id = 0
        for sentence_id in range(len(lengths)):
            while sentence_id >= int(document_ends[document_id]):
                document_id += 1
            start = int(sentence_offsets[sentence_id])
            end = start + int(lengths[sentence_id, 0])
            record = normalize_sentence_record(
                np.asarray(tree[start:end]),
                vocab,
                first_in_document=sentence_id == int(document_starts[document_id]),
                last_in_document=sentence_id + 1 == int(document_ends[document_id]),
            )
            projections = {
                "tree": record,
                "terminal": terminal_projection(record, vocab),
                "tg": tg_projection(record, vocab),
            }
            for fmt, values in projections.items():
                expected = int(sentence_lengths[fmt][sentence_id])
                if len(values) != expected:
                    raise AssertionError(
                        f"{fmt} sentence {sentence_id}: {len(values)} != {expected}"
                    )
                cursor = cursors[fmt]
                outputs[fmt][cursor : cursor + expected] = values
                cursors[fmt] += expected

        for fmt in FORMATS:
            if cursors[fmt] != totals[fmt]:
                raise AssertionError(f"{fmt}: wrote {cursors[fmt]} != {totals[fmt]}")
            outputs[fmt].flush()
            del outputs[fmt]
            os.replace(temporary_paths[fmt], destinations[fmt])
    except BaseException:
        outputs.clear()
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)
        raise

    for fmt in FORMATS:
        fmt_dir = output_root / fmt
        atomic_save(fmt_dir / "test_sent_index.npy", sentence_lengths[fmt])
        atomic_save(fmt_dir / "test_doc_index.npy", document_counts)

    verify_outputs(output_root, sentence_lengths, document_counts, vocab)

    manifest = {
        "format_version": 1,
        "source": {
            "tree_300": str(tree_path),
            "tree_sent_index": str(sent_path),
            "tree_doc_index": str(doc_path),
            "tokenizer": str(tokenizer_path),
        },
        "contract": {
            "candidate": 0,
            "samples_per_sentence": SAMPLES_PER_SENTENCE,
            "documents": len(document_counts),
            "sentences": len(lengths),
            "document_boundary": "one BOS before first sentence and one EOS after last sentence",
            "historical_source_special_counts": old_special_counts,
        },
        "outputs": {},
    }
    for fmt in FORMATS:
        path = destinations[fmt]
        array = np.load(path, mmap_mode="r")
        manifest["outputs"][fmt] = {
            "path": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_file(path),
            "sentence_index_sha256": sha256_file(
                output_root / fmt / "test_sent_index.npy"
            ),
            "document_index_sha256": sha256_file(
                output_root / fmt / "test_doc_index.npy"
            ),
        }
    temporary_manifest = output_root / ".manifest.json.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_manifest, output_root / "manifest.json")

    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
