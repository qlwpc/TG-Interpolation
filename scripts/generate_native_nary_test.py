#!/usr/bin/env python3
"""Generate shared native n-ary top-K trees for GPST and Pushdown.

The canonical evaluation boundary is ``testppl_tree`` (4,966 documents), while
``tree/test.npy`` supplies the terminal-alignment source relation.  Benepar is
used only for its labeled span score chart.  Labels are marginalized, unary
choices are removed, and exact native n-ary K-best decoding runs on CPU workers
while the next chart batches are scored on one GPU.

Output is sharded and mmap-ready.  Every sentence has 300 physical candidate
slots plus a logical ``valid_count``.  Pushdown stores only real n-ary spans;
GPST stores deterministic right-binary merge orders and a mass-preserving map
from shared n-ary candidates to unique GPST trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import socket
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# The environment used by the released Benepar checkpoint combines an older
# sentencepiece-generated protobuf module with a newer protobuf runtime.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datatools.native_binary import (  # noqa: E402
    adapt_native_nary_candidates,
    decode_labeled_scores_nary_topk,
)
from olmo.data.parse_align import TreeVocab  # noqa: E402


SLOTS = 300
FORMAT_VERSION = 1
DEFAULT_TREE = REPO / "dataset/bbc-news/tree/test.npy"
DEFAULT_PPL = REPO / "dataset/bbc-news/testppl_tree"
DEFAULT_TOKENIZER = REPO / "dataset/bbc-news/TG_GPT2_tokenizer.json"
DEFAULT_OUTPUT = REPO / "dataset/bbc-news/native_nary_300"


@dataclass(frozen=True)
class SentenceInput:
    global_sentence_id: int
    document_id: int
    terminal_tokens: tuple[int, ...]
    content_start: int
    content_end: int
    words: tuple[str, ...]
    word_piece_ids: tuple[tuple[int, ...], ...]


def _json_dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _tree_terminals_and_record(block: Sequence[int], vocab: TreeVocab):
    record = []
    content = []
    content_start = None
    content_end = None
    depth = 0
    roots = 0
    for raw in block:
        token = int(raw)
        if vocab.is_opening(token):
            if depth == 0:
                roots += 1
                content_start = len(record)
            depth += 1
        elif vocab.is_closing(token):
            depth -= 1
            if depth < 0:
                raise ValueError("closing non-terminal without an opening token")
            if depth == 0:
                content_end = len(record)
        else:
            record.append(token)
            if depth:
                content.append(token)
    if depth or roots != 1 or content_start is None or content_end is None:
        raise ValueError(f"expected one complete top-level tree, got roots={roots}, depth={depth}")
    if tuple(record[content_start:content_end]) != tuple(content):
        raise ValueError("tree content is not a contiguous terminal segment")
    return tuple(record), tuple(content), int(content_start), int(content_end)


def _scan_document_terminals(tree_path: Path, vocab: TreeVocab):
    data = np.load(tree_path, mmap_mode="r").reshape(-1)
    documents = []
    current_document = None
    current_sentence = None
    depth = 0
    for raw in data:
        token = int(raw)
        if token == vocab.pad:
            continue
        if token == vocab.bos and depth == 0:
            current_document = []
            continue
        if vocab.is_opening(token):
            if depth == 0:
                current_sentence = []
            depth += 1
        elif vocab.is_closing(token):
            depth -= 1
            if depth == 0:
                if current_document is None or current_sentence is None:
                    raise ValueError("tree/test.npy contains a tree outside a document")
                current_document.append(tuple(current_sentence))
                current_sentence = None
        elif token == vocab.eos and depth == 0:
            if current_document is not None:
                documents.append(tuple(current_document))
                current_document = None
        elif depth:
            current_sentence.append(token)
    return documents


class CanonicalPPLCorpus:
    def __init__(self, ppl_dir: Path, tokenizer_path: Path) -> None:
        self.ppl_dir = ppl_dir
        self.tree = np.load(ppl_dir / "tree_300.npy", mmap_mode="r")
        lengths = np.load(ppl_dir / "tree_sent_index.npy", mmap_mode="r")
        if lengths.size % SLOTS:
            raise ValueError("tree_sent_index length is not divisible by 300")
        self.lengths = lengths.reshape(-1, SLOTS)
        self.document_counts = np.asarray(
            np.load(ppl_dir / "tree_doc_index.npy", mmap_mode="r"), dtype=np.int64
        )
        if int(self.document_counts.sum()) != len(self.lengths):
            raise ValueError("tree_doc_index does not cover all tree_300 sentences")
        totals = self.lengths.sum(axis=1, dtype=np.uint64)
        self.sentence_offsets = np.empty(len(totals) + 1, dtype=np.uint64)
        self.sentence_offsets[0] = 0
        np.cumsum(totals, dtype=np.uint64, out=self.sentence_offsets[1:])
        if int(self.sentence_offsets[-1]) != len(self.tree):
            raise ValueError("tree_sent_index lengths do not cover tree_300.npy")
        self.document_ends = np.cumsum(self.document_counts, dtype=np.int64)
        self.vocab = TreeVocab.from_tokenizer_file(str(tokenizer_path))
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def document_id(self, sentence_id: int) -> int:
        return int(np.searchsorted(self.document_ends, sentence_id, side="right"))

    def candidate_zero_block(self, sentence_id: int) -> np.ndarray:
        start = int(self.sentence_offsets[sentence_id])
        end = start + int(self.lengths[sentence_id, 0])
        return self.tree[start:end]

    def sentence_input(self, sentence_id: int) -> SentenceInput:
        record, content, content_start, content_end = _tree_terminals_and_record(
            self.candidate_zero_block(sentence_id), self.vocab
        )
        # TG serialization inserts a space before every ordinary parser leaf,
        # which GPT-2 records as a ``Ġ`` word start. Recover groups directly
        # from the existing IDs: decoding and re-encoding is not identity for
        # literal strings such as "-LSB-", because the second pass recognizes
        # them as added tokens. PTB bracket added tokens also ignore surrounding
        # whitespace, so each must be treated as one atomic parser word.
        ptb_atoms = {"-LRB-", "-RRB-", "-LCB-", "-RCB-", "-LSB-", "-RSB-"}
        piece_rows = []
        current = []
        for token in content:
            spelling = self.tokenizer.id_to_token(int(token))
            if spelling in ptb_atoms:
                if current:
                    piece_rows.append(tuple(current))
                    current = []
                piece_rows.append((int(token),))
            else:
                if current and spelling.startswith("Ġ"):
                    piece_rows.append(tuple(current))
                    current = []
                current.append(int(token))
        if current:
            piece_rows.append(tuple(current))
        word_piece_ids = tuple(piece_rows)
        words = tuple(
            self.tokenizer.decode(list(pieces), skip_special_tokens=False).strip()
            for pieces in word_piece_ids
        )
        if not words or any(not word for word in words):
            raise ValueError(f"sentence {sentence_id} has an empty recovered parser word")
        if tuple(token for pieces in word_piece_ids for token in pieces) != content:
            raise AssertionError("word grouping changed terminal IDs")
        return SentenceInput(
            global_sentence_id=sentence_id,
            document_id=self.document_id(sentence_id),
            terminal_tokens=record,
            content_start=content_start,
            content_end=content_end,
            words=words,
            word_piece_ids=word_piece_ids,
        )

    def shard_bounds(self, shard_id: int, num_shards: int):
        if not 0 <= shard_id < num_shards:
            raise ValueError("shard_id must be in [0, num_shards)")
        targets = np.linspace(0, len(self.lengths), num_shards + 1)
        doc_bounds = [0]
        for target in targets[1:-1]:
            doc_bounds.append(int(np.searchsorted(self.document_ends, target, side="left") + 1))
        doc_bounds.append(len(self.document_counts))
        doc_bounds = np.maximum.accumulate(np.asarray(doc_bounds, dtype=np.int64))
        sentence_bounds = np.concatenate(([0], self.document_ends))
        doc_start, doc_end = map(int, doc_bounds[shard_id : shard_id + 2])
        return doc_start, doc_end, int(sentence_bounds[doc_start]), int(sentence_bounds[doc_end])


def _digest_document(sentences: Iterable[Sequence[int]]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for sentence in sentences:
        values = np.asarray(sentence, dtype=np.uint16)
        digest.update(values.tobytes())
    return digest.hexdigest()


def audit_alignment(args) -> None:
    corpus = CanonicalPPLCorpus(args.ppl_dir, args.tokenizer)
    test_documents = _scan_document_terminals(args.test_tree, corpus.vocab)
    offset = len(test_documents) - len(corpus.document_counts)
    if offset < 0:
        raise ValueError("test.npy contains fewer documents than testppl_tree")
    ppl_documents = []
    cursor = 0
    for count in corpus.document_counts:
        document = []
        for sentence_id in range(cursor, cursor + int(count)):
            _record, content, _start, _end = _tree_terminals_and_record(
                corpus.candidate_zero_block(sentence_id), corpus.vocab
            )
            document.append(content)
        ppl_documents.append(tuple(document))
        cursor += int(count)
    exceptions = []
    for ppl_id, (source, canonical) in enumerate(zip(test_documents[offset:], ppl_documents)):
        source_hash = _digest_document(source)
        canonical_hash = _digest_document(canonical)
        if source_hash != canonical_hash:
            exceptions.append(
                {
                    "ppl_document_id": ppl_id,
                    "test_document_id": ppl_id + offset,
                    "test_sentence_count": len(source),
                    "ppl_sentence_count": len(canonical),
                    "test_terminal_count": sum(map(len, source)),
                    "ppl_terminal_count": sum(map(len, canonical)),
                    "test_hash": source_hash,
                    "ppl_hash": canonical_hash,
                }
            )
    result = {
        "format_version": FORMAT_VERSION,
        "test_document_count": len(test_documents),
        "test_sentence_count": sum(map(len, test_documents)),
        "ppl_document_count": len(ppl_documents),
        "ppl_sentence_count": sum(map(len, ppl_documents)),
        "document_offset": offset,
        "exact_document_count": len(ppl_documents) - len(exceptions),
        "exceptions": exceptions,
        "test_tree": str(args.test_tree),
        "ppl_dir": str(args.ppl_dir),
        "tokenizer": str(args.tokenizer),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    _json_dump(args.output / "alignment_audit.json", result)
    print(json.dumps(result, indent=2))


def _decode_and_adapt(payload):
    local_id, labeled_scores, word_piece_ids, content_start, k = payload
    candidates = decode_labeled_scores_nary_topk(labeled_scores, k=k)
    adapted = adapt_native_nary_candidates(
        candidates, word_piece_ids, pushdown_token_offset=content_start
    )
    return local_id, candidates, adapted


def _open_array(path: Path, shape, dtype, fill=None):
    if path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != tuple(shape) or array.dtype != np.dtype(dtype):
            raise ValueError(f"resume array mismatch for {path}: {array.shape} {array.dtype}")
        return array
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        array[...] = fill
    array.flush()
    return array


def _prepare_shard_arrays(shard_dir: Path, records: Sequence[SentenceInput]):
    count = len(records)
    token_lengths = np.asarray([len(row.terminal_tokens) for row in records], dtype=np.uint64)
    token_offsets = np.empty(count + 1, dtype=np.uint64)
    token_offsets[0] = 0
    np.cumsum(token_lengths, out=token_offsets[1:])
    gpst_widths = np.asarray(
        [SLOTS * max(row.content_end - row.content_start - 1, 0) for row in records],
        dtype=np.uint64,
    )
    gpst_offsets = np.empty(count + 1, dtype=np.uint64)
    gpst_offsets[0] = 0
    np.cumsum(gpst_widths, out=gpst_offsets[1:])
    push_widths = np.asarray(
        [SLOTS * max(len(row.words) - 1, 0) * 3 for row in records], dtype=np.uint64
    )
    push_offsets = np.empty(count + 1, dtype=np.uint64)
    push_offsets[0] = 0
    np.cumsum(push_widths, out=push_offsets[1:])
    word_offsets = np.empty(count + 1, dtype=np.uint64)
    word_offsets[0] = 0
    np.cumsum(np.asarray([len(row.words) for row in records], dtype=np.uint64), out=word_offsets[1:])

    shard_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("terminal_offsets.npy", token_offsets),
        ("gpst_offsets.npy", gpst_offsets),
        ("pushdown_offsets.npy", push_offsets),
        ("word_offsets.npy", word_offsets),
    ):
        if not (shard_dir / name).exists():
            np.save(shard_dir / name, values)
    arrays = {
        "terminal_tokens": _open_array(shard_dir / "terminal_tokens.npy", (int(token_offsets[-1]),), np.uint16),
        "content_bounds": _open_array(shard_dir / "content_bounds.npy", (count, 2), np.uint16),
        "word_starts": _open_array(shard_dir / "word_starts.npy", (int(word_offsets[-1]),), np.uint16),
        "sentence_ids": _open_array(shard_dir / "global_sentence_ids.npy", (count,), np.uint32),
        "document_ids": _open_array(shard_dir / "document_ids.npy", (count,), np.uint32),
        "valid_counts": _open_array(shard_dir / "valid_counts.npy", (count,), np.uint16),
        "proposal_scores": _open_array(shard_dir / "proposal_scores.npy", (count, SLOTS), np.float32, -np.inf),
        "pushdown_span_counts": _open_array(shard_dir / "pushdown_span_counts.npy", (count, SLOTS), np.uint16),
        "pushdown_spans": _open_array(shard_dir / "pushdown_spans.npy", (int(push_offsets[-1]),), np.int16, -1),
        "gpst_merge_orders": _open_array(shard_dir / "gpst_merge_orders.npy", (int(gpst_offsets[-1]),), np.int16, -1),
        "candidate_to_gpst": _open_array(shard_dir / "candidate_to_gpst.npy", (count, SLOTS), np.int16, -1),
        "gpst_unique_counts": _open_array(shard_dir / "gpst_unique_counts.npy", (count,), np.uint16),
        "gpst_source_slots": _open_array(shard_dir / "gpst_source_slots.npy", (count, SLOTS), np.int16, -1),
        "gpst_multiplicities": _open_array(shard_dir / "gpst_multiplicities.npy", (count, SLOTS), np.uint16),
        "gpst_log_masses": _open_array(shard_dir / "gpst_log_masses.npy", (count, SLOTS), np.float32, -np.inf),
        "completed": _open_array(shard_dir / "completed.npy", (count,), np.bool_, False),
    }
    for local_id, row in enumerate(records):
        ts, te = map(int, token_offsets[local_id : local_id + 2])
        arrays["terminal_tokens"][ts:te] = row.terminal_tokens
        arrays["content_bounds"][local_id] = (row.content_start, row.content_end)
        arrays["sentence_ids"][local_id] = row.global_sentence_id
        arrays["document_ids"][local_id] = row.document_id
        ws, we = map(int, word_offsets[local_id : local_id + 2])
        cursor = 0
        starts = []
        for pieces in row.word_piece_ids:
            starts.append(cursor)
            cursor += len(pieces)
        arrays["word_starts"][ws:we] = starts
    return arrays, token_offsets, gpst_offsets, push_offsets


def _write_result(local_id, candidates, adapted, records, arrays, gpst_offsets, push_offsets):
    row = records[local_id]
    valid = len(candidates)
    arrays["valid_counts"][local_id] = valid
    arrays["proposal_scores"][local_id, :valid] = [candidate.score for candidate in candidates]

    push_width = max(len(row.words) - 1, 0)
    ps, pe = map(int, push_offsets[local_id : local_id + 2])
    push_view = arrays["pushdown_spans"][ps:pe].reshape(SLOTS, push_width, 3)
    padding_spans = adapted.pushdown_spans[0]
    for slot in range(SLOTS):
        spans = adapted.pushdown_spans[slot] if slot < valid else padding_spans
        arrays["pushdown_span_counts"][local_id, slot] = len(spans)
        if spans:
            push_view[slot, : len(spans)] = spans

    gpst_width = max(row.content_end - row.content_start - 1, 0)
    gs, ge = map(int, gpst_offsets[local_id : local_id + 2])
    gpst_view = arrays["gpst_merge_orders"][gs:ge].reshape(SLOTS, gpst_width)
    for slot in range(SLOTS):
        source = slot if slot < valid else 0
        unique_id = adapted.candidate_to_gpst[source]
        if gpst_width:
            gpst_view[slot] = adapted.gpst_candidates[unique_id].merge_orders
    arrays["candidate_to_gpst"][local_id, :valid] = adapted.candidate_to_gpst
    unique = len(adapted.gpst_candidates)
    arrays["gpst_unique_counts"][local_id] = unique
    arrays["gpst_source_slots"][local_id, :unique] = adapted.gpst_source_slots
    arrays["gpst_multiplicities"][local_id, :unique] = adapted.gpst_multiplicities
    arrays["gpst_log_masses"][local_id, :unique] = adapted.gpst_log_masses
    arrays["completed"][local_id] = True


def generate_shard(args) -> None:
    audit_path = args.output / "alignment_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(f"run the audit command first: missing {audit_path}")
    with audit_path.open() as handle:
        audit = json.load(handle)
    if audit["ppl_document_count"] != 4966 or audit["ppl_sentence_count"] != 148836:
        raise ValueError("alignment audit does not match the fixed BBC test-PPL contract")

    corpus = CanonicalPPLCorpus(args.ppl_dir, args.tokenizer)
    doc_start, doc_end, sent_start, sent_end = corpus.shard_bounds(args.shard_id, args.num_shards)
    if args.max_sentences is not None:
        if args.max_sentences < 1:
            raise ValueError("max_sentences must be positive")
        sent_end = min(sent_end, sent_start + args.max_sentences)
    shard_dir = args.output / f"shard-{args.shard_id:05d}-of-{args.num_shards:05d}"
    records = [corpus.sentence_input(i) for i in range(sent_start, sent_end)]
    arrays, _token_offsets, gpst_offsets, push_offsets = _prepare_shard_arrays(shard_dir, records)
    pending_ids = [i for i in range(len(records)) if not bool(arrays["completed"][i])]

    run = {
        "format_version": FORMAT_VERSION,
        "status": "running",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "document_start": doc_start,
        "document_end": doc_end,
        "sentence_start": sent_start,
        "sentence_end": sent_end,
        "sentence_count": len(records),
        "dev_sentence_limit": args.max_sentences,
        "slots": SLOTS,
        "model": args.model,
        "device": args.device,
        "cpu_workers": args.cpu_workers,
        "parser_batch_tokens": args.parser_batch_tokens,
        "command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at_unix": time.time(),
    }
    _json_dump(shard_dir / "run.json", run)
    if not pending_ids:
        run["status"] = "complete"
        _json_dump(shard_dir / "run.json", run)
        print(f"{shard_dir} already complete")
        return

    import benepar
    import torch

    parser = benepar.Parser(args.model, batch_size=args.parser_batch_size)
    parser._parser.to(torch.device(args.device))
    executor = ProcessPoolExecutor(
        max_workers=args.cpu_workers, mp_context=mp.get_context("spawn")
    )
    futures = set()
    completed_now = 0
    started = time.time()

    def drain(block: bool) -> None:
        nonlocal futures, completed_now
        if not futures:
            return
        done, futures = wait(futures, return_when=FIRST_COMPLETED if block else FIRST_COMPLETED, timeout=None if block else 0)
        for future in done:
            local_id, candidates, adapted = future.result()
            _write_result(local_id, candidates, adapted, records, arrays, gpst_offsets, push_offsets)
            completed_now += 1
            if completed_now % args.flush_every == 0:
                for array in arrays.values():
                    array.flush()
                elapsed = time.time() - started
                print(
                    f"shard={args.shard_id} completed={completed_now}/{len(pending_ids)} "
                    f"rate={completed_now / max(elapsed, 1e-6):.2f} sent/s",
                    flush=True,
                )

    try:
        cursor = 0
        while cursor < len(pending_ids):
            batch_ids = []
            cost = 0
            while cursor < len(pending_ids) and len(batch_ids) < args.parser_batch_size:
                local_id = pending_ids[cursor]
                length = len(records[local_id].words)
                if batch_ids and cost + length > args.parser_batch_tokens:
                    break
                batch_ids.append(local_id)
                cost += length
                cursor += 1
            examples = [
                parser._with_missing_fields_filled(benepar.InputSentence(words=records[i].words))
                for i in batch_ids
            ]
            charts = parser._parser.parse(
                examples, return_scores=True, subbatch_max_tokens=args.parser_batch_tokens
            )
            for local_id, chart in zip(batch_ids, charts):
                row = records[local_id]
                futures.add(
                    executor.submit(
                        _decode_and_adapt,
                        (local_id, chart, row.word_piece_ids, row.content_start, args.k),
                    )
                )
            while len(futures) >= args.max_pending:
                drain(True)
            drain(False)
        while futures:
            drain(True)
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        for array in arrays.values():
            array.flush()

    if not bool(np.all(arrays["completed"])):
        raise RuntimeError("shard ended with incomplete sentences")
    run.update(
        status="complete",
        completed_at_unix=time.time(),
        elapsed_seconds=time.time() - started,
        valid_candidate_min=int(arrays["valid_counts"].min()),
        valid_candidate_max=int(arrays["valid_counts"].max()),
        gpst_unique_min=int(arrays["gpst_unique_counts"].min()),
        gpst_unique_max=int(arrays["gpst_unique_counts"].max()),
    )
    _json_dump(shard_dir / "run.json", run)
    print(json.dumps(run, indent=2))


def finalize(args) -> None:
    shards = sorted(args.output.glob("shard-*-of-*"))
    if len(shards) != args.num_shards:
        raise ValueError(f"found {len(shards)} shards, expected {args.num_shards}")
    runs = []
    sentence_total = 0
    document_total = 0
    expected_sentence = 0
    expected_document = 0
    for shard in shards:
        with (shard / "run.json").open() as handle:
            run = json.load(handle)
        if run["status"] != "complete":
            raise ValueError(f"incomplete shard: {shard}")
        if run["sentence_start"] != expected_sentence or run["document_start"] != expected_document:
            raise ValueError(f"non-contiguous shard boundary at {shard}")
        completed = np.load(shard / "completed.npy", mmap_mode="r")
        if not bool(np.all(completed)):
            raise ValueError(f"completion mask is false in {shard}")
        valid = np.load(shard / "valid_counts.npy", mmap_mode="r")
        unique = np.load(shard / "gpst_unique_counts.npy", mmap_mode="r")
        if np.any(valid < 1) or np.any(valid > SLOTS) or np.any(unique < 1) or np.any(unique > valid):
            raise ValueError(f"invalid candidate counts in {shard}")
        expected_sentence = run["sentence_end"]
        expected_document = run["document_end"]
        sentence_total += run["sentence_count"]
        document_total += run["document_end"] - run["document_start"]
        runs.append(run)
    if sentence_total != 148836 or document_total != 4966:
        raise ValueError(f"unexpected totals: documents={document_total}, sentences={sentence_total}")
    manifest = {
        "format_version": FORMAT_VERSION,
        "status": "complete",
        "document_count": document_total,
        "sentence_count": sentence_total,
        "candidate_slots": SLOTS,
        "shard_count": len(shards),
        "shards": [path.name for path in shards],
        "alignment_audit": "alignment_audit.json",
        "semantic_contract": {
            "shared_candidates": "unlabeled unary-free native n-ary trees over parser words",
            "pushdown": "real n-ary BPE constituent spans only",
            "gpst": "deterministic right-binary BPE merge orders",
            "gpst_collision_mass": "logsumexp of shared n-ary proposal scores",
        },
    }
    _json_dump(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "generate", "finalize"))
    parser.add_argument("--test-tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--ppl-dir", type=Path, default=DEFAULT_PPL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=SLOTS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--model", default="benepar_en3_large")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--parser-batch-size", type=int, default=64)
    parser.add_argument("--parser-batch-tokens", type=int, default=4000)
    parser.add_argument("--max-pending", type=int, default=32)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument(
        "--max-sentences", type=int, default=None,
        help="Development smoke limit; never use with finalize.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k != SLOTS:
        raise ValueError("this production format fixes k=300")
    if args.command == "audit":
        audit_alignment(args)
    elif args.command == "generate":
        generate_shard(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
