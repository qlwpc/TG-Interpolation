"""Memory-mapped loader for independent GPST and Pushdown top-K candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NativeModelTopKSentence:
    global_sentence_id: int
    document_id: int
    tokens: np.ndarray
    content_bounds: tuple[int, int]
    pushdown_valid_count: int
    pushdown_proposal_scores: np.ndarray
    pushdown_spans: np.ndarray
    pushdown_span_counts: np.ndarray
    gpst_valid_count: int
    gpst_proposal_scores: np.ndarray
    gpst_merge_orders: np.ndarray


class NativeModelTopKShard:
    """Load one v2 shard without conflating model-specific candidate IDs."""

    slots = 300

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with (self.path / "run.json").open() as handle:
            self.run = json.load(handle)
        if self.run.get("status") != "complete":
            raise ValueError(f"native model top-K shard is incomplete: {self.path}")
        names = (
            "terminal_offsets", "gpst_offsets", "pushdown_offsets",
            "terminal_tokens", "content_bounds", "global_sentence_ids",
            "document_ids", "pushdown_valid_counts", "pushdown_proposal_scores",
            "pushdown_span_counts", "pushdown_spans", "gpst_valid_counts",
            "gpst_proposal_scores", "gpst_merge_orders", "pushdown_completed",
            "gpst_completed",
        )
        for name in names:
            setattr(self, name, np.load(self.path / f"{name}.npy", mmap_mode="r"))
        for component in ("pushdown", "gpst"):
            if not bool(np.all(getattr(self, f"{component}_completed"))):
                raise ValueError(f"{component} completion mask is false: {self.path}")

    def __len__(self) -> int:
        return len(self.pushdown_valid_counts)

    def sentence(self, index: int) -> NativeModelTopKSentence:
        if not 0 <= index < len(self):
            raise IndexError(index)
        pushdown_valid = int(self.pushdown_valid_counts[index])
        gpst_valid = int(self.gpst_valid_counts[index])
        token_start, token_end = map(int, self.terminal_offsets[index : index + 2])
        push_start, push_end = map(int, self.pushdown_offsets[index : index + 2])
        gpst_start, gpst_end = map(int, self.gpst_offsets[index : index + 2])
        content_left, content_right = map(int, self.content_bounds[index])
        push_width = (push_end - push_start) // (self.slots * 3)
        gpst_width = (gpst_end - gpst_start) // self.slots
        pushdown_rows = self.pushdown_spans[push_start:push_end].reshape(
            self.slots, push_width, 3
        )
        gpst_rows = self.gpst_merge_orders[gpst_start:gpst_end].reshape(
            self.slots, gpst_width
        )
        return NativeModelTopKSentence(
            global_sentence_id=int(self.global_sentence_ids[index]),
            document_id=int(self.document_ids[index]),
            tokens=self.terminal_tokens[token_start:token_end],
            content_bounds=(content_left, content_right),
            pushdown_valid_count=pushdown_valid,
            pushdown_proposal_scores=self.pushdown_proposal_scores[index, :pushdown_valid],
            pushdown_spans=pushdown_rows[:pushdown_valid],
            pushdown_span_counts=self.pushdown_span_counts[index, :pushdown_valid],
            gpst_valid_count=gpst_valid,
            gpst_proposal_scores=self.gpst_proposal_scores[index, :gpst_valid],
            gpst_merge_orders=gpst_rows[:gpst_valid],
        )


class NativeModelTopKCorpus:
    """Traverse finalized v2 shards without concatenating multi-GB arrays."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with (self.path / "manifest.json").open() as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format_version") != 2:
            raise ValueError("native model top-K corpus requires format version 2")
        if self.manifest.get("status") != "complete":
            raise ValueError("native model top-K corpus manifest is incomplete")
        self.shards = tuple(
            NativeModelTopKShard(self.path / name) for name in self.manifest["shards"]
        )
        counts = np.asarray([len(shard) for shard in self.shards], dtype=np.int64)
        self.ends = np.cumsum(counts)
        if not len(self.ends) or int(self.ends[-1]) != int(self.manifest["sentence_count"]):
            raise ValueError("native model top-K shard counts do not match manifest")
        # Only ~0.6 MB for BBC test-PPL. Keeping this index in RAM lets callers
        # split evaluation on document boundaries without touching large mmap
        # candidate arrays or accidentally dropping a document prefix.
        self.document_ids = np.concatenate(
            [np.asarray(shard.document_ids, dtype=np.uint32) for shard in self.shards]
        )
        if np.any(self.document_ids[1:] < self.document_ids[:-1]):
            raise ValueError("native model top-K document IDs are not monotone")

    def __len__(self) -> int:
        return int(self.ends[-1])

    def sentence(self, index: int) -> NativeModelTopKSentence:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_id = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if shard_id == 0 else int(self.ends[shard_id - 1])
        return self.shards[shard_id].sentence(index - start)

    def document_sentence_range(self, start_document: int = 0, end_document: int | None = None) -> tuple[int, int]:
        """Return the half-open sentence interval for complete document IDs."""
        total = int(self.manifest["document_count"])
        if end_document is None:
            end_document = total
        if not 0 <= start_document <= end_document <= total:
            raise ValueError(f"invalid document interval [{start_document}, {end_document})")
        start = int(np.searchsorted(self.document_ids, start_document, side="left"))
        end = int(np.searchsorted(self.document_ids, end_document, side="left"))
        if start_document < end_document:
            if start == len(self.document_ids) or int(self.document_ids[start]) != start_document:
                raise ValueError(f"document {start_document} is absent")
        return start, end

    def sentences(self, indices: Sequence[int]):
        return tuple(self.sentence(int(index)) for index in indices)
