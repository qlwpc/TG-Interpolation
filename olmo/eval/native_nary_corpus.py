"""Memory-mapped loader for shared native n-ary GPST/Pushdown candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NativeNarySentence:
    global_sentence_id: int
    document_id: int
    tokens: np.ndarray
    content_bounds: tuple[int, int]
    valid_count: int
    proposal_scores: np.ndarray
    pushdown_spans: np.ndarray
    pushdown_span_counts: np.ndarray
    gpst_merge_orders: np.ndarray
    candidate_to_gpst: np.ndarray
    gpst_source_slots: np.ndarray
    gpst_multiplicities: np.ndarray
    gpst_log_masses: np.ndarray

    @property
    def gpst_unique_count(self) -> int:
        return len(self.gpst_source_slots)

    @property
    def gpst_unique_merge_orders(self) -> np.ndarray:
        return self.gpst_merge_orders[self.gpst_source_slots]


class NativeNaryShard:
    """Load one shard with one contiguous mmap slice per structure family."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with (self.path / "run.json").open() as handle:
            self.run = json.load(handle)
        if self.run.get("status") != "complete":
            raise ValueError(f"native n-ary shard is incomplete: {self.path}")
        names = (
            "terminal_offsets", "gpst_offsets", "pushdown_offsets",
            "terminal_tokens", "content_bounds", "global_sentence_ids",
            "document_ids", "valid_counts", "proposal_scores",
            "pushdown_span_counts", "pushdown_spans", "gpst_merge_orders",
            "candidate_to_gpst", "gpst_unique_counts", "gpst_source_slots",
            "gpst_multiplicities", "gpst_log_masses", "completed",
        )
        for name in names:
            setattr(self, name, np.load(self.path / f"{name}.npy", mmap_mode="r"))
        if not bool(np.all(self.completed)):
            raise ValueError(f"native n-ary completion mask is false: {self.path}")

    def __len__(self) -> int:
        return len(self.valid_counts)

    def sentence(self, index: int) -> NativeNarySentence:
        if not 0 <= index < len(self):
            raise IndexError(index)
        valid = int(self.valid_counts[index])
        unique = int(self.gpst_unique_counts[index])
        token_start, token_end = map(int, self.terminal_offsets[index : index + 2])
        push_start, push_end = map(int, self.pushdown_offsets[index : index + 2])
        gpst_start, gpst_end = map(int, self.gpst_offsets[index : index + 2])
        content_left, content_right = map(int, self.content_bounds[index])
        push_width = (push_end - push_start) // (300 * 3)
        gpst_width = (gpst_end - gpst_start) // 300
        return NativeNarySentence(
            global_sentence_id=int(self.global_sentence_ids[index]),
            document_id=int(self.document_ids[index]),
            tokens=self.terminal_tokens[token_start:token_end],
            content_bounds=(content_left, content_right),
            valid_count=valid,
            proposal_scores=self.proposal_scores[index, :valid],
            pushdown_spans=self.pushdown_spans[push_start:push_end].reshape(300, push_width, 3)[:valid],
            pushdown_span_counts=self.pushdown_span_counts[index, :valid],
            gpst_merge_orders=self.gpst_merge_orders[gpst_start:gpst_end].reshape(300, gpst_width)[:valid],
            candidate_to_gpst=self.candidate_to_gpst[index, :valid],
            gpst_source_slots=self.gpst_source_slots[index, :unique],
            gpst_multiplicities=self.gpst_multiplicities[index, :unique],
            gpst_log_masses=self.gpst_log_masses[index, :unique],
        )


class NativeNaryCorpus:
    """Traverse finalized shards without concatenating their multi-GB arrays."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with (self.path / "manifest.json").open() as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("status") != "complete":
            raise ValueError("native n-ary corpus manifest is incomplete")
        self.shards = tuple(NativeNaryShard(self.path / name) for name in self.manifest["shards"])
        counts = np.asarray([len(shard) for shard in self.shards], dtype=np.int64)
        self.ends = np.cumsum(counts)
        if int(self.ends[-1]) != int(self.manifest["sentence_count"]):
            raise ValueError("native n-ary shard counts do not match manifest")

    def __len__(self) -> int:
        return int(self.ends[-1])

    def sentence(self, index: int) -> NativeNarySentence:
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_id = int(np.searchsorted(self.ends, index, side="right"))
        start = 0 if shard_id == 0 else int(self.ends[shard_id - 1])
        return self.shards[shard_id].sentence(index - start)

    def sentences(self, indices: Sequence[int]):
        return tuple(self.sentence(int(index)) for index in indices)
