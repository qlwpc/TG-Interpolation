"""Document-level GPST perplexity conditioned on prescribed gold trees.

This evaluator mirrors the repository's OLMo ``TG_doc`` protocol, with one
important model-specific adaptation: GPST consumes an unlabeled binary merge
trajectory instead of serialized non-terminal tokens.  For every sentence it
scores the supplied 300 parses directly; it never invokes beam search.

Previous sentences use candidate 0 as the shared document prefix, exactly as
the OLMo evaluator updates its KV cache from row 0.  The implementation uses
full-prefix teacher forcing rather than GPST's incomplete legacy cache API.
This is slower but makes the probability being measured unambiguous and gives
a correctness reference for a future optimized cache implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from olmo.data.parse_align import TreeVocab, parse_block_segments
from olmo.eval.native_model_topk_corpus import NativeModelTopKCorpus
from olmo.gpst.reader.dataset_gold import GoldTreeCollator, tree_to_merge_orders


@dataclass(frozen=True)
class GoldSegment:
    """One independently composed GPST segment."""

    tokens: Tuple[int, ...]
    merge_orders: Tuple[int, ...]

    @property
    def action_count(self) -> int:
        return max(2 * len(self.tokens) - 1, 0)


def _plain_leaf_merge_orders(length: int) -> Tuple[int, ...]:
    """Use a deterministic right-recursive tree for unparsed separators."""
    return tuple(range(length - 2, -1, -1))


def parse_gold_tree_candidate(
    block: Sequence[int], vocab: TreeVocab, direction: str = "right"
) -> Tuple[GoldSegment, ...]:
    """Convert one serialized gold parse to GPST segments.

    BOS/EOS/PAD are control symbols and are not content leaves: GPST supplies
    its own learned BOS embedding.  Ordinary plain leaves between/after trees
    (notably the BBC whitespace/newline suffix) remain scored terminals.
    """
    segments: List[GoldSegment] = []
    special = {vocab.bos, vocab.eos, vocab.pad}
    for kind, data in parse_block_segments(block, vocab):
        if kind == "tree":
            leaves, orders = tree_to_merge_orders(data, direction=direction)
        else:
            leaves = [int(token) for token in data if int(token) not in special]
            orders = list(_plain_leaf_merge_orders(len(leaves)))
        if not leaves:
            continue
        if len(orders) != len(leaves) - 1:
            raise ValueError(
                f"invalid gold segment: {len(leaves)} leaves but {len(orders)} merges"
            )
        segments.append(GoldSegment(tuple(map(int, leaves)), tuple(map(int, orders))))
    if not segments:
        raise ValueError("gold parse candidate contains no GPST terminal segments")
    return tuple(segments)


class GoldTree300Corpus:
    """Memory-mapped OLMo ``tree_300`` corpus, grouped by sentence/document."""

    def __init__(
        self,
        tree_path: str,
        sentence_index_path: str,
        document_index_path: str,
        tokenizer_path: str,
        samples_per_sentence: int = 300,
        max_sentences: Optional[int] = None,
        direction: str = "right",
    ) -> None:
        self.tree = np.load(tree_path, mmap_mode="r")
        lengths = np.load(sentence_index_path, mmap_mode="r")
        self.document_counts = np.asarray(
            np.load(document_index_path, mmap_mode="r"), dtype=np.int64
        )
        self.vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
        self.samples_per_sentence = int(samples_per_sentence)
        self.direction = direction
        if self.samples_per_sentence <= 0:
            raise ValueError("samples_per_sentence must be positive")
        total_sentences, remainder = divmod(len(lengths), self.samples_per_sentence)
        if remainder:
            raise ValueError(
                f"{len(lengths)} tree records is not divisible by "
                f"samples_per_sentence={self.samples_per_sentence}"
            )
        self.num_sentences = min(total_sentences, max_sentences) \
            if max_sentences is not None else total_sentences
        used_records = self.num_sentences * self.samples_per_sentence
        # Store only the requested prefix for smoke runs.  A full evaluation
        # necessarily needs all boundaries, matching OLMo's cumsum preparation.
        self.offsets = np.empty(used_records + 1, dtype=np.uint64)
        self.offsets[0] = 0
        np.cumsum(lengths[:used_records], dtype=np.uint64, out=self.offsets[1:])
        self.document_ends = np.cumsum(self.document_counts, dtype=np.int64)
        if self.document_ends.size == 0 or self.document_ends[-1] < self.num_sentences:
            raise ValueError("document_index does not cover all requested sentences")

    def __len__(self) -> int:
        return self.num_sentences

    def document_id(self, sentence_index: int) -> int:
        return int(np.searchsorted(self.document_ends, sentence_index, side="right"))

    def sentence_candidates(self, sentence_index: int) -> Tuple[Tuple[GoldSegment, ...], ...]:
        if sentence_index < 0 or sentence_index >= self.num_sentences:
            raise IndexError(sentence_index)
        first = sentence_index * self.samples_per_sentence
        candidates = []
        for flat_index in range(first, first + self.samples_per_sentence):
            start = int(self.offsets[flat_index])
            end = int(self.offsets[flat_index + 1])
            block = self.tree[start:end].astype(np.int64, copy=False).tolist()
            candidates.append(parse_gold_tree_candidate(block, self.vocab, self.direction))
        reference_tokens = tuple(token for seg in candidates[0] for token in seg.tokens)
        for candidate_id, candidate in enumerate(candidates[1:], start=1):
            tokens = tuple(token for seg in candidate for token in seg.tokens)
            if tokens != reference_tokens:
                raise ValueError(
                    f"sentence {sentence_index} candidate {candidate_id} has different terminals"
                )
        return tuple(candidates)

    def __iter__(self) -> Iterator[Tuple[int, Tuple[Tuple[GoldSegment, ...], ...]]]:
        for sentence_index in range(len(self)):
            yield self.document_id(sentence_index), self.sentence_candidates(sentence_index)


class NativeGPSTTopKCorpus:
    """Zero-reparse document corpus backed by native-model-topk v2 mmaps."""

    def __init__(self, native_path: str, tokenizer_path: str, max_sentences: Optional[int] = None,
                 start_document: int = 0, end_document: Optional[int] = None) -> None:
        self.native = NativeModelTopKCorpus(native_path)
        self.vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
        self.samples_per_sentence = int(self.native.manifest["candidate_slots"])
        self.start_sentence, end_sentence = self.native.document_sentence_range(start_document, end_document)
        self.num_sentences = end_sentence - self.start_sentence
        if max_sentences is not None:
            self.num_sentences = min(self.num_sentences, max_sentences)

    def __len__(self) -> int:
        return self.num_sentences

    def sentence_candidates(self, sentence_index: int) -> Tuple[Tuple[GoldSegment, ...], ...]:
        row = self.native.sentence(self.start_sentence + sentence_index)
        tokens = tuple(map(int, row.tokens.tolist()))
        left, right = row.content_bounds
        special = {self.vocab.bos, self.vocab.eos, self.vocab.pad}
        before = tuple(token for token in tokens[:left] if token not in special)
        after = tuple(token for token in tokens[right:] if token not in special)
        content = tokens[left:right]
        if not content:
            raise ValueError(f"native GPST sentence {sentence_index} has no parsed terminals")
        result = []
        for orders in row.gpst_merge_orders:
            order = tuple(map(int, orders[: max(len(content) - 1, 0)].tolist()))
            segments = []
            if before:
                segments.append(GoldSegment(before, _plain_leaf_merge_orders(len(before))))
            segments.append(GoldSegment(content, order))
            if after:
                segments.append(GoldSegment(after, _plain_leaf_merge_orders(len(after))))
            result.append(tuple(segments))
        return tuple(result)

    def __iter__(self) -> Iterator[Tuple[int, Tuple[Tuple[GoldSegment, ...], ...]]]:
        for index in range(len(self)):
            yield self.native.sentence(self.start_sentence + index).document_id, self.sentence_candidates(index)


def _candidate_signature(candidate: Sequence[GoldSegment]) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    return tuple((segment.tokens, segment.merge_orders) for segment in candidate)


def _as_collator_item(segments: Sequence[GoldSegment]) -> dict:
    text: List[int] = []
    splits: List[int] = []
    orders: List[np.ndarray] = []
    for segment in segments:
        text.extend(segment.tokens)
        splits.append(len(text))
        orders.append(np.asarray(segment.merge_orders, dtype=np.int32))
    return {
        "text": np.asarray(text, dtype=np.int32),
        "sentence_splits": splits,
        "merge_orders": orders,
    }


def _count_tokens(segments: Sequence[GoldSegment]) -> int:
    return sum(len(segment.tokens) for segment in segments)


def _count_actions(segments: Sequence[GoldSegment]) -> int:
    return sum(segment.action_count for segment in segments)


def aggregate_candidate_nll(
    nll: torch.Tensor,
    normalize_mixture: bool = False,
    multiplicities: Optional[torch.Tensor] = None,
) -> float:
    """Return a sentence log likelihood from its fixed-tree joint NLLs.

    ``multiplicities`` allows identical tree candidates to be evaluated once
    without changing the probability represented by the original candidate
    list.  This matters for tree_300: label and unary-chain differences often
    collapse to the same unlabeled GPST merge trajectory.
    """
    if nll.ndim != 1 or nll.numel() == 0:
        raise ValueError("candidate NLL must be a non-empty one-dimensional tensor")
    terms = -nll.to(dtype=torch.float64)
    candidate_count = nll.numel()
    if multiplicities is not None:
        if multiplicities.shape != nll.shape:
            raise ValueError("multiplicities must have the same shape as candidate NLL")
        multiplicities = multiplicities.to(dtype=torch.float64, device=nll.device)
        if bool((multiplicities <= 0).any()):
            raise ValueError("multiplicities must be positive")
        terms = terms + multiplicities.log()
        candidate_count = int(multiplicities.sum().item())
    value = torch.logsumexp(terms, dim=0).item()
    if normalize_mixture:
        value -= math.log(candidate_count)
    return value


def _compress_candidates(
    candidates: Sequence[Tuple[GoldSegment, ...]],
) -> Tuple[Tuple[Tuple[GoldSegment, ...], ...], torch.Tensor]:
    """Return stable unique candidates and their slot multiplicities."""
    unique: List[Tuple[GoldSegment, ...]] = []
    counts: List[int] = []
    positions = {}
    for candidate in candidates:
        signature = _candidate_signature(candidate)
        position = positions.get(signature)
        if position is None:
            positions[signature] = len(unique)
            unique.append(candidate)
            counts.append(1)
        else:
            counts[position] += 1
    return tuple(unique), torch.tensor(counts, dtype=torch.float64)


def _trim_prefix(
    prefix: Sequence[GoldSegment],
    current: Sequence[GoldSegment],
    max_action_nodes: int,
    max_terminals: int,
) -> Tuple[GoldSegment, ...]:
    current_actions = _count_actions(current)
    current_tokens = _count_tokens(current)
    if current_actions > max_action_nodes or current_tokens > max_terminals:
        raise ValueError(
            "one sentence exceeds the GPST context limit: "
            f"actions={current_actions}/{max_action_nodes}, "
            f"terminals={current_tokens}/{max_terminals}"
        )
    kept: List[GoldSegment] = []
    actions = current_actions
    tokens = current_tokens
    for segment in reversed(prefix):
        if actions + segment.action_count > max_action_nodes:
            break
        if tokens + len(segment.tokens) > max_terminals:
            break
        kept.append(segment)
        actions += segment.action_count
        tokens += len(segment.tokens)
    kept.reverse()
    return tuple(kept)


def _retain_prefix_for_any_future_sentence(
    prefix: Sequence[GoldSegment],
    max_action_nodes: int,
    max_terminals: int,
) -> Tuple[GoldSegment, ...]:
    """Bound retained document state without changing future contexts.

    ``_trim_prefix`` requires every future sentence to occupy at least one
    terminal and one GPST action.  Consequently no later call can consume more
    than ``limit - 1`` prefix terminals/actions.  Keeping precisely that
    largest fitting suffix is lossless for all future calls to
    ``_trim_prefix`` while preventing a single unusually long document from
    retaining every historical ``GoldSegment`` in Python memory.
    """
    prefix_action_limit = max_action_nodes - 1
    prefix_token_limit = max_terminals - 1
    if prefix_action_limit < 0 or prefix_token_limit < 0:
        return ()
    kept: List[GoldSegment] = []
    actions = 0
    tokens = 0
    for segment in reversed(prefix):
        segment_actions = segment.action_count
        segment_tokens = len(segment.tokens)
        if (actions + segment_actions > prefix_action_limit
                or tokens + segment_tokens > prefix_token_limit):
            break
        kept.append(segment)
        actions += segment_actions
        tokens += segment_tokens
    kept.reverse()
    return tuple(kept)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _fast_gold_batch(candidates: Sequence[Sequence[GoldSegment]]) -> dict:
    """Vectorized native-only replacement for :class:`GoldTreeCollator`.

    All candidates for one fixed sentence share terminals and segment lengths;
    only their merge trajectories differ.  The generic collator repeatedly
    builds Python lists and pads each row, which dominated the CUDA profile.
    """
    if not candidates:
        raise ValueError("candidates cannot be empty")
    template = candidates[0]
    lengths = [len(segment.tokens) for segment in template]
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("gold candidates need non-empty segments")
    width = sum(lengths)
    segment_count = len(lengths)
    for candidate in candidates:
        if len(candidate) != segment_count or [len(segment.tokens) for segment in candidate] != lengths:
            raise ValueError("native candidates do not share segment layout")
        if any(segment.tokens != template[i].tokens for i, segment in enumerate(candidate)):
            raise ValueError("native candidates do not share terminals")
    batch_size = len(candidates)
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    text = np.asarray([token for segment in template for token in segment.tokens], dtype=np.int64)
    chunk_input_ids = np.broadcast_to(text, (batch_size, width)).copy()
    chunk_masks = np.zeros((batch_size, width), dtype=np.int64)
    for segment_id, (start, length) in enumerate(zip(starts, lengths), start=1):
        chunk_masks[:, start:start + length] = segment_id
    max_segment = max(lengths)
    input_ids = np.zeros((batch_size * segment_count, max_segment), dtype=np.int64)
    masks = np.zeros_like(input_ids)
    for segment_id, segment in enumerate(template):
        rows = np.arange(batch_size) * segment_count + segment_id
        size = len(segment.tokens)
        input_ids[rows, :size] = np.asarray(segment.tokens, dtype=np.int64)
        masks[rows, :size] = 1
    orders = np.empty((batch_size, max(width - 1, 0)), dtype=np.int64)
    for row, candidate in enumerate(candidates):
        trajectory = []
        boundaries = []
        for segment_id, (start, segment) in enumerate(zip(starts, candidate)):
            trajectory.extend(start + int(order) for order in segment.merge_orders)
            if segment_id + 1 < segment_count:
                boundaries.append(start + len(segment.tokens) - 1)
        trajectory.extend(boundaries)
        if sorted(trajectory) != list(range(max(width - 1, 0))):
            raise ValueError("native merge orders are not a global gap permutation")
        orders[row] = trajectory
    return {
        "chunk_input_ids": torch.from_numpy(chunk_input_ids),
        "chunk_masks": torch.from_numpy(chunk_masks),
        "input_ids": torch.from_numpy(input_ids),
        "masks": torch.from_numpy(masks),
        "group_ids": np.repeat(np.arange(batch_size, dtype=np.int64), segment_count),
        "span_ids": [],
        "external_vocab_ids": None,
        "merge_orders": torch.from_numpy(orders),
    }


@torch.no_grad()
def _score_items(
    model: torch.nn.Module,
    items: Sequence[dict],
    collator: GoldTreeCollator,
    device: torch.device,
    prefix_tokens: int,
    current_tokens: int,
    prefix_actions: int,
    current_actions: int,
) -> torch.Tensor:
    batch = _move_batch(collator(items), device)
    output = model(
        **batch,
        force_gold_tree=True,
        score_token_range=(prefix_tokens, prefix_tokens + current_tokens),
        score_action_range=(prefix_actions, prefix_actions + current_actions),
    )
    if output.logits is None or output.action_logits is None:
        raise RuntimeError("GPST generative token/action heads are required for perplexity")
    token_loss = F.cross_entropy(
        output.logits.transpose(1, 2),
        output.token_targets,
        ignore_index=-100,
        reduction="none",
    ).sum(dim=1)
    action_loss = F.cross_entropy(
        output.action_logits.transpose(1, 2),
        output.action_targets,
        ignore_index=-1,
        reduction="none",
    ).sum(dim=1)
    # Eq. (3)/(4) joint likelihood: no parser/inside-outside loss and no 0.5
    # training coefficient.  Each candidate is an observed (x, y) pair.
    return (token_loss + action_loss).to(dtype=torch.float64).cpu()


@torch.no_grad()
def _score_native_candidates(
    model: torch.nn.Module,
    candidates: Sequence[Sequence[GoldSegment]],
    device: torch.device,
    prefix_tokens: int,
    current_tokens: int,
    prefix_actions: int,
    current_actions: int,
) -> torch.Tensor:
    """Score a native candidate batch without the generic Python collator."""
    batch = _move_batch(_fast_gold_batch(candidates), device)
    output = model(
        **batch,
        force_gold_tree=True,
        score_token_range=(prefix_tokens, prefix_tokens + current_tokens),
        score_action_range=(prefix_actions, prefix_actions + current_actions),
    )
    if output.logits is None or output.action_logits is None:
        raise RuntimeError("GPST generative token/action heads are required for perplexity")
    token_loss = F.cross_entropy(output.logits.transpose(1, 2), output.token_targets,
                                 ignore_index=-100, reduction="none").sum(dim=1)
    action_loss = F.cross_entropy(output.action_logits.transpose(1, 2), output.action_targets,
                                  ignore_index=-1, reduction="none").sum(dim=1)
    return (token_loss + action_loss).to(dtype=torch.float64).cpu()


@dataclass(frozen=True)
class GoldTreePPLResult:
    perplexity: float
    log_likelihood: float
    terminal_count: int
    sentence_count: int
    document_count: int
    samples_per_sentence: int
    normalized_mixture: bool
    deduplicated_trees: bool
    candidate_slots: int
    model_candidate_forwards: int

    def as_dict(self) -> dict:
        result = dict(self.__dict__)
        result["candidate_compression_ratio"] = (
            self.candidate_slots / self.model_candidate_forwards
            if self.model_candidate_forwards else math.nan
        )
        return result


def evaluate_gold_tree_document_ppl(
    model: torch.nn.Module,
    corpus: GoldTree300Corpus | NativeGPSTTopKCorpus,
    device: torch.device | str,
    eval_batch_size: int = 4,
    max_action_nodes: int = 2048,
    max_terminals: int = 2048,
    max_batch_actions: int = 65536,
    normalize_mixture: bool = False,
    deduplicate_trees: bool = False,
    progress: Optional[Callable[[int, int, int], None]] = None,
) -> GoldTreePPLResult:
    """Evaluate document PPL from fixed gold trees, without beam search.

    ``normalize_mixture=False`` reproduces the existing OLMo metric exactly:
    ``logsumexp(log p(x,y_k))``.  Set it to true for the conventional uniform
    mixture, which subtracts ``log(K)`` per sentence.
    """
    if eval_batch_size <= 0 or max_batch_actions <= 0:
        raise ValueError("batch limits must be positive")
    device = torch.device(device)
    model.eval()
    collator = GoldTreeCollator()
    prefix: Tuple[GoldSegment, ...] = ()
    previous_doc: Optional[int] = None
    seen_documents = 0
    log_likelihood = 0.0
    terminal_count = 0
    candidate_slots = 0
    model_candidate_forwards = 0

    for sentence_index, (doc_id, original_candidates) in enumerate(corpus):
        if doc_id != previous_doc:
            prefix = ()
            previous_doc = doc_id
            seen_documents += 1
        # Native v2 GPST rows are already unique strict-binary CKY trees.  Do
        # not hash hundreds of long Python signatures at every sentence.
        if isinstance(corpus, NativeGPSTTopKCorpus):
            candidates = original_candidates
            multiplicities = torch.ones(len(candidates), dtype=torch.float64)
        else:
            candidates, multiplicities = _compress_candidates(original_candidates)
        candidate_slots += len(original_candidates)
        model_candidate_forwards += len(candidates)
        if deduplicate_trees:
            # Historical diagnostic mode: treat each distinct unlabeled tree as
            # one mixture component.  Normal evaluation retains all original
            # slots through their multiplicities while still forwarding each
            # distinct tree only once.
            multiplicities.fill_(1)
        current = candidates[0]
        current_tokens = _count_tokens(current)
        current_actions = _count_actions(current)
        context = _trim_prefix(
            prefix, current, max_action_nodes=max_action_nodes, max_terminals=max_terminals
        )
        prefix_tokens = _count_tokens(context)
        prefix_actions = _count_actions(context)
        nll_parts: List[torch.Tensor] = []
        batch_size = min(eval_batch_size, max(1, max_batch_actions // max(
            prefix_actions + current_actions, 1
        )))
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            full_candidates = [context + tuple(candidate) for candidate in chunk]
            if isinstance(corpus, NativeGPSTTopKCorpus):
                nll_parts.append(_score_native_candidates(
                    model, full_candidates, device, prefix_tokens, current_tokens,
                    prefix_actions, current_actions,
                ))
            else:
                items = [_as_collator_item(candidate) for candidate in full_candidates]
                nll_parts.append(_score_items(
                    model, items, collator, device,
                    prefix_tokens, current_tokens, prefix_actions, current_actions,
                ))
        nll = torch.cat(nll_parts)
        sentence_log_likelihood = aggregate_candidate_nll(
            nll, normalize_mixture, multiplicities=multiplicities
        )
        log_likelihood += sentence_log_likelihood
        terminal_count += current_tokens
        # OLMo commits candidate 0 to the shared document cache.  Retain only
        # the suffix that can ever enter a later bounded-context forward pass:
        # this is exactly equivalent to keeping the whole document history,
        # but avoids unbounded host-memory growth on very long documents.
        prefix = _retain_prefix_for_any_future_sentence(
            prefix + tuple(original_candidates[0]),
            max_action_nodes=max_action_nodes,
            max_terminals=max_terminals,
        )
        if progress is not None:
            progress(sentence_index + 1, len(corpus), doc_id)

    perplexity = math.exp(-log_likelihood / terminal_count) if terminal_count else math.nan
    return GoldTreePPLResult(
        perplexity=perplexity,
        log_likelihood=log_likelihood,
        terminal_count=terminal_count,
        sentence_count=len(corpus),
        document_count=seen_documents,
        samples_per_sentence=corpus.samples_per_sentence,
        normalized_mixture=normalize_mixture,
        deduplicated_trees=deduplicate_trees,
        candidate_slots=candidate_slots,
        model_candidate_forwards=model_candidate_forwards,
    )
