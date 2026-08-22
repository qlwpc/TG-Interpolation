"""Gold-tree document perplexity for terminal-only Pushdown OLMo models.

The evaluator intentionally does full-prefix teacher forcing.  It is the
correctness reference for a future Pushdown KV-cache implementation and never
calls :meth:`OLMo.pushdown_beam_search`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from olmo.attachment import derive_gold_attachment_actions
from olmo.data.parse_align import TreeVocab, parse_chunk_slice
from olmo.model import OLMo


@dataclass(frozen=True)
class PushdownGoldCandidate:
    tokens: Tuple[int, ...]
    spans: Tuple[Tuple[int, int, int], ...]
    sentence_ids: Tuple[int, ...]
    attachment_targets: Tuple[int, ...]
    legal_attachment_targets: Tuple[Tuple[int, ...], ...]


def _candidate_from_block(block: Sequence[int], vocab: TreeVocab) -> PushdownGoldCandidate:
    parsed = parse_chunk_slice(
        block, vocab, direction="right", binarize=True, collapse_unary=True,
        drop_singleton_spans=True,
    )
    tokens = tuple(map(int, parsed["input_ids"].tolist()))
    spans = tuple(tuple(map(int, x)) for x in parsed["spans"].tolist())
    sentence_ids = tuple(map(int, parsed["sentence_ids"].tolist()))
    span_tensor = torch.tensor(spans or [(-1, -1, -1)], dtype=torch.long).unsqueeze(0)
    sid_tensor = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0)
    targets, legal = derive_gold_attachment_actions(span_tensor, sid_tensor)
    return PushdownGoldCandidate(
        tokens=tokens, spans=spans, sentence_ids=sentence_ids,
        attachment_targets=tuple(map(int, targets[0].tolist())),
        legal_attachment_targets=tuple(tuple(x) for x in legal[0]),
    )


def _drop_leading_bos(candidate: PushdownGoldCandidate, bos: int) -> PushdownGoldCandidate:
    """Remove a per-record BOS when appending a later sentence to a document."""
    if not candidate.tokens or candidate.tokens[0] != bos:
        return candidate
    def shift_span(span: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return tuple(x - 1 for x in span)  # type: ignore[return-value]
    spans = tuple(shift_span(span) for span in candidate.spans if span[0] > 0)
    targets = tuple(-1 if target < 0 else target - 1 for target in candidate.attachment_targets[1:])
    legal = tuple(tuple(key - 1 for key in keys) for keys in candidate.legal_attachment_targets[1:])
    return PushdownGoldCandidate(candidate.tokens[1:], spans, candidate.sentence_ids[1:], targets, legal)


class PushdownGold300Corpus:
    """Memory-mapped ``tree_300`` records organized as sentences/documents."""
    def __init__(
        self, tree_path: str, sentence_index_path: str, document_index_path: str,
        tokenizer_path: str, samples_per_sentence: int = 300,
        max_sentences: Optional[int] = None,
    ) -> None:
        self.tree = np.load(tree_path, mmap_mode="r")
        lengths = np.load(sentence_index_path, mmap_mode="r")
        self.vocab = TreeVocab.from_tokenizer_file(tokenizer_path)
        self.samples_per_sentence = int(samples_per_sentence)
        if self.samples_per_sentence <= 0 or len(lengths) % self.samples_per_sentence:
            raise ValueError("tree record count must be divisible by samples_per_sentence")
        total = len(lengths) // self.samples_per_sentence
        self.num_sentences = min(total, max_sentences) if max_sentences is not None else total
        used = self.num_sentences * self.samples_per_sentence
        self.offsets = np.empty(used + 1, dtype=np.uint64)
        self.offsets[0] = 0
        np.cumsum(lengths[:used], dtype=np.uint64, out=self.offsets[1:])
        self.document_ends = np.cumsum(np.asarray(np.load(document_index_path, mmap_mode="r"), dtype=np.int64))
        if not len(self.document_ends) or self.document_ends[-1] < self.num_sentences:
            raise ValueError("document_index does not cover requested sentences")

    def __len__(self) -> int:
        return self.num_sentences

    def document_id(self, sentence_index: int) -> int:
        return int(np.searchsorted(self.document_ends, sentence_index, side="right"))

    def sentence_candidates(self, sentence_index: int) -> Tuple[PushdownGoldCandidate, ...]:
        if not 0 <= sentence_index < len(self):
            raise IndexError(sentence_index)
        first = sentence_index * self.samples_per_sentence
        candidates = []
        for i in range(first, first + self.samples_per_sentence):
            block = self.tree[int(self.offsets[i]):int(self.offsets[i + 1])].astype(np.int64, copy=False)
            candidates.append(_candidate_from_block(block, self.vocab))
        reference = candidates[0].tokens
        if any(candidate.tokens != reference for candidate in candidates[1:]):
            raise ValueError(f"sentence {sentence_index} gold candidates have different terminals")
        return tuple(candidates)

    def __iter__(self) -> Iterator[Tuple[int, Tuple[PushdownGoldCandidate, ...]]]:
        for i in range(len(self)):
            yield self.document_id(i), self.sentence_candidates(i)


@dataclass(frozen=True)
class PushdownCandidateScores:
    joint_nll: torch.Tensor
    token_nll: torch.Tensor
    attachment_nll: torch.Tensor


def _signature(c: PushdownGoldCandidate) -> Tuple[Tuple[int, int, int], ...]:
    return c.spans


def _compress_candidates(
    candidates: Sequence[PushdownGoldCandidate],
) -> Tuple[Tuple[PushdownGoldCandidate, ...], torch.Tensor]:
    """Return stable unique structures and their original slot counts."""
    unique: List[PushdownGoldCandidate] = []
    counts: List[int] = []
    positions = {}
    for candidate in candidates:
        signature = _signature(candidate)
        position = positions.get(signature)
        if position is None:
            positions[signature] = len(unique)
            unique.append(candidate)
            counts.append(1)
        else:
            counts[position] += 1
    return tuple(unique), torch.tensor(counts, dtype=torch.float64)


def _weighted_logsumexp(nll: torch.Tensor, multiplicities: torch.Tensor) -> torch.Tensor:
    if nll.ndim != 1 or nll.shape != multiplicities.shape or nll.numel() == 0:
        raise ValueError("NLL and multiplicities must be non-empty vectors of equal shape")
    if bool((multiplicities <= 0).any()):
        raise ValueError("multiplicities must be positive")
    return torch.logsumexp(
        -nll.to(torch.float64) + multiplicities.to(nll.device, torch.float64).log(), 0
    )


def _trim_prefix(prefix: Sequence[PushdownGoldCandidate], current: PushdownGoldCandidate, max_length: int) -> Tuple[PushdownGoldCandidate, ...]:
    if len(current.tokens) > max_length:
        raise ValueError(f"one sentence has {len(current.tokens)} tokens, exceeding max_sequence_length={max_length}")
    total = len(current.tokens)
    kept: List[PushdownGoldCandidate] = []
    for sentence in reversed(prefix):
        if total + len(sentence.tokens) > max_length:
            break
        kept.append(sentence)
        total += len(sentence.tokens)
    return tuple(reversed(kept))


def _compose(prefix: Sequence[PushdownGoldCandidate], current: PushdownGoldCandidate) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]], int]:
    """Concatenate complete sentences, remapping span and sentence coordinates."""
    all_sentences = tuple(prefix) + (current,)
    token_offset = 0
    sentence_offset = 0
    tokens: List[int] = []; spans: List[Tuple[int, int, int]] = []; sids: List[int] = []
    targets: List[int] = []; legal: List[List[int]] = []
    for sentence in all_sentences:
        tokens.extend(sentence.tokens)
        spans.extend(tuple(x + token_offset for x in span) for span in sentence.spans)
        for sid in sentence.sentence_ids:
            sids.append(-1 if sid < 0 else sid + sentence_offset)
        targets.extend(-1 if target < 0 else target + token_offset for target in sentence.attachment_targets)
        legal.extend([key + token_offset for key in keys] for keys in sentence.legal_attachment_targets)
        sentence_offset += max((sid for sid in sentence.sentence_ids if sid >= 0), default=-1) + 1
        token_offset += len(sentence.tokens)
    return (
        torch.tensor(tokens, dtype=torch.long), torch.tensor(spans or [(-1, -1, -1)], dtype=torch.long),
        torch.tensor(sids, dtype=torch.long), torch.tensor(targets, dtype=torch.long), legal,
        token_offset - len(current.tokens),
    )


@torch.no_grad()
def score_pushdown_gold_candidates(
    model: OLMo, prefix: Sequence[PushdownGoldCandidate], candidates: Sequence[PushdownGoldCandidate],
    device: torch.device | str, eval_batch_size: int = 4, include_attachment_probability: bool = True,
) -> PushdownCandidateScores:
    if not candidates:
        raise ValueError("candidates cannot be empty")
    device = torch.device(device)
    token_losses: List[torch.Tensor] = []; attachment_losses: List[torch.Tensor] = []
    for start in range(0, len(candidates), eval_batch_size):
        packed = [_compose(prefix, candidate) for candidate in candidates[start:start + eval_batch_size]]
        total_length = int(packed[0][0].numel()); prefix_length = packed[0][-1]
        if any(int(row[0].numel()) != total_length or row[-1] != prefix_length for row in packed):
            raise ValueError("all candidate token sequences must have the same length")
        max_spans = max(row[1].shape[0] for row in packed)
        input_ids = torch.stack([row[0] for row in packed]).to(device)
        sentence_ids = torch.stack([row[2] for row in packed]).to(device)
        tree_spans = torch.full((len(packed), max_spans, 3), -1, dtype=torch.long, device=device)
        for b, row in enumerate(packed): tree_spans[b, :row[1].shape[0]] = row[1].to(device)
        target_start = max(prefix_length, 1); target_end = total_length
        logit_range = (target_start - 1, target_end - 1)
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                    tree_spans=tree_spans, pushdown_sentence_ids=sentence_ids,
                    compute_attachment_logits=include_attachment_probability,
                    logits_range=logit_range, attachment_query_range=(prefix_length, total_length))
        labels = input_ids[:, target_start:target_end]
        token_losses.append(F.cross_entropy(out.logits.float().transpose(1, 2), labels, reduction="none").sum(dim=1).to(torch.float64).cpu())
        if not include_attachment_probability:
            attachment_losses.append(torch.zeros(len(packed), dtype=torch.float64))
            continue
        if out.attachment_logits is None:
            raise RuntimeError("joint Pushdown PPL requires attachment logits")
        q = total_length - prefix_length
        legal_mask = torch.zeros((len(packed), q, total_length), dtype=torch.bool, device=device)
        targets = torch.full((len(packed), q), -100, dtype=torch.long, device=device)
        for b, row in enumerate(packed):
            for global_q in range(prefix_length, total_length):
                keys = row[4][global_q]
                if keys:
                    legal_mask[b, global_q - prefix_length, torch.tensor(keys, device=device)] = True
                    target = int(row[3][global_q])
                    if target not in keys: raise ValueError("gold attachment target is outside its legal action set")
                    targets[b, global_q - prefix_length] = target
        masked = out.attachment_logits.float().masked_fill(~legal_mask, float("-inf"))
        attachment_losses.append(F.cross_entropy(masked.transpose(1, 2), targets, ignore_index=-100, reduction="none").sum(dim=1).to(torch.float64).cpu())
    token_nll = torch.cat(token_losses); attachment_nll = torch.cat(attachment_losses)
    return PushdownCandidateScores(token_nll + attachment_nll, token_nll, attachment_nll)


@dataclass(frozen=True)
class PushdownDocumentPPLResult:
    legacy_perplexity: float; uniform_mixture_perplexity: float; token_only_perplexity: float
    legacy_log_likelihood: float; uniform_mixture_log_likelihood: float
    terminal_count: int; sentence_count: int; document_count: int; samples_per_sentence: int
    deduplicated_trees: bool; beam_search: bool = False
    candidate_slots: int = 0; model_candidate_forwards: int = 0
    def as_dict(self) -> dict:
        result = dict(self.__dict__)
        result["candidate_compression_ratio"] = (
            self.candidate_slots / self.model_candidate_forwards
            if self.model_candidate_forwards else math.nan
        )
        return result


def evaluate_pushdown_document_ppl(
    model: OLMo, corpus: PushdownGold300Corpus, device: torch.device | str, eval_batch_size: int = 4,
    max_sequence_length: int = 2048, deduplicate_trees: bool = False,
    include_attachment_probability: bool = True, progress: Optional[Callable[[int, int, int], None]] = None,
) -> PushdownDocumentPPLResult:
    if eval_batch_size <= 0: raise ValueError("eval_batch_size must be positive")
    if include_attachment_probability and not hasattr(model, "pushdown_attachment_head"):
        raise RuntimeError("joint Pushdown PPL requires a checkpoint with attachment-head weights; use token-only explicitly")
    model.eval(); prefix: Tuple[PushdownGoldCandidate, ...] = (); previous_doc: Optional[int] = None
    legacy_ll = mixture_ll = token_ll = 0.0; terminals = documents = 0
    candidate_slots = model_candidate_forwards = 0
    for index, (doc_id, original) in enumerate(corpus):
        first = doc_id != previous_doc
        if first:
            prefix = (); previous_doc = doc_id; documents += 1
        candidates = original if first else tuple(_drop_leading_bos(c, corpus.vocab.bos) for c in original)
        scored, multiplicities = _compress_candidates(candidates)
        candidate_slots += len(candidates)
        model_candidate_forwards += len(scored)
        if deduplicate_trees:
            # Diagnostic semantics: distinct structures receive equal mass.
            # Otherwise counts restore the exact original 300-slot mixture.
            multiplicities.fill_(1)
        current = candidates[0]; context = _trim_prefix(prefix, current, max_sequence_length)
        scores = score_pushdown_gold_candidates(model, context, scored, device, eval_batch_size, include_attachment_probability)
        joint_ll = _weighted_logsumexp(scores.joint_nll, multiplicities)
        token_sentence_ll = _weighted_logsumexp(scores.token_nll, multiplicities)
        legacy_ll += joint_ll.item()
        mixture_ll += (joint_ll - math.log(int(multiplicities.sum().item()))).item()
        token_ll += token_sentence_ll.item()
        terminals += len(current.tokens) - (1 if first and current.tokens and current.tokens[0] == corpus.vocab.bos else 0)
        prefix = prefix + (current,)
        if progress: progress(index + 1, len(corpus), doc_id)
    def ppl(ll: float) -> float: return math.exp(-ll / terminals) if terminals else math.nan
    return PushdownDocumentPPLResult(
        ppl(legacy_ll), ppl(mixture_ll), ppl(token_ll), legacy_ll, mixture_ll,
        terminals, len(corpus), documents, corpus.samples_per_sentence,
        deduplicate_trees, False, candidate_slots, model_candidate_forwards,
    )
