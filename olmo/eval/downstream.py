import abc
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Union, Callable, Set

import datasets
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torchmetrics import Metric
import numpy as np
import json, os
import evaluate

from olmo.util import load_hf_dataset, load_oe_eval_requests, get_global_rank

from ..data.tg_mask import TG_attention_bias, SentencepieceVocab
from ..tokenizer import Tokenizer
from ..data.util import (
    encode_TG_string,
    convert_TG_format,
    pause_input_ids,
    pause_spec_from_grammar_type,
    pause_expanded_len,
    pause_trailing_trim,
)
from ..data.collator import DataCollator
from ..data.parse_align import TreeVocab, parse_chunk_slice
from ..config import PaddingDirection

log = logging.getLogger(__name__)

# ``setup_logging()`` filters INFO-and-below records from non-zero ranks by
# default. XSum predictions are experiment output rather than routine progress
# logging, so give them a dedicated level just above INFO. This preserves the
# normal rank-0-only log policy while ensuring every rank's generations reach
# stdout (and includes the global rank for post-hoc parsing).
XSUM_PREDICTION_LOG_LEVEL = logging.INFO + 1
logging.addLevelName(XSUM_PREDICTION_LOG_LEVEL, "XSUM_PREDICTION")
SG_SCORE_LOG_LEVEL = logging.INFO + 2
logging.addLevelName(SG_SCORE_LOG_LEVEL, "SG_SCORE")

# Map from oe-eval metrics to metrics used here
METRIC_FROM_OE_EVAL = {"acc_raw": "acc", "acc_per_char": "len_norm", "acc_uncond": "pmi_dc"}
LOG_2_OF_E = 1.44269504089


def _parse_treereg_tree_tokens(
    tree_tokens: Sequence[int], tree_vocab: TreeVocab,
) -> Dict[str, List[Any]]:
    """Use the pretraining/eval parser to build the complete TreeReg contract."""
    parsed = parse_chunk_slice(
        tree_tokens, tree_vocab, direction="right", binarize=True,
        collapse_unary=True,
    )
    return {
        "input_ids": [int(x) for x in parsed["input_ids"].tolist()],
        "tree_spans": [tuple(map(int, span)) for span in parsed["spans"].tolist()],
        "treereg_word_boundaries": [bool(x) for x in parsed["word_boundaries"].tolist()],
        "treereg_sentence_ids": [int(x) for x in parsed["sentence_ids"].tolist()],
    }


def _parse_pushdown_tree_tokens(
    tree_tokens: Sequence[int], tree_vocab: TreeVocab, direction: str,
) -> Dict[str, List[Any]]:
    """Convert parsed TG tokens to Pushdown's terminal/unary-span contract.

    Pushdown never receives bracket tokens as language-model inputs.  Instead,
    it consumes the terminal stream together with the unary-collapsed,
    binarized constituent spans used to build its stale stack tape.  Singleton
    preterminal spans are SHIFT operations rather than REDUCE operations and
    are therefore intentionally omitted.
    """
    parsed = parse_chunk_slice(
        tree_tokens,
        tree_vocab,
        direction=direction,
        binarize=True,
        collapse_unary=True,
        drop_singleton_spans=True,
    )
    return {
        "input_ids": [int(x) for x in parsed["input_ids"].tolist()],
        "tree_spans": [tuple(map(int, span)) for span in parsed["spans"].tolist()],
        "pushdown_sentence_ids": [int(x) for x in parsed["sentence_ids"].tolist()],
    }


def _join_pushdown_parts(parts: Sequence[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
    """Concatenate parsed Pushdown fragments without crossing tree boundaries."""
    out: Dict[str, List[Any]] = {
        "input_ids": [], "tree_spans": [], "pushdown_sentence_ids": [],
    }
    token_offset = sentence_offset = 0
    for part in parts:
        out["input_ids"].extend(part["input_ids"])
        out["tree_spans"].extend(
            (left + token_offset, split + token_offset, right + token_offset)
            for left, split, right in part["tree_spans"]
        )
        sentence_ids = part["pushdown_sentence_ids"]
        out["pushdown_sentence_ids"].extend(
            sentence_id + sentence_offset if sentence_id >= 0 else -1
            for sentence_id in sentence_ids
        )
        token_offset += len(part["input_ids"])
        if any(sentence_id >= 0 for sentence_id in sentence_ids):
            sentence_offset += max(
                sentence_id for sentence_id in sentence_ids if sentence_id >= 0
            ) + 1
    return out


def _slice_pushdown_parse(
    parsed: Dict[str, List[Any]], start: int, end: int,
) -> Dict[str, List[Any]]:
    """Slice terminal coordinates and invalidate trees cut by a context window."""
    start, end = max(0, int(start)), max(0, int(end))
    all_sentence_ids = parsed["pushdown_sentence_ids"]
    kept_sentence_ids = all_sentence_ids[start:end]
    cut_ids = {
        sentence_id for sentence_id in kept_sentence_ids if sentence_id >= 0
        and any(
            other == sentence_id
            for other in all_sentence_ids[:start] + all_sentence_ids[end:]
        )
    }
    return {
        "input_ids": parsed["input_ids"][start:end],
        "tree_spans": [
            (left - start, split - start, right - start)
            for left, split, right in parsed["tree_spans"]
            if start <= left and right < end and all_sentence_ids[left] not in cut_ids
        ],
        "pushdown_sentence_ids": [
            -1 if sentence_id in cut_ids else sentence_id
            for sentence_id in kept_sentence_ids
        ],
    }


def _slice_treereg_parse(
    parsed: Dict[str, List[Any]], start: int, end: int,
) -> Dict[str, List[Any]]:
    """Slice terminal coordinates and exclude top-level trees cut by the window."""
    start, end = max(0, int(start)), max(0, int(end))
    all_sentence_ids = parsed["treereg_sentence_ids"]
    kept_sentence_ids = all_sentence_ids[start:end]
    cut_ids = {
        sid for sid in kept_sentence_ids if sid >= 0
        and any(other == sid for other in all_sentence_ids[:start] + all_sentence_ids[end:])
    }
    return {
        "input_ids": parsed["input_ids"][start:end],
        "tree_spans": [
            (left - start, split - start, right - start)
            for left, split, right in parsed["tree_spans"]
            if start <= left and right < end and all_sentence_ids[left] not in cut_ids
        ],
        "treereg_word_boundaries": parsed["treereg_word_boundaries"][start:end],
        "treereg_sentence_ids": [
            -1 if sid in cut_ids else sid for sid in kept_sentence_ids
        ],
    }


def _join_treereg_parts(parts: Sequence[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
    """Concatenate parsed tree streams with globally unique sentence ids."""
    out: Dict[str, List[Any]] = {
        "input_ids": [], "tree_spans": [], "treereg_word_boundaries": [],
        "treereg_sentence_ids": [],
    }
    offset = sentence_offset = 0
    for part in parts:
        out["input_ids"].extend(part["input_ids"])
        out["treereg_word_boundaries"].extend(part["treereg_word_boundaries"])
        out["tree_spans"].extend(
            (left + offset, split + offset, right + offset)
            for left, split, right in part["tree_spans"]
        )
        ids = part["treereg_sentence_ids"]
        out["treereg_sentence_ids"].extend(
            sid + sentence_offset if sid >= 0 else -1 for sid in ids
        )
        offset += len(part["input_ids"])
        if any(sid >= 0 for sid in ids):
            sentence_offset += max(sid for sid in ids if sid >= 0) + 1
    return out


def _world_size() -> int:
    """Number of distributed ranks (1 if distributed is unavailable/uninitialized)."""
    import torch.distributed as _dist
    if _dist.is_available() and _dist.is_initialized():
        return _dist.get_world_size()
    return 1


def _all_reduce_tensor(t: torch.Tensor) -> torch.Tensor:
    """SUM all-reduce a fixed-size tensor metric state across ranks.

    For metrics whose state is a pre-allocated tensor scattered into *disjoint*
    slots by ``sent_id``/position (``BLiMPMetric``, ``TGPerplexityDocumentLevelMetric``).
    Each rank writes only its own partition; unwritten slots stay 0, so a SUM
    all-reduce reconstructs the global tensor regardless of how many ``update()``
    calls each rank made. It is also used for fixed-size additive sufficient
    statistics, such as SG's per-suite correct and sample counts.

    This is the count-insensitive replacement for torchmetrics'
    ``sync_on_compute=True`` + ``dist_reduce_fx="sum"``, which deadlocks when
    ranks call ``update()`` a different number of times (the case under
    ``DistributedEvalSampler`` when the dataset size isn't divisible by the world
    size — counts differ by at most one).
    """
    if _world_size() > 1:
        import torch.distributed as _dist
        t = t.clone()  # don't mutate the rank-local view in place
        _dist.all_reduce(t, op=_dist.ReduceOp.SUM)
    return t


def _gather_list(lst: list) -> list:
    """All-gather + concatenate a list metric state across ranks.

    For metrics whose state is a list appended per ``update()`` (``ICLMetric``,
    ``BeamSearchICLMetric``, ``DecomposedICLMetric``, ``RougeMetric``,
    ``TGPerplexitySentenceLevelMetric``).

    List gather is **count-insensitive**: each rank contributes however many
    items it appended, and the concatenation is the global list regardless of
    per-rank update counts. This replaces torchmetrics' ``sync_on_compute=True``
    gather, which requires equal update counts per rank and deadlocks otherwise.
    """
    if _world_size() <= 1:
        return lst
    import torch.distributed as _dist
    world = _dist.get_world_size()
    gathered = [None] * world
    _dist.all_gather_object(gathered, [lst])
    out = []
    for rank_payload in gathered:
        # rank_payload is the [lst] wrapper we sent; None/empty means that rank
        # sent nothing (no updates this eval).
        if rank_payload:
            out.extend(rank_payload[0])
    return out


class ICLMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(self, metric_type="acc", vocab_path=None, tree_eval_type=None,
                 doc_group: List[str]=None, tokenizer=None,
                 save_per_example_path: Optional[str] = None) -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb

        If ``save_per_example_path`` is set, ``compute()`` also writes a JSON file
        with per-instance ``(doc_id, pred, label, correct, scores)`` for post-hoc
        bootstrap significance testing. Only rank 0 writes (``compute()`` runs
        after a full cross-rank state gather due to ``sync_on_compute=True``).
        """
        # sync_on_compute=False: we gather list state explicitly in compute()
        # via _gather_list (count-insensitive). torchmetrics' built-in sync
        # deadlocks when ranks call update() unequal times (DistributedEvalSampler
        # with N not divisible by world size).
        super().__init__(sync_on_compute=False)

        self.metric_type = metric_type
        self.tree_eval_type = tree_eval_type
        self.doc_group = doc_group
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.tokenizer = tokenizer
        self.save_per_example_path = save_per_example_path
        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)
        self.add_state("loglikelihoods_term", default=[], dist_reduce_fx=None)

    def reset(self):
        self.loglikelihoods = []
        self.labels = []
        self.loglikelihoods_term = []

    def update(self, batch: Dict[str, Any], lm_logits: torch.Tensor, dc_lm_logits=None):
        lm_logits = F.log_softmax(lm_logits, dim=-1, dtype=torch.float32)
        if self.metric_type == "pmi_dc":
            assert dc_lm_logits is not None, "PMI_DC acc type selected but no domain conditional logits provided"

        for idx, (doc_id, cont_id) in enumerate(zip(batch["doc_id"], batch["cont_id"])):
            # [cont_len]: continuation is padded for batching
            cont_tokens = batch["continuation"][idx][: batch["cont_len"][idx]].unsqueeze(-1)
            # get logits from LM for the continuation: [cont_len, vocab]
            # batch['input_ids'][idx] -> ctx + cont + padding
            # -1 in both indices: lm_logits will be left shifted 1 pos as 0th pos in input generates next token in the 0th pos of lm_logits
            lm_cont_logits = lm_logits[idx][
                batch["ctx_len"][idx] - 1 : batch["ctx_len"][idx] + batch["cont_len"][idx] - 1
            ]
            lm_log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens).squeeze()
            cont_mask = None
            if "cont_mask" in batch:
                cont_mask = batch["cont_mask"][idx][: batch["cont_len"][idx]]

            # Always compute terminal mask for decomposition
            cont_tokens_seq = cont_tokens.squeeze()
            term_mask = (self.vocab.opening_non_terminals[0] > cont_tokens_seq) | (cont_tokens_seq > self.vocab.closing_non_terminals[1])
            # term_mask is bool; coerce masks to float so downstream arithmetic
            # (lm_log_likelihood * mask, cont_mask *= term_mask) never depends on
            # implicit bool→float promotion (fragile if a mask arrives as int/bool
            # from the collator or get_mask).
            term_mask = term_mask.to(torch.float32)

            # Compute terminal-only log-likelihood BEFORE any mask is applied
            if cont_mask is not None:
                lm_log_likelihood_term = (lm_log_likelihood * cont_mask.float()).sum()
            else:
                lm_log_likelihood_term = (lm_log_likelihood).sum()

            if self.tree_eval_type == "terminal":
                if cont_mask is None:
                    cont_mask = term_mask
                else:
                    cont_mask = cont_mask.float() * term_mask

            if cont_mask is not None:
                lm_log_likelihood *= cont_mask

            log_likelihood: torch.Tensor
            if self.metric_type == "pmi_dc":
                assert dc_lm_logits is not None
                # get domain conditional continuation logits: [cont_len, vocab]
                dc_lm_cont_logits = dc_lm_logits[idx][
                    batch["dc_len"][idx] - 1 : batch["dc_len"][idx] + batch["cont_len"][idx] - 1
                ]

                # gather log-probs at continuation token indices but divide by domain conditional prob
                log_likelihood = (
                    lm_log_likelihood.sum()
                    / torch.gather(dc_lm_cont_logits, 1, cont_tokens).sum()
                )
            elif self.metric_type == "acc" or self.metric_type == "f1":
                # gather log-probs at continuation token indices
                log_likelihood = lm_log_likelihood.sum()
            elif self.metric_type == "len_norm" or self.metric_type == "ce_loss":
                log_likelihood = (
                    lm_log_likelihood.sum() / batch["cont_str_len"][idx]
                )
                if self.metric_type == "ce_loss":
                    log_likelihood = -log_likelihood
            elif self.metric_type == "bpb":
                # bits per byte
                log_likelihood = (
                    -lm_log_likelihood.sum()
                    / batch["cont_byte_len"][idx]
                    * LOG_2_OF_E
                )
            else:
                raise ValueError(self.metric_type)

            # Store terminal-only score.
            # IMPORTANT: normalize by terminal-only string/byte lengths,
            # NOT the full string lengths (which include non-terminals).
            if hasattr(self, 'loglikelihoods_term'):
                if self.metric_type in ("len_norm", "ce_loss", "bpb"):
                    # Decode terminal-only continuation to get correct lengths
                    term_cont_ids = cont_tokens_seq[term_mask].tolist()
                    if self.tokenizer is not None and len(term_cont_ids) > 0:
                        term_cont_str = self.tokenizer.decode(term_cont_ids)
                        cont_str_len_term = len(term_cont_str) - 1  # leading blank
                        cont_byte_len_term = len(term_cont_str[1:].encode("utf-8"))
                    else:
                        # Fallback: use full lengths (old behavior)
                        cont_str_len_term = batch["cont_str_len"][idx]
                        cont_byte_len_term = batch["cont_byte_len"][idx]

                    if self.metric_type in ("len_norm", "ce_loss"):
                        div = cont_str_len_term if cont_str_len_term > 0 else 1
                        term_log_likelihood = lm_log_likelihood_term / div
                        if self.metric_type == "ce_loss":
                            term_log_likelihood = -term_log_likelihood
                    else:  # bpb
                        div = cont_byte_len_term if cont_byte_len_term > 0 else 1
                        term_log_likelihood = -lm_log_likelihood_term / div * LOG_2_OF_E
                else:
                    term_log_likelihood = lm_log_likelihood_term

                self.loglikelihoods_term.append(
                    torch.Tensor((doc_id, cont_id, term_log_likelihood)).to(batch["continuation"][idx].device)
                )

            # because metric states cannot be dict/list of tuples, store this tuple as tensor: (doc_id, cont_id, metric_state)
            self.loglikelihoods.append(
                torch.Tensor((doc_id, cont_id, log_likelihood)).to(batch["continuation"][idx].device)
            )
            self.labels.append(
                torch.LongTensor((doc_id, cont_id, batch["label_id"][idx])).to(batch["label_id"][idx].device)
            )

    def compute(self) -> torch.Tensor:
        # Gather per-rank list state across ranks (count-insensitive). Replaces
        # torchmetrics' sync_on_compute=True gather, which deadlocks when ranks
        # update unequal times under DistributedEvalSampler.
        self.loglikelihoods = _gather_list(self.loglikelihoods)
        self.labels = _gather_list(self.labels)
        self.loglikelihoods_term = _gather_list(self.loglikelihoods_term)
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        loglikelihood_dict: Dict[int, Dict[int, float]] = {}
        label_dict : Dict[int, Set[int]] = {}

        # collect labels
        for doc_id, cont_id, label_id in self.labels:
            doc_id = doc_id.item()
            if doc_id not in label_dict:
                label_dict[doc_id] = set()
            label_dict[doc_id].add(label_id.item())

        # collect loglikelihoods
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            if int(doc_id.item()) not in loglikelihood_dict:
                loglikelihood_dict[int(doc_id.item())] = {}

            if int(cont_id.item()) not in loglikelihood_dict[int(doc_id.item())]:
                loglikelihood_dict[int(doc_id.item())][int(cont_id.item())] = loglikelihood

        # compute acc
        correct = {}
        preds: Optional[List[float]] = None
        labels: Optional[List[int]] = None
        if self.metric_type == "f1":
            preds = {}
            labels = {}

        # Per-instance records for post-hoc bootstrap significance testing.
        # Populated only for ranking metrics (acc/f1); dumped from rank 0 below.
        per_example: List[Dict[str, Any]] = []

        for doc_id in loglikelihood_dict:
            # each doc_id might have a different number of continuation
            group = "_" if self.doc_group is None else self.doc_group[doc_id]
            if group not in correct:
                correct[group] = []
            num_continuations = len(loglikelihood_dict[doc_id].keys())
            loglikelihoods = torch.tensor([-float("inf")] * num_continuations)

            skip_document = False
            for cont_id in loglikelihood_dict[doc_id]:
                try:
                    loglikelihoods[cont_id] = loglikelihood_dict[doc_id][cont_id]
                except IndexError:
                    # We didn't process all of the continuations, so skip this document.
                    skip_document = True
                    break

            if skip_document:
                continue
            if self.metric_type in ["ce_loss", "bpb"]:
                correct[group].append(loglikelihoods[0])  # Only one answer is scored
            else:
                # print(f"{doc_id} is {'correct' if torch.argmax(loglikelihoods).item() == label_dict[doc_id] else 'wrong'}")
                pred_id = torch.argmax(loglikelihoods).item()
                is_correct = 1.0 if pred_id in label_dict[doc_id] else 0.0
                correct[group].append(is_correct)
                # Capture per-instance (doc_id, pred, label, correct, scores) for bootstrap.
                label_id = next(iter(label_dict[doc_id]))
                per_example.append({
                    "doc_id": int(doc_id),
                    "pred": int(pred_id),
                    "label": int(label_id),
                    "correct": int(is_correct),
                    "n_choices": int(num_continuations),
                    "score_pred": float(loglikelihoods[pred_id].item()),
                    "score_label": float(loglikelihoods[label_id].item())
                                   if 0 <= label_id < num_continuations else float("-inf"),
                })

            if self.metric_type == "f1":
                assert preds is not None
                assert labels is not None
                if group not in pred:
                    preds[group] = []
                    labels[group] = []
                pred = torch.argmax(loglikelihoods).item()
                preds[group].append(pred)
                labels[group].append(label_dict[doc_id][0] if pred not in label_dict[doc_id] else pred)

        # Dump per-instance predictions for bootstrap significance testing.
        # sync_on_compute=True gathers state across ranks before compute(), so
        # rank 0 sees the full (deduplicated) dataset; only rank 0 writes.
        if self.save_per_example_path is not None and get_global_rank() == 0:
            save_dir = os.path.dirname(self.save_per_example_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            with open(self.save_per_example_path, "w") as f:
                json.dump(per_example, f)
            log.info(f"Saved {len(per_example)} per-instance predictions to {self.save_per_example_path}")

        score_dict = {}
        if self.metric_type == "f1":
            assert preds is not None
            assert labels is not None
            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            for group in labels:
                score_dict[group] = f1_score(labels[group], preds[group], pos_label=0)
        else:
            for group in correct:
                score_dict[group] = sum(correct[group]) / len(correct[group])
        
        all = 0.0
        for group in score_dict:
            all += score_dict[group]
        score_dict["_"] = all / len(score_dict)
        return score_dict


class BeamSearchICLMetric(ICLMetric):
    """ICL metric for beam-search-based evaluation.

    Inherits ``compute()`` and state from :class:`ICLMetric`, but overrides
    ``update()`` to accept a pre-computed log-likelihood scalar instead of
    deriving it from batch + raw logits.

    Supports all metric types: ``acc``, ``f1``, ``len_norm``, ``ce_loss``, ``bpb``.
    """

    def __init__(
        self,
        metric_type: str = "acc",
        doc_group: Optional[List[str]] = None,
    ) -> None:
        # sync_on_compute=False: ICLMetric.compute() gathers list state via
        # _gather_list (count-insensitive), avoiding the unequal-count deadlock.
        Metric.__init__(self, sync_on_compute=False)
        self.metric_type = metric_type
        self.tree_eval_type = None
        self.doc_group = doc_group
        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def update(
        self,
        doc_id: int,
        cont_id: int,
        log_likelihood: float,
        label_id: int,
        cont_str_len: Optional[int] = None,
        cont_byte_len: Optional[int] = None,
    ) -> None:
        # Apply same normalizations as ICLMetric.update()
        if self.metric_type == "len_norm" and cont_str_len is not None:
            log_likelihood = log_likelihood / cont_str_len
        elif self.metric_type == "ce_loss" and cont_str_len is not None:
            log_likelihood = -log_likelihood / cont_str_len
        elif self.metric_type == "bpb" and cont_byte_len is not None:
            log_likelihood = -log_likelihood / cont_byte_len * LOG_2_OF_E

        self.loglikelihoods.append(
            torch.Tensor((doc_id, cont_id, log_likelihood))
        )
        self.labels.append(
            torch.LongTensor((doc_id, cont_id, label_id))
        )


class DecomposedICLMetric(ICLMetric):
    """ICLMetric that records both full and terminal-only scores per choice.

    Used to quantify non-terminal noise in MC evaluation.
    ``compute()`` returns a dict with ``_`` (full accuracy), ``_term``
    (terminal-only accuracy), and flip statistics.

    When ``save_per_example_path`` is set, ``compute()`` also writes a JSON
    file with per-example full/term scores for post-hoc analysis
    (e.g. depth-stratified evaluation).
    """

    def __init__(self, metric_type="acc", vocab_path=None, tree_eval_type="default",
                 doc_group=None, tokenizer=None, save_per_example_path=None):
        super().__init__(metric_type=metric_type, vocab_path=vocab_path,
                         tree_eval_type=tree_eval_type, doc_group=doc_group,
                         tokenizer=tokenizer)
        self.save_per_example_path = save_per_example_path

    def compute(self):
        # Temporarily null save_per_example_path so the parent's compute()
        # doesn't write its own per_example dump (ICLMetric schema) to the same
        # path just before _save_per_example (decomp schema) overwrites it — a
        # double-write that wastes work and leaves a mixed-schema file if the
        # process dies between the two writes. We write ours below instead.
        _saved_path = self.save_per_example_path
        self.save_per_example_path = None
        full_result = super().compute()
        self.save_per_example_path = _saved_path

        # Rebuild terminal-only rankings
        loglikelihood_dict_term = {}
        for doc_id, cont_id, loglikelihood in self.loglikelihoods_term:
            d = int(doc_id.item())
            c = int(cont_id.item())
            if d not in loglikelihood_dict_term:
                loglikelihood_dict_term[d] = {}
            loglikelihood_dict_term[d][c] = loglikelihood

        # Rebuild full rankings from parent state
        loglikelihood_dict_full = {}
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            d = int(doc_id.item())
            c = int(cont_id.item())
            if d not in loglikelihood_dict_full:
                loglikelihood_dict_full[d] = {}
            loglikelihood_dict_full[d][c] = loglikelihood

        # Build label dict
        label_dict = {}
        for doc_id, cont_id, label_id in self.labels:
            d = doc_id.item()
            if d not in label_dict:
                label_dict[d] = set()
            label_dict[d].add(label_id.item())

        # Every rank has the same gathered state after ``super().compute()``.
        # Only rank 0 may publish the shared output file.
        if self.save_per_example_path and get_global_rank() == 0:
            self._save_per_example(loglikelihood_dict_full, loglikelihood_dict_term,
                                   label_dict)

        # Compare rankings
        correct_term = 0
        n_flips = 0
        n_flip_to_correct = 0
        n_flip_to_wrong = 0
        total = 0

        for doc_id in loglikelihood_dict_full:
            if doc_id not in loglikelihood_dict_term or doc_id not in label_dict:
                continue
            full_choices = loglikelihood_dict_full[doc_id]
            term_choices = loglikelihood_dict_term[doc_id]
            if len(full_choices) != len(term_choices):
                continue

            full_best = max(full_choices, key=full_choices.get)
            term_best = max(term_choices, key=term_choices.get)
            label = next(iter(label_dict[doc_id]))

            full_correct = (full_best == label)
            term_correct = (term_best == label)
            flipped = (full_best != term_best)

            if term_correct:
                correct_term += 1
            total += 1
            if flipped:
                n_flips += 1
                if term_correct and not full_correct:
                    n_flip_to_correct += 1
                if full_correct and not term_correct:
                    n_flip_to_wrong += 1

        acc_full = full_result.get("_", 0.0)
        acc_term = correct_term / total if total > 0 else 0.0
        flip_rate = n_flips / total if total > 0 else 0.0

        result = {"_": acc_full}
        result["_term"] = acc_term
        result["_flip_rate"] = flip_rate
        result["_flip_to_correct"] = n_flip_to_correct / total if total > 0 else 0.0
        result["_flip_to_wrong"] = n_flip_to_wrong / total if total > 0 else 0.0
        result["_total"] = float(total)
        return result

    def _save_per_example(self, loglikelihood_dict_full, loglikelihood_dict_term,
                          label_dict):
        """Atomically save full/terminal scores from global rank 0 only."""
        # Keep the guard inside the writer as well as at its normal call site so
        # future/direct callers cannot accidentally reintroduce a multi-rank
        # write race.
        if get_global_rank() != 0:
            return

        import json as _json
        per_example = []
        for doc_id in loglikelihood_dict_full:
            if doc_id not in loglikelihood_dict_term or doc_id not in label_dict:
                continue
            full_choices = loglikelihood_dict_full[doc_id]
            term_choices = loglikelihood_dict_term[doc_id]
            label = next(iter(label_dict[doc_id]))
            for cont_id in full_choices:
                per_example.append({
                    "doc_id": doc_id,
                    "cont_id": cont_id,
                    "label": label,
                    "full_score": float(full_choices.get(cont_id, float("-inf"))),
                    "term_score": float(term_choices.get(cont_id, float("-inf"))),
                })
        output_path = os.fspath(self.save_per_example_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # os.replace() is atomic when source and destination are in the same
        # directory, so readers never observe a partially written JSON file.
        tmp_path = f"{output_path}.tmp.rank0.{os.getpid()}"
        try:
            with open(tmp_path, "w") as f:
                _json.dump(per_example, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class ICLMultiChoiceTaskDataset(metaclass=abc.ABCMeta):
    """Only supports zero-shot for now."""

    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="test",
        metric_type=None,  # Override default metric type
        prompts=[None],  # List of prompt variants to use
        local_datasets=True,
        shots_num=0,
        transformer_grammar_type="",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type="default",
        pause_token_id=None,
        parse_binarize_direction: str = "right",
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        # Checkpoint configs for the ordinary terminal model historically use
        # ``null`` here.  Treat it as the empty grammar rather than requiring
        # every direct evaluator caller to special-case it.
        self.transformer_grammar_type = transformer_grammar_type or ""
        if self.ispause:
            # Shrink the real-token budget so the expanded (paused) length still
            # fits model_ctx_len. For spec (p, q), expansion factor is (q+p)/q,
            # so the real-token budget is model_ctx_len * q // (q+p). The slice-
            # based continuation alignment (see sample build) is correct even
            # when model_ctx_len is not divisible by (q+p), so no assert here.
            p, q = self.pause_spec
            self.model_ctx_len = self.model_ctx_len * q // (q + p)
        self.prompts = prompts
        self.current_prompt = None
        self.shots_num = shots_num
        self.shots_prompt = ""
        self.split = split
        if metric_type is not None:
            self.metric_type = metric_type
        self.tree_eval_type = tree_eval_type
        self.log_instances = 5  # Set to > 0 to log the first few instances as a sanity check
        self.generate_TG_attention_bias = generate_TG_attention_bias
        print(f"vocab path is {vocab_path}")
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.parse_binarize_direction = parse_binarize_direction
        self.pushdown_tree_vocab = (
            TreeVocab.from_tokenizer_file(vocab_path)
            if self.transformer_grammar_type == "pushdown" else None
        )
        self.treereg_vocab = (
            TreeVocab.from_tokenizer_file(vocab_path)
            if self.transformer_grammar_type == "treereg" else None
        )
        self.pause_token_id = pause_token_id
        self.doc_group = None

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        if not local_datasets:
            dataset_list = []
            for ds_name in dataset_names:
                dataset = load_hf_dataset(self.dataset_path, ds_name, split)
                dataset_list.append(dataset)
            self.dataset = datasets.concatenate_datasets(dataset_list)
        else:
            self.load_local_datasets()

        if shots_num!=0 and split!="train":
            self.prepare_shots()
        # prep examples
        self.prep_examples()

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    @property
    def pause_spec(self) -> "tuple[int, int]":
        """Rational pause spec ``(p, q)`` parsed from ``transformer_grammar_type``.

        ``(0, 1)`` means no pauses (non-pause grammar types).
        """
        return pause_spec_from_grammar_type(self.transformer_grammar_type)

    @property
    def ispause(self) -> int:
        """Number of pause tokens per block (``p``); 0 means no pauses.

        Used as a truthy flag (``if self.ispause:``). For the rational spec
        ``(p, q)``, pass ``self.transformer_grammar_type`` (not this int) as
        ``pause_num`` to :func:`pause_input_ids`.
        """
        return self.pause_spec[0]

    def convert_grammar_input(self, input_ids, grammar_type=None) -> List[int]:
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)
        if grammar_type is None:
            grammar_type = self.transformer_grammar_type
        if (
            grammar_type[:8] == "terminal"
            or grammar_type[:5] == "pause"
            or grammar_type == "pushdown"
            or grammar_type == "treereg"
        ):
            # Pushdown models are trained on terminal-only sequences. Their
            # structure is carried separately as spans / inferred attachment
            # actions, never as TG non-terminal tokens.
            input_ids = self.vocab.convert_treenpy_to_terminal(input_ids)
        elif grammar_type == "tgtree":
            # TGTree data is already converted to LIN2 (TG) format in token_encode
            # (convert_treenpy_to_TG duplicates each CNT). No-op here to avoid
            # double-converting — converting again would re-duplicate CNTs and
            # inflate continuation length past the logits slice (see token_encode).
            pass
        elif grammar_type == "tree_noont":
            # Convert TG → tree, then strip opening non-terminals
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            input_ids = self.vocab.convert_treenpy_to_noont(input_ids)
        elif grammar_type == "tree_compress":
            # Convert TG → tree, then merge consecutive CNTs
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            input_ids = self.vocab.convert_treenpy_to_compress(input_ids)
        elif grammar_type == "tree_triplecnt":
            # Convert TG → tree, then triple CNTs (3 copies instead of 1)
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            input_ids = self.vocab.convert_treenpy_to_triplecnt(input_ids)
        elif grammar_type[:4] == "tree":
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
        return input_ids.tolist()

    def encode_pushdown_with_metadata(self, string: str) -> Dict[str, List[Any]]:
        """Encode a parsed string as terminals, unary spans, and tree ids."""
        if self.pushdown_tree_vocab is None:
            raise RuntimeError("Pushdown parser requested outside a Pushdown run")
        return _parse_pushdown_tree_tokens(
            encode_TG_string(self.tokenizer, string, string_with_POS_tags=False),
            self.pushdown_tree_vocab,
            self.parse_binarize_direction,
        )

    def encode_pushdown_with_spans(self, string: str) -> tuple[List[int], List[tuple[int, int, int]]]:
        """Backward-compatible terminal/spans view of Pushdown parse metadata."""
        parsed = self.encode_pushdown_with_metadata(string)
        return parsed["input_ids"], parsed["tree_spans"]

    def encode_treereg_with_metadata(self, string: str) -> Dict[str, List[Any]]:
        if self.treereg_vocab is None:
            raise RuntimeError("TreeReg parser requested outside a TreeReg run")
        return _parse_treereg_tree_tokens(
            encode_TG_string(self.tokenizer, string, string_with_POS_tags=False),
            self.treereg_vocab,
        )

    def get_shots(self, shots_split):
        self.shots = []
        for shot_id in self.shots_list:
            self.shots.append(shots_split[shot_id])
        return self.shots

    def prepare_shots(self, split="train"):
        shots_split = self.load_local_datasets(split=split, ret=True)
        self.shots = self.get_shots(shots_split)
        self.shots_prompt = ""
        for i in range(self.shots_num):
            doc = self.shots[i]
            self.shots_prompt += self.doc_to_text(doc, single_shot=True) + self.doc_to_continuations(doc, single_shot=True)[self.doc_to_label(doc)] + " \n \n"

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        doc_id = 0
        for doc in self.dataset:
            for prompt in self.prompts:
                self.current_prompt = prompt
                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                continuations = self.doc_to_continuations(doc)
                label_id = self.doc_to_label(doc)
                doc_text = self.doc_to_text(doc)
                if self.transformer_grammar_type == "pushdown":
                    ctx_parsed = self.encode_pushdown_with_metadata(doc_text)
                    ctx, ctx_spans = ctx_parsed["input_ids"], ctx_parsed["tree_spans"]
                    dc = self.encode_pushdown_with_metadata(
                        self.doc_to_domain_conditional(doc)
                    )["input_ids"]
                    # Appendix B initializes every context window with a
                    # dedicated ROOT token. The tokenizer's BOS is that ROOT;
                    # parser spans are shifted into the resulting coordinates.
                    ctx = [int(self.vocab.bos)] + ctx
                    ctx_spans = [
                        (left + 1, split + 1, right + 1)
                        for left, split, right in ctx_spans
                    ]
                    ctx_parsed = {
                        "input_ids": ctx,
                        "tree_spans": ctx_spans,
                        "pushdown_sentence_ids": [-1] + ctx_parsed[
                            "pushdown_sentence_ids"
                        ],
                    }
                    dc = [int(self.vocab.bos)] + dc
                    ctx_treereg = None
                elif self.transformer_grammar_type == "treereg":
                    ctx_treereg = self.encode_treereg_with_metadata(doc_text)
                    ctx, ctx_spans = ctx_treereg["input_ids"], ctx_treereg["tree_spans"]
                    dc = self.encode_treereg_with_metadata(
                        self.doc_to_domain_conditional(doc)
                    )["input_ids"]
                else:
                    ctx = self.token_encode(doc_text)
                    dc = self.token_encode(self.doc_to_domain_conditional(doc))
                    ctx_spans = []
                    ctx_treereg = None

                for cont_id, continuation_str in enumerate(continuations):
                    # cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    # cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                    if self.transformer_grammar_type == "pushdown":
                        continuation_parsed = self.encode_pushdown_with_metadata(
                            continuation_str
                        )
                        continuation = continuation_parsed["input_ids"]
                        continuation_spans = continuation_parsed["tree_spans"]
                        continuation_treereg = None
                    elif self.transformer_grammar_type == "treereg":
                        continuation_treereg = self.encode_treereg_with_metadata(continuation_str)
                        continuation = continuation_treereg["input_ids"]
                        continuation_spans = continuation_treereg["tree_spans"]
                    else:
                        continuation = self.token_encode(continuation_str)
                        continuation_spans = []
                        continuation_treereg = None
                    

                    # query, remove last token from continuation, truncate from left is longer than model ctx length
                    # when train, should keep last token
                    query = ctx + continuation
                    dc_query = dc + continuation
                    query_spans = list(ctx_spans) + [
                        (left + len(ctx), split + len(ctx), right + len(ctx))
                        for left, split, right in continuation_spans
                    ]
                    treereg_metadata = (
                        _join_treereg_parts((ctx_treereg, continuation_treereg))
                        if ctx_treereg is not None and continuation_treereg is not None
                        else None
                    )
                    pushdown_metadata = (
                        _join_pushdown_parts((ctx_parsed, continuation_parsed))
                        if self.transformer_grammar_type == "pushdown" else None
                    )
                    # if self.split=="train":
                    #     query = ctx + continuation
                    # else:
                    #     query = ctx + continuation[:-1]
                        
                    trim_left = max(len(query) - self.model_ctx_len, 0)
                    if trim_left and self.transformer_grammar_type == "pushdown":
                        # Keep a ROOT/BOS even when left truncation is necessary.
                        # The retained suffix is treated as a fresh context forest;
                        # crossing constituents are dropped, fully retained spans
                        # are translated after the inserted root.
                        trim_left = len(query) - (self.model_ctx_len - 1)
                        query = [int(self.vocab.bos)] + query[trim_left:]
                        query_spans = [
                            (
                                left - trim_left + 1,
                                split - trim_left + 1,
                                right - trim_left + 1,
                            )
                            for left, split, right in query_spans
                            if left >= trim_left
                        ]
                        if pushdown_metadata is not None:
                            retained = _slice_pushdown_parse(
                                pushdown_metadata,
                                trim_left,
                                trim_left + len(query) - 1,
                            )
                            pushdown_metadata = _join_pushdown_parts((
                                {
                                    "input_ids": [int(self.vocab.bos)],
                                    "tree_spans": [],
                                    "pushdown_sentence_ids": [-1],
                                },
                                retained,
                            ))
                    else:
                        query = query[trim_left:]
                        if trim_left:
                            query_spans = [
                                (left - trim_left, split - trim_left, right - trim_left)
                                for left, split, right in query_spans
                                if left >= trim_left
                            ]
                    if treereg_metadata is not None:
                        treereg_metadata = _slice_treereg_parse(
                            treereg_metadata, trim_left, trim_left + len(query)
                        )
                        query, query_spans = (
                            treereg_metadata["input_ids"], treereg_metadata["tree_spans"]
                        )
                    if pushdown_metadata is not None:
                        query, query_spans = (
                            pushdown_metadata["input_ids"], pushdown_metadata["tree_spans"]
                        )
                    query = self.convert_grammar_input(query)
                    dc_query = self.convert_grammar_input(dc_query)
                    continuation = self.convert_grammar_input(continuation)
                    mask = None
                    if hasattr(self, "get_mask"):
                        mask = self.get_mask(continuation, doc)
                    # For the pause path we pause-expand the FULL ctx+cont query (see
                    # below), so keep an untrimmed copy before the eval-time ``[:-1]``
                    # trim drops the continuation's last real token.
                    full_query = query
                    full_dc_query = dc_query
                    if self.split!="train":
                        query = query[:-1]
                        dc_query = dc_query[:-1]
                        query_spans = [
                            span for span in query_spans if span[2] < len(query)
                        ]
                        if treereg_metadata is not None:
                            treereg_metadata = _slice_treereg_parse(
                                treereg_metadata, 0, len(query)
                            )
                        if pushdown_metadata is not None:
                            pushdown_metadata = _slice_pushdown_parse(
                                pushdown_metadata, 0, len(query)
                            )
                    # Grammar conversions that INFLATE the sequence (e.g.
                    # tree_triplecnt tripling every CNT token) can push an
                    # already-trimmed query past model_ctx_len. The collator's
                    # pad_tokens_until_max would then silently left-truncate
                    # the padded input while ctx_len still refers to positions
                    # in the untruncated query, so ICLMetric.update gathers
                    # from an empty logits slice and crashes ("expected index
                    # [1, 1] to be no larger than self [0, vocab]"). Truncate
                    # here instead so ctx_len, spans and the actual input stay
                    # aligned; the continuation sits at the tail and always
                    # survives. Pause types are unaffected: their real-token
                    # budget was pre-shrunk and expansion happens later on
                    # full_query, outside this path.
                    overflow = len(query) - self.model_ctx_len
                    if overflow > 0:
                        query = query[overflow:]
                        query_spans = [
                            (left - overflow, mid - overflow, right - overflow)
                            for left, mid, right in query_spans
                            if left >= overflow and right < len(query)
                        ]
                    continuation_str = self.token_decode(continuation)
                    # this will be different from len(ctx) when truncated by model_ctx_len
                    actual_ctx_len = len(query) - len(continuation) + 1

                    # get domain conditional query
                    # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left

                    cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                    cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                    if self.tree_eval_type=="terminal":
                        terminal_continuation = self.convert_grammar_input(continuation, grammar_type="terminal")
                        terminal_continuation_str = self.token_decode(terminal_continuation)
                        cont_str_len = len(terminal_continuation_str) - 1  # continuation contain leading blank
                        cont_byte_len = len(terminal_continuation_str[1:].encode("utf-8"))


                    if self.ispause:
                        # Rational pause spec (p, q): insert p pauses after every q real
                        # tokens. Pause the full ctx+cont query as one sequence so the
                        # layout matches training. The scored continuation must be a
                        # contiguous slice of the expanded query: logits are gathered at
                        # [ctx_len-1 : ctx_len+cont_len-1] (see ICLMetric.update), so
                        # continuation[k] must equal query[ctx_len+k].
                        #
                        # We expand ``full_query`` (ctx+cont, untrimmed) rather than the
                        # eval-trimmed ``query``: at eval time ``query = query[:-1]``
                        # drops the continuation's last real token, so for a single-token
                        # continuation (e.g. BoolQ " yes"/" no") the trimmed query would
                        # contain NO continuation tokens and the slice below would be
                        # empty, collapsing the score to the majority class. Expanding the
                        # full ctx+cont keeps the continuation in the expanded query.
                        #
                        # split = expanded position of the first continuation real token
                        #   = pause_expanded_len(ctx_real) = ctx_real + (ctx_real//q)*p,
                        # which is correct for ANY ctx_real (divisible or not), because
                        # pause_input_ids places real token i at position i + (i//q)*p.
                        # Trailing pauses after the last real token exist only when the
                        # full real length completes a block (n % q == 0); trim them so
                        # the model is not scored on predicting a trailing pause.
                        p, q = self.pause_spec
                        gtype = self.transformer_grammar_type
                        # ctx_real = number of real ctx tokens in full_query.
                        # full_query = ctx + continuation (real tokens, untrimmed), so
                        # ctx_real = len(full_query) - len(continuation). We must NOT use
                        # ``actual_ctx_len`` here: that value carries a ``+1`` offset meant
                        # to compensate for the eval-time ``query[:-1]`` trim in the
                        # non-pause metric path, and for ``split=="train"`` the trim never
                        # runs, so actual_ctx_len would be len(ctx)+1 and make the slice
                        # below overshoot past the end of the expanded query (empty
                        # continuation -> degenerate majority-class score).
                        ctx_real = len(full_query) - len(continuation)
                        cont_real = len(continuation)
                        query = pause_input_ids(full_query, self.pause_token_id, pause_num=gtype)
                        dc_query = pause_input_ids(full_dc_query, self.pause_token_id, pause_num=gtype)
                        split = pause_expanded_len(ctx_real, p, q)
                        trim = pause_trailing_trim(ctx_real + cont_real, p, q)
                        continuation = query[split : len(query) - trim]
                        actual_ctx_len = split
                        if mask is not None:
                            # mask is aligned to the real continuation (length cont_real).
                            # To align it with continuation = query[split:end] we must
                            # expand it on the SAME block grid as the full query: prepend
                            # ctx_real zeros (ctx tokens are not continuation), pause with
                            # None (broadcasts each real's mask value to its pause slots),
                            # then slice [split:end]. This stays correct when ctx_real is
                            # not divisible by q, because the boundary block is shared with
                            # the query expansion above.
                            full_mask = [0] * ctx_real + list(mask)
                            full_mask = pause_input_ids(full_mask, pause_token_id=None, pause_num=gtype)
                            mask = full_mask[split : len(query) - trim]

                    self.samples.append(
                        {
                            "doc_id": doc_id,
                            "cont_id": cont_id,
                            # "ctx": ctx,
                            "continuation": continuation,
                            "ctx_len": actual_ctx_len,
                            "dc_len": len(dc),
                            "cont_len": len(
                                continuation
                            ),  # even if query has last token removed, LM will output same cont len
                            "cont_str_len": cont_str_len,
                            "cont_byte_len": cont_byte_len,
                            "query": query,  # remove last token from continuation
                            "dc_query": dc_query,
                            "label_id": label_id if isinstance(label_id, int) else (cont_id if cont_id in label_id else label_id[0]), 
                                                # since some benchmarks have multiple correct labels
                            "cont_mask": mask,
                            "tree_spans": query_spans,
                            "treereg_word_boundaries": (
                                treereg_metadata["treereg_word_boundaries"]
                                if treereg_metadata is not None else None
                            ),
                            "treereg_sentence_ids": (
                                treereg_metadata["treereg_sentence_ids"]
                                if treereg_metadata is not None else None
                            ),
                            "pushdown_sentence_ids": (
                                pushdown_metadata["pushdown_sentence_ids"]
                                if pushdown_metadata is not None else None
                            ),
                        }
                    )
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f" \ndoc_text: {doc_text} \ncontinuations: {continuations} \n" +
                        f"input_ids is {self.token_decode(query)}\n" + 
                        f"query is {query}\n" +  
                        f"continuation is {continuation}\n" + 
                        f"ctx_len is {actual_ctx_len}"
                    )

                doc_id += 1

    def pad_tokens_until_max(self, tokens, max_len=2048, max_model_len=None):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        # model_ctx_len is the real-token budget; expand it to the paused length.
        p, q = self.pause_spec
        model_ctx_len = max_model_len or pause_expanded_len(self.model_ctx_len, p, q)
        if len(tokens) > model_ctx_len:
            return tokens[-model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > model_ctx_len
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - len(tokens))

            if len(tokens) > model_ctx_len:
                tokens = tokens[: model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        # 'ctx', 'continuation', 'query' can all have variable length
        max_ctx_len = 0
        max_cont_len = 0
        max_query_len = 0
        max_dc_query_len = 0

        for sample in data:
            # if len(sample["ctx"]) > max_ctx_len:
            #     max_ctx_len = len(sample["ctx"])

            if len(sample["continuation"]) > max_cont_len:
                max_cont_len = len(sample["continuation"])

            if len(sample["query"]) > max_query_len:
                max_query_len = len(sample["query"])

            if len(sample["dc_query"]) > max_dc_query_len:
                max_dc_query_len = len(sample["dc_query"])

        doc_ids = []
        cont_ids = []
        ctxs = []
        continuations = []
        ctx_lens = []
        dc_lens = []
        cont_lens = []
        cont_str_lens = []
        cont_byte_lens = []
        queries = []
        dc_queries = []
        label_ids = []
        all_attention_bias = []
        all_label_mask = []
        all_cont_mask = []
        all_tree_spans = []
        all_treereg_word_boundaries = []
        all_treereg_sentence_ids = []
        all_pushdown_sentence_ids = []

        # pad according to max_lengths
        for sample in data:
            input_ids = sample["query"]
            if self.transformer_grammar_type[:12] == "tree_shuffle":
                if not isinstance(input_ids, np.ndarray):
                    input_ids = np.array(input_ids)
                input_ids = self.vocab.random_shuffle_tree(input_ids)
                input_ids = input_ids.tolist()
            p, q = self.pause_spec
            input_ids = torch.LongTensor(self.pad_tokens_until_max(input_ids, max_len=max_query_len, max_model_len=pause_expanded_len(self.model_ctx_len, p, q)))
            queries.append(input_ids)
            all_tree_spans.append(sample.get("tree_spans", []))
            if self.transformer_grammar_type == "treereg":
                word_boundaries = sample["treereg_word_boundaries"]
                sentence_ids = sample["treereg_sentence_ids"]
                if word_boundaries is None or sentence_ids is None:
                    raise RuntimeError("TreeReg sample is missing parser metadata")
                all_treereg_word_boundaries.append(F.pad(
                    torch.tensor(word_boundaries, dtype=torch.bool),
                    (0, max_query_len - len(word_boundaries)), value=False,
                ))
                all_treereg_sentence_ids.append(F.pad(
                    torch.tensor(sentence_ids, dtype=torch.int32),
                    (0, max_query_len - len(sentence_ids)), value=-1,
                ))
            if self.transformer_grammar_type == "pushdown":
                sentence_ids = sample["pushdown_sentence_ids"]
                if sentence_ids is None:
                    raise RuntimeError("Pushdown sample is missing parser sentence metadata")
                all_pushdown_sentence_ids.append(F.pad(
                    torch.tensor(sentence_ids, dtype=torch.int32),
                    (0, max_query_len - len(sentence_ids)), value=-1,
                ))

            label_mask = None
            if self.generate_TG_attention_bias is not None:
                attention_bias, label_mask = self.generate_TG_attention_bias(input_ids)
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                all_attention_bias.append(attention_bias)
            if self.transformer_grammar_type == "tree_shuffle_mask":
                cur_label_mask = self.vocab.get_non_terminal_mask(input_ids)
                if label_mask is not None:
                    label_mask = torch.bitwise_and(label_mask, torch.tensor(cur_label_mask))
                else:
                    label_mask = cur_label_mask
            if label_mask is not None:
                all_label_mask.append(label_mask)
            
            doc_ids.append(sample["doc_id"])
            cont_ids.append(sample["cont_id"])

            # ctxs.append(torch.LongTensor(self.pad_tokens_until_max(sample["ctx"], max_len=max_ctx_len)))
            continuations.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["continuation"], max_len=max_cont_len))
            )
            if sample["cont_mask"] is not None:
                all_cont_mask.append(
                    torch.tensor(self.pad_tokens_until_max(sample["cont_mask"], max_len=max_cont_len), dtype=torch.bool)
                )

            ctx_lens.append(sample["ctx_len"])
            dc_lens.append(sample["dc_len"])
            cont_lens.append(sample["cont_len"])
            cont_str_lens.append(sample["cont_str_len"])
            cont_byte_lens.append(sample["cont_byte_len"])

            dc_queries.append(
                torch.LongTensor(self.pad_tokens_until_max(sample["dc_query"], max_len=max_dc_query_len))
            )
            label_ids.append(sample["label_id"])
            # print(self.token_decode(sample["query"]))
            # print(sample["query"])
            # print(f"ctx_len is {sample['ctx_len']}")
            # print(f'cont_len is {sample["cont_len"]}')


        batch = {
            "doc_id": torch.LongTensor(doc_ids),
            "cont_id": torch.LongTensor(cont_ids),
            # "ctx": torch.stack(ctxs),
            "continuation": torch.stack(continuations),
            "ctx_len": torch.LongTensor(ctx_lens),
            "dc_len": torch.LongTensor(dc_lens),
            "cont_len": torch.LongTensor(cont_lens),  # since query has last token removed from continuation
            "cont_str_len": torch.LongTensor(cont_str_lens),
            "cont_byte_len": torch.LongTensor(cont_byte_lens),
            "input_ids": torch.stack(queries),
            "dc_input_ids": torch.stack(dc_queries),
            "label_id": torch.LongTensor(label_ids),
        }
        if all_attention_bias:
            batch["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            batch["label_mask"] = torch.stack(all_label_mask)
        if all_cont_mask:
            batch["cont_mask"] = torch.stack(all_cont_mask)
        if all_treereg_word_boundaries:
            batch["treereg_word_boundaries"] = torch.stack(all_treereg_word_boundaries)
            batch["treereg_sentence_ids"] = torch.stack(all_treereg_sentence_ids)
        if all_pushdown_sentence_ids:
            batch["pushdown_sentence_ids"] = torch.stack(all_pushdown_sentence_ids)
        if any(all_tree_spans):
            max_spans = max(len(spans) for spans in all_tree_spans)
            tree_spans = torch.full(
                (len(data), max_spans, 3), -1, dtype=torch.long
            )
            tree_span_mask = torch.zeros(
                (len(data), max_spans), dtype=torch.bool
            )
            for row, spans in enumerate(all_tree_spans):
                if spans:
                    count = len(spans)
                    tree_spans[row, :count] = torch.tensor(spans, dtype=torch.long)
                    tree_span_mask[row, :count] = True
            batch["tree_spans"] = tree_spans
            batch["tree_span_mask"] = tree_span_mask

        return batch

    def token_encode(self, string: str) -> List[int]:
        ids = encode_TG_string(self.tokenizer, string, string_with_POS_tags=False)
        if self.transformer_grammar_type == "tgtree":
            # TGTree: trained on LIN2 (TG) data — convert_treenpy_to_TG duplicates
            # each closing NT (1 ONT + 2 CNT). Must happen here, BEFORE ctx+cont
            # assembly and model_ctx_len truncation, so the duplicated CNTs are
            # counted in the length budget (converting later in
            # convert_grammar_input would inflate the continuation past the
            # logits slice). convert_grammar_input is a no-op for tgtree.
            # Return a list to match the contract of the other branch (.tolist()).
            ids = self.vocab.convert_treenpy_to_TG(ids).tolist()
        else:
            ids = self.convert_grammar_input(ids)
        return ids

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False)

    @abc.abstractmethod
    def doc_to_text(self, doc) -> str:
        """Match EAI eval harness
        returns a single context string
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_continuations(self, doc) -> List[str]:
        """Match EAI eval harness
        returns a list of continuations
        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_label(self, doc) -> int:
        """Match EAI eval harness
        returns continuation id which corresponds to true label
        """
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc) -> str:
        """Provide string for domain conditional normalization
        by default its blank string, continuation normalized by prob conditioned on a blank
        """
        del doc
        return " "
    
    def load_local_datasets(self, split, ret):
        raise NotImplementedError


class XsumDataset(metaclass=abc.ABCMeta):
    def __init__(self,
        tokenizer: Tokenizer,
        dataset_path: str,
        model_ctx_len: int = 2048,
        split="test",
        metric_type="sent",
        generate_TG_attention_bias: Optional[Callable] = None,
        transformer_grammar_type:str = "",
        vocab_path: str = None,
        pause_token_id : int = None,
        parse_binarize_direction: str = "right",
        **kwargs):

        self.tokenizer = tokenizer
        self.transformer_grammar_type = transformer_grammar_type
        self.collator = DataCollator(pad_direction=PaddingDirection.left, pad_token_id=self.tokenizer.pad_token_id, 
                                        generate_attention_mask=True, shuffle_tree=transformer_grammar_type)
        self.MAX_SUMMARY_LENGTH = 150
        self.collator.vocab = self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.treereg_vocab = (
            TreeVocab.from_tokenizer_file(vocab_path)
            if transformer_grammar_type == "treereg" else None
        )
        self.pushdown_tree_vocab = (
            TreeVocab.from_tokenizer_file(vocab_path)
            if transformer_grammar_type == "pushdown" else None
        )
        self.parse_binarize_direction = parse_binarize_direction
        self.model_ctx_len = model_ctx_len
        self.generate_TG_attention_bias = generate_TG_attention_bias
        self.prompts = " \n<(S><(VP> Summarize<(NP> the above article<NP)><(PP> in<(NP> 1 sentence<NP)><PP)><VP)> .<S)> \n"
        self.prompts_tokens = encode_TG_string(self.tokenizer, self.prompts, string_with_POS_tags=False)
        self.prompts_TG_tokens = self.vocab.convert_treenpy_to_TG(self.prompts_tokens)
        self.pause_token_id = pause_token_id

        if transformer_grammar_type[:5] == "pause":
            # Shrink the real-token budget so the expanded (paused) length still
            # fits model_ctx_len. For spec (p, q), expansion factor is (q+p)/q.
            p, q = pause_spec_from_grammar_type(transformer_grammar_type)
            self.model_ctx_len = self.model_ctx_len * q // (q + p)
            self.prompts_TG_tokens = self.vocab.convert_treenpy_to_terminal(self.prompts_TG_tokens)
        passages = []
        gold_summary = []
        with open(os.path.join(dataset_path, f"gold_{split}_summary.jsonl"), 'r') as file:
            for line in file:
                summary = json.loads(line.strip())
                gold_summary.append(summary)

        with open(os.path.join(dataset_path, f"xsum_{split}.txt"), 'r') as file:
            for line in file:
                passages.append(line.strip())

        if split=="train":
            train_summary = []
            with open(os.path.join(dataset_path, f"save_ids.json"), 'r') as file:
                train_ids = json.load(file)
            with open(os.path.join(dataset_path, "xsum_train_summary.txt"), 'r') as file:
                for line in file:
                    train_summary.append(line.strip())
            train_ids = set(train_ids)
            self.passages = []
            self.train_summary = []
            self.gold_summary = []
            for passage, summary, gold in zip(passages, train_summary, gold_summary):
                if gold["id"] in train_ids:
                    self.passages.append(passage)
                    self.train_summary.append(summary)
                    self.gold_summary.append(gold["summary"])
        else:
            self.passages = passages
            self.gold_summary = [line["summary"] for line in gold_summary]
            self.train_summary = None

    @property
    def pause_spec(self) -> "tuple[int, int]":
        """Rational pause spec ``(p, q)`` parsed from ``transformer_grammar_type``.

        ``(0, 1)`` means no pauses (non-pause grammar types).
        """
        return pause_spec_from_grammar_type(self.transformer_grammar_type)

    @property
    def ispause(self) -> int:
        """Number of pause tokens per block (``p``); 0 means no pauses.

        Used as a truthy flag (``if self.ispause:``). For the rational spec
        ``(p, q)``, pass ``self.transformer_grammar_type`` (not this int) as
        ``pause_num`` to :func:`pause_input_ids`.
        """
        return self.pause_spec[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        '''
        truncate input as the length of TG.
        '''
        passage = self.passages[index]
        passage_tokens = encode_TG_string(self.tokenizer, passage)
        if self.transformer_grammar_type == "pushdown":
            if self.pushdown_tree_vocab is None:
                raise RuntimeError("Pushdown XSum dataset is missing its tree vocabulary")
            parse = lambda tokens: _parse_pushdown_tree_tokens(
                tokens, self.pushdown_tree_vocab, self.parse_binarize_direction
            )
            passage_parsed = parse(passage_tokens)
            prompt_parsed = parse(self.prompts_tokens)
            prefix = {
                "input_ids": [int(self.vocab.bos)], "tree_spans": [],
                "pushdown_sentence_ids": [-1],
            }
            suffix = {
                "input_ids": [int(self.vocab.eos)], "tree_spans": [],
                "pushdown_sentence_ids": [-1],
            }
            if self.train_summary is not None:
                summary_parsed = parse(
                    encode_TG_string(self.tokenizer, self.train_summary[index])
                )
                passage_budget = (
                    self.model_ctx_len - len(summary_parsed["input_ids"])
                    - len(prompt_parsed["input_ids"]) - 2
                )
                # The XSum input is a prefix window; never retain a constituent
                # whose closing terminal is outside that window.
                passage_parsed = {
                    "input_ids": passage_parsed["input_ids"][:passage_budget],
                    "tree_spans": [
                        span for span in passage_parsed["tree_spans"]
                        if span[2] < passage_budget
                    ],
                    "pushdown_sentence_ids": passage_parsed[
                        "pushdown_sentence_ids"
                    ][:passage_budget],
                }
                parsed = _join_pushdown_parts(
                    (prefix, passage_parsed, prompt_parsed, summary_parsed, suffix)
                )
                label_mask = torch.zeros(len(parsed["input_ids"]), dtype=torch.bool)
                label_mask[-(len(summary_parsed["input_ids"]) + 1):] = True
            else:
                passage_budget = (
                    self.model_ctx_len - self.MAX_SUMMARY_LENGTH
                    - len(prompt_parsed["input_ids"]) - 2
                )
                passage_parsed = {
                    "input_ids": passage_parsed["input_ids"][:passage_budget],
                    "tree_spans": [
                        span for span in passage_parsed["tree_spans"]
                        if span[2] < passage_budget
                    ],
                    "pushdown_sentence_ids": passage_parsed[
                        "pushdown_sentence_ids"
                    ][:passage_budget],
                }
                parsed = _join_pushdown_parts((prefix, passage_parsed, prompt_parsed))
                label_mask = None
            return {
                "attention_bias": None,
                "gold_summary": self.gold_summary[index],
                "label_mask": label_mask,
                "input_ids": np.asarray(parsed["input_ids"], dtype=np.int64),
                "tree_spans": parsed["tree_spans"],
                "pushdown_sentence_ids": parsed["pushdown_sentence_ids"],
            }
        if self.transformer_grammar_type == "treereg":
            if self.treereg_vocab is None:
                raise RuntimeError("TreeReg XSum dataset is missing its tree vocabulary")
            parse = lambda tokens: _parse_treereg_tree_tokens(tokens, self.treereg_vocab)
            passage_parsed = parse(passage_tokens)
            prompt_parsed = parse(self.prompts_tokens)
            prefix = {
                "input_ids": [int(self.vocab.bos)], "tree_spans": [],
                "treereg_word_boundaries": [False], "treereg_sentence_ids": [-1],
            }
            suffix = {
                "input_ids": [int(self.vocab.eos)], "tree_spans": [],
                "treereg_word_boundaries": [False], "treereg_sentence_ids": [-1],
            }
            if self.train_summary is not None:
                summary_parsed = parse(encode_TG_string(self.tokenizer, self.train_summary[index]))
                passage_budget = self.model_ctx_len - len(summary_parsed["input_ids"]) - len(prompt_parsed["input_ids"]) - 2
                passage_parsed = _slice_treereg_parse(passage_parsed, 0, passage_budget)
                parsed = _join_treereg_parts(
                    (prefix, passage_parsed, prompt_parsed, summary_parsed, suffix)
                )
                label_mask = torch.zeros(len(parsed["input_ids"]), dtype=torch.bool)
                label_mask[-(len(summary_parsed["input_ids"]) + 1):] = True
            else:
                passage_budget = self.model_ctx_len - self.MAX_SUMMARY_LENGTH - len(prompt_parsed["input_ids"]) - 2
                passage_parsed = _slice_treereg_parse(passage_parsed, 0, passage_budget)
                parsed = _join_treereg_parts((prefix, passage_parsed, prompt_parsed))
                label_mask = None
            return {
                "attention_bias": None, "gold_summary": self.gold_summary[index],
                "label_mask": label_mask,
                "input_ids": np.asarray(parsed["input_ids"], dtype=np.int64),
                "tree_spans": parsed["tree_spans"],
                "treereg_word_boundaries": parsed["treereg_word_boundaries"],
                "treereg_sentence_ids": parsed["treereg_sentence_ids"],
            }
        passage_TG_tokens = self.vocab.convert_treenpy_to_TG(passage_tokens)
        if self.transformer_grammar_type[:5] == "pause":
            passage_TG_tokens = self.vocab.convert_treenpy_to_terminal(passage_tokens)
        if self.train_summary is not None:
            train_summary = self.train_summary[index]
            train_summary_tokens = encode_TG_string(self.tokenizer, train_summary)
            train_summary_TG_tokens = self.vocab.convert_treenpy_to_TG(train_summary_tokens)
            if self.transformer_grammar_type[:5] == "pause":
                train_summary_TG_tokens = self.vocab.convert_treenpy_to_terminal(train_summary_TG_tokens)
            passage_truncate_length = self.model_ctx_len - len(train_summary_TG_tokens) - len(self.prompts_TG_tokens) - 1 - 1 # one for bos and one for eos
            input_ids = np.concatenate([
                np.array([self.vocab.bos]),
                passage_TG_tokens[:passage_truncate_length],
                self.prompts_TG_tokens,
                train_summary_TG_tokens,
                np.array([self.vocab.eos])
            ])
            loss_tokens = np.concatenate([
                train_summary_TG_tokens, 
                np.array([self.vocab.eos])
            ])
        else:
            passage_truncate_length = self.model_ctx_len - self.MAX_SUMMARY_LENGTH - len(self.prompts_TG_tokens) - 1 - 1
            input_ids = np.concatenate([
                np.array([self.vocab.bos]),
                passage_TG_tokens[:passage_truncate_length],
                self.prompts_TG_tokens,
            ])
            loss_tokens = None
        
        attention_bias, label_mask, TG_label_mask = None, None, None
        if self.transformer_grammar_type == "terminal":
            input_ids = self.vocab.convert_treenpy_to_terminal(input_ids)
            if loss_tokens is not None:
                loss_tokens = self.vocab.convert_treenpy_to_terminal(loss_tokens)
        elif self.transformer_grammar_type == "tree_noont":
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            input_ids = self.vocab.convert_treenpy_to_noont(input_ids)
            if loss_tokens is not None:
                loss_tokens = self.vocab.convert_TGnpy_to_tree(loss_tokens)
                loss_tokens = self.vocab.convert_treenpy_to_noont(loss_tokens)
        elif self.transformer_grammar_type == "tree_compress":
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            input_ids = self.vocab.convert_treenpy_to_compress(input_ids)
            if loss_tokens is not None:
                loss_tokens = self.vocab.convert_TGnpy_to_tree(loss_tokens)
                loss_tokens = self.vocab.convert_treenpy_to_compress(loss_tokens)
        elif self.transformer_grammar_type == "tree_triplecnt":
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            input_ids = self.vocab.convert_treenpy_to_triplecnt(input_ids)
            if loss_tokens is not None:
                loss_tokens = self.vocab.convert_TGnpy_to_tree(loss_tokens)
                loss_tokens = self.vocab.convert_treenpy_to_triplecnt(loss_tokens)
        elif self.transformer_grammar_type == "tree":
            input_ids = self.vocab.convert_TGnpy_to_tree(input_ids)
            if loss_tokens is not None:
                loss_tokens = self.vocab.convert_TGnpy_to_tree(loss_tokens)
        elif self.transformer_grammar_type[:5] == "pause":
            input_ids = pause_input_ids(input_ids, self.pause_token_id, pause_num=self.transformer_grammar_type)
        elif self.generate_TG_attention_bias is not None:
            input_ids = torch.tensor(input_ids)
            attention_bias, TG_label_mask = self.generate_TG_attention_bias(input_ids)
        
        if loss_tokens is not None:
            label_mask = torch.zeros((input_ids.shape[0], ), dtype=torch.bool)
            label_mask[label_mask.shape[0] - loss_tokens.shape[0]:] = True
            if TG_label_mask is not None:
                label_mask = torch.bitwise_and(label_mask, TG_label_mask)
        return {
            "attention_bias": attention_bias,
            "gold_summary" : self.gold_summary[index],
            "label_mask": label_mask,
            "input_ids": input_ids,
        }

    def __len__(self):
        return len(self.passages)
    
    def collate_fn(self, data):
        return self.collator(data)

class RougeMetric(Metric):
    def __init__(self, 
                 metric_type:str = "rouge",
        vocab_path:str = None,
        tokenizer:Tokenizer = None
        ) -> None:
        # sync_on_compute=False: compute() gathers list state via _gather_list
        # (count-insensitive), avoiding the unequal-count deadlock.
        super().__init__(sync_on_compute=False)
        self.add_state("predictions", default=[], dist_reduce_fx=None)
        self.add_state("references", default=[], dist_reduce_fx=None)
        self.tokenizer = tokenizer

    def update(self, batch, predictions, references):
        input_ids = batch["input_ids"].cpu()
        for b in range(predictions.shape[0]):
            # Decoding and printing the complete source+150-token output for
            # every example is expensive in a CPU-starved multi-rank Slurm job
            # and produces multi-GB logs. Keep it as an explicit diagnostic;
            # ROUGE state below is unchanged when logging is disabled.
            if os.environ.get("OLMO_LOG_XSUM_PREDICTIONS", "1") != "0":
                passage = self.tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=False)
                prediction = self.tokenizer.decode(predictions[b].tolist(), skip_special_tokens=False)
                log.log(
                    XSUM_PREDICTION_LOG_LEVEL,
                    f"[global_rank={get_global_rank()}] <New Passage>: {passage} {prediction}",
                )
            # all_gather_object restores CUDA tensors onto the GPU that receives
            # them. Keeping metric state on CPU prevents a world-size-multiplied
            # late GPU-memory spike during ROUGE aggregation.
            self.predictions.append(predictions[b].cpu())

        for gold in references:
            self.references.append(torch.tensor(self.tokenizer.encode(gold, add_special_tokens=False)))

    def reset(self):
        self.predictions = []
        self.references = []

    def compute(self):
        # Gather per-rank list state across ranks (count-insensitive).
        self.predictions = _gather_list(self.predictions)
        self.references = _gather_list(self.references)
        rouge = evaluate.load('rouge')
        predictions_str = []
        references_str = []
        for prediction in self.predictions:
            predictions_str.append(self.tokenizer.decode(prediction.tolist()))
        for reference in self.references:
            references_str.append(self.tokenizer.decode(reference.tolist()))
        results = rouge.compute(
            predictions=predictions_str,
            references=references_str,
            use_stemmer=True,
            rouge_types=['rouge1', 'rouge2', 'rougeL'],  
            use_aggregator=True,  # ave scores
        )
        results["R-AVG"] = sum(results.values()) / 3
        return results

class TGPerplexityDocumentLevelMetric(Metric):
    full_state_update: bool = False
    
    def __init__(
            self, 
            metric_type="doc_ppl", 
            vocab_path = None,
            term_length = None, 
            device_eval_batch_size = None, 
            dataset_length = None,
            samples_per_sent = 300,
        ) -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        super().__init__(sync_on_compute=False) # since we use one device to eval, sync could be false

        self.metric_type = "doc_ppl"
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.term_length = term_length
        self.samples_per_sent = samples_per_sent
        self.cur_sent = 0
        self.cur_batch = 0
        self.device_eval_batch_size = device_eval_batch_size 
        self.add_state("loglikelihoods", default=torch.zeros((dataset_length//self.samples_per_sent, self.samples_per_sent), dtype=torch.float32), dist_reduce_fx=None)

    def reset(self):
        # ``compute()`` replaces this rank-local state with the globally
        # all-reduced tensor. A later evaluation in the same Trainer must start
        # from zeros; otherwise slots owned by other ranks retain their previous
        # global values and are summed again at the next all-reduce.
        self.loglikelihoods.zero_()
        # Retained for compatibility with older code that inspected these
        # counters, although update() now scatters by global sample index.
        self.cur_sent = 0
        self.cur_batch = 0

    def update(self, batch: Dict[str, Any], ce_loss:torch.Tensor, lm_logits: Optional[torch.Tensor] = None, dc_lm_logits=None):
        # Scatter by GLOBAL flat index so multi-rank docppl writes disjoint slots:
        # each rank processes whole documents (DistributedEvalSampler
        # group_starts), so its indices index distinct slots of the pre-allocated
        # [n_sent, SENT_SIZE] tensor. row = index//SENT_SIZE, col = index%SENT_SIZE
        # — one unique slot per tree (sent_id is shared by all SENT_SIZE trees of
        # a sentence, so it cannot distinguish them). Unwritten slots stay 0;
        # compute() SUM all-reduces via _all_reduce_tensor. Replaces the old local
        # cur_sent/cur_batch counters, which assumed single-rank in-order arrival
        # and would alias across ranks.
        idx = batch["index"]
        if idx.dim() == 0:
            idx = idx.unsqueeze(0)
            ce_loss = ce_loss.unsqueeze(0)
        row = idx // self.samples_per_sent
        col = idx % self.samples_per_sent
        self.loglikelihoods[row, col] = ce_loss

    def compute(self) -> torch.Tensor:
        # SUM all-reduce the fixed-size loglikelihoods tensor across ranks
        # (count-insensitive: each rank wrote its own disjoint slots). Required
        # under multi-GPU DistributedEvalSampler; a no-op for single-device eval.
        self.loglikelihoods = _all_reduce_tensor(self.loglikelihoods)
        data_numwords = sum(self.term_length)
        ppl = torch.logsumexp(-self.loglikelihoods, dim=1).sum().item()
        ppl = np.exp(-ppl / data_numwords)
        return torch.tensor(ppl)


class TerminalDocumentPerplexityMetric(Metric):
    """Global token-weighted K=1 document PPL for the terminal projection."""

    full_state_update: bool = False

    def __init__(self, term_length: Sequence[int], dataset_length: int) -> None:
        super().__init__(sync_on_compute=False)
        self.metric_type = "doc_ppl"
        self.data_numwords = int(sum(term_length))
        self.expected_records = int(dataset_length)
        if self.data_numwords <= 0 or self.expected_records <= 0:
            raise ValueError("terminal document PPL requires non-empty data")
        self.add_state(
            "total_nll", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx=None
        )
        self.add_state(
            "evaluated_records", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx=None
        )

    def update(self, batch: Dict[str, Any], ce_loss: torch.Tensor, **kwargs) -> None:
        self.total_nll += ce_loss.detach().to(dtype=torch.float64).sum()
        self.evaluated_records += int(ce_loss.numel())

    def compute(self) -> torch.Tensor:
        total_nll = _all_reduce_tensor(self.total_nll)
        evaluated_records = _all_reduce_tensor(self.evaluated_records)
        if int(evaluated_records.item()) != self.expected_records:
            raise RuntimeError(
                "terminal document PPL must evaluate every sentence exactly once: "
                f"{int(evaluated_records.item())} != {self.expected_records}"
            )
        return torch.exp(total_nll / self.data_numwords)

# Deprecated please use Document Level ppl metric
class TGPerplexitySentenceLevelMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(
            self, 
            metric_type="sent_ppl", 
            vocab_path : str = None,
            term_length = None
        ) -> None:
        """metric_type: f1, acc, len_norm, pmi_dc, ce_loss, bpb"""
        # sync_on_compute=False: compute() gathers list state via _gather_list
        # (count-insensitive), avoiding the unequal-count deadlock.
        super().__init__(sync_on_compute=False)

        self.metric_type = "sent_ppl"
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.term_length = term_length
        self.add_state("loglikelihoods", default=[], dist_reduce_fx=None)

    def reset(self):
        self.loglikelihoods = []

    def update(self, batch: Dict[str, Any], lm_logits: torch.Tensor, dc_lm_logits=None):
        logits_for_loss = lm_logits[..., :-1, :].to(dtype=torch.float32).contiguous()
        # print_tensor_data(batch["input_ids"])
        # print_tensor_data(batch["attention_bias"])
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, self.vocab.pad)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, self.vocab.pad)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=self.vocab.pad)
        labels = labels[..., 1:].contiguous()
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss = F.cross_entropy(
            logits_for_loss, labels, ignore_index=self.vocab.pad, reduction="none")
        # print_tensor_data(ce_loss.view(batch["input_ids"].shape[0], -1))
        ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1).sum(dim=1)
        device = batch["input_ids"].device

        for idx, sent_id in enumerate(batch["sent_id"]):
            # [cont_len]: continuation is padded for batching
            # sent = tokens[idx]
            # sent_length = sum([self.vocab.is_terminal(sent[i]) for i in range(batch["input_ids"].shape[1])]) + 1 # add predict eos

            # because metric states cannot be dict/list of tuples, store this tuple as tensor: (doc_id, cont_id, metric_state)
            self.loglikelihoods.append(
                torch.Tensor((sent_id, ce_loss[idx])).to(device)
            )

    def compute(self) -> torch.Tensor:
        # Gather per-rank list state across ranks (count-insensitive). Replaces
        # torchmetrics' sync_on_compute=True gather, which deadlocks when ranks
        # update unequal times under DistributedEvalSampler.
        self.loglikelihoods = _gather_list(self.loglikelihoods)
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        samples_per_sent = 300
        sent_cnt = len(self.loglikelihoods)//samples_per_sent
        loglikelihood_dict = torch.zeros(sent_cnt, dtype=torch.int32)
        loglikelihood_tensor = torch.empty(
            sent_cnt, 
            samples_per_sent, 
            dtype=torch.float32,
            device=self.loglikelihoods[0][0].device
        )
        # collect loglikelihoods
        for sent_id, loglikelihood in self.loglikelihoods:
            sent_id = int(sent_id.item()) - 1  # data sent_id count from 1
            loglikelihood_tensor[sent_id, loglikelihood_dict[sent_id]] = loglikelihood
            # log.info(f"eval likeli is {loglikelihood}")
            loglikelihood_dict[sent_id] += 1
        
        ppl = 0.0
        data_numwords = sum(self.term_length)

        ppl = torch.logsumexp(-loglikelihood_tensor, dim=1).sum().item()
        # for sent_id in loglikelihood_dict:
        #     sent_logLs = torch.tensor(loglikelihood_dict[sent_id])
        #     cur_loss = torch.logsumexp(-sent_logLs, dim=0).item()
        #     ppl += cur_loss
        #     data_numwords += self.term_length[sent_id]

        ppl = np.exp(-ppl / data_numwords)
        return torch.tensor(ppl)


def normalize_testppl_document_record(
    input_ids: np.ndarray,
    *,
    first_in_document: bool,
    last_in_document: bool,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> np.ndarray:
    """Idempotently frame one testppl candidate with document BOS/EOS.

    Historical testppl data has one boundary special per document: the first
    document of each source split has BOS only and later documents have EOS
    only.  This helper accepts both that layout and the normalized layout.  It
    rejects misplaced/duplicated specials instead of silently hiding corrupt
    record boundaries.
    """
    values = np.asarray(input_ids)
    if values.ndim != 1:
        raise ValueError(f"testppl candidate must be one-dimensional, got {values.shape}")
    bos_positions = np.flatnonzero(values == bos_token_id)
    eos_positions = np.flatnonzero(values == eos_token_id)
    if np.any(values == pad_token_id):
        raise ValueError("testppl candidate unexpectedly contains PAD")
    if len(bos_positions) > 1 or len(eos_positions) > 1:
        raise ValueError("testppl candidate contains duplicated BOS/EOS")
    if len(bos_positions) and (not first_in_document or int(bos_positions[0]) != 0):
        raise ValueError("BOS occurs outside the start of a document's first sentence")
    if len(eos_positions) and (
        not last_in_document or int(eos_positions[0]) != len(values) - 1
    ):
        raise ValueError("EOS occurs outside the end of a document's last sentence")

    pieces = []
    if first_in_document and not len(bos_positions):
        pieces.append(np.asarray([bos_token_id], dtype=values.dtype))
    pieces.append(values)
    if last_in_document and not len(eos_positions):
        pieces.append(np.asarray([eos_token_id], dtype=values.dtype))
    normalized = np.concatenate(pieces) if len(pieces) > 1 else values
    if first_in_document != bool(len(normalized) and int(normalized[0]) == bos_token_id):
        raise AssertionError("normalized BOS contract failed")
    if last_in_document != bool(len(normalized) and int(normalized[-1]) == eos_token_id):
        raise AssertionError("normalized EOS contract failed")
    return normalized


class TGPerplexityApproximationDataset(metaclass=abc.ABCMeta):
    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: str = None, # tg or tree
        model_ctx_len: int = 2048,
        split="validation",
        metric_type="sent",  # Override default metric type, whether be sent/doc
        generate_TG_attention_bias: Optional[TG_attention_bias] = None,
        vocab_path: str = None,
        device_eval_batch_size: int = 60,
        transformer_grammar_type: str = "",
        pause_token_id: int = None,
        normalize_document_boundaries: Optional[bool] = None,
        samples_per_sentence: int = 300,
        data_filename: Optional[str] = None,
        sentence_index_filename: Optional[str] = None,
        document_index_filename: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.dataset_path = dataset_path
        # Ordinary terminal checkpoints historically serialize this field as
        # null.  Direct document-PPL evaluation must accept that representation.
        self.transformer_grammar_type = transformer_grammar_type or ""

        # For tree_noont / tree_compress / tree_triplecnt, the underlying .npy data
        # is in tree format so we keep dataset_name as "tree" for loading.
        if transformer_grammar_type in ("tree_noont", "tree_compress", "tree_triplecnt"):
            self.dataset_name = "tree"
        else:
            self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.metric_type = metric_type
        self.log_instances = 2  # Set to > 0 to log the first few instances as a sanity check
        self.batch_size = device_eval_batch_size
        if samples_per_sentence <= 0:
            raise ValueError("samples_per_sentence must be positive")
        self.SENT_SIZE = int(samples_per_sentence)
        self.pause_token_id = pause_token_id
        self.normalize_document_boundaries = (
            metric_type == "doc"
            if normalize_document_boundaries is None
            else bool(normalize_document_boundaries)
        )

        self.samples: List[Dict[str, Any]] = []
        self.term_len: List[int] = []

        log.info(
                f"Starting loading {self.dataset_name}_approx_ppl dataset"
            )

        data_filename = data_filename or f"{self.dataset_name}_300.npy"
        sentence_index_filename = (
            sentence_index_filename or f"{self.dataset_name}_sent_index.npy"
        )
        document_index_filename = (
            document_index_filename or f"{self.dataset_name}_doc_index.npy"
        )
        self.dataset = np.load(
            os.path.join(self.dataset_path, data_filename), mmap_mode="r"
        )
        self.sent_index = torch.LongTensor(
            np.load(os.path.join(self.dataset_path, sentence_index_filename))
        )
        self.doc_index = torch.LongTensor(
            np.load(os.path.join(self.dataset_path, document_index_filename))
        )
        if len(self.sent_index) % self.SENT_SIZE:
            raise ValueError(
                f"{self.dataset_name}_sent_index.npy length must be divisible by "
                f"{self.SENT_SIZE}, got {len(self.sent_index)}"
            )
        document_counts = self.doc_index.numpy().astype(np.int64, copy=False)
        if not len(document_counts) or np.any(document_counts <= 0):
            raise ValueError("testppl document sentence counts must all be positive")
        sentence_count = len(self.sent_index) // self.SENT_SIZE
        if int(document_counts.sum(dtype=np.int64)) != sentence_count:
            raise ValueError(
                "testppl document index does not cover the sentence index: "
                f"{int(document_counts.sum(dtype=np.int64))} != {sentence_count}"
            )
        self.document_ends = np.cumsum(document_counts, dtype=np.int64)
        self.document_starts = np.concatenate(
            (np.zeros(1, dtype=np.int64), self.document_ends[:-1])
        )
        self.length = len(self.sent_index)
        self.generate_TG_attention_bias = generate_TG_attention_bias
        self.prep_examples()
        self.reset()
        log.info(f"Loading Dataset finished")

    def __getitem__(self, index):
        input_ids = self.dataset[self.sent_index[index]:self.sent_index[index+1]]
        if self.normalize_document_boundaries:
            sentence_index = int(index) // self.SENT_SIZE
            document_id = int(
                np.searchsorted(self.document_ends, sentence_index, side="right")
            )
            input_ids = normalize_testppl_document_record(
                input_ids,
                first_in_document=(
                    sentence_index == int(self.document_starts[document_id])
                ),
                last_in_document=(
                    sentence_index + 1 == int(self.document_ends[document_id])
                ),
                bos_token_id=self.vocab.bos,
                eos_token_id=self.vocab.eos,
                pad_token_id=self.vocab.pad,
            )
        input_ids = self._convert_sequence(input_ids)
        return {
            "sent_id" : index//self.SENT_SIZE + 1,
            # Flat 0-based index: unique per tree (unlike sent_id, which is shared
            # by all SENT_SIZE trees of a sentence). Used by the docppl metric to
            # scatter into a unique [n_sent, SENT_SIZE] slot: row=index//SENT_SIZE,
            # col=index%SENT_SIZE. Robust to unequal per-rank update counts.
            "index": index,
            "doc_id": self.sent_doc_id[index//self.SENT_SIZE],
            "input_ids": input_ids,
        }

    def _convert_sequence(self, input_ids):
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)
        if self.transformer_grammar_type == "tree_noont":
            input_ids = self.vocab.convert_treenpy_to_noont(input_ids)
        elif self.transformer_grammar_type == "tree_compress":
            input_ids = self.vocab.convert_treenpy_to_compress(input_ids)
        elif self.transformer_grammar_type == "tree_triplecnt":
            input_ids = self.vocab.convert_treenpy_to_triplecnt(input_ids)
        return input_ids

    def __len__(self):
        return self.length

    def get_term_length(self):
        return self.term_len

    def reset(self) -> None:
        self.cur_doc_id = 0
        self.sent_to_add = None
        self.num_evaled = 0

    def prep_examples(self):
        """Append doc_ids to each example so that they are processed together in the metric"""
        self.sent_index = torch.cat([torch.LongTensor([0]), torch.cumsum(self.sent_index, dim=0)])
        self.doc_index = torch.cumsum(self.doc_index, dim=0)
        self.sent_doc_id = torch.zeros((self.length // self.SENT_SIZE + 1), dtype=torch.int)
        self.term_len = [0] * (self.length // self.SENT_SIZE + 1)
        self.sent_doc_id[0] = 1
        self.sent_doc_id[self.doc_index] = 1
        self.sent_doc_id = torch.cumsum(self.sent_doc_id, dim=0)

        for i in range(1, len(self.term_len)):
            sent = self[self.SENT_SIZE * (i-1)]
            self.term_len[i] = int(
                sent.get(
                    "term_count",
                    sum(
                        self.vocab.is_terminal(token) or token == self.vocab.eos
                        for token in sent["input_ids"]
                    ),
                )
            )

    def pad_tokens_until_max(self, tokens, max_len=2048):
        """truncate from left if len(tokens) > model_ctx_len, max_len is not considered then
        queries are already truncated at max length of model_ctx_len
        this acts as additional check for all types of sequences in the batch
        """
        if len(tokens) > self.model_ctx_len:
            return tokens[-self.model_ctx_len :]
        else:
            # pad to max_len, but check again if this padding exceeded self.model_ctx_len
            # this time truncate from right side of the sequence because additional padding caused len(tokens) > self.model_ctx_len
            tokens = np.concatenate([tokens, [self.vocab.pad] * (max_len - len(tokens))], axis=0)

            if len(tokens) > self.model_ctx_len:
                tokens = tokens[: self.model_ctx_len]

            return tokens

    def collate_fn(self, data):
        # pad to max length
        if self.metric_type=="doc" and data[0]["doc_id"] > self.cur_doc_id:
            self.cur_doc_id = data[0]["doc_id"]
            if self.generate_TG_attention_bias is not None:
                self.generate_TG_attention_bias.reset_state()
        
        self.num_evaled += len(data)
        max_input_len = 0
        for sample in data:
            if len(sample["input_ids"]) > max_input_len:
                max_input_len = len(sample["input_ids"])

        sent_ids = []
        flat_indices = []
        input_ids = []
        all_attention_bias = []
        all_label_mask = []
        # pad according to max_lengths
        for sample in data:
            pad_shape = (
                0, (max_input_len - len(sample["input_ids"]))
            )
            sent_ids.append(sample["sent_id"])
            flat_indices.append(sample["index"])
            # make sure Gen TG bias have the correct length
            cur_input_id = torch.LongTensor(self.pad_tokens_until_max(sample["input_ids"], max_len=max_input_len))

            attention_bias, label_mask = None, None
            if self.generate_TG_attention_bias is not None:
                attention_bias, label_mask = self.generate_TG_attention_bias(cur_input_id)
            sample_label_mask = sample.get("label_mask")
            if sample_label_mask is not None:
                sample_label_mask = np.asarray(sample_label_mask, dtype=np.bool_)
                if len(sample_label_mask) > len(cur_input_id):
                    # pad_tokens_until_max truncates a too-long record from the
                    # left, so keep the matching suffix of its label mask.
                    sample_label_mask = sample_label_mask[-len(cur_input_id) :]
                elif len(sample_label_mask) < len(cur_input_id):
                    sample_label_mask = np.pad(
                        sample_label_mask,
                        (0, len(cur_input_id) - len(sample_label_mask)),
                        constant_values=False,
                    )
                sample_label_mask = torch.from_numpy(sample_label_mask)
                label_mask = (
                    sample_label_mask
                    if label_mask is None
                    else torch.bitwise_and(label_mask, sample_label_mask)
                )
            input_ids.append(cur_input_id)
            
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                all_attention_bias.append(attention_bias)

            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(label_mask)

        batch = {
            "doc_id": data[0]["doc_id"] if self.metric_type=="doc" else None,
            "sent_id": torch.LongTensor(sent_ids),
            "index": torch.LongTensor(flat_indices),
            "input_ids": torch.stack(input_ids),
        }
        if all_attention_bias:
            batch["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            batch["label_mask"] = torch.stack(all_label_mask)

        if self.metric_type=="doc":
            if self.num_evaled % self.SENT_SIZE == self.batch_size or self.batch_size == self.SENT_SIZE:
                # Make sure bias has the same length with kv cache, we must pass pad into GenBias
                self.sent_to_add = torch.LongTensor(self.pad_tokens_until_max(data[0]["input_ids"], max_len=max_input_len))
                batch["add_len"] = data[0]["input_ids"].shape[0]
            if self.num_evaled % self.SENT_SIZE == 0:
                if self.generate_TG_attention_bias is not None:
                    self.generate_TG_attention_bias(self.sent_to_add, True)
        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


class TerminalDocumentPerplexityDataset(TGPerplexityApproximationDataset):
    """One-path terminal projection for exact document-level PPL comparison."""

    def __init__(self, tokenizer: Tokenizer, dataset_path: str, **kwargs):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name="test",
            metric_type="doc",
            samples_per_sentence=1,
            data_filename="test.npy",
            sentence_index_filename="test_sent_index.npy",
            document_index_filename="test_doc_index.npy",
            **kwargs,
        )

    def prep_examples(self):
        """Prepare pause phase offsets before the base class indexes records."""
        self._pause_enabled = self.transformer_grammar_type[:5] == "pause"
        if self._pause_enabled:
            self._pause_p, self._pause_q = pause_spec_from_grammar_type(
                self.transformer_grammar_type
            )
            raw_lengths = self.sent_index.numpy().astype(np.int64, copy=False)
            raw_document_counts = self.doc_index.numpy().astype(np.int64, copy=False)
            self._pause_start_phase = np.zeros(len(raw_lengths), dtype=np.int64)
            sentence_id = 0
            for document_count in raw_document_counts:
                phase = 0
                for _ in range(int(document_count)):
                    self._pause_start_phase[sentence_id] = phase
                    phase = (phase + int(raw_lengths[sentence_id])) % self._pause_q
                    sentence_id += 1
        super().prep_examples()

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        input_ids = np.asarray(sample["input_ids"])
        sample["term_count"] = sum(
            self.vocab.is_terminal(token) or token == self.vocab.eos
            for token in input_ids
        )
        if not self._pause_enabled:
            return sample

        phase = int(self._pause_start_phase[int(index)])
        expanded = []
        label_mask = []
        for raw_token in input_ids:
            token = int(raw_token)
            expanded.append(token)
            label_mask.append(True)
            phase = (phase + 1) % self._pause_q
            if phase == 0:
                pause_value = token if self.pause_token_id is None else self.pause_token_id
                expanded.extend([pause_value] * self._pause_p)
                label_mask.extend([False] * self._pause_p)
        sample["input_ids"] = np.asarray(expanded, dtype=input_ids.dtype)
        sample["label_mask"] = np.asarray(label_mask, dtype=np.bool_)
        return sample


formula_dict = {
    'center_embed': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    'center_embed_mod': ['[ (%plaus%) ] < [ (%implaus%) ]'],
    
    'cleft': ['[ (%np_mismatch%) - (%np_match%) ] + [ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    'cleft_modifier': ['[ (%np_mismatch%) - (%np_match%) ]+[ [ (%vp_mismatch%) ] - [ (%vp_match%) ] ]>0'],
    
    'fgd_subject': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_object': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_pp': ['[ (%what_nogap%) > (%that_nogap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd-embed3': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd-embed4': ['[ (%what_no-gap%) > (%that_no-gap%) ] & [ (%what_gap%) < (%that_gap%) ]'],
    'fgd_hierarchy': ['[ (%what_nogap%) > (%that_nogap%)] & [ (%what_subjgap%) <  (%that_subjgap%) ]', '[ (%what_nogap%) = (%that_nogap%) ] & [ (%what_subjgap%) = (%that_subjgap%) ]'],   
    #TODO: why two formulas in fgd_hierarchy? we use the first formula
    
    'mvrr': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%) ] > [ (%reduced_unambig%) - (%unreduced_unambig%) ] ]'],
    'mvrr_mod': ['[ (%reduced_ambig%) > (%unreduced_ambig%) ] & [ (%reduced_ambig%) > (%reduced_unambig%) ] & [ [ (%reduced_ambig%) - (%unreduced_ambig%)] > [(%reduced_unambig%) - (%unreduced_unambig%)] ]'],
    
    # nn-nv-rpl is intentionally unreachable at eval time: it is excluded from
    # SGDataset.task_list and test_suite_dict, so no task ever looks it up here.
    # This is a SG branch that should NOT be evaluated — kept only as a reference
    # of the formula. Do not wire it in without an explicit decision to score it.
    'nn-nv-rpl': ['(%nn_ambig%)>(%nn_unambig%)', '(%nv_ambig%)>(%nv_unambig%)'],
    
    'npi_orc_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_orc_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_any': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'],
    'npi_src_ever': ['[ (%neg_pos%) < (%pos_pos%) ] & [ (%neg_neg%) < (%pos_neg%) ] & [ (%neg_pos%) < (%pos_neg%) ]'], 
    
    'npz_ambig': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_ambig_mod': ['[ (%ambig_nocomma%) > (%ambig_comma%) ] &  [ (%ambig_nocomma%) > (%unambig_nocomma%) ]  & [ [ (%ambig_nocomma%) - (%ambig_comma%) ] > [ (%unambig_nocomma%) - (%unambig_comma%) ] ]'],
    'npz_obj': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    'npz_obj_mod': ['[ (%no-obj_no-comma%) > (%no-obj_comma%) ] &  [ (%no-obj_no-comma%) > (%obj_no-comma%) ] & [ [ (%no-obj_no-comma%) - (%no-obj_comma%) ] > [ (%obj_no-comma%) - (%obj_comma%) ] ]'],
    
    'number_orc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_prep': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'number_src': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'reflexive_orc_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_orc_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_prep_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_prep_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    'reflexive_src_fem': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'], 
    'reflexive_src_masc': ['[ (%match_sing%) < (%mismatch_sing%) ] & [ (%match_plural%) < (%mismatch_plural%) ]'],
    
    'subordination': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_orc-orc': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_pp-pp': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]'], 
    'subordination_src-src': ['[ (%sub_no-matrix%) > (%no-sub_no-matrix%) ] & [ (%sub_matrix%) < (%no-sub_matrix%) ]']
}
test_suite_dict = {
    "Agreement" : ["number_orc", "number_prep", "number_src"], 
    "Center_Embedding" : ["center_embed", "center_embed_mod"],
    "Garden_Path_Effects" : ["mvrr", "mvrr_mod", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod"],
    "Gross_Syntactic_Expectation" : ["subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src"],
    "Licensing" : ["npi_orc_any", "npi_orc_ever", "npi_src_any", "npi_src_ever", \
            "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc"],
    "Long_Distance_Dependencies" : ["fgd_subject", "fgd_object", "fgd_pp", "fgd-embed3", "fgd-embed4", "fgd_hierarchy", "cleft", "cleft_modifier"],
    # "nn-nv-rpl" : ["nn-nv-rpl"] # extra test in SG but not in test suites
}

class SyntacticGeneralizationMetric(Metric):
    def __init__(
            self,
            metric_type="syntactic_generation",
            tree_eval_type="default",
        ) -> None:
        # sync_on_compute=False: compute() explicitly all-reduces fixed-size
        # correct/count statistics, avoiding both unequal-count deadlocks and
        # object-gathered tensors that retain different CUDA devices.
        super().__init__(sync_on_compute=False)

        self.metric_type = metric_type
        self.tree_eval_type = tree_eval_type
        self.map_task_dict = {}
        for key in test_suite_dict:
            # Synchronization is performed explicitly in compute() by reducing
            # fixed-size sufficient statistics. Keeping these as rank-local
            # lists avoids object-gathering CUDA tensors from different devices.
            self.add_state(key, default=[], dist_reduce_fx=None)
            for task in test_suite_dict[key]:
                self.map_task_dict[task] = key

    def reset(self):
        for key in test_suite_dict:
            setattr(self, key, [])
    
    def update(self, task, score_dict):
        '''
        input: task, condition probability variables, then eval with formula
        '''
        # Default logging suppresses INFO records from nonzero ranks. Use a
        # dedicated level above INFO so every rank's disjoint SG task subset
        # is visible without enabling all-rank noise. It must differ from the
        # XSum level because Python logging assigns one name per numeric level.
        log.log(
            SG_SCORE_LOG_LEVEL,
            f"[global_rank={get_global_rank()}] task is {task} score is {score_dict}",
        )
        formula = formula_dict[task][0]
        keys = re.findall(r"%([\w|-]+)%", formula)
        keys = set(keys)
        for key in keys:
            # Coerce to float and emit a literal eval() can always parse.
            # str(float('inf')) is the bare token `inf`, which is NOT a valid
            # Python name → NameError. That crashed the whole eval run when
            # bf16 underflow gave a tagged token probability 0 (CE = inf).
            # float('inf') / float('nan') are valid expressions, and the
            # comparisons behave correctly (inf is "very large surprise" →
            # fails a `<` test, which is the intended "model got it wrong").
            val = float(score_dict[key])
            if math.isfinite(val):
                literal = repr(val)
            elif math.isnan(val):
                literal = "float('nan')"
            elif val > 0:
                literal = "float('inf')"
            else:
                literal = "float('-inf')"
            formula = formula.replace("(%{}%)".format(key), "({})".format(literal))
        formula = formula.replace("[", "(")
        formula = formula.replace("]", ")")

        result = eval(formula)
        log.info(f"result is {result}")
        # Pin bool scalars to CPU: _gather_list (all_gather_object) does not move
        # tensors across devices, so per-rank tensors keep their rank's device.
        # compute() then does sum() over the gathered list, which raises
        # "Expected all tensors to be on the same device" when ranks span
        # multiple GPUs. Booleans are CPU-native, so keeping them on the host
        # sidesteps the cross-device sum and costs nothing.
        getattr(self, self.map_task_dict[task]).append(torch.tensor(result, dtype=torch.bool, device="cpu"))

    def compute(self) -> Dict[str, float]:
        # Reduce only additive sufficient statistics. all_gather_object would
        # preserve each serialized CUDA tensor's source device, producing a
        # mixed-device list (cuda:0, cuda:1, ...) that fails in sum(). A fixed
        # [num_suites, 2] tensor stays on the current rank's metric device and
        # also tolerates unequal update counts across ranks.
        suite_names = list(test_suite_dict)
        stats = torch.zeros((len(suite_names), 2), dtype=torch.long, device=self.device)
        for idx, key in enumerate(suite_names):
            values = getattr(self, key)
            if values:
                stats[idx, 0] = torch.stack(values).to(dtype=torch.long).sum()
                stats[idx, 1] = len(values)
        stats = _all_reduce_tensor(stats)

        acc_dict = {}
        avg_acc = 0.0
        for idx, key in enumerate(suite_names):
            correct = int(stats[idx, 0].item())
            count = int(stats[idx, 1].item())
            acc_dict[key] = correct / count if count else 0.0
            # nn-nv-rpl is excluded from test_suite_dict (see SGDataset.task_list),
            # so every key here contributes to the average.
            avg_acc += acc_dict[key]
        acc_dict["avg"] = avg_acc / len(test_suite_dict)
        return acc_dict

class SGDataset(metaclass=abc.ABCMeta):
    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split="validation",
        metric_type="SG",
        vocab_path: str = None,
        transformer_grammar_type: str = "",
        samples_per_sent: int = 300,
        tree_eval_type: str = "default",
        pause_token_id: int = None,
        **kwargs
    ):
        self.task_list = ["center_embed", "center_embed_mod", "cleft", "cleft_modifier", "fgd_subject", "fgd_object", "fgd_pp", "fgd-embed3", \
            "fgd-embed4", "fgd_hierarchy", "mvrr", "mvrr_mod", "npi_orc_any", "npi_orc_ever", "npi_src_any", \
            "npi_src_ever", "npz_ambig", "npz_ambig_mod", "npz_obj", "npz_obj_mod", "number_orc", "number_prep", "number_src", \
            "reflexive_orc_fem", "reflexive_orc_masc", "reflexive_prep_fem", "reflexive_prep_masc", "reflexive_src_fem", "reflexive_src_masc", \
            "subordination", "subordination_orc-orc", "subordination_pp-pp", "subordination_src-src",
            # "nn-nv-rpl" don't include this test
        ]
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.metric_type = metric_type
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.vocab_path = vocab_path
        self.transformer_grammar_type = transformer_grammar_type
        self.samples_per_sent = samples_per_sent
        self.tree_eval_type = tree_eval_type
        self.pause_token_id = pause_token_id
        self.sg_nc_ratio = kwargs.get("sg_nc_ratio", 1.0)
        self.sg_pc = kwargs.get("sg_pc", 3)
        # Qwen3 tokenizer uses different tokenization → load Qwen3-aligned tags.
        self.is_qwen3 = "qwen3" in (vocab_path or "").lower()
        if self.is_qwen3:
            self.dataset_path = os.path.join(self.dataset_path, "qwen3")
        self.prep_examples()

    @property
    def pause_spec(self) -> "tuple[int, int]":
        """Rational pause spec ``(p, q)`` parsed from ``transformer_grammar_type``.

        ``(0, 1)`` means no pauses (non-pause grammar types).
        """
        return pause_spec_from_grammar_type(self.transformer_grammar_type)

    @property
    def ispause(self) -> int:
        """Number of pause tokens per block (``p``); 0 means no pauses.

        Used as a truthy flag (``if self.ispause:``). For the rational spec
        ``(p, q)``, pass ``self.transformer_grammar_type`` (not this int) as
        ``pause_num`` to :func:`pause_input_ids`.
        """
        return self.pause_spec[0]

    def prep_examples(self):
        self.samples : List[List[Dict[str, List]]] = []
        is_gpt2 = "gpt2" in (self.vocab_path or "").lower()
        # Resume support: two env vars (SG_RESUME_TASKS takes precedence).
        #   SG_RESUME_TASKS=task_a,task_b  -> keep ONLY those tasks (all cases).
        #     Used to re-run specific tasks whose cases were split across ranks
        #     and partially missed by a from-offset resume.
        #   SG_RESUME_FROM_TASK=<task>:<case_offset> -> drop every case before
        #     <task>, and the first <case_offset> cases within <task>.
        #     Example: subordination_pp-pp:7 keeps pp-pp cases 7..end + every
        #     task after it (subordination_src-src).
        _resume_tasks = [t.strip() for t in os.environ.get("SG_RESUME_TASKS", "").split(",") if t.strip()]
        _resume = os.environ.get("SG_RESUME_FROM_TASK", "").strip()
        _resume_task = None
        _resume_offset = 0
        if _resume:
            if ":" in _resume:
                _resume_task, _off = _resume.split(":", 1)
                _resume_offset = int(_off)
            else:
                _resume_task, _resume_offset = _resume, 0
        if _resume_tasks:
            # Whitelist takes full precedence: ignore SG_RESUME_FROM_TASK entirely
            # (it may be leaked in the environment from a previous resume run).
            log.info(f"[SG resume] task whitelist: {_resume_tasks} (ignoring SG_RESUME_FROM_TASK={_resume!r})")
            _resume_task = None
            _resume_offset = 0
        elif _resume_task is not None:
            log.info(f"[SG resume] dropping cases before task '{_resume_task}' (offset {_resume_offset})")
        for task in self.task_list:
            if _resume_tasks:
                if task not in _resume_tasks:
                    continue
            elif _resume_task is not None and task != _resume_task and _resume_task in self.task_list:
                # still before the resume task -> skip entirely
                if self.task_list.index(task) < self.task_list.index(_resume_task):
                    continue
            # if task not in test_suite_dict["Agreement"]: continue
            with open(os.path.join(self.dataset_path, task+".json"), 'r', encoding='utf-8') as file:
                dataset = json.load(file)
                for case_idx, case in enumerate(dataset["data"]):
                    if _resume_task is not None and task == _resume_task and case_idx < _resume_offset:
                        continue
                    cur = []
                    for sent in case:
                        sent["task"] = task
                        tokens = self.tokenizer.encode(" " + sent["input"], add_special_tokens=False)
                        # GPT-2 was trained with a leading BOS; Qwen3 was not. Prepend BOS to the
                        # input iff the tokenizer matches the training convention, and pad the tag
                        # with a leading 0 in lockstep so tag and input_ids always share one length
                        # and one coordinate system (SG_eval_step then unconditionally does tag[1:]).
                        if is_gpt2:
                            sent["input_ids"] = torch.LongTensor([self.vocab.bos] + tokens).unsqueeze(0)
                            sent["tag"][0] = [0] + sent["tag"][0]
                        else:
                            sent["input_ids"] = torch.LongTensor(tokens).unsqueeze(0)
                        start = end = -1
                        for i,x in enumerate(sent["tag"][0]):
                            if x == 1:
                                end = i
                                if start == -1:
                                    start = i - 1
                        # tag_start/tag_end index into the same (BOS-aware) coordinate system as
                        # input_ids, so no bos_offset shift is needed.
                        sent["tag_start"] = start
                        sent["tag_end"] = end
                        assert(sum(sent['tag'][0]) == end-start)
                        if self.ispause:
                            # pause_input_ids expands input and tag by the same (p, q) factor, so
                            # they stay equal in length; SG_eval_step's tag[1:] still aligns.
                            sent["tag"][0] = pause_input_ids(sent["tag"][0], pause_token_id=None, pause_num=self.transformer_grammar_type)
                            sent_paused = pause_input_ids(sent["input_ids"][0], self.pause_token_id, pause_num=self.transformer_grammar_type)
                            sent["input_ids"] = sent_paused.unsqueeze(0)
                        if sent["condition_name"] in formula_dict[task][0]:
                            cur.append(sent)
                    self.samples.append(cur)
    
    def __getitem__(self, index):
        return self.samples[index]
    
    def reset(self) -> None:
        return
    
    def __len__(self):
        return len(self.samples)
    
    def collate_fn(self, data):
        return data[0]


BLiMP_TASK_ANAPHOR_AGR = ["anaphor_gender_agreement", "anaphor_number_agreement"]
BLiMP_TASK_ARG_STRUCTURE = ["animate_subject_passive", "animate_subject_trans", "causative",
                            "drop_argument", "inchoative", "intransitive", "passive_1", "passive_2", "transitive"]
BLiMP_TASK_BINDING = ["principle_A_c_command", "principle_A_case_1", "principle_A_case_2",
                      "principle_A_domain_1", "principle_A_domain_2", "principle_A_domain_3",
                      "principle_A_reconstruction"]
BLiMP_TASK_CONTROL_RAISING = ["existential_there_object_raising", "existential_there_subject_raising",
                              "expletive_it_object_raising", "tough_vs_raising_1", "tough_vs_raising_2"]
BLiMP_TASK_DET_NOUN_AGR = ["determiner_noun_agreement_1", "determiner_noun_agreement_2",
                           "determiner_noun_agreement_irregular_1", "determiner_noun_agreement_irregular_2",
                           "determiner_noun_agreement_with_adj_2", "determiner_noun_agreement_with_adj_irregular_1",
                           "determiner_noun_agreement_with_adj_irregular_2", "determiner_noun_agreement_with_adjective_1"]
BLiMP_TASK_ELLIPSIS = ["ellipsis_n_bar_1", "ellipsis_n_bar_2"]
BLiMP_TASK_FILLER_GAP = ["wh_questions_object_gap", "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance", 
                         "wh_vs_that_no_gap", "wh_vs_that_no_gap_long_distance", "wh_vs_that_with_gap", 
                         "wh_vs_that_with_gap_long_distance"]
BLiMP_TASK_IRREGULAR_FORMS = ["irregular_past_participle_adjectives", "irregular_past_participle_verbs"]
BLiMP_TASK_ISLAND_EFFECTS = ["adjunct_island", "complex_NP_island", "coordinate_structure_constraint_complex_left_branch",
                             "coordinate_structure_constraint_object_extraction", "left_branch_island_echo_question",
                             "left_branch_island_simple_question", "sentential_subject_island", "wh_island"]
BLiMP_TASK_NPI_LICENSING = ["matrix_question_npi_licensor_present", "npi_present_1", "npi_present_2",
                            "only_npi_licensor_present", "only_npi_scope", "sentential_negation_npi_licensor_present",
                            "sentential_negation_npi_scope"]
BLiMP_TASK_QUANTIFIERS = ["existential_there_quantifiers_1", "existential_there_quantifiers_2",
                          "superlative_quantifiers_1", "superlative_quantifiers_2"]
BLiMP_TASK_SUBJECT_VERB_AGR = ["distractor_agreement_relational_noun", "distractor_agreement_relative_clause",
                               "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2",
                               "regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2"]
BLiMP_TASK_DICT = {
    "anaphor_agreement" : BLiMP_TASK_ANAPHOR_AGR,
    "argument_structure" : BLiMP_TASK_ARG_STRUCTURE,
    "binding" : BLiMP_TASK_BINDING,
    "control_raising" : BLiMP_TASK_CONTROL_RAISING,
    "determiner_noun_agreement" : BLiMP_TASK_DET_NOUN_AGR,
    "ellipsis" : BLiMP_TASK_ELLIPSIS,
    "filler_gap_dependency" : BLiMP_TASK_FILLER_GAP,
    "irregular_forms" : BLiMP_TASK_IRREGULAR_FORMS,
    "island_effects" : BLiMP_TASK_ISLAND_EFFECTS,
    "npi_licensing" : BLiMP_TASK_NPI_LICENSING,
    "quantifiers" : BLiMP_TASK_QUANTIFIERS,
    "subject_verb_agreement" : BLiMP_TASK_SUBJECT_VERB_AGR, 
}
BLiMP_TASK_LIST = [x for v in BLiMP_TASK_DICT.values() for x in v]


class BLiMPMetric(Metric):
    full_state_update: bool = False
    
    def __init__(
            self,
            metric_type="BLiMP",
            dataset_name: Optional[str] = None, # explicit terminal, tree_300, tg_300, or tree_300_qwen
            vocab_path = None,
            device_eval_batch_size = None,
            dataset_length = None,
            samples_per_sent = 300,
            pair_per_task = 1000,
            tree_eval_type = "default",
            save_beam_trees_path: Optional[str] = None,
        ) -> None:
        # sync_on_compute=False: compute() SUM all-reduces the fixed-size
        # loglikelihoods tensor via _all_reduce_tensor (count-insensitive: each
        # rank wrote its own disjoint sent_id slots). Avoids the unequal-count
        # deadlock under DistributedEvalSampler when N % world_size != 0.
        super().__init__(sync_on_compute=False)

        self.metric_type = metric_type
        self.task_dict = BLiMP_TASK_DICT
        self.task_list = BLiMP_TASK_LIST
        self.pair_per_task = pair_per_task
        self.device_eval_batch_size = device_eval_batch_size
        self.dataset_length = dataset_length
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.tree_eval_type = tree_eval_type
        # Optional beam-tree dump (env-gated OLMO_BEAM_DUMP=1 in build_downstream_evaluator).
        # Trainer.BLiMP_beam_eval_step calls record_beams() per sentence with the decoded
        # bracketed tree strings + logprobs; compute() writes the accumulated list to JSON.
        self.save_beam_trees_path = save_beam_trees_path
        self._beam_records: List[Dict[str, Any]] = []

        if dataset_name[:8] == "terminal" or dataset_name[:5] == "pause" or samples_per_sent==1:
            self.SENT_SIZE = 1
            self.add_state("loglikelihoods", default=torch.zeros((dataset_length), dtype=torch.float32), dist_reduce_fx="sum")
        else:
            self.SENT_SIZE = samples_per_sent
            self.add_state("loglikelihoods", default=torch.zeros((dataset_length//self.SENT_SIZE, self.SENT_SIZE), dtype=torch.float32), dist_reduce_fx="sum")

    def reset(self):
        if self.SENT_SIZE == 1:
            self.loglikelihoods = torch.zeros((self.dataset_length), dtype=torch.float32, device=self.device)
        else:
            self.loglikelihoods = torch.zeros((self.dataset_length//self.SENT_SIZE, self.SENT_SIZE), dtype=torch.float32, device=self.device)

    def _get_terminal_mask(self, labels):
        """Return float mask: 1.0 for terminal tokens, 0.0 for non-terminals."""
        mask = (self.vocab.opening_non_terminals[0] > labels) | (labels > self.vocab.closing_non_terminals[1])
        return mask.float()

    def update(self, batch: Dict[str, Any], lm_logits:torch.Tensor):
        logits_for_loss = lm_logits[..., :-1, :].to(dtype=torch.float32).contiguous()
        # tokenizer = Tokenizer.from_file("/home/wangpch/TG-Interpolation/dataset/bbc-news/TG_GPT2_tokenizer.json")
        # print(self.device)
        
        # print_tensor_data(batch["input_ids"])
        # print_tensor_data(batch["attention_bias"])
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels, label_mask, attention_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, self.vocab.pad)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, self.vocab.pad)
        labels = labels[..., 1:].contiguous()
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss = F.cross_entropy(
            logits_for_loss, labels, ignore_index=self.vocab.pad, reduction="none")

        if self.tree_eval_type == "terminal":
            ce_loss = ce_loss * self._get_terminal_mask(labels)

        # for sent_id, loglikelihood in zip(batch["sent_id"], ce_loss):
        sample_id = batch["sent_id"] % self.SENT_SIZE
        sent_id = batch["sent_id"] // self.SENT_SIZE


        ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1).sum(dim=1)
        actual_batch = ce_loss.shape[0]
        flat_sent_ids = batch.get("sent_ids")
        if flat_sent_ids is not None:
            flat_sent_ids = flat_sent_ids.to(device=self.device, dtype=torch.long)
            if self.SENT_SIZE == 1:
                self.loglikelihoods[flat_sent_ids] = ce_loss
            else:
                self.loglikelihoods[
                    torch.div(flat_sent_ids, self.SENT_SIZE, rounding_mode="floor"),
                    flat_sent_ids.remainder(self.SENT_SIZE),
                ] = ce_loss
        elif self.SENT_SIZE==1:
            self.loglikelihoods[sent_id : sent_id + actual_batch] = ce_loss
        else:
            self.loglikelihoods[sent_id, sample_id : sample_id + actual_batch] = ce_loss
        
        # self.loglikelihoods[self.cur_sent, self.cur_batch:self.cur_batch + self.device_eval_batch_size] = ce_loss
        # self.cur_batch += self.device_eval_batch_size
        # if self.cur_batch == self.SENT_SIZE:
        #    self.cur_batch = 0
        #    self.cur_sent += 1

    def update_beam(self, batch: Dict[str, Any], log_likelihoods: torch.Tensor):
        """Scatter precomputed (beam-search) per-sentence log-likelihoods.

        Used by ``Trainer.BLiMP_beam_eval_step``: instead of deriving CE from
        teacher-forced logits (``update``), the beam path calls
        ``OLMo.word_sync_beam_search`` and passes the resulting
        ``logsumexp(beam logprob)`` here.

        Sign convention: the ``loglikelihoods`` slot holds a *loss* (+CE for the
        teacher-forcing path), and ``compute()`` negates it to get a log-prob
        (``loglikelihoods = -self.loglikelihoods`` for SENT_SIZE==1). The beam
        quantity is already a log-prob, so we store ``-LL`` here so that
        ``compute()``'s negation yields ``+LL`` — matching the teacher-forcing
        convention and keeping ``compute()`` unchanged.

        Args:
            batch: must contain ``sent_id`` (scalar or 0-d tensor when
                ``device_eval_batch_size==1``; the BLiMP collate_fn sets it to
                ``min(sent_ids)``).
            log_likelihoods: ``(B,)`` tensor of per-sentence log-probs
                (higher = more probable).
        """
        sid = batch["sent_id"]
        sid = int(sid) if torch.is_tensor(sid) else sid
        sample_id = sid % self.SENT_SIZE
        sent_id = sid // self.SENT_SIZE
        ll = log_likelihoods.to(self.device).to(self.loglikelihoods.dtype).neg()  # store -LL
        B = ll.shape[0]
        if self.SENT_SIZE == 1:
            self.loglikelihoods[sent_id : sent_id + B] = ll
        else:
            self.loglikelihoods[sent_id, sample_id : sample_id + B] = ll


    def record_beams(self, sent_id, task_name, pair_id, is_bad, terminal_str, beams, topk=5):
        """Append a per-sentence beam-tree record for offline dump.

        Called by ``Trainer.BLiMP_beam_eval_step`` only when
        ``save_beam_trees_path`` is set. All args are JSON-serializable: the
        caller decodes token ids to strings via the HF tokenizer.

        Args:
            sent_id: flat dataset index (even=good, odd=bad within a task).
            task_name: BLiMP task name (e.g. ``anaphor_gender_agreement``).
            pair_id: minimal-pair index within the task.
            is_bad: True for the ungrammatical member of the pair.
            terminal_str: decoded terminal sentence (NT tokens stripped).
            beams: list of beam dicts (``input_ids``, ``logprob``,
                ``terminal_logprob``) from ``OLMo.word_sync_beam_search``.
                Already decoded to ``tree`` strings by the caller.
            topk: number of top beams (by logprob) to record.
        """
        if self.save_beam_trees_path is None:
            return
        # `beams` entries carry a pre-decoded ``tree`` string + scalar logprobs.
        top = sorted(beams, key=lambda b: b.get("logprob", float("-inf")), reverse=True)[:topk]
        self._beam_records.append({
            "sent_id": int(sent_id),
            "task": task_name,
            "pair_id": int(pair_id),
            "good_bad": "bad" if is_bad else "good",
            "terminal": terminal_str,
            "beams": [
                {
                    "tree": b["tree"],
                    "logprob": float(b.get("logprob", 0.0)),
                    "terminal_logprob": float(b.get("terminal_logprob", 0.0)),
                }
                for b in top
            ],
        })


    def _save_beam_trees(self):
        """Write accumulated beam-tree records to JSON (per-rank files).

        Each rank scored its own disjoint subset of sentences
        (DistributedEvalSampler), so each rank writes its own
        ``<base>_rank{R}.jsonl`` file. The offline comparison script globs all
        ``_rank*.jsonl`` files to aggregate. Under single-GPU (world_size==1)
        this still writes ``_rank0.jsonl``.
        """
        if self.save_beam_trees_path is None:
            return
        rank = get_global_rank()
        base = self.save_beam_trees_path
        # Insert _rank{R} before a trailing .jsonl (else append _rank{R}.jsonl).
        if base.endswith(".jsonl"):
            out_path = f"{base[:-6]}_rank{rank}.jsonl"
        else:
            out_path = f"{base}_rank{rank}.jsonl"
        save_dir = os.path.dirname(out_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(self._beam_records, f)
        log.info(f"Saved {len(self._beam_records)} beam-tree records to {out_path}")


    def compute(self) -> torch.Tensor:
        # SUM all-reduce the fixed-size loglikelihoods tensor across ranks
        # (count-insensitive: each rank wrote its own disjoint sent_id slots).
        # Required under multi-GPU DistributedEvalSampler; no-op single-device.
        self.loglikelihoods = _all_reduce_tensor(self.loglikelihoods)
        cnt_dict = {}
        if self.SENT_SIZE!=1:
            if self.tree_eval_type == "terminal":
                # Average terminal log-probability across K parse trees.
                # loglikelihoods stores per-tree CE losses (terminal-only when
                # tree_eval_type=="terminal").  -mean(CE) = mean(log p).
                loglikelihoods = -self.loglikelihoods.mean(dim=1)
            else:
                # Tree-marginalized probability: log(Σ p(x, y_k)).
                loglikelihoods = torch.logsumexp(-self.loglikelihoods, dim=1)
        else:
            loglikelihoods = -self.loglikelihoods
        
        expected = len(self.task_list) * self.pair_per_task * 2
        if loglikelihoods.numel() != expected:
            raise RuntimeError(
                f"BLiMP score tensor has {loglikelihoods.numel()} sentences; "
                f"expected {expected}"
            )
        pairs = loglikelihoods.reshape(
            len(self.task_list), self.pair_per_task, 2
        )
        good = pairs[..., 0]
        bad = pairs[..., 1]
        good_wins = good > bad
        bad_wins = bad > good
        unresolved = ~(good_wins | bad_wins)  # ties and NaNs
        unresolved_count = int(unresolved.sum().item())
        if unresolved_count:
            log.warning(
                "BLiMP contains %d tied or non-finite minimal pairs",
                unresolved_count,
            )
        task_counts = good_wins.sum(dim=1).to(device="cpu").tolist()
        cnt_dict = {
            task: int(task_counts[task_id])
            for task_id, task in enumerate(self.task_list)
        }

        acc_dict = {}
        total_cnt = 0
        for term, term_task_list in self.task_dict.items():
            term_cnt = 0
            for task in term_task_list: 
                acc_dict[term + '/' + task] = cnt_dict[task] / self.pair_per_task
                term_cnt += cnt_dict[task]
            acc_dict[term + '/overall'] = term_cnt / (self.pair_per_task * len(term_task_list))
            total_cnt += term_cnt

        acc_dict['overall/overall'] = total_cnt / (self.pair_per_task * len(self.task_list))

        self._save_beam_trees()

        return acc_dict


class BLiMPApproximationDataset(metaclass=abc.ABCMeta):
    metric_type: str

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Optional[str] = None, # explicit terminal, tree_300, tg_300, or tree_300_qwen
        model_ctx_len: int = 2048,
        split="test",
        metric_type="BLiMP",
        generate_TG_attention_bias: Optional[Callable | str] = None,
        transformer_grammar_type: str = "",
        vocab_path: str = None,
        device_eval_batch_size: int = 60,
        samples_per_sent: int = 300,
        pair_per_task: int = 1000,
        pause_token_id: int = None,
        tree_eval_type: str = "default",
        force_terminal: bool = False,
        pushdown_gold: bool = False,
        parse_binarize_direction: str = "right",
    ):

        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.vocab_path = vocab_path
        self.dataset_path = dataset_path
        self.metric_type = metric_type
        self.task_list = BLiMP_TASK_LIST
        self.batch_size = device_eval_batch_size
        self.pair_per_task = pair_per_task
        self.transformer_grammar_type = transformer_grammar_type
        self.pause_token_id = pause_token_id
        self.tree_eval_type = tree_eval_type
        self.force_terminal = force_terminal
        self.pushdown_gold = pushdown_gold
        self.parse_binarize_direction = parse_binarize_direction
        if force_terminal and pushdown_gold:
            raise ValueError("force_terminal and pushdown_gold are mutually exclusive")
        self._tree_vocab = (
            TreeVocab.from_tokenizer_file(vocab_path) if pushdown_gold else None
        )

        self.is_qwen3 = "qwen3" in (vocab_path or "").lower()

        if pushdown_gold:
            self.SENT_SIZE = samples_per_sent
        elif force_terminal:
            # Beam-search path: always load the terminal-only data and score one
            # sequence per sentence (the model generates its own NT structure
            # during word_sync_beam_search; feeding a fixed tree/tg sequence would
            # defeat the parse-marginalization). Overrides grammar-type selection.
            self.SENT_SIZE = 1
        elif self.is_qwen3:
            if transformer_grammar_type[:8] == "terminal" or transformer_grammar_type[:5] == "pause":
                # Qwen3 terminal/pause: load tree_300 data, convert to terminal/pause
                # via _convert_sequence.  Only 1 sample per sentence (all 300 trees
                # convert to identical terminal sequences).  __getitem__ remaps flat
                # indices to the first tree of each sentence group.
                self.SENT_SIZE = 1
            else:
                self.SENT_SIZE = samples_per_sent
        elif transformer_grammar_type[:8] == "terminal" or transformer_grammar_type[:5] == "pause":
            self.SENT_SIZE = 1
        else:
            self.SENT_SIZE = samples_per_sent
        self.TASK_SIZE = 2 * pair_per_task * self.SENT_SIZE
        self.length = len(self.task_list) * self.TASK_SIZE

        self.samples: List[Dict[str, Any]] = []
        if pushdown_gold:
            self.dataset_name = "tree_300"
        elif force_terminal:
            # Qwen checkpoints have a different vocabulary and therefore need
            # their own terminal-tokenized BLiMP array. Reusing the GPT-2 file
            # would silently score the wrong IDs.
            self.dataset_name = "terminal_qwen" if self.is_qwen3 else "terminal"
        elif transformer_grammar_type[:8] == "terminal" or transformer_grammar_type[:5] == "pause":
            self.dataset_name = "terminal"
        elif transformer_grammar_type[:4] == "tree":
            self.dataset_name = "tree_300"
        else:
            self.dataset_name = "tg_300"
        # Qwen3: all non-beam grammar types load from tree_300_qwen (tree
        # format), then _convert_sequence handles format-specific conversion.
        # The beam path uses the compact terminal_qwen array selected above.
        if self.is_qwen3 and not force_terminal:
            self.dataset_name = "tree_300_qwen"
        if dataset_name is not None:
            self.dataset_name = dataset_name
        self.dataset_file = os.path.join(self.dataset_path, f"blimp_{self.dataset_name}.npy")
        self.dataset = np.load(self.dataset_file, mmap_mode="r")
        self.input_len = self.dataset.shape[1]
        self.generate_TG_attention_bias = generate_TG_attention_bias

        self.prep_examples()
        self.reset()
        log.info(f"Loading Dataset finished")

    @property
    def pause_spec(self) -> "tuple[int, int]":
        """Rational pause spec ``(p, q)`` parsed from ``transformer_grammar_type``.

        ``(0, 1)`` means no pauses (non-pause grammar types).
        """
        return pause_spec_from_grammar_type(self.transformer_grammar_type)

    @property
    def ispause(self) -> int:
        """Number of pause tokens per block (``p``); 0 means no pauses.

        Used as a truthy flag (``if self.ispause:``). For the rational spec
        ``(p, q)``, pass ``self.transformer_grammar_type`` (not this int) as
        ``pause_num`` to :func:`pause_input_ids`.
        """
        return self.pause_spec[0]

    def prep_examples(self):
        return

    def _convert_sequence(self, input_ids):
        if not isinstance(input_ids, np.ndarray):
            input_ids = np.array(input_ids)
        if self.is_qwen3 and self.transformer_grammar_type in ("tg", "tgtree"):
            # Qwen3 data is tree format; convert to TG for tg/tgtree grammar
            input_ids = self.vocab.convert_treenpy_to_TG(input_ids)
        elif self.is_qwen3 and (self.transformer_grammar_type[:8] == "terminal" or self.transformer_grammar_type[:5] == "pause"):
            # Qwen3 data is tree format; convert to terminal for terminal/pause grammar
            input_ids = self.vocab.convert_treenpy_to_terminal(input_ids)
        elif self.transformer_grammar_type == "tree_noont":
            input_ids = self.vocab.convert_treenpy_to_noont(input_ids)
        elif self.transformer_grammar_type == "tree_compress":
            input_ids = self.vocab.convert_treenpy_to_compress(input_ids)
        elif self.transformer_grammar_type == "tree_triplecnt":
            input_ids = self.vocab.convert_treenpy_to_triplecnt(input_ids)
        return input_ids

    def __getitem__(self, index):
        task_idx = index // self.TASK_SIZE
        sample_idx = index % self.TASK_SIZE

        if self.is_qwen3 and self.SENT_SIZE < 300:
            # Remap flat indices to K=SENT_SIZE trees per sentence from the
            # tree_300_qwen data.  Data layout is 300 trees per sentence:
            #   [Good_0×300, Bad_0×300, Good_1×300, Bad_1×300, …]
            # We take the first K trees of each sentence group.
            K = self.SENT_SIZE
            pair_idx = sample_idx // (2 * K)
            in_pair = sample_idx % (2 * K)
            is_bad = 1 if in_pair >= K else 0
            tree_within = in_pair % K
            data_idx = pair_idx * 600 + is_bad * 300 + tree_within
        else:
            data_idx = sample_idx

        input_ids = self.dataset[task_idx, data_idx].copy()
        tree_spans = None
        if self.pushdown_gold:
            # The fixed-width tree_300 rows are right padded. Parse the real tree
            # using exactly the unary-collapse + CNF convention used to train the
            # terminal-only Pushdown checkpoint, then expose only its terminal
            # leaves to the LM. The parser's whitespace tokenization is retained;
            # this is intentionally the gold300 protocol, distinct from BLiMP's
            # primal terminal tokenization.
            assert self._tree_vocab is not None
            input_ids = input_ids[input_ids != self._tree_vocab.pad]
            parsed = parse_chunk_slice(
                input_ids,
                self._tree_vocab,
                self.parse_binarize_direction,
                binarize=True,
                collapse_unary=True,
                drop_singleton_spans=True,
            )
            input_ids = parsed["input_ids"]
            tree_spans = parsed["spans"]
        elif self.ispause:
            if self.is_qwen3:
                input_ids = self.vocab.convert_treenpy_to_terminal(input_ids)
            input_ids = pause_input_ids(input_ids, self.pause_token_id, pause_num=self.transformer_grammar_type)
        else:
            input_ids = self._convert_sequence(input_ids)
        sample = {
            "sent_id" : index,
            "input_ids": torch.LongTensor(input_ids),
        }
        if tree_spans is not None:
            sample["tree_spans"] = torch.as_tensor(tree_spans, dtype=torch.long)
        return sample

    def __len__(self):
        return self.length

    def reset(self) -> None:
        return

    def __getstate__(self):
        # DataLoader uses spawn in this project. Never pickle a multi-GB memmap:
        # workers receive only metadata and reopen the same file read-only.
        state = self.__dict__.copy()
        state["dataset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.dataset = np.load(self.dataset_file, mmap_mode="r")

    def collate_fn(self, data):
        sent_ids = []
        input_ids = []
        all_attention_bias = []
        all_label_mask = []
        all_tree_spans = []
        # pad according to max_lengths
        max_len = 0
        for sample in data:
            max_len = max(max_len, sample["input_ids"].shape[0])
            
        for sample in data:
            cur_input_id = sample["input_ids"]
            cur_input_id = F.pad(cur_input_id, (0, max_len - cur_input_id.shape[0]), value=self.vocab.pad)

            attention_bias, label_mask = None, None
            if self.generate_TG_attention_bias is not None:
                attention_bias, label_mask = self.generate_TG_attention_bias(cur_input_id)
            sent_ids.append(sample["sent_id"])
            input_ids.append(cur_input_id)

            if "tree_spans" in sample:
                all_tree_spans.append(sample["tree_spans"])

            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                all_attention_bias.append(attention_bias)

            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(label_mask)

        batch = {
            # Keep the legacy scalar for evaluators that consume contiguous batches,
            # and expose every flat ID so BLiMP can scatter batches spanning
            # multiple 300-parse sentence groups without corrupting rows.
            "sent_id": min(sent_ids),
            "sent_ids": torch.as_tensor(sent_ids, dtype=torch.long),
            "input_ids": torch.stack(input_ids),
        }
        if all_attention_bias:
            batch["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            batch["label_mask"] = torch.stack(all_label_mask)
        if all_tree_spans:
            max_spans = max(spans.shape[0] for spans in all_tree_spans)
            padded_spans = []
            span_masks = []
            for spans in all_tree_spans:
                n_spans = spans.shape[0]
                padded_spans.append(
                    F.pad(spans, (0, 0, 0, max_spans - n_spans), value=-1)
                )
                span_masks.append(
                    F.pad(
                        torch.ones(n_spans, dtype=torch.bool),
                        (0, max_spans - n_spans),
                        value=False,
                    )
                )
            batch["tree_spans"] = torch.stack(padded_spans)
            batch["tree_span_mask"] = torch.stack(span_masks)
        return batch

    def token_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def token_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)


class BoolQ(ICLMultiChoiceTaskDataset):
    """Prompt: "{passage} \nQuestion: {question}? \nAnswer:"
    continuation: yes, no

    acc, random at 50% (SuperGLUE)

    {
        "question": "is ncis new orleans over for the season",
        "passage": "NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.",
        "label": true
    }
    """

    metric_type = "acc"
    BoolQPATH = "./dataset/SuperGLUE/BoolQ/"
    shots_list = [0, 1, 7, 11, 3, 4, 5]

    def __init__(
        self,
        tokenizer,
        dataset_path="boolq",
        dataset_name=None,
        model_ctx_len=2048,
        split="val",
        shots_num=3,
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = []
        with open(os.path.join(self.BoolQPATH, f"{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        for key in ["passage", "question"]:
            with open(os.path.join(self.BoolQPATH, f"{split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    dataset[idx][key] = convert_TG_format(line.strip())
        
        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset
    
    def get_shots(self, train):
        self.shots = []
        for shot_id in self.shots_list:
            self.shots.append(train[shot_id])
        return self.shots

    def doc_to_text(self, doc, single_shot=False):
        return (self.shots_prompt if single_shot==False else "") + doc["passage"] + " \n<(SQ><(NP> Question<NP)> :" + doc["question"] + " ?<SQ)> \n<(S><(NP> The answer<NP)><(VP> is<(NP>"

    def doc_to_continuations(self, doc, single_shot=False):
        label = not doc["label"]
        del doc
        # add spaces in front of continuation
        # (S (NP (DT The) (NN answer)) (VP (VBZ is) (NP (UH yes))) (. .))
        if single_shot:
            return [" yes<NP)><VP)> .<S)>", " no<NP)><VP)> .<S)>"]
        else:
            if self.split=="train":
                return [[" yes", " no"][label]]
            return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "<(S><(NP> The answer<NP)><(VP> is<(NP>"



class CommitmentBank(ICLMultiChoiceTaskDataset):
    """Prompt: "{premise} \nQuestion:{hypothesis}. True, False or Neither? \nAnswer:"
    continuations: True, False, Neither.

    implement PMI_DC
    acc, random at 33%

    {
        "premise": "It was a complex language. Not written down but handed down. One might say it was peeled down.",
        "hypothesis": "the language was peeled down",
        "label": "entailment",
    }
    """

    metric_type = "acc"
    CBPATH = "./dataset/SuperGLUE/CB/"
    LABEL_DICT = {"entailment": 0, "contradiction": 1, "neutral": 2}
    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="cb",
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )

    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.CBPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["premise", "hypothesis"]:
            with open(os.path.join(self.CBPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())
    
    def doc_to_text(self, doc):
        # Convert hypothesis part in a full sentence. 
        def convert_hypothesis(sent):
            tokens = sent.split()
            return ' '.join(tokens[:-1]) + " . " + tokens[-1]
        
        return doc["premise"] + " \n<(NP><(NP> Question<NP)> :" + convert_hypothesis(doc["hypothesis"]) + "<NP)><(FRAG> True, False or Neither ?<FRAG)> \n<(NP><(NP> Answer<NP)> :<(NP>"

    def doc_to_continuations(self, doc):
        label = self.LABEL_DICT[doc["label"]]
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [[" True", " False", " Neither"][label]]
        else:
            return [" True", " False", " Neither"]

    def doc_to_label(self, doc):
        return self.LABEL_DICT[doc["label"]]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "<(NP><(NP> Answer<NP)> :<(NP>"


# TODO: 
class COPA(ICLMultiChoiceTaskDataset):
    """Prompt: "{premise.strip()[:-1]} {because/therefore}"
    Req_loglikelihood('The pair of students came under scrutiny by the teacher because', ' the students both received excellent grades.'
    "question":
        "cause": "because",
        "effect": "therefore",
    continuations: {choice1}/{choice2}

    implement PMI_DC
    acc, random at 50%

    {
        "premise": "The pair of students came under scrutiny by the teacher.",
        "choice1": "The students both received excellent grades.",
        "choice2": "Their responses on the assignment were identical.",
        "question": "cause",
        "label": 1,
        "idx": 42
    }
    """

    metric_type = "acc"
    COPAPATH = "./dataset/SuperGLUE/COPA/"
    LABEL_DICT = {"entailment": 0, "contradiction": 1, "neutral": 2}
    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="copa",
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )

    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.COPAPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["premise", "choice1", "choice2"]:
            with open(os.path.join(self.COPAPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())

    def doc_to_text(self, doc):
        # Remove the tail part for inserting choices.
        def convert_premise(sent):
            tokens = sent.split()
            return ' '.join(tokens[:-3])

        connector = "because" if doc["question"] == "cause" else "therefore"
        return convert_premise(doc["premise"]) + "<(SBAR>" + connector

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        def convert_choice(sent):
            tokens = sent.split()
            for i, token in enumerate(tokens):
                if token[0] != '(':
                    tokens[i] = token[0].lower() + token[1:]
                    break
            return ' '.join(tokens[:-2])
        
        choices_list = [" " + convert_choice(doc["choice1"]), " " + convert_choice(doc["choice2"])]
        label = doc["label"]
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [choices_list[label]]
        else:
            return choices_list

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        connector = "because" if doc["question"] == "cause" else "therefore"
        del doc
        return "<(SBAR>" + connector


# TODO: 
class MultiRC(ICLMultiChoiceTaskDataset):
    """Prompt: {passage} \nQuestion: {Question} \nAnswer: {Answer} \nIs the answer correct? {yes/no}

    {
        "passage": {
            "text": "Should places at the same distance from the equator have the same climate? You might think they should. Unfor- tunately, you would not be correct to think this. Climate types vary due to other factors besides distance from the equator. So what are these factors? How can they have such a large impact on local climates? For one thing, these factors are big. You may wonder, are they as big as a car. Think bigger. Are they bigger than a house? Think bigger. Are they bigger than a football stadium? You are still not close. We are talking about mountains and oceans. They are big features and big factors. Oceans and mountains play a huge role in climates around the world. You can see this in Figure above . Only one of those factors is latitude, or distance from the equator. ",
            "questions": [
                {
                    "question": "Name at least one factor of climate",
                    "answers": [{ "text": "Oceans", "label": 1 },
                                { "text": "Houses", "label": 0 },
                                { "text": "Day length", "label": 0 },
                                { "text": "Latitude", "label": 1 },
                                { "text": "Longitude", "label": 0 },
                                { "text": "Season", "label": 0 },
                                { "text": "Distance from the equator", "label": 1 },
                                { "text": "The stars", "label": 0 },
                                { "text": "Moutains", "label": 1 },
                                { "text": "Mountains, ocean, longitude, latitude", "label": 1 },
                                { "text": "Cars", "label": 0 },
                                { "text": "Figures", "label": 0 },
                                { "text": "Mountains", "label": 1 },
                                { "text": "Football stadiums", "label": 0 },
                                { "text": "Same climate", "label": 0 } ],
                }
            ]
        }
    }
    """

    metric_type = "acc"
    MultiRCATH = "./dataset/SuperGLUE/MultiRC/"
    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="multirc",
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )
    
    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.MultiRCATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in []: # TODO: 
            with open(os.path.join(self.MultiRCATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())

    def doc_to_text(self, doc):
        raise NotImplementedError

    def doc_to_continuations(self, doc):
        raise NotImplementedError

    def doc_to_label(self, doc):
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc):
        raise NotImplementedError


# TODO: 
class ReCoRD(ICLMultiChoiceTaskDataset):
    """Prompt: "{passage[text]} \n"

    {
        "source": "Daily mail",
        "passage": {
            "text": "The harrowing stories of women and children locked up for so-called 'moral crimes' in Afghanistan's notorious female prison have been revealed after cameras were allowed inside. Mariam has been in Badam Bagh prison for three months after she shot a man who just raped her at gunpoint and then turned the weapon on herself - but she has yet to been charged. Nuria has eight months left to serve of her sentence for trying to divorce her husband. She gave birth in prison to her son and they share a cell together. Scroll down for video Nuria was jailed for trying to divorce her husband. Her son is one of 62 children living at Badam Bagh prison \n@highlight \nMost of the 202 Badam Bagh inmates are jailed for so-called 'moral crimes' \n@highlight \nCrimes include leaving their husbands or refusing an arrange marriage \n@highlight \n62 children live there and share cells with their mothers and five others",
            "entities": [ { "start": 86, "end": 96 },
                          { "start": 178, "end": 183 },
                          { "start": 197, "end": 206 },
                          { "start": 357, "end": 361 },
                          { "start": 535, "end": 539 },
                          { "start": 627, "end": 636 },
                          { "start": 672, "end": 681 } ]
        },
        "qas": [
            {
                "query": "The baby she gave birth to is her husbands and he has even offered to have the courts set her free if she returns, but @placeholder has refused.",
                "answers": [ { "start": 535, "end": 539, "text": "Nuria" } ],
                "idx": 0
            }
        ],
    }
    """

    metric_type = "acc"
    ReCoRDPATH = "./dataset/SuperGLUE/ReCoRD/"
    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="record",
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )
    
    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.ReCoRDPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["text", "query"]:
            with open(os.path.join(self.ReCoRDPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())

    def doc_to_text(self, doc):
        raise NotImplementedError

    def doc_to_continuations(self, doc):
        raise NotImplementedError

    def doc_to_label(self, doc):
        raise NotImplementedError

    def doc_to_domain_conditional(self, doc):
        raise NotImplementedError


class RTE(ICLMultiChoiceTaskDataset):
    """Prompt: "{sentence1} \nQuestion: {sentence2} True or False? \nAnswer:"
    continuations: True, False

    implement PMI_DC
    acc, random at 50% (GLUE)

    {
        "premise": "The number of Danes opposed to swapping the krone for the euro has increased slightly to 35.3 percent, up from 34.6 percent in April, according to a poll published on Thursday by Danske Bank.",
        "hypothesis": "The introduction of the euro has been opposed.",
        "label": "entailment",
    }
    """

    metric_type = "acc"
    RTEPATH = "./dataset/SuperGLUE/RTE/"
    LABEL_DICT = {"entailment": 0, "not_entailment": 1}
    def __init__(
        self,
        tokenizer,
        dataset_path="rte",
        dataset_name=None,
        model_ctx_len=2048,
        split="val",
        shots_num=0, 
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )
    
    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.RTEPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["premise", "hypothesis"]:
            with open(os.path.join(self.RTEPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())

    def doc_to_text(self, doc):
        return doc["premise"] + " \n<(S><(NP> Question<NP)> :" + doc["hypothesis"] + "<S)><(ADJP><(ADJP> True or False<ADJP)> ?<ADJP)> \n<(NP><(NP> Answer<NP)> :<(NP>"

    def doc_to_continuations(self, doc):
        label = self.LABEL_DICT[doc["label"]]
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [[" True", " False"][label]]
        else:
            return [" True", " False"]

    def doc_to_label(self, doc):
        return self.LABEL_DICT[doc["label"]]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "<(NP> <(NP> Answer <NP)> : (NP"
    

# TODO: 
class WiC(ICLMultiChoiceTaskDataset):
    """Prompt: "Sentence 1: {sentence1} \nSentence 2: {sentence2} \nQuestion: Is the word '{word}' used in the same way in the two sentences above? \nAnswer: "

    acc, random at 50% (SuperGLUE) 
    continuation: yes, no

    {
        "word": "place",
        "sentence1": "Do you want to come over to my place later?",
        "sentence2": "A political system with no place for the less prominent groups.",
        "label": false,
    }
    """

    metric_type = "acc"
    WiCPATH = "./dataset/SuperGLUE/WiC/"
    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="wic",
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )
    
    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.WiCPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["sentence1", "sentence2"]:
            with open(os.path.join(self.WiCPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())

    def doc_to_text(self, doc):
        return "<(NP><(NP> Sentence 1<NP)> :" + doc["sentence1"] + "<NP)> \n<(NP><(NP> Sentence 2<NP)> :" + doc["sentence2"] + "<NP)> \n<(SQ><(NP> Question<NP)> :<(SQ> Is<(NP><(NP> the word<NP)> '<(NP>" + doc["word"] + "<NP)> '<NP)><(VP> used<(PP> in<(NP> the same way<NP)><PP)><(PP> in<(NP><(NP> the two sentences<NP)><(ADVP> above<ADVP)><NP)><PP)><VP)><SQ)> ?<SQ)> \n<(NP><(NP> Answer :<NP)><(NP>"

    def doc_to_continuations(self, doc):
        label = doc["label"]
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [[" yes", " no"][not label]]
        else:
            return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "<(NP><(NP> Answer<NP)> :<(NP>"


# TODO: 
class WSC(ICLMultiChoiceTaskDataset):
    """Prompt: "{text} \nQuestion: In the passage above, does the pronoun {span1_text} refer to {span2_text}? \nAnswer: "

    acc, random at 50% (SuperGLUE) 
    continuation: yes, no

    {
        "text": "I poured water from the bottle into the cup until it was full.",
        "target": {
            "span1_text": "the cup",
            "span2_text": "it"
        },
        "label": true
    }
    """

    metric_type = "acc"
    WSCPATH = "./dataset/SuperGLUE/WSC/"
    def __init__(
        self,
        tokenizer,
        dataset_path="super_glue",
        dataset_name="wsc",
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path
        )
    
    def load_local_datasets(self):
        self.dataset = []
        with open(os.path.join(self.WSCPATH, f"{self.split}.jsonl"), "r") as file:
            for line in file:
                self.dataset.append(json.loads(line.strip()))
        for key in ["text"]:
            with open(os.path.join(self.WSCPATH, f"{self.split}_{key}.txt"), "r") as file:
                for idx, line in enumerate(file):
                    self.dataset[idx][key] = convert_TG_format(line.strip())

    def doc_to_text(self, doc):
        return doc["text"] + " \n<(NP> Question :<NP)><(SQ><(PP> In<(NP><(NP> the passage<NP)><(ADVP> above<ADVP)><NP)><PP)> , does<(NP><(NP> the pronoun<NP)> '<(NP> " + doc["target"]["span1_text"] + "<NP)> '<NP)><(VP> refer<(PP> to '<(NP> " + doc["target"]["span2_text"] + "<NP)> '<PP)><VP)> ?<SQ)> \n<(NP><(NP> Answer :<NP)><(NP>"

    def doc_to_continuations(self, doc):
        label = doc["label"]
        del doc
        # add spaces in front of continuation
        if self.split=="train":
            return [[" yes", " no"][not label]]
        else:
            return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['answer'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "<(NP><(NP> Answer<NP)> :<(NP>"
    

class HellaSwag(ICLMultiChoiceTaskDataset):
    """HellaSwag concats "ACTIVITY_LABEL: CTX_A CTX_B.capitalize()" to form context and then sends endings as continuations
        space added as prefix to each continuation

    {
        'activity_label': 'Roof shingle removal',
        'ctx_a': 'A man is sitting on a roof.',
        'ctx_b': 'he',
        'ctx': 'A man is sitting on a roof. he',
        'endings': ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.'],
        'label': '3'
    }
    """

    metric_type = "len_norm"
    SwagPATH = "./dataset/hellaswag/"
    shots_list = ["wikihow~19", "wikihow~66", "activitynet~v_-2dxp-mv2zo", "activitynet~v_-Xl95IW5H_s",
                  "wikihow~62", "activitynet~v_-QuFk_ThRNg", "activitynet~v_-YjGbsbDoxs", "activitynet~v_-fMxoShIXiM",
                  "activitynet~v_-1IBHYS3L-Y", "activitynet~v_-JqLjPz-07E", "activitynet~v_-fBTCykx4gM"] 

    def __init__(
        self,
        tokenizer,
        dataset_path="hellaswag",
        dataset_name=None,
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        shots_num=5,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\] ", "", text)
        text = text.replace("..", ".")
        text = re.sub("\\[.*?\\] ", "", text)
        text = text.replace("..", ".")
        text = text.replace("  ", " ")
        return text

    def load_local_datasets(self, split=None, ret=False):
        dataset = []
        split = self.split if split is None else split
        with open(os.path.join(self.SwagPATH, f"hellaswag_{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
                dataset[-1]["ctx_b"] = " " + dataset[-1]["ctx_b"].capitalize()
        with open(os.path.join(self.SwagPATH, f"hellaswag_{split}.txt"), "r") as file:
            for idx, line in enumerate(file):
                id_entry, num = idx // 5, idx % 5
                if num==0:
                    dataset[id_entry]["ctx_a"] = convert_TG_format(line.strip())
                else:
                    dataset[id_entry]["endings"][num-1] = convert_TG_format(line.strip())

        # if split!='train':
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset
    
    def get_mask(self, continuation, doc):
        ctx_b_ids = self.token_encode(doc["ctx_b"])
        mask = [1] * len(continuation)
        i,j = (0,0)
        while i<len(continuation) and j<len(ctx_b_ids):
            if ctx_b_ids[j] == continuation[i]:
                mask[i] = 0
                j += 1
            i += 1
        return mask

    def get_shots(self, train):
        shots = {}
        self.shots = []
        for data in train:
            if data["source_id"] in self.shots_list:
                shots[data["source_id"]] = data
        for shot_id in self.shots_list:
            self.shots.append(shots[shot_id])
        return self.shots

    def doc_to_text(self, doc, single_shot=False):
        return (self.shots_prompt if single_shot==False else "") + "<(NP> " + doc["activity_label"] + "<NP)> :" + doc["ctx_a"] # " "    + doc["ctx_b"] + " "

    def doc_to_continuations(self, doc, single_shot=False):
        return [ending for ending in doc["endings"]]

    def doc_to_label(self, doc):
        return int(doc["label"])

    def doc_to_domain_conditional(self, doc):
        return doc["ctx_a"].split(" ")[-1]


class WinoGrande(ICLMultiChoiceTaskDataset):
    """Prompt: split sentence at _ "SENTENCE[:idx] + OPTION1/OPTION2", where idx = SENTENCE.index("_")
        implement PMI_DC
        acc, random at 50%
        continuation is everything in setnence after '_' (" SENTENCE[idx:].strip()")

        Req_loglikelihood('People think Samantha', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')
        Req_loglikelihood('People think Rebecca', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')

    {
        'sentence': 'People think _ is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.',
        'option1': 'Samantha',
        'option2': 'Rebecca',
        'answer': '2'
    }

    TODO: might need to write custom metric for Winogrande
    """

    metric_type = "acc"
    WinoPATH = "./dataset/winogrande/"
    shots_list = [19875, 26035, 2302, 13568, 7412]

    def __init__(
        self,
        tokenizer,
        dataset_path="winogrande",
        dataset_name=None,
        model_ctx_len=2048,
        split="val",
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        shots_num=5,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        # all winogrande datasets have same val set
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            shots_num=shots_num,
            split=split,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = datasets.load_from_disk(f"./dataset/winogrande/{split}")
        dataset = list(dataset)
        with open(os.path.join(self.WinoPATH, f"winogrande_{split}.txt"), "r") as file:
            for idx, line in enumerate(file):
                id_entry, num = idx // 2, idx % 2
                if num==0:
                    dataset[id_entry]["parsed"] = ["", ""]
                dataset[id_entry]["parsed"][num] = convert_TG_format(line.strip())

        for idx in range(len(dataset)):
            answer_label = int(dataset[idx]["answer"])
            tree_tokens = dataset[idx]["parsed"][answer_label-1].split(" ")
            left_text, right_text = dataset[idx]["sentence"].split("_", 1)
            left_text = left_text + dataset[idx][f"option{answer_label}"]
            i, j = 0, 0
            while j<len(left_text) and i<len(tree_tokens):
                for char in tree_tokens[i]:
                    while j<len(left_text) and re.match(r"\s", left_text[j]):
                        j += 1
                    if j<len(left_text) and char == left_text[j]:
                        j += 1
                i += 1
            if j<len(left_text):
                raise NotImplementedError
            
            doc_str = " ".join(tree_tokens[:i-1])
            answer_tokens = tree_tokens[i-1]
            answer = dataset[idx][f"option{answer_label}"]
            cont_tokens = tree_tokens[i:]
            cont_str = " " + " ".join(cont_tokens)
            dataset[idx]["continuation"] = cont_str
            dataset[idx]["ctxs"] = [
                doc_str + " "  + answer_tokens.replace(answer, dataset[idx]["option1"]),
                doc_str + " "  + answer_tokens.replace(answer, dataset[idx]["option2"]),
            ]


        if ret:
            return dataset
        else:
            self.dataset = dataset

    def prepare_shots(self):
        train = self.load_local_datasets(split="train", ret=True)
        self.shots = []
        for shot_id in self.shots_list:
            self.shots.append(train[shot_id])
        
        self.shots_prompt = ""
        for i in range(self.shots_num):
            doc = self.shots[i]
            self.shots_prompt += self.doc_to_text(doc, single_shot=True)[self.doc_to_label(doc)] + self.doc_to_continuations(doc) + " \n \n"

    def prep_examples(self):
        """Overwrite for WinoGrande as multiple ctx, single continuation"""
        import copy
        doc_id = 0
        for doc in self.dataset:
            # here ctx is a list
            ctxs = self.doc_to_text(doc)
            dcs = self.doc_to_domain_conditional(doc)

            continuation_str = self.doc_to_continuations(doc)
            label_id = self.doc_to_label(doc)
            cont_str_len = len(continuation_str) - 1  # continuations contain leading blank space
            cont_byte_len = len(continuation_str[1:].encode("utf-8"))

            # tokenize
            ori_continuation = self.token_encode(continuation_str)
            for cont_id, (ctx, dc) in enumerate(zip(ctxs, dcs)):
                doc_text = ctx
                ctx = self.token_encode(ctx)
                dc = self.token_encode(dc)
                continuation = copy.deepcopy(ori_continuation)
                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation
                dc_query = dc + continuation
                query = query[-self.model_ctx_len :]
                query = self.convert_grammar_input(query)
                dc_query = self.convert_grammar_input(dc_query)
                continuation = self.convert_grammar_input(continuation)
                mask = None
                if hasattr(self, "getmask"):
                    mask = self.getmask(continuation, doc)
                if self.split!="train":
                    query = query[:-1]
                    dc_query = dc_query[:-1]
                actual_ctx_len = len(query) - len(continuation) + 1

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left

                cont_str_len = len(self.token_decode(continuation)) - 1

                if self.ispause:
                    # See ICLMultiChoiceTaskDataset for the alignment rationale:
                    # pause the full ctx+cont query, then derive the scored
                    # continuation as a contiguous slice query[split:end] so the
                    # logits gather at [ctx_len-1 : ctx_len+cont_len-1] aligns for
                    # any divisibility of ctx_real by q.
                    #
                    # Expand the UNTRIMMED full_query (ctx+cont before the eval-time
                    # [:-1] trim) and compute ctx_real from it — NOT from
                    # actual_ctx_len. actual_ctx_len carries a +1 offset (meant to
                    # compensate for the [:-1] trim in the non-pause metric path) and
                    # would make split = pause_expanded_len(ctx_real, p, q) overshoot
                    # past the end of the expanded query: empty continuation ->
                    # degenerate majority-class score. Mirrors
                    # ICLMultiChoiceTaskDataset.prep_examples.
                    full_query = ctx + continuation
                    full_dc_query = dc + continuation
                    full_query = self.convert_grammar_input(full_query)
                    full_dc_query = self.convert_grammar_input(full_dc_query)
                    full_query = full_query[-self.model_ctx_len :]
                    p, q = self.pause_spec
                    gtype = self.transformer_grammar_type
                    ctx_real = len(full_query) - len(continuation)
                    cont_real = len(continuation)
                    query = pause_input_ids(full_query, self.pause_token_id, pause_num=gtype)
                    dc_query = pause_input_ids(full_dc_query, self.pause_token_id, pause_num=gtype)
                    split = pause_expanded_len(ctx_real, p, q)
                    trim = pause_trailing_trim(ctx_real + cont_real, p, q)
                    continuation = query[split : len(query) - trim]
                    actual_ctx_len = split
                    if mask is not None:
                        full_mask = [0] * ctx_real + list(mask)
                        full_mask = pause_input_ids(full_mask, pause_token_id=None, pause_num=gtype)
                        mask = full_mask[split : len(query) - trim]

                # form a sample
                self.samples.append(
                    {
                        "doc_id": doc_id,
                        "cont_id": cont_id,
                        # "ctx": ctx,
                        "continuation": continuation,
                        "ctx_len": actual_ctx_len,
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "cont_byte_len": cont_byte_len,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                        "cont_mask": mask,
                    }
                )

                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}, {self.current_prompt}):"
                        + f" \ndoc_text: {doc_text} \ncontinuations: {continuation_str} \n" +
                        f"input_ids is {self.token_decode(query)}"
                    )

            doc_id += 1

    def doc_to_text(self, doc, single_shot=False):
        # special case where there are multiple ctx and single continuation
        return [(self.shots_prompt if single_shot==False else "") + ctx for ctx in doc["ctxs"]]

    def doc_to_continuations(self, doc, single_shot=False):
        # add spaces in front of continuation
        return doc["continuation"]

    def doc_to_label(self, doc):
        return int(doc["answer"]) - 1

    def doc_to_domain_conditional(self, doc):
        """same number of domain conditionals as context"""
        return [doc["option1"], doc["option2"]]


# TODO:
class PIQA(ICLMultiChoiceTaskDataset):
    """PIQA sends context in the following fashion: "Question: GOAL \nAnswer:"
    space added as prefix to each continuation

    implement PMI_DC

    {
        'goal': "How do I ready a guinea pig cage for it's new occupants?",
        'sol1': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.',
        'sol2': 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.',
        'label': 0
    }
    """

    metric_type = "len_norm"
    PIQAPATH = "./dataset/piqa"
    shots_list = [9, 3, 4, 5, 6, 7, 8]

    def __init__(
        self,
        tokenizer,
        dataset_path="piqa",
        dataset_name="plain_text",
        model_ctx_len=2048,
        split="validation",
        shots_num=3,
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = []
        with open(os.path.join(self.PIQAPATH, f"piqa_{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        with open(os.path.join(self.PIQAPATH, f"piqa_{split}.txt"), "r") as file:
            item_list = ["goal", "sol1", "sol2"]
            for idx, line in enumerate(file):
                id_entry, num = idx // 3, idx % 3
                dataset[id_entry][item_list[num]] = convert_TG_format(line.strip())
        
        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset

    def get_shots(self, train):
        self.shots = []
        for shot_id in self.shots_list:
            self.shots.append(train[shot_id])
        return self.shots

    def doc_to_text(self, doc, single_shot=False):
        return (self.shots_prompt if single_shot==False else "") + "<(NP> Goal<NP)> :" + doc["goal"] + " \n<(NP> Answer<NP)> :"

    def doc_to_continuations(self, doc, single_shot=False):
        return [doc["sol1"], doc["sol2"]]

    def doc_to_label(self, doc):
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        return "<(NP> Answer<NP)> :"


class OpenBookQA(ICLMultiChoiceTaskDataset):
    """OBQA: question_stem is sent as context (no special prompt format) and choices are sent as continuation
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question_stem': 'Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as',
        'choices': {'text': ['Deep sea animals', 'fish', 'Long Sea Fish', 'Far Sea Animals'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "len_norm"
    OpenBookQAPATH = "./dataset/openbookqa"
    shots_list = ["7-584", "7-870", "9-732", "9-782", "8-72", "9-87", "1046", "1591", "7-1167"]

    def __init__(
        self,
        tokenizer,
        dataset_path="openbookqa",
        dataset_name="main",
        model_ctx_len=2048,
        split="test",
        shots_num=5,
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = []
        with open(os.path.join(self.OpenBookQAPATH, f"openbookqa_{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        with open(os.path.join(self.OpenBookQAPATH, f"openbookqa_{split}.txt"), "r") as file:
            for idx, line in enumerate(file):
                id_entry, num = idx // 4, idx % 4
                dataset[id_entry]["choices"]["text"][num] = convert_TG_format(line.strip())
        for item in dataset:
            choices = item["choices"]["text"]
            idx = os.path.commonprefix(choices).rfind('>') + 1
            # stem = item["question_stem"]
            item["question_stem"] = choices[0][:idx]
            item["choices"]["text"] = [s[idx:] for s in choices]
            #if len(item["question_stem"]) < len(stem):
            #    print(f'Oh no! {item["question_stem"]=} {item["choices"]["text"]=}')
        
        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset

    def get_shots(self, train):
        shots = {}
        self.shots = []
        for data in train:
            if data["id"] in self.shots_list:
                shots[data["id"]] = data
        for shot_id in self.shots_list:
            self.shots.append(shots[shot_id])
        return self.shots

    def doc_to_text(self, doc, single_shot=False):
        return (self.shots_prompt if single_shot==False else "") + doc["question_stem"]
        # return (self.shots_prompt if single_shot==False else "") + " \n<(NP> Question<NP)> :" + doc["question_stem"] + " \n<(NP> Answer<NP)> :"

    def doc_to_continuations(self, doc, single_shot=False):
        return doc["choices"]["text"]

    def doc_to_label(self, doc):
        return ["A", "B", "C", "D"].index(doc["answerKey"].strip())

    def doc_to_domain_conditional(self, doc):
        return doc["question_stem"].strip().split(" ")[-1]


class SciQ(ICLMultiChoiceTaskDataset):
    """SciQ sends context as "SUPPORT \nQuestion: QUESTION \nAnswer:" and then distractors + correct_answer as continuations
        space added as prefix to each continuation

        implement PMI_DC

    {
        'question': 'Who proposed the theory of evolution by natural selection?',
        'distractor3': 'Scopes',
        'distractor1': 'Linnaeus',
        'distractor2': 'shaw',
        'correct_answer': 'darwin',
        'support': ''
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="sciq",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return doc["support"].strip() + " \nQuestion: " + doc["question"] + " \nAnswer:"

    def doc_to_continuations(self, doc):
        # add spaces in front of continuation
        return [
            " " + doc["distractor1"],
            " " + doc["distractor2"],
            " " + doc["distractor3"],
            " " + doc["correct_answer"],
        ]

    def doc_to_label(self, doc):
        del doc
        return 3

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


# TODO:
class ArcEasy(ICLMultiChoiceTaskDataset):
    """ArcEasy creates context with "Question: QUESTION \nAnswer:" and sends the choices as continuations
        space added as prefix to each continuation

    {
        'question': 'Which technology was developed most recently?',
        'choices': {'text': ['cellular telephone', 'television', 'refrigerator', 'airplane'],
        'label': ['A', 'B', 'C', 'D']},
        'answerKey': 'A'
    }
    """

    metric_type = "len_norm"
    ArcPATH = "./dataset/ai2_arc/ARC-Easy"
    shots_list = ["MCAS_2007_8_5189", "Mercury_SC_401169", "MCAS_2004_8_27",
                  "NYSEDREGENTS_2006_8_10", "Mercury_7013388", "Mercury_7179953",
                  "Mercury_7205118", "MCAS_2016_8_13"]

    def __init__(
        self,
        tokenizer,
        dataset_path="ai2_arc",
        dataset_name="ARC-Easy",
        model_ctx_len=2048,
        split="validation",
        shots_num=3,
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = []
        with open(os.path.join(self.ArcPATH, f"arc_easy_{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        with open(os.path.join(self.ArcPATH, f"arc_easy_{split}.txt"), "r") as file:
            for idx, line in enumerate(file):
                id_entry, num = idx // 6, idx % 6
                if num==0:
                    dataset[id_entry]["question"] = convert_TG_format(line.strip())
                elif num-1 < len(dataset[id_entry]["choices"]["label"]):
                    dataset[id_entry]["choices"]["text"][num-1] = convert_TG_format(line.strip())
        
        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset

    def get_shots(self, train):
        shots = {}
        self.shots = []
        for data in train:
            if data["id"] in self.shots_list:
                shots[data["id"]] = data
        for shot_id in self.shots_list:
            self.shots.append(shots[shot_id])
        return self.shots

    def doc_to_text(self, doc, single_shot=False):
        return (self.shots_prompt if single_shot==False else "") + "<(NP> Question<NP)> :" + doc["question"] + " \n<(S><(NP> The answer<NP)><(VP> is"

    def doc_to_continuations(self, doc, single_shot=False):
        return doc["choices"]["text"]

    def doc_to_label(self, doc):
        # some doc["answerKey"] are stored as numbers
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}

        if doc["answerKey"] in num_to_letter:
            doc["answerKey"] = num_to_letter[doc["answerKey"]]

        return ["A", "B", "C", "D", "E"].index(doc["answerKey"])

    def doc_to_domain_conditional(self, doc):
        return "<(S><(NP> The answer<NP)><(VP> is"


# TODO:
class ArcChallenge(ArcEasy):
    """ArcChallenge follows the same prompt format as ArcEasy.
    implement PMI_DC
    """

    metric_type = "len_norm"
    ArcPATH = "./dataset/ai2_arc/ARC-Challenge"
    shots_list = ["Mercury_SC_415702", "MCAS_2009_5_6516", "Mercury_7233695", 
                  "Mercury_7041615", "MCAS_1998_4_3", "Mercury_7041860", 
                  "ACTAAP_2013_5_11", "MDSA_2008_5_30", "MEA_2016_8_14",
                  "Mercury_SC_401653", "Mercury_7106908"]

    def __init__(
        self,
        tokenizer,
        dataset_path="ai2_arc",
        dataset_name="ARC-Challenge",
        model_ctx_len=2048,
        split="validation",
        shots_num=3,
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = []
        with open(os.path.join(self.ArcPATH, f"arc_challenge_{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        with open(os.path.join(self.ArcPATH, f"arc_challenge_{split}.txt"), "r") as file:
            for idx, line in enumerate(file):
                id_entry, num = idx // 6, idx % 6
                if num==0:
                    dataset[id_entry]["question"] = convert_TG_format(line.strip())
                elif num-1 < len(dataset[id_entry]["choices"]["label"]):
                    dataset[id_entry]["choices"]["text"][num-1] = convert_TG_format(line.strip())
        
        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset


class BasicArithmetic(ArcEasy):
    """This is a basic arithmetic task follows the same prompt format as ArcEasy.
    Example:
    {"id": "q85_1d1d_max1d_plus",
    "question": "Calculate 2 + 5 =",
    "choices": {"text": ["8", "7", "6", "17"],
    "label": ["A", "B", "C", "D"]},
    "answerKey": "B", "type_tag": "easy"}

    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="allenai/basic_arithmetic",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )


class CommonsenseQA(ICLMultiChoiceTaskDataset):
    """CommonsenseQA
    Example:
    {'id': 'e68fb2448fd74e402aae9982aa76e527',
    'question': 'Where are  you likely to find a hamburger?',
    'question_concept': 'hamburger',
    'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
    'text': ['fast food restaurant', 'pizza', 'ground up dead cows', 'mouth', 'cow carcus']},
    'answerKey': 'A'}
    """

    metric_type = "len_norm"
    CSQAPATH = "./dataset/commonsense_qa"
    shots_list = ["61fe6e879ff18686d7552425a36344c8", "02e821a3e53cb320790950aab4489e85", 
                  "23505889b94e880c3e89cff4ba119860", "a76403b4921a9281b6ee2a7241a5ec9f", 
                  "6dc921840aa1e5dda3333b79007f630b", "e8a8b3a2061aa0e6d7c6b522e9612824", 
                  "527e72eb38950b8031ee6217ef531960"]

    def __init__(
        self,
        tokenizer,
        dataset_path="tau/commonsense_qa",
        dataset_name=None,
        model_ctx_len=2048,
        split="validation",
        shots_num=3,
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = []
        with open(os.path.join(self.CSQAPATH, f"commonsense_qa_{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        with open(os.path.join(self.CSQAPATH, f"commonsense_qa_{split}.txt"), "r") as file:
            for idx, line in enumerate(file):
                id_entry, num = idx // 6, idx % 6
                if num==0:
                    dataset[id_entry]["question"] = convert_TG_format(line.strip())
                else:
                    dataset[id_entry]["choices"]["text"][num-1] = convert_TG_format(line.strip())
        
        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset

    def get_shots(self, train):
        shots = {}
        self.shots = []
        for data in train:
            if data["id"] in self.shots_list:
                shots[data["id"]] = data
        for shot_id in self.shots_list:
            self.shots.append(shots[shot_id])
        return self.shots

    def doc_to_text(self, doc, single_shot=False):
        return (self.shots_prompt if single_shot==False else "") + "<(NP> Question<NP)> :" + doc["question"] + " \n<(S><(NP> The answer<NP)><(VP> is"

    def doc_to_continuations(self, doc, single_shot=False):
        return doc["choices"]["text"]

    def doc_to_label(self, doc):
        return ["A", "B", "C", "D", "E"].index(doc["answerKey"].strip())

    def doc_to_domain_conditional(self, doc):
        return "<(S><(NP> The answer<NP)><(VP> is"


class SocialIQa(ICLMultiChoiceTaskDataset):
    """SocialIQa
    Example:
    {'context': 'Jordan was in charge of taking the food on the camping trip and left all the food at home.',
     'question': 'How would Jordan feel afterwards?',
     'answerA': 'horrible that he let his friends down on the camping trip',
     'answerB': "happy that he doesn't need to do the cooking on the trip",
     'answerC': 'very proud and accomplished about the camping trip', 'label': '1'}
    """

    metric_type = "len_norm"
    SIQAPATH = "./dataset/social_i_qa"
    shots_list = [0, 8, 1, 13, 2, 7, 18, 9]

    def __init__(
        self,
        tokenizer,
        dataset_path="social_i_qa",
        dataset_name=None,
        model_ctx_len=2048,
        split="validation",
        shots_num=3,
        transformer_grammar_type:str = "",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type=None,
        pause_token_id=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )

    def load_local_datasets(self, split=None, ret=False):
        split = self.split if split is None else split
        dataset = []
        with open(os.path.join(self.SIQAPATH, f"social_i_qa_{split}.jsonl"), "r") as file:
            for line in file:
                dataset.append(json.loads(line.strip()))
        with open(os.path.join(self.SIQAPATH, f"social_i_qa_{split}.txt"), "r") as file:
            item_list = ["context", "question", "answerA", "answerB", "answerC"]
            for idx, line in enumerate(file):
                id_entry, num = idx // 5, idx % 5
                dataset[id_entry][item_list[num]] = convert_TG_format(line.strip())
        
        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset

    def get_shots(self, train):
        self.shots = []
        for shot_id in self.shots_list:
            self.shots.append(train[shot_id])
        return self.shots

    def doc_to_text(self, doc, single_shot=False):
        return (self.shots_prompt if single_shot==False else "") + "<(NP> Question<NP)> :" + doc["context"] + doc["question"] + " \n<(S><(NP> The answer<NP)><(VP> is<(NP>"

    def doc_to_continuations(self, doc, single_shot=False):
        return [doc["answerA"], doc["answerB"], doc["answerC"]]

    def doc_to_label(self, doc):
        return int(doc["label"]) - 1

    def doc_to_domain_conditional(self, doc):
        return "<(S><(NP> The answer<NP)><(VP> is<(NP>"


class MRPC(ICLMultiChoiceTaskDataset):
    """Prompt for MRPC is formed using "Sentence 1: SENTENCE1 \nSentence 2: SENTENCE2 \nQuestion: Do both sentences mean the same thing? \nAnswer:"
    acc/F1, random at 50% acc. (GLUE)
    continuations: yes and no

    {
        'sentence1': 'In fiction : Edward P. Jones ( " The Known World " ) and Scott Spencer ( " A Ship Made of Paper " ) .',
        'sentence2': 'The fifth nominee for fiction is Scott Spencer , for A Ship Made of Paper .',
        'label': 0
    }
    """

    metric_type = "f1"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="mrpc",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return (
            "Sentence 1: "
            + self.preprocess(doc["sentence1"])
            + " \nSentence 2: "
            + self.preprocess(doc["sentence2"])
            + " \nQuestion: Do both sentences mean the same thing? \nAnswer:"
        )

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        return [" yes", " no"]

    def doc_to_label(self, doc):
        # if doc['label'] is True, return index of " yes" which is 0
        if doc["label"]:
            return 0
        else:
            return 1

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class SST2(ICLMultiChoiceTaskDataset):
    """SST2 task formats prompts as "SENTENCE \nQuestion: Is this sentence positive or negative? \nAnswer:"
    some preprocessing done on sentence

    constructs 2 requests, 1 for positive and another for negative
    positive and negative have just 1 token in tokenizer
    positive: 1313
    negative: 2430

    implement PMI_DC
    acc, random at 50% (GLUE)

    {
        'sentence': "harrison 's flowers puts its heart in the right place , but its brains are in no particular place at all . ",
        'label': 1,
    }
    """

    metric_type = "acc"

    def __init__(
        self,
        tokenizer,
        dataset_path="glue",
        dataset_name="sst2",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    @classmethod
    def preprocess(cls, string: str) -> str:
        string = string.replace(" n't", "n't")
        string = string.replace(" )", ")")
        string = string.replace("( ", "(")
        string = string.replace('" ', '"')
        string = string.replace(' "', '"')

        string = re.sub(r" (['.,])", r"\1", string)

        return string

    def doc_to_text(self, doc):
        return self.preprocess(doc["sentence"]) + " \nQuestion: Is this sentence positive or negative? \nAnswer:"

    def doc_to_continuations(self, doc):
        del doc
        # add spaces in front of continuation
        # # {1: "positive", 0: "negative"}
        return [" negative", " positive"]

    def doc_to_label(self, doc):
        # {1: "positive", 0: "negative"}
        return doc["label"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class MMLU(ICLMultiChoiceTaskDataset):
    """MMLU creates context with "Question: QUESTION \nAnswer:" and sends the choices as continuations
           space added as prefix to each continuation

       {
           'question': "Which of the following terms describes the body's ability to maintain its normal state?",
           'subject': 'anatomy',
           'choices': ['Anabolism', 'Catabolism', 'Tolerance', 'Homeostasis'],
    '       answer': 3
        }
    """

    metric_type = "len_norm"  # Ideally pmi_dc

    _subcategories = {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }

    _categories = {
        "stem": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
        "humanities": ["history", "philosophy", "law"],
        "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
        "other": ["other", "business", "health"],
    }

    _MMLUPATH = "./dataset/mmlu"
    _REDUXPATH = "./dataset/mmluredux"
    shots_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str = "edinburgh-dawg/mmlu-redux-2.0",
        dataset_name: Union[str, Sequence[str], None] = "all",
        model_ctx_len: int = 2048,
        split="test",
        metric_type=None,  # Override default metric type
        prompts=[None],  # List of prompt variants to use
        local_datasets=True,
        shots_num=5,
        transformer_grammar_type="",
        generate_TG_attention_bias=None,
        vocab_path=None,
        tree_eval_type="default",
        pause_token_id=None,
        mc_labels=False,
    ):
        self.mc_labels = mc_labels
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            model_ctx_len=model_ctx_len,
            split=split,
            shots_num=shots_num,
            transformer_grammar_type=transformer_grammar_type,
            generate_TG_attention_bias=generate_TG_attention_bias,
            vocab_path=vocab_path,
            tree_eval_type=tree_eval_type,
            pause_token_id=pause_token_id,
        )
        self.doc_group = []
        for doc in self.dataset:
            self.doc_group.append(doc["subject"])
    
    def prepare_shots(self, split="validation"):
        self.shots_prompt : Dict[str, str] = {}
        shots_split = self.load_local_datasets(split=split, ret=True, dataset_path="cais/mmlu")
        self.shots = {}
        for doc in shots_split:
            subject = doc["subject"]
            if subject not in self.shots:
                self.shots[subject] = []
            self.shots[subject].append(doc)
        for category in self._subcategories:
            prompt = ""
            for i in range(self.shots_num):
                doc = self.shots[category][i]
                prompt += self.doc_to_text(doc, single_shot=True) + self.doc_to_continuations(doc, single_shot=True)[self.doc_to_label(doc)] + " \n \n"
            self.shots_prompt[category] = prompt
    
    def load_local_datasets(self, split=None, ret=False, dataset_path=None):
        split = self.split if split is None else split
        dataset_path = self.dataset_path if dataset_path is None else dataset_path
        def correct_redux(record: Dict[str, Any]):
            # NOTE: This is intentionally NOT called here — the mmluredux data
            # loaded by load_local_datasets is ALREADY corrected offline using
            # exactly this logic (no_correct_answer / wrong_groundtruth /
            # multiple_correct_answers). Kept as a self-documenting reference of
            # the preprocessing applied to the raw redux records, not as dead code.
            error_type = record['error_type']
            error_type = record['error_type']
            choices = record['choices']
            target_index_list = [int(record['answer'])]
            correct_answer = record['correct_answer']
            if error_type == 'no_correct_answer' and correct_answer:
                choices[target_index_list[0]] = correct_answer
            elif error_type == 'wrong_groundtruth' and correct_answer:
                try:
                    target_index_list = [int(correct_answer)]
                except ValueError:
                    choice_index = ord(correct_answer) - ord('A')
                    target_index_list = [choice_index]
            elif error_type == 'multiple_correct_answers' and correct_answer:
                correct_answer = correct_answer.strip('()')
                try:
                    correct_answer = correct_answer.replace(' and ', ',').replace(' or ', ',')
                    target_index_list = list(map(int, correct_answer.split(',')))
                except ValueError:
                    try:
                        target_index_list = [ord(c) - ord('A') for c in correct_answer.split(',')]
                    except TypeError:
                        # find the index of the correct answer in choices
                        target_index_list = [choices.index(c) for c in correct_answer.split(',') if c in choices]
                        if target_index_list == []:
                            target_index_list = [int(record['answer'])]
            record["choices"] = choices
            record["answer"] = target_index_list
            return record
        
        def load_local_mmlu(file_path, dataset):
            with open(file_path, "r") as file:
                for idx, line in enumerate(file):
                    id_entry, num = idx // 5, idx % 5
                    if num==0:
                        dataset[id_entry]["question"] = convert_TG_format(line.strip())
                    else:
                        dataset[id_entry]["choices"][num - 1] = convert_TG_format(line.strip())
            return
        if dataset_path=="cais/mmlu":
            dataset = datasets.load_from_disk(os.path.join(self._MMLUPATH, split)) #(dataset_path, self.dataset_name, split=split)
            dataset = list(dataset)
            load_local_mmlu(os.path.join(self._MMLUPATH, f"mmlu{split}.txt"), dataset)
        else:
            dataset = datasets.load_from_disk(os.path.join(self._REDUXPATH, "all"))
            dataset = list(dataset)
            load_local_mmlu(os.path.join(self._REDUXPATH, f"all.txt"), dataset)

        # if split!="train":
        #     dataset = dataset[:1000]
        if ret:
            return dataset
        else:
            self.dataset = dataset

    def doc_to_text(self, doc, single_shot=False):
        return ("" if single_shot==True else f"<(S><(NP> The following<NP)><(VP> are<(NP><(NP><(NML> multiple choice<NML)> questions<NP)> (<(PP> with<(NP> answers<NP)><PP)> )<(PP> about<(NP> {doc['subject']}<NP)><PP)><NP)><VP)> .<S)> \n")  \
           +  (self.shots_prompt[doc["subject"]] if single_shot==False else "")   \
           + "<(SQ><(NP> Question<NP)> :" + doc["question"] + " ?<SQ)> \n<(S><(NP> The answer<NP)><(VP> is"

    def doc_to_continuations(self, doc, single_shot=False):
        # add spaces in front of continuation
        if self.mc_labels:
            choices = [" A", " B", " C", " D"]
        else:
            choices = doc["choices"]
        if self.metric_type in ["ce_loss", "bpb"]:
            # Only need correct answer for these metrics
            return [choices[doc["answer"]]]
        else:
            return choices

    def doc_to_label(self, doc):
        if self.metric_type in ["ce_loss", "bpb"]:
            # Only the correct answer is provided for these metrics
            return 0
        return doc["answer"]

    def doc_to_domain_conditional(self, doc):
        del doc
        return "<(S><(NP> The answer<NP)><(VP> is"


class TriviaQACELoss(ICLMultiChoiceTaskDataset):
    """Sample TriviaQA entity with some fields suppressed. For CE Loss we only consider the "value"
    field as the answer to score.

    {
        'question': 'Which Lloyd Webber musical premiered in the US on 10th December 1993?',
        'question_id': 'tc_33',
        'answer': {
            'aliases': ['Sunset Blvd', ...],
            'normalized_aliases': ['sunset boulevard', ...],
            'normalized_value': 'sunset boulevard',
            'value': 'Sunset Boulevard'
        }
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="trivia_qa",
        dataset_name="rc.wikipedia.nocontext",
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return " \nQuestion: " + doc["question"] + " \nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"]["value"]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class NaturalQuestionsCELoss(ICLMultiChoiceTaskDataset):
    """Sample NaturalQuestions entity. For CE Loss we only consider the first answer entry to score.

    {
        'question': 'when was the last time anyone was on the moon',
        'answer': ['14 December 1972 UTC', 'December 1972']
    }
    """

    metric_type = "ce_loss"

    def __init__(
        self,
        tokenizer,
        dataset_path="nq_open",
        dataset_name=None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
        )

    def doc_to_text(self, doc):
        return " \nQuestion: " + doc["question"] + " \nAnswer:"

    def doc_to_continuations(self, doc):
        return [" " + doc["answer"][0]]

    def doc_to_label(self, doc):
        return 0

    def doc_to_domain_conditional(self, doc):
        del doc
        return "Answer:"


class OEEvalTask(ICLMultiChoiceTaskDataset):
    """Generic class for OE evaluation tasks"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        dataset_path: str,
        dataset_name: Union[str, Sequence[str], None] = None,
        model_ctx_len: int = 2048,
        split=None,
        metric_type=None,
        prompts=[None],  # List of prompt variants to use
    ):
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.model_ctx_len = model_ctx_len
        self.log_instances = 0  # Set to > 0 to log the first few instances as a sanity check

        self.samples: List[Dict[str, Any]] = []
        dataset_names: Sequence[Optional[str]]
        if isinstance(dataset_name, str) or dataset_name is None:
            dataset_names = [dataset_name]
        else:
            dataset_names = dataset_name

        requests_list = []
        configs = []
        for ds_name in dataset_names:
            config, requests = load_oe_eval_requests(self.dataset_path, ds_name, split)
            requests_list.append(requests)
            configs.append(config)
        if metric_type is not None:
            self.metric_type = metric_type
        else:
            # Use metric type from associated task config
            for config in configs:
                if config is not None:
                    metric_type_raw = config["task_config"].get("primary_metric")
                    if metric_type_raw is not None:
                        # acc, len_norm, pmi_dc
                        metric_type = METRIC_FROM_OE_EVAL[metric_type_raw]
                        if self.metric_type is not None and self.metric_type != metric_type:
                            raise ValueError(f"Conflicting metric types: {self.metric_type} and {metric_type}")
                        self.metric_type = metric_type
        self.dataset = requests_list

        # prep examples
        self.prep_examples()

    def prep_examples(self):
        current_doc_id_offset = 0
        max_doc_id = 0
        for requests in self.dataset:
            current_doc_id_offset += max_doc_id
            max_doc_id = 0  # Max doc id seen in this dataset
            for request in requests:
                doc = request["doc"]
                doc_id = request["doc_id"]
                if doc_id >= 1000000:
                    # Hacky implementation of unconditional requests in oe-eval
                    # Not supported here for now
                    continue
                if doc_id > max_doc_id:
                    max_doc_id = doc_id
                assert (
                    request["request_type"] == "loglikelihood"
                ), f"Unsupported request type: {request['request_type']}"

                # from EAI harness
                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                request_dict = request["request"]
                continuation_str = request_dict["continuation"]
                label_id = request["label"]
                cont_id = request["idx"]
                if self.metric_type in ["ce_loss", "bpb"]:
                    if label_id != cont_id:
                        # Skip non-target continuations for ce_loss and bpb
                        continue
                    else:
                        # Treat as instance with just one continuation
                        cont_id = 0
                        label_id = 0
                doc_text = request_dict["context"]
                ctx = self.token_encode(doc_text)
                dc = self.token_encode(self.doc_to_domain_conditional(doc))
                if self.log_instances > 0:
                    self.log_instances -= 1
                    ds_name = self.dataset_name
                    if isinstance(ds_name, list):
                        ds_name = ds_name[0]
                    log.info(
                        f"Sample doc from ({self.dataset_path}, {ds_name}):"
                        + f" \ndoc_text: {doc_text} \ncontinuation: {continuation_str}"
                    )
                cont_str_len = len(continuation_str) - 1  # continuation contain leading blank
                cont_byte_len = len(continuation_str[1:].encode("utf-8"))
                continuation = self.token_encode(continuation_str)

                # query, remove last token from continuation, truncate from left is longer than model ctx length
                query = ctx + continuation[:-1]
                query = query[-self.model_ctx_len :]
                # this will be different from len(ctx) when truncated by model_ctx_len
                actual_ctx_len = len(query) - len(continuation) + 1

                # get domain conditional query
                # we don't expect this to be longer than self.model_ctx_len and it won't make sense to truncate from left
                dc_query = dc + continuation[:-1]

                # form a sample
                self.samples.append(
                    {
                        "doc_id": doc_id + current_doc_id_offset,
                        "cont_id": cont_id,
                        "ctx": ctx,
                        "continuation": continuation,
                        "ctx_len": actual_ctx_len,
                        "dc_len": len(dc),
                        "cont_len": len(
                            continuation
                        ),  # even if query has last token removed, LM will output same cont len
                        "cont_str_len": cont_str_len,
                        "cont_byte_len": cont_byte_len,
                        "query": query,  # remove last token from continuation
                        "dc_query": dc_query,
                        "label_id": label_id,
                    }
                )

    def doc_to_text(self, doc) -> str:
        raise NotImplementedError

    def doc_to_continuations(self, doc) -> List[str]:
        raise NotImplementedError

    def doc_to_label(self, doc) -> int:
        raise NotImplementedError


TG_path = "./dataset/bbc-news/testppl_tg/"
TXLTREE_path = "./dataset/bbc-news/testppl_tree/"
TERMINAL_path = "./dataset/bbc-news/terminal/"
TESTOR_TG_PATH = "./dataset/bbc-news/testor_tg/"
TESTOR_TREE_PATH = "./dataset/bbc-news/testor_tree/"
BLiMP_PATH = "./dataset/BLiMP/tree300/"
BLiMP_RAW_PATH = "./dataset/BLiMP/raw_data/"

TG_task_map = {
    "tg_approx_sent": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "newppl.json", "metric_type": "sent"}),
    "tg_approx_sent_testor": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "smallppl.json", "metric_type": "sent"}),
    "txl_approx_sent": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "newppl.json", "metric_type": "sent"}),
    "txl_approx_sent_testor": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "smallppl.json", "metric_type": "sent"}),
    "tg_approx_doc": (TGPerplexityApproximationDataset, {"dataset_path": TG_path, "dataset_name": "tg", "metric_type": "doc"}),
    "tg_approx_doc_testor": (TGPerplexityApproximationDataset, {"dataset_path": TESTOR_TG_PATH, "dataset_name": "CC-MAIN-2022-49", "metric_type": "doc"}),
    "txl_approx_doc": (TGPerplexityApproximationDataset, {"dataset_path": TXLTREE_path, "dataset_name": "tree", "metric_type": "doc"}),
    "txl_approx_doc_testor": (TGPerplexityApproximationDataset, {"dataset_path": TESTOR_TREE_PATH, "dataset_name": "CC-MAIN-2022-49", "metric_type": "doc"}),
    "terminal_doc_ppl": (
        TerminalDocumentPerplexityDataset,
        {
            "dataset_path": TERMINAL_path,
        },
    ),
    "syntactic_generalization": (SGDataset, {"dataset_path": "./evaluation/SG/tokenized"}), 
    "BLiMP": (BLiMPApproximationDataset, {"dataset_path": BLiMP_PATH}), 
    "xsum": (XsumDataset, {"dataset_path":"./dataset/Xsum", "metric_type": "rouge"}),
    "xsum_valid": (XsumDataset, {"dataset_path":"./dataset/Xsum", "metric_type": "rouge", "split":"validation"})
}

Super_GLUE = {
    "boolq": BoolQ,
    "cb": CommitmentBank,
    "copa": COPA,
    "multirc": MultiRC, 
    "record": ReCoRD,
    "rte": RTE,
    "wic": WiC, 
    "wsc": WSC,
    "hellaswag": HellaSwag,
    "mmlu": (MMLU, {"dataset_path": "cais/mmlu"}),
    "mmluredux": (MMLU, {"dataset_path": "edinburgh-dawg/mmlu-redux-2.0"}),
}

label_to_task_map = {
    "piqa": PIQA,
    "winogrande": WinoGrande,
    "openbook_qa": OpenBookQA,
    "sciq": SciQ,
    "arc_easy": ArcEasy,
    "arc_challenge": ArcChallenge,
    "basic_arithmetic": BasicArithmetic,
    "commitment_bank": CommitmentBank,
    "mrpc": MRPC,
    "sst2": SST2,
    "commonsense_qa": CommonsenseQA,
    "social_iqa": SocialIQa,
    "trivia_qa_wiki_ppl": TriviaQACELoss,
    "natural_qs_open_ppl": NaturalQuestionsCELoss,
    "arc_easy_term": (ArcEasy, {"tree_eval_type": "terminal"}),
    "arc_challenge_term": (ArcChallenge, {"tree_eval_type": "terminal"}),
    "piqa_term": (PIQA, {"tree_eval_type": "terminal"}),
    "winogrande_term" : (WinoGrande, {"tree_eval_type": "terminal"}),
    "hellaswag_term" : (HellaSwag, {"tree_eval_type": "terminal"}),
    "social_iqa_term": (SocialIQa, {"tree_eval_type": "terminal"}),
    "commonsense_qa_term": (CommonsenseQA, {"tree_eval_type": "terminal"}),
    "openbook_qa_term": (OpenBookQA, {"tree_eval_type": "terminal"}),
    "mmlu_term": (MMLU, {"dataset_path": "cais/mmlu", "tree_eval_type": "terminal"}),
    "mmluredux_term": (MMLU, {"dataset_path": "edinburgh-dawg/mmlu-redux-2.0", "tree_eval_type": "terminal"}),
    "hellaswag_decomp": (HellaSwag, {"tree_eval_type": "default"}),
    "piqa_decomp": (PIQA, {"tree_eval_type": "default"}),
    "winogrande_decomp": (WinoGrande, {"tree_eval_type": "default"}),
    "arc_easy_decomp": (ArcEasy, {"tree_eval_type": "default"}),
    "arc_challenge_decomp": (ArcChallenge, {"tree_eval_type": "default"}),
    "social_iqa_decomp": (SocialIQa, {"tree_eval_type": "default"}),
    "commonsense_qa_decomp": (CommonsenseQA, {"tree_eval_type": "default"}),
    "openbook_qa_decomp": (OpenBookQA, {"tree_eval_type": "default"}),
    "mmlu_stem_test": (MMLU, {"dataset_name": "stem", "split": "test"}),
    "mmlu_humanities_test": (MMLU, {"dataset_name": "humanities", "split": "test"}),
    "mmlu_social_sciences_test": (MMLU, {"dataset_name": "social_sciences", "split": "test"}),
    "mmlu_other_test": (MMLU, {"dataset_name": "other", "split": "test"}),
    "mmlu_stem": (MMLU, {"dataset_name": "stem"}),
    "mmlu_humanities": (MMLU, {"dataset_name": "humanities"}),
    "mmlu_social_sciences": (MMLU, {"dataset_name": "social_sciences"}),
    "mmlu_other": (MMLU, {"dataset_name": "other"}),
    # "mmlu_stem_bpb": (MMLU, {"dataset_name": "stem", "metric_type": "bpb"}),
    # "mmlu_humanities_bpb": (MMLU, {"dataset_name": "humanities", "metric_type": "bpb"}),
    # "mmlu_social_sciences_bpb": (MMLU, {"dataset_name": "social_sciences", "metric_type": "bpb"}),
    # "mmlu_other_bpb": (MMLU, {"dataset_name": "other", "metric_type": "bpb"}),
    # "mmlu_stem_var": (MMLU, {"dataset_name": "stem", "prompts": 1}),
    # "mmlu_humanities_var": (MMLU, {"dataset_name": "humanities", "prompts": 1}),
    # "mmlu_social_sciences_var": (MMLU, {"dataset_name": "social_sciences", "prompts": 1}),
    # "mmlu_other_var": (MMLU, {"dataset_name": "other", "prompts": 1}),
    # "mmlu_stem_var_bpb": (MMLU, {"dataset_name": "stem", "prompts": 1, "metric_type": "bpb"}),
    # "mmlu_humanities_var_bpb": (
    #     MMLU,
    #     {"dataset_name": "humanities", "prompts": 1, "metric_type": "bpb"},
    # ),
    # "mmlu_social_sciences_var_bpb": (
    #     MMLU,
    #     {"dataset_name": "social_sciences", "prompts": 1, "metric_type": "bpb"},
    # ),
    # "mmlu_other_var_bpb": (MMLU, {"dataset_name": "other", "prompts": 1, "metric_type": "bpb"}),
    # "mmlu_stem_mc_5shot": (MMLU, {"dataset_name": "stem", "prompts": 2, "mc_labels": True}),
    # "mmlu_humanities_mc_5shot": (MMLU, {"dataset_name": "humanities", "prompts": 2, "mc_labels": True}),
    # "mmlu_social_sciences_mc_5shot": (
    #     MMLU,
    #     {"dataset_name": "social_sciences", "prompts": 2, "mc_labels": True},
    # ),
    # "mmlu_other_mc_5shot": (MMLU, {"dataset_name": "other", "prompts": 2, "mc_labels": True}),
    # "mmlu_stem_mc_5shot_test": (
    #     MMLU,
    #     {"dataset_name": "stem", "split": "test", "prompts": 2, "mc_labels": True},
    # ),
    # "mmlu_humanities_mc_5shot_test": (
    #     MMLU,
    #     {"dataset_name": "humanities", "split": "test", "prompts": 2, "mc_labels": True},
    # ),
    # "mmlu_social_sciences_mc_5shot_test": (
    #     MMLU,
    #     {"dataset_name": "social_sciences", "split": "test", "prompts": 2, "mc_labels": True},
    # ),
    # "mmlu_other_mc_5shot_test": (
    #     MMLU,
    #     {"dataset_name": "other", "split": "test", "prompts": 2, "mc_labels": True},
    # ),
    # Paste in all oe-eval tasks from output of scripts/list_evals_from_oe_eval.py
    "arc_challenge_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "arc_challenge_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "arc_easy_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "arc_easy_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_mc_5shot": (OEEvalTask, {"dataset_path": "boolq", "dataset_name": "mc_5shot", "metric_type": "acc"}),
    "boolq_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "boolq_rc_0shot": (OEEvalTask, {"dataset_path": "boolq", "dataset_name": "rc_0shot", "metric_type": "acc"}),
    "boolq_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "boolq_rc_5shot": (OEEvalTask, {"dataset_path": "boolq", "dataset_name": "rc_5shot", "metric_type": "acc"}),
    "boolq_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "copa_rc_0shot": (OEEvalTask, {"dataset_path": "copa", "dataset_name": "rc_0shot", "metric_type": "acc"}),
    "copa_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "copa", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "copycolors_10way": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "10way", "metric_type": "acc"},
    ),
    "copycolors_10way_bpb": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "10way", "metric_type": "bpb"},
    ),
    "copycolors_xl_10way": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "xl_10way", "metric_type": "acc"},
    ),
    "copycolors_xl_10way_bpb": (
        OEEvalTask,
        {"dataset_path": "copycolors", "dataset_name": "xl_10way", "metric_type": "bpb"},
    ),
    "csqa_mc_5shot": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "mc_5shot", "metric_type": "acc"}),
    "csqa_mc_5shot_bpb": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "mc_5shot", "metric_type": "bpb"}),
    "csqa_rc_0shot": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"}),
    "csqa_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "csqa_rc_5shot": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"}),
    "csqa_rc_5shot_bpb": (OEEvalTask, {"dataset_path": "csqa", "dataset_name": "rc_5shot", "metric_type": "bpb"}),
    "hellaswag_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "hellaswag_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "hellaswag_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "openbookqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "openbookqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_mc_5shot": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "mc_5shot", "metric_type": "acc"}),
    "piqa_mc_5shot_bpb": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "mc_5shot", "metric_type": "bpb"}),
    "piqa_rc_0shot": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"}),
    "piqa_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "piqa_rc_5shot": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"}),
    "piqa_rc_5shot_bpb": (OEEvalTask, {"dataset_path": "piqa", "dataset_name": "rc_5shot", "metric_type": "bpb"}),
    "sciq_rc_0shot": (OEEvalTask, {"dataset_path": "sciq", "dataset_name": "rc_0shot", "metric_type": "acc"}),
    "sciq_rc_0shot_bpb": (OEEvalTask, {"dataset_path": "sciq", "dataset_name": "rc_0shot", "metric_type": "bpb"}),
    "socialiqa_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_0shot", "metric_type": "len_norm"},
    ),
    "socialiqa_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "socialiqa_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_5shot", "metric_type": "len_norm"},
    ),
    "socialiqa_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "mc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_rc_0shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_0shot", "metric_type": "acc"},
    ),
    "winogrande_rc_0shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_0shot", "metric_type": "bpb"},
    ),
    "winogrande_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_5shot", "metric_type": "acc"},
    ),
    "winogrande_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "rc_5shot", "metric_type": "bpb"},
    ),
}

# This standardizes the metrics we should eval for the ladder.
# Train and test sets are added when applicable.
# No subsampling happens in these sets.
label_to_task_map_new = {
    "arc_challenge_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_challenge_test_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_challenge_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "arc_challenge_test_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_challenge", "dataset_name": "test_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),  # this used to be acc
    "arc_easy_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_easy_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "arc_easy_test_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "arc_easy_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "arc_easy_test_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "arc_easy", "dataset_name": "test_mc_5shot", "metric_type": "bpb"},
    ),
    "boolq_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_rc_5shot", "metric_type": "acc"},
    ),  # kept acc here, since len_norm can bias towards "yes"
    "boolq_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "boolq_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "boolq_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_rc_5shot", "metric_type": "acc"},
    ),
    "boolq_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "boolq_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "boolq_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "boolq", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "csqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "csqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "csqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "csqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "csqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "csqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "csqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "csqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "csqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "hellaswag_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "hellaswag_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "hellaswag_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "hellaswag", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_test_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_rc_5shot", "metric_type": "len_norm"},
    ),
    "openbookqa_test_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_rc_5shot", "metric_type": "bpb"},
    ),
    "openbookqa_test_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_mc_5shot", "metric_type": "acc"},
    ),
    "openbookqa_test_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "openbookqa", "dataset_name": "test_mc_5shot", "metric_type": "bpb"},
    ),
    "piqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "piqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "piqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "piqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "piqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "piqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "piqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "piqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),
    "socialiqa_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "socialiqa_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "socialiqa_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "socialiqa_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "socialiqa", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_train_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_rc_5shot", "metric_type": "len_norm"},
    ),  # this used to be acc
    "winogrande_train_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_train_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_train_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "train_mc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_val_rc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_rc_5shot", "metric_type": "len_norm"},
    ),
    "winogrande_val_rc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_rc_5shot", "metric_type": "bpb"},
    ),
    "winogrande_val_mc_5shot": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_mc_5shot", "metric_type": "acc"},
    ),
    "winogrande_val_mc_5shot_bpb": (
        OEEvalTask,
        {"dataset_path": "winogrande", "dataset_name": "val_mc_5shot", "metric_type": "bpb"},
    ),
    "mmlu_stem_val_rc_var": (MMLU, {"dataset_name": "stem", "prompts": 1}),
    "mmlu_stem_val_rc_var_bpb": (MMLU, {"dataset_name": "stem", "prompts": 1, "metric_type": "bpb"}),
    "mmlu_stem_val_rc_5shot": (MMLU, {"dataset_name": "stem", "prompts": 2}),
    "mmlu_stem_val_rc_5shot_bpb": (MMLU, {"dataset_name": "stem", "prompts": 2, "metric_type": "bpb"}),
    "mmlu_stem_val_mc_5shot": (MMLU, {"dataset_name": "stem", "prompts": 2, "mc_labels": True}),
    "mmlu_stem_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "stem", "prompts": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_rc_var": (MMLU, {"dataset_name": "stem", "split": "test", "prompts": 1}),
    "mmlu_stem_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompts": 1, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_rc_5shot": (MMLU, {"dataset_name": "stem", "split": "test", "prompts": 2}),
    "mmlu_stem_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompts": 2, "metric_type": "bpb"},
    ),
    "mmlu_stem_test_mc_5shot": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompts": 2, "mc_labels": True},
    ),
    "mmlu_stem_test_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "stem", "split": "test", "prompts": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_rc_var": (MMLU, {"dataset_name": "humanities", "prompts": 1}),
    "mmlu_humanities_val_rc_var_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompts": 1, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_rc_5shot": (MMLU, {"dataset_name": "humanities", "prompts": 2}),
    "mmlu_humanities_val_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompts": 2, "metric_type": "bpb"},
    ),
    "mmlu_humanities_val_mc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "prompts": 2, "mc_labels": True},
    ),
    "mmlu_humanities_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "humanities", "prompts": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_humanities_test_rc_var": (MMLU, {"dataset_name": "humanities", "split": "test", "prompts": 1}),
    "mmlu_humanities_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompts": 1, "metric_type": "bpb"},
    ),
    "mmlu_humanities_test_rc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompts": 2},
    ),
    "mmlu_humanities_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompts": 2, "metric_type": "bpb"},
    ),
    "mmlu_humanities_test_mc_5shot": (
        MMLU,
        {"dataset_name": "humanities", "split": "test", "prompts": 2, "mc_labels": True},
    ),
    "mmlu_humanities_test_mc_5shot_bpb": (
        MMLU,
        {
            "dataset_name": "humanities",
            "split": "test",
            "prompts": 2,
            "mc_labels": True,
            "metric_type": "bpb",
        },
    ),
    "mmlu_social_sciences_val_rc_var": (MMLU, {"dataset_name": "social_sciences", "prompts": 1}),
    "mmlu_social_sciences_val_rc_var_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompts": 1, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_val_rc_5shot": (MMLU, {"dataset_name": "social_sciences", "prompts": 2}),
    "mmlu_social_sciences_val_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompts": 2, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_val_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "prompts": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "prompts": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_test_rc_var": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompts": 1},
    ),
    "mmlu_social_sciences_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompts": 1, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_test_rc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompts": 2},
    ),
    "mmlu_social_sciences_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompts": 2, "metric_type": "bpb"},
    ),
    "mmlu_social_sciences_test_mc_5shot": (
        MMLU,
        {"dataset_name": "social_sciences", "split": "test", "prompts": 2, "mc_labels": True},
    ),
    "mmlu_social_sciences_test_mc_5shot_bpb": (
        MMLU,
        {
            "dataset_name": "social_sciences",
            "split": "test",
            "prompts": 2,
            "mc_labels": True,
            "metric_type": "bpb",
        },
    ),
    "mmlu_other_val_rc_var": (MMLU, {"dataset_name": "other", "prompts": 1}),
    "mmlu_other_val_rc_var_bpb": (MMLU, {"dataset_name": "other", "prompts": 1, "metric_type": "bpb"}),
    "mmlu_other_val_rc_5shot": (MMLU, {"dataset_name": "other", "prompts": 2}),
    "mmlu_other_val_rc_5shot_bpb": (MMLU, {"dataset_name": "other", "prompts": 2, "metric_type": "bpb"}),
    "mmlu_other_val_mc_5shot": (MMLU, {"dataset_name": "other", "prompts": 2, "mc_labels": True}),
    "mmlu_other_val_mc_5shot_bpb": (
        MMLU,
        {"dataset_name": "other", "prompts": 2, "mc_labels": True, "metric_type": "bpb"},
    ),
    "mmlu_other_test_rc_var": (MMLU, {"dataset_name": "other", "split": "test", "prompts": 1}),
    "mmlu_other_test_rc_var_bpb": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompts": 1, "metric_type": "bpb"},
    ),
    "mmlu_other_test_rc_5shot": (MMLU, {"dataset_name": "other", "split": "test", "prompts": 2}),
    "mmlu_other_test_rc_5shot_bpb": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompts": 2, "metric_type": "bpb"},
    ),
    "mmlu_other_test_mc_5shot": (
        MMLU,
        {"dataset_name": "other", "split": "test", "prompts": 2, "mc_labels": True},
    ),
    "mmlu_other_test_mc_5shot_bpb": (
        MMLU,
        {
            "dataset_name": "other",
            "split": "test",
            "prompts": 2,
            "mc_labels": True,
            "metric_type": "bpb",
        },
    ),
}

label_to_task_map = {
    **TG_task_map,
    **label_to_task_map,
    **label_to_task_map_new,
    **Super_GLUE,
}
