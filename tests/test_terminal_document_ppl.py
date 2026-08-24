"""Exact terminal document-PPL dataset and metric regression tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from olmo.config import EvaluatorType
from olmo.eval.downstream import (
    TerminalDocumentPerplexityDataset,
    TerminalDocumentPerplexityMetric,
)
from olmo.train import Trainer


REPO = Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "dataset/bbc-news/TG_GPT2_tokenizer.json"
BOS = 50257
EOS = 50256


def build_tiny_terminal_corpus(
    tmp_path: Path, grammar_type: str = ""
) -> TerminalDocumentPerplexityDataset:
    # Two documents and three sentence records. BOS is context-only; both EOS
    # tokens are scored, so the exact denominator is 4 ordinary terminals + 2.
    records = (
        np.asarray([BOS, 10, 11], dtype=np.uint16),
        np.asarray([12, EOS], dtype=np.uint16),
        np.asarray([BOS, 13, EOS], dtype=np.uint16),
    )
    np.save(tmp_path / "test.npy", np.concatenate(records))
    np.save(
        tmp_path / "test_sent_index.npy",
        np.asarray([len(record) for record in records], dtype=np.uint16),
    )
    np.save(tmp_path / "test_doc_index.npy", np.asarray([2, 1], dtype=np.uint16))
    return TerminalDocumentPerplexityDataset(
        tokenizer=None,
        dataset_path=str(tmp_path),
        vocab_path=str(TOKENIZER),
        device_eval_batch_size=1,
        model_ctx_len=2048,
        transformer_grammar_type=grammar_type,
    )


def test_terminal_document_dataset_uses_one_path_and_exact_denominator(tmp_path):
    dataset = build_tiny_terminal_corpus(tmp_path)
    assert EvaluatorType("terminal_doc") is EvaluatorType.terminal_doc
    assert dataset.SENT_SIZE == 1
    assert len(dataset) == 3
    assert dataset.document_starts.tolist() == [0, 2]
    assert dataset.document_ends.tolist() == [2, 3]
    assert dataset.sent_doc_id[0:3].tolist() == [1, 1, 2]
    assert dataset[0]["input_ids"].tolist() == [BOS, 10, 11]
    assert dataset[1]["input_ids"].tolist() == [12, EOS]
    assert dataset[2]["input_ids"].tolist() == [BOS, 13, EOS]
    assert dataset.get_term_length() == [0, 2, 2, 2]
    assert sum(dataset.get_term_length()) == 6


def test_terminal_document_collation_commits_every_single_path(tmp_path):
    dataset = build_tiny_terminal_corpus(tmp_path)
    first = dataset.collate_fn([dataset[0]])
    assert first["doc_id"] == 1
    assert first["add_len"] == 3
    assert first["input_ids"].tolist() == [[BOS, 10, 11]]
    second = dataset.collate_fn([dataset[1]])
    assert second["doc_id"] == 1
    assert second["add_len"] == 2
    third = dataset.collate_fn([dataset[2]])
    assert third["doc_id"] == 2
    assert third["add_len"] == 3


def test_pause_document_projection_preserves_phase_and_masks_pause_tokens(tmp_path):
    # pause1/2 inserts a copied pause token after every two *document-global*
    # raw positions.  The second sentence of document 1 therefore begins with
    # phase 1, whereas the first sentence of document 2 resets to phase 0.
    dataset = build_tiny_terminal_corpus(tmp_path, grammar_type="pause1/2")
    assert dataset[0]["input_ids"].tolist() == [BOS, 10, 10, 11]
    assert dataset[0]["label_mask"].tolist() == [True, True, False, True]
    assert dataset[1]["input_ids"].tolist() == [12, 12, EOS]
    assert dataset[1]["label_mask"].tolist() == [True, False, True]
    assert dataset[2]["input_ids"].tolist() == [BOS, 13, 13, EOS]
    assert dataset[2]["label_mask"].tolist() == [True, True, False, True]
    # The denominator remains raw terminal/EOS tokens, never pause positions.
    assert dataset.get_term_length() == [0, 2, 2, 2]
    batch = dataset.collate_fn([dataset[0]])
    assert batch["label_mask"].tolist() == [[True, True, False, True]]


def test_terminal_document_metric_is_global_token_weighted(tmp_path):
    dataset = build_tiny_terminal_corpus(tmp_path)
    metric = TerminalDocumentPerplexityMetric(
        term_length=dataset.get_term_length(),
        dataset_length=len(dataset),
    )
    for index, nll in enumerate((1.0, 2.0, 3.0)):
        metric.update(
            {"index": torch.tensor([index])},
            torch.tensor([nll], dtype=torch.float32),
        )
    # Total NLL / exact scored-token count = 6 / 6 = 1.
    assert metric.compute().item() == pytest.approx(math.e, rel=1e-6)


def test_terminal_document_metric_rejects_partial_evaluation(tmp_path):
    dataset = build_tiny_terminal_corpus(tmp_path)
    metric = TerminalDocumentPerplexityMetric(
        term_length=dataset.get_term_length(), dataset_length=len(dataset)
    )
    metric.update({"index": torch.tensor([0])}, torch.tensor([1.0]))
    with pytest.raises(RuntimeError, match="every sentence exactly once"):
        metric.compute()


class _UniformCacheModel:
    def __init__(self, vocab_size: int = 50259):
        self.vocab_size = vocab_size
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, use_cache=False, **kwargs):
        batch, length = input_ids.shape
        past = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        self.calls.append((input_ids.clone(), past, use_cache))
        logits = torch.zeros(batch, length, self.vocab_size)
        if len(self.calls) == 2:
            # Make the current sentence's final distribution strongly prefer
            # its own first token. Using this instead of the frozen preceding
            # distribution would make the regression assertion fail.
            logits[:, -1, 12] = 10.0
        cache = torch.zeros(batch, 1, past + length, 1)
        return SimpleNamespace(
            logits=logits,
            attn_key_values=[(cache, cache.clone())],
        )


class _LossCollector:
    def __init__(self):
        self.losses = []
        self.eval_loader = SimpleNamespace(dataset=SimpleNamespace(SENT_SIZE=1))

    def update_metrics(self, batch, ce_loss, logits=None):
        self.losses.extend(ce_loss.detach().cpu().tolist())


def test_terminal_document_step_scores_sentence_initial_from_previous_cache():
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.cfg = SimpleNamespace(
        model=SimpleNamespace(max_sequence_length=8, pad_token_id=63),
        autocast_precision=torch.float32,
    )
    trainer.dist_model = _UniformCacheModel()
    trainer.num_evaled = 0
    trainer.cur_length = 0
    trainer.cur_doc_id = 0
    trainer.doc_kv_cache = None
    trainer.kv_to_update = None
    trainer.past_key_values = None
    trainer.last_logProb = None
    trainer.logits_to_update = None
    evaluator = _LossCollector()

    trainer.TG_doc_eval_step(
        {
            "doc_id": 1,
            "input_ids": torch.tensor([[BOS, 10, 11]]),
            "add_len": 3,
        },
        evaluator,
    )
    trainer.TG_doc_eval_step(
        {
            "doc_id": 1,
            "input_ids": torch.tensor([[12, EOS]]),
            "add_len": 2,
        },
        evaluator,
    )

    log_vocab = math.log(50259)
    # First sentence scores 10,11. Second scores its cross-sentence initial 12
    # from the frozen previous-sentence distribution, then EOS internally.
    assert evaluator.losses == pytest.approx([2 * log_vocab, 2 * log_vocab])
    assert trainer.dist_model.calls[0][1:] == (0, True)
    assert trainer.dist_model.calls[1][1:] == (3, True)
