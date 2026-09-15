"""Model-selected document histories must agree across cache, mask and metrics."""
from types import SimpleNamespace

import pytest
import torch

from olmo.config import EvaluatorType
from olmo.eval.downstream import TGPerplexityDocumentLevelMetric
from olmo.eval.evaluator import Evaluator
from olmo.train import Trainer
from test_terminal_document_ppl import TOKENIZER


class RecordingBias:
    def __init__(self):
        self.commits = []
        self.resets = 0
        self.history = 0

    def reset_state(self):
        self.resets += 1
        self.history = 0

    def __call__(self, row, update_state=False):
        if update_state:
            self.commits.append(row.clone())
            self.history += int((row != 511).sum())
        return torch.zeros(len(row), self.history + len(row)), torch.ones(len(row), dtype=torch.bool)


class CandidateCacheModel:
    def __init__(self):
        self.pasts = []

    def __call__(self, input_ids, past_key_values, use_cache, attention_bias, **kwargs):
        assert use_cache
        batch, length = input_ids.shape
        past = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        assert attention_bias.shape == (batch, 1, length, past + length)
        self.pasts.append(None if past_key_values is None else past_key_values[0][0].clone())
        logits = torch.zeros(batch, length, 512)
        # The unique winner is candidate257 in the third microbatch.
        winner = input_ids[:, 1] == 258
        logits[winner, 0, 258] = 10
        cache = input_ids[:, None, :, None].float()
        if past_key_values is not None:
            cache = torch.cat((past_key_values[0][0], cache), dim=2)
        return SimpleNamespace(logits=logits, attn_key_values=[(cache, cache.clone())])


def test_parse_tree_300_candidates_commit_late_winner_and_reset():
    bias = RecordingBias()
    dataset = SimpleNamespace(SENT_SIZE=300, generate_TG_attention_bias=bias)
    metric = TGPerplexityDocumentLevelMetric(vocab_path=str(TOKENIZER), term_length=[6],
                                            dataset_length=900, samples_per_sent=300)
    evaluator = Evaluator("parse_tree", EvaluatorType.tg_doc,
                          SimpleNamespace(dataset=dataset), metric)
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.cfg = SimpleNamespace(model=SimpleNamespace(max_sequence_length=32, pad_token_id=511),
                                  autocast_precision=torch.float32)
    trainer.dist_model = CandidateCacheModel()
    trainer.num_evaled = 0
    trainer.cur_doc_id = None
    for sentence, doc_id in enumerate((1, 1, 2)):
        for start in range(0, 300, 100):
            ids = torch.tensor([[0, i + 1, 3] for i in range(start, start + 100)])
            lengths = torch.full((100,), 3)
            if sentence == 0 and start == 200:
                ids[57, 2] = 511
                lengths[57] = 2
            if sentence > 0:
                ids[:, 1] = 1  # All candidates tie; retain candidate0.
            trainer.TG_doc_eval_step({"doc_id": doc_id, "input_ids": ids,
                                     "candidate_lengths": lengths,
                                     "index": torch.arange(sentence * 300 + start, sentence * 300 + start + 100)}, evaluator)
            assert len(bias.commits) == sentence + int(start == 200)
        if sentence == 0:
            assert trainer.doc_kv_cache[0][0].flatten().tolist() == [0, 258]
            assert bias.commits[0].tolist() == [0, 258, 511]
    assert bias.resets == 2
    assert trainer.dist_model.pasts[3][0].flatten().tolist() == [0, 258]
    assert trainer.dist_model.pasts[6] is None
    assert trainer.doc_kv_cache[0][0].flatten().tolist() == [0, 1, 3]
    metrics = evaluator.compute_metrics()
    assert metrics["eval/downstream/parse_tree_non_candidate0_ratio"] == pytest.approx(1 / 3)
    evaluator.reset_metrics()
    assert not bool(metric.evaluated_candidates.any())


def test_document_collation_does_not_advance_mask_history(tmp_path):
    from test_terminal_document_ppl import build_tiny_terminal_corpus
    dataset = build_tiny_terminal_corpus(tmp_path)
    bias = RecordingBias()
    dataset.generate_TG_attention_bias = bias
    # Collate multiple sentences ahead of scoring to simulate DataLoader prefetch.
    batches = [dataset.collate_fn([dataset[i]]) for i in range(3)]
    assert bias.resets == 0
    assert bias.commits == []
    assert all("attention_bias" not in batch for batch in batches)
    assert [batch["candidate_lengths"].tolist() for batch in batches] == [[3], [2], [3]]
