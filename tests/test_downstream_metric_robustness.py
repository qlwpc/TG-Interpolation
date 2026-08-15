import json

import torch

import olmo.eval.downstream as downstream
from olmo.eval.downstream import (
    DecomposedICLMetric,
    TGPerplexityDocumentLevelMetric,
)


VOCAB = "./dataset/bbc-news/TG_GPT2_tokenizer.json"


def test_doc_ppl_reset_clears_all_reduced_state():
    metric = TGPerplexityDocumentLevelMetric(
        vocab_path=VOCAB,
        term_length=[4, 5],
        dataset_length=6,
        samples_per_sent=3,
    )
    metric.to(torch.device("cpu"))
    metric.loglikelihoods.fill_(7.0)
    metric.cur_sent = 9
    metric.cur_batch = 2

    metric.reset()

    assert torch.count_nonzero(metric.loglikelihoods).item() == 0
    assert metric.cur_sent == 0
    assert metric.cur_batch == 0


def test_decomposed_dump_is_rank0_only_and_atomic(tmp_path, monkeypatch):
    output_path = tmp_path / "nested" / "per_example.json"
    metric = DecomposedICLMetric(
        vocab_path=VOCAB,
        save_per_example_path=str(output_path),
    )
    full = {0: {0: torch.tensor(-1.0), 1: torch.tensor(-2.0)}}
    term = {0: {0: torch.tensor(-1.5), 1: torch.tensor(-2.5)}}
    labels = {0: {0}}

    monkeypatch.setattr(downstream, "get_global_rank", lambda: 1)
    metric._save_per_example(full, term, labels)
    assert not output_path.exists()

    monkeypatch.setattr(downstream, "get_global_rank", lambda: 0)
    metric._save_per_example(full, term, labels)

    with output_path.open() as f:
        rows = json.load(f)
    assert rows == [
        {
            "doc_id": 0,
            "cont_id": 0,
            "label": 0,
            "full_score": -1.0,
            "term_score": -1.5,
        },
        {
            "doc_id": 0,
            "cont_id": 1,
            "label": 0,
            "full_score": -2.0,
            "term_score": -2.5,
        },
    ]
    assert list(output_path.parent.glob("*.tmp.rank0.*")) == []
