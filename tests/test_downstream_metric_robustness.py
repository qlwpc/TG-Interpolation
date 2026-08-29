import json

import torch

import olmo.eval.downstream as downstream
from olmo.eval.downstream import (
    DecomposedICLMetric,
    ICLMetric,
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


def test_decomposed_metric_applies_terminal_mask_and_records_raw_evidence(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "per_choice.json"
    metric = DecomposedICLMetric(
        metric_type="acc",
        vocab_path=VOCAB,
        save_per_example_path=str(output_path),
    )
    monkeypatch.setattr(downstream, "get_global_rank", lambda: 0)

    terminal_id = 100
    nonterminal_id = metric.vocab.opening_non_terminals[0]
    continuations = torch.tensor(
        [
            [terminal_id, nonterminal_id, terminal_id],
            [terminal_id, nonterminal_id, terminal_id],
        ],
        dtype=torch.long,
    )
    vocab_size = metric.vocab.closing_non_terminals[1] + 1
    logits = torch.zeros((2, 3, vocab_size), dtype=torch.float32)
    # Choice 0 wins on lexical terminals; choice 1 wins only after its very
    # likely non-terminal is included in the full-format score.
    logits[0, 0, terminal_id] = 4.0
    logits[0, 1, nonterminal_id] = 0.0
    logits[0, 2, terminal_id] = 4.0
    logits[1, 0, terminal_id] = 0.0
    logits[1, 1, nonterminal_id] = 12.0
    logits[1, 2, terminal_id] = 0.0
    batch = {
        "doc_id": torch.tensor([0, 0]),
        "cont_id": torch.tensor([0, 1]),
        "continuation": continuations,
        "cont_len": torch.tensor([3, 3]),
        "ctx_len": torch.tensor([1, 1]),
        "dc_len": torch.tensor([1, 1]),
        "cont_str_len": torch.tensor([3, 3]),
        "cont_byte_len": torch.tensor([3, 3]),
        "label_id": torch.tensor([0, 0]),
    }

    metric.update(batch, logits)
    result = metric.compute()

    assert result["_raw_full"] == 0.0
    assert result["_raw_term"] == 1.0
    assert result["_raw_flip_rate"] == 1.0
    assert result["_raw_flip_to_correct"] == 1.0
    assert result["_raw_flip_to_wrong"] == 0.0
    assert result["_n_terminal_tokens"] == 4.0
    assert result["_n_nonterminal_tokens"] == 2.0
    assert result["_nt_logp_gap"] > 0.0

    rows = json.loads(output_path.read_text())
    assert len(rows) == 2
    assert all(row["n_terminal"] == 2 for row in rows)
    assert all(row["n_nonterminal"] == 1 for row in rows)
    assert all(abs(row["decomposition_residual"]) < 1e-6 for row in rows)
    assert rows[0]["term_score_raw"] > rows[1]["term_score_raw"]
    assert rows[0]["full_score_raw"] < rows[1]["full_score_raw"]


def test_metric_preserves_single_token_continuation_shape():
    metric = ICLMetric(metric_type="acc", vocab_path=VOCAB)
    token_id = 100
    logits = torch.zeros((1, 1, 256), dtype=torch.float32)
    logits[0, 0, token_id] = 4.0
    batch = {
        "doc_id": torch.tensor([0]),
        "cont_id": torch.tensor([0]),
        "continuation": torch.tensor([[token_id]], dtype=torch.long),
        "cont_len": torch.tensor([1]),
        "ctx_len": torch.tensor([1]),
        "dc_len": torch.tensor([1]),
        "cont_mask": torch.tensor([[1.0]]),
        "cont_str_len": torch.tensor([1]),
        "cont_byte_len": torch.tensor([1]),
        "label_id": torch.tensor([0]),
    }

    metric.update(batch, logits)
    result = metric.compute()

    assert result["_"] == 1.0


def test_decomposed_rankings_materialize_python_scalars(monkeypatch):
    metric = DecomposedICLMetric(metric_type="acc", vocab_path=VOCAB)
    metric.loglikelihoods = [
        torch.tensor([0.0, 0.0, -2.0]),
        torch.tensor([0.0, 1.0, -1.0]),
    ]
    metric.loglikelihoods_term = [
        torch.tensor([0.0, 0.0, -1.0]),
        torch.tensor([0.0, 1.0, -2.0]),
    ]
    metric.labels = [torch.tensor([0, 0, 0]), torch.tensor([0, 1, 0])]
    metric.decomposition_stats = [
        torch.tensor([0.0, 0.0, -2.0, -1.0, -1.0, 1.0, 1.0]),
        torch.tensor([0.0, 1.0, -1.0, -2.0, 1.0, 1.0, 1.0]),
    ]

    observed = []
    original = metric._ranking_summary

    def checked(full_scores, term_scores, label_dict):
        observed.extend(
            value
            for scores in (full_scores, term_scores)
            for choices in scores.values()
            for value in choices.values()
        )
        return original(full_scores, term_scores, label_dict)

    monkeypatch.setattr(metric, "_ranking_summary", checked)
    result = metric.compute()

    assert observed
    assert all(isinstance(value, float) for value in observed)
    assert result["_raw_flip_rate"] == 1.0
