from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from olmo.eval.downstream import (
    RougeMetric,
    SG_SCORE_LOG_LEVEL,
    SyntacticGeneralizationMetric,
    XSUM_PREDICTION_LOG_LEVEL,
    XsumDataset,
)


def _mock_xsum_dataset(grammar_type: str, pause_token_id: int | None):
    dataset = object.__new__(XsumDataset)
    dataset.passages = ["passage"]
    dataset.train_summary = ["summary"]
    dataset.gold_summary = ["gold"]
    dataset.model_ctx_len = 64
    dataset.prompts_TG_tokens = np.array([30, 31])
    dataset.transformer_grammar_type = grammar_type
    dataset.generate_TG_attention_bias = None
    dataset.pause_token_id = pause_token_id

    vocab = MagicMock()
    vocab.bos = 1
    vocab.eos = 2
    vocab.convert_treenpy_to_TG.side_effect = lambda values: np.asarray(values)
    vocab.convert_treenpy_to_terminal.side_effect = lambda values: np.asarray(values)
    dataset.vocab = vocab
    dataset.tokenizer = MagicMock()
    return dataset


@pytest.mark.parametrize(
    ("grammar_type", "conversion_name"),
    [
        ("tree_noont", "convert_treenpy_to_noont"),
        ("tree_compress", "convert_treenpy_to_compress"),
        ("tree_triplecnt", "convert_treenpy_to_triplecnt"),
    ],
)
def test_structural_xsum_label_mask_uses_converted_target_length(
    grammar_type, conversion_name
):
    dataset = object.__new__(XsumDataset)
    dataset.passages = ["passage"]
    dataset.train_summary = ["summary"]
    dataset.gold_summary = ["gold"]
    dataset.model_ctx_len = 64
    dataset.prompts_TG_tokens = np.array([30, 31])
    dataset.transformer_grammar_type = grammar_type
    dataset.generate_TG_attention_bias = None
    dataset.pause_token_id = None

    vocab = MagicMock()
    vocab.bos = 1
    vocab.eos = 2
    vocab.convert_treenpy_to_TG.side_effect = lambda values: np.asarray(values)
    vocab.convert_TGnpy_to_tree.side_effect = lambda values: np.asarray(values)
    conversions = {
        "convert_treenpy_to_noont": lambda values: np.asarray(values)[::2],
        "convert_treenpy_to_compress": lambda values: np.asarray(values)[:-1],
        "convert_treenpy_to_triplecnt": lambda values: np.repeat(np.asarray(values), 3),
    }
    for name, fn in conversions.items():
        getattr(vocab, name).side_effect = fn
    dataset.vocab = vocab

    encoded = {
        "passage": np.array([10, 11, 12]),
        "summary": np.array([20, 21, 22, 23]),
    }
    with patch(
        "olmo.eval.downstream.encode_TG_string",
        side_effect=lambda tokenizer, text, **kwargs: encoded[text],
    ):
        dataset.tokenizer = MagicMock()
        item = dataset[0]

    raw_target = np.array([20, 21, 22, 23, vocab.eos])
    intended_target = conversions[conversion_name](raw_target)
    assert int(item["label_mask"].sum()) == len(intended_target)
    assert item["label_mask"][-len(intended_target):].all()


def test_rouge_metric_keeps_gather_state_on_cpu_and_logs_every_rank():
    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda ids, **kwargs: " ".join(map(str, ids))
    tokenizer.encode.return_value = [7, 8]
    metric = RougeMetric(tokenizer=tokenizer)
    predictions = torch.tensor([[4, 5, 6]])

    with patch("olmo.eval.downstream.get_global_rank", return_value=3), patch(
        "olmo.eval.downstream.log.log"
    ) as log_call:
        metric.update({"input_ids": torch.tensor([[1, 2]])}, predictions, ["gold"])

    assert metric.predictions[0].device.type == "cpu"
    assert metric.references[0].device.type == "cpu"
    assert XSUM_PREDICTION_LOG_LEVEL > 20  # survives the default INFO rank filter
    level, message = log_call.call_args.args
    assert level == XSUM_PREDICTION_LOG_LEVEL
    assert "[global_rank=3] <New Passage>:" in message


def test_sg_metric_logs_each_rank_with_rank_tag():
    metric = SyntacticGeneralizationMetric()
    scores = {
        "sub_no-matrix": 2.0,
        "no-sub_no-matrix": 1.0,
        "sub_matrix": 1.0,
        "no-sub_matrix": 2.0,
    }

    with patch("olmo.eval.downstream.get_global_rank", return_value=2), patch(
        "olmo.eval.downstream.log.log"
    ) as log_call:
        metric.update("subordination", scores)

    assert SG_SCORE_LOG_LEVEL > 20
    assert SG_SCORE_LOG_LEVEL != XSUM_PREDICTION_LOG_LEVEL
    level, message = log_call.call_args.args
    assert level == SG_SCORE_LOG_LEVEL
    assert "[global_rank=2] task is subordination" in message


def test_generated_xsum_config_preserves_eval_batch_size_one(tmp_path):
    from scripts.init_cfg_and_sbatch import generate_config

    output = tmp_path / "xsum.yaml"
    generate_config(output, [], Device="RTX3090", modelname="terminal", task="xsum_finetune")
    text = output.read_text()

    assert "device_eval_batch_size: 1" in text
    assert "device_eval_batch_size: ${device_train_microbatch_size}" not in text


@pytest.mark.parametrize(
    ("grammar_type", "expected_factor"),
    [("pause1", 2), ("pause2", 3)],
)
def test_pause_xsum_finetune_uses_sep_and_expands_full_summary_mask(
    grammar_type, expected_factor
):
    dataset = _mock_xsum_dataset(grammar_type, pause_token_id=99)
    encoded = {
        "passage": np.array([10, 11, 12]),
        "summary": np.array([20, 21, 22, 23]),
    }
    with patch(
        "olmo.eval.downstream.encode_TG_string",
        side_effect=lambda tokenizer, text, **kwargs: encoded[text],
    ):
        item = dataset[0]

    # Four summary tokens plus EOS, with every owned pause target included in
    # ordinary (non-label) pause finetuning.
    expected_targets = expected_factor * 5
    mask = np.asarray(item["label_mask"], dtype=np.bool_)
    assert int(mask.sum()) == expected_targets
    assert mask[-expected_targets:].all()
    assert not mask[:-expected_targets].any()
    assert np.count_nonzero(np.asarray(item["input_ids"])[mask] == 99) == (
        expected_targets - 5
    )


def test_pause_label_xsum_masks_forced_pause_targets():
    dataset = _mock_xsum_dataset("pause1_label", pause_token_id=99)
    encoded = {
        "passage": np.array([10, 11, 12]),
        "summary": np.array([20, 21, 22, 23]),
    }
    with patch(
        "olmo.eval.downstream.encode_TG_string",
        side_effect=lambda tokenizer, text, **kwargs: encoded[text],
    ):
        item = dataset[0]

    mask = np.asarray(item["label_mask"], dtype=np.bool_)
    assert int(mask.sum()) == 5
    assert not np.any(np.asarray(item["input_ids"])[mask] == 99)


def test_finetune_loader_forwards_checkpoint_pause_contract(tmp_path):
    from olmo.data import build_train_dataloader

    captured = {}

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    cfg = MagicMock()
    cfg.device_train_batch_size = 1
    cfg.finetune_task = "xsum"
    cfg.model.max_sequence_length = 2049
    cfg.model.transformer_grammar_type = "pause2"
    cfg.model.pause_token_id = 50261
    cfg.tokenizer.vocabulary = "tokenizer.json"
    cfg.save_folder = str(tmp_path)
    cfg.save_overwrite = True
    cfg.data.seed = None
    cfg.seed = 6198
    cfg.epoch = 0
    cfg.global_train_batch_size = 1
    cfg.data.drop_last = False
    cfg.data.index_world_size = None
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.prefetch_factor = None
    cfg.data.persistent_workers = False
    cfg.data.timeout = 0

    with patch.dict(
        "olmo.eval.downstream.label_to_task_map", {"xsum": FakeDataset}, clear=False
    ), patch("olmo.data.Tokenizer.from_train_config", return_value=MagicMock()), patch(
        "olmo.data.DataCollator.from_train_config", return_value=MagicMock()
    ), patch("olmo.data.get_TG_generate_bias_func", return_value=None), patch(
        "olmo.data.get_global_rank", return_value=0
    ), patch("olmo.data.barrier"), patch(
        "olmo.data.IterableDataset", side_effect=lambda dataset, *args, **kwargs: dataset
    ), patch("olmo.data.DataLoader", side_effect=lambda dataset, **kwargs: dataset):
        build_train_dataloader(cfg)

    assert captured["model_ctx_len"] == 2049
    assert captured["pause_token_id"] == 50261
    assert captured["transformer_grammar_type"] == "pause2"
