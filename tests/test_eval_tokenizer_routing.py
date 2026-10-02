"""Corpus routing regressions; fixtures require no checkpoints or large datasets."""

import json
import numpy as np
import pytest
import torch
import yaml

from olmo.config import TrainConfig
from olmo.exceptions import OLMoCliError
from scripts import init_cfg_and_sbatch as campaign


FINEWEB = [f"{name}-fwedu-1B" for name in ("terminal", "tree", "tgtree", "pause1", "pause2")]
BBC = ["terminal", "tree", "tgtree", "pause1", "pause2", "terminal-bbc-1B", "tree-bbc-1B",
       "terminal-1B", "tree-1B"]


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)

    def write(modelname, **model_overrides):
        qwen = modelname in FINEWEB
        grammar = campaign.Models[modelname]["model.transformer_grammar_type"]
        model = dict(
            vocab_size=151732 if qwen else 50320,
            eos_token_id=151643 if qwen else 50256,
            pad_token_id=151670 if qwen else 50258,
            transformer_grammar_type=grammar,
            pause_token_id=(151673 if qwen else 50261) if grammar.startswith("pause") else None,
        )
        model.update(model_overrides)
        path = tmp_path / campaign.model_paths[modelname].lstrip("/") / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(dict(
            run_name="old_training_run", workspace="${workspace}/TG-Interpolation", model=model,
            tokenizer=dict(identifier="old-tokenizer.json", vocabulary="old-tokenizer.json",
                           use_bracket_mapping=not qwen),
            data=dict(paths=["old-shard.npy", "another-old-shard.npy"],
                      memmap_dtype="uint16" if qwen else "uint32",
                      label_mask_paths=["old-mask.npy"]),
            distributed_strategy="fsdp", finetune_task="boolq", eval_subset_num_batches=3,
        )))
        return path

    return write


@pytest.mark.parametrize("modelname", BBC + FINEWEB)
@pytest.mark.parametrize("task", ["SG", "blimp"])
def test_generated_configs_keep_corpus_and_model_identity(checkpoint, tmp_path, modelname, task):
    checkpoint(modelname)
    output = tmp_path / "eval.yaml"
    campaign.generate_config(output, [], "RTX3090", modelname, task)
    cfg = TrainConfig.load(output, overrides=[f"workspace={tmp_path}"], validate_paths=False)
    qwen = modelname in FINEWEB
    expected = "dataset/TG_QWEN3_tokenizer.json" if qwen else "dataset/bbc-news/TG_GPT2_tokenizer.json"
    assert cfg.tokenizer.identifier == cfg.tokenizer.vocabulary == str(tmp_path / expected)
    assert cfg.tokenizer.use_bracket_mapping is qwen
    assert cfg.data.memmap_dtype == ("uint32" if qwen else "uint16")
    assert len(cfg.data.paths) == 1
    assert cfg.data.label_mask_paths is None
    assert cfg.finetune_task is None
    assert cfg.eval_no_save and cfg.reset_optimizer_state and cfg.reset_trainer_state
    assert cfg.max_duration == "0ep" and cfg.stop_at == 0 and cfg.eval_subset_num_batches == -1
    if cfg.model.transformer_grammar_type.startswith("pause"):
        assert cfg.model.pause_token_id == (151673 if qwen else 50261)
    if task == "SG":
        assert cfg.evaluators[0].sg_dataset_path == str(tmp_path / "evaluation/SG/tokenized")
    if qwen:
        assert cfg.distributed_strategy == "ddp" and cfg.fsdp is None
        from olmo.data import build_memmap_dataset

        dataset = build_memmap_dataset(cfg, cfg.data)
        assert len(dataset) >= 2
        assert torch.all(dataset[0]["input_ids"] >= 151643)
        buffer = np.load(cfg.data.paths[0])
        assert buffer.dtype == np.uint32 and np.all(buffer == 151643)
    else:
        assert "/dataset/bbc-news/" in cfg.data.paths[0]


@pytest.mark.parametrize("modelname,wrong_vocab", [("terminal", 151732), (FINEWEB[0], 50320)])
def test_rejects_checkpoint_from_other_tokenizer_family(checkpoint, tmp_path, modelname, wrong_vocab):
    checkpoint(modelname, vocab_size=wrong_vocab)
    with pytest.raises(OLMoCliError, match="family mismatch"):
        campaign.generate_config(tmp_path / "eval.yaml", [], "RTX3090", modelname, "SG")


def test_overrides_cannot_relabel_checkpoint_token_ids(checkpoint, tmp_path):
    checkpoint(FINEWEB[0])
    with pytest.raises(OLMoCliError, match="family mismatch"):
        campaign.generate_config(tmp_path / "eval.yaml", ["model.pad_token_id=50258"],
                                 "RTX3090", FINEWEB[0], "blimp")


def test_qwen_pause_cannot_use_gpt2_sep(checkpoint, tmp_path):
    checkpoint(FINEWEB[3], pause_token_id=50261)
    with pytest.raises(OLMoCliError, match="pause_token_id=151673"):
        campaign.generate_config(tmp_path / "eval.yaml", [], "RTX3090", FINEWEB[3], "blimp")


@pytest.mark.parametrize("task", ["docppl", "xsum_finetune", "boolq", "pretrain_terminal"])
def test_fineweb_cannot_fall_back_to_bbc_only_recipes(tmp_path, task):
    with pytest.raises(OLMoCliError, match="SG/blimp only"):
        campaign.generate_config(tmp_path / "eval.yaml", [], "RTX3090", FINEWEB[0], task)


@pytest.mark.parametrize("grammar,force_terminal", [
    ("terminal", False), ("pause1", False), ("pause2", False),
    ("terminal", True), ("pause1", True), ("tree", True), ("tgtree", True),
    ("tree", False), ("tgtree", False),
])
def test_qwen_blimp_reads_qwen_candidates_in_all_protocols(tmp_path, grammar, force_terminal):
    from olmo.eval.downstream import BLiMPApproximationDataset

    # Minimal vocabulary consumed by the real C++ format converters. Distinct
    # candidate rows expose wrong K=1 remapping; high IDs expose uint16 mixing.
    vocab = tmp_path / "TG_QWEN3_tokenizer.json"
    vocab.write_text(json.dumps({"added_tokens": [
        {"id": i, "content": t} for i, t in (
            (151643, "<|endoftext|>"), (151669, "<|beginoftext|>"),
            (151670, "<|pad|>"), (151673, "<|SEP|>"),
            (151674, "<(S>"), (151675, "<S)>"),
        )
    ]}))
    data = np.empty((1, 600, 3), dtype=np.uint32)
    data[:, :, 0] = 151674
    data[0, :, 1] = np.arange(600) + 70000
    data[:, :, 2] = 151675
    np.save(tmp_path / "blimp_tree_300_qwen.npy", data)
    # No GPT-2 file is present: an incorrect fallback fails at dataset creation.
    ds = BLiMPApproximationDataset(
        tokenizer=None, dataset_path=str(tmp_path), vocab_path=str(vocab),
        transformer_grammar_type=grammar, force_terminal=force_terminal,
        pair_per_task=1, pause_token_id=151673,
    )
    terminal = force_terminal or grammar.startswith(("terminal", "pause"))
    assert ds.SENT_SIZE == (1 if terminal else 300)
    first = ds[0]["input_ids"].tolist()
    bad = ds[1 if terminal else 300]["input_ids"].tolist()
    assert 70000 in first and 70300 in bad
    if terminal:
        assert 151674 not in first and 151675 not in first
        assert first.count(151673) == ({"pause1": 1, "pause2": 2}.get(grammar, 0))
    else:
        assert first.count(151675) == (2 if grammar == "tgtree" else 1)
