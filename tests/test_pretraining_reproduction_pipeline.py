from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from datatools.parse_pretrain_data.assemble_streams import assemble
from datatools.parse_pretrain_data.build_pretrain_data import (
    allocate_counts,
    fineweb_group,
    make_split_indices,
    read_bbc_configs,
    validate_fineweb_arrow_shards,
)
from datatools.parse_pretrain_data.get_TG_tokenizer import validate_paper_layout
from scripts.prepare_paper_pretraining import (
    MANIFEST_PATH,
    REPO_ROOT,
    audit_source_config,
    data_paths,
    load_manifest,
    materialize_config,
    validate_training_inputs,
)


def test_corpus_task_manifests_are_complete():
    configs = read_bbc_configs()
    assert len(configs) == 94
    assert configs[0] == "CC-MAIN-2013-20"
    assert configs[-1] == "CC-MAIN-2023-50"
    groups = [fineweb_group(index) for index in range(246)]
    assert len(groups) == len(set(groups)) == 246
    assert groups[0] == ".*-00(000|001|002|003).*arrow"
    assert groups[-1] == ".*-00(980|981|982|983).*arrow"
    validate_fineweb_arrow_shards(
        [
            Path(f"fineweb-edu-train-{index:05d}-of-00984.arrow")
            for index in range(984)
        ]
    )


def test_available_paper_tokenizers_have_the_registered_layout():
    from tokenizers import Tokenizer

    for model_name, relative in (
        ("gpt2", "dataset/bbc-news/TG_GPT2_tokenizer.json"),
        ("qwen3", "dataset/TG_QWEN3_tokenizer.json"),
    ):
        path = REPO_ROOT / relative
        if path.is_file():
            validate_paper_layout(model_name, Tokenizer.from_file(str(path)))


def test_split_allocation_and_indices_are_exact_and_disjoint(tmp_path):
    counts = {"a": 5, "b": 7, "c": 11}
    allocated = allocate_counts(counts, 9)
    assert sum(allocated.values()) == 9
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    for stem, count in counts.items():
        (parsed / f"{stem}.txt").write_text("doc\n" * count)
    dev, test = make_split_indices(parsed, list(counts), dev_size=5, test_size=6, seed=42)
    assert sum(map(len, dev.values())) == 5
    assert sum(map(len, test.values())) == 6
    for stem in counts:
        assert set(dev[stem]).isdisjoint(test[stem])


def _write_shard(path: Path, documents: list[list[int]], dtype=np.uint16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = np.asarray([token for document in documents for token in document], dtype=dtype)
    np.save(path, stream)


def test_stream_assembly_preserves_format_alignment(tmp_path):
    input_root = tmp_path / "shards"
    output_root = tmp_path / "final"
    order = ["a", "b"]
    # Every document begins with the synthetic BOS=99. Structural formats have
    # different lengths but identical document counts and selections.
    for input_format, extra in (("terminal", []), ("tree", [70]), ("tg", [70, 70])):
        _write_shard(
            input_root / input_format / "a.npy",
            [[99, 1, *extra], [99, 2, *extra], [99, 3, *extra]],
        )
        _write_shard(
            input_root / input_format / "b.npy",
            [[99, 4, *extra], [99, 5, *extra]],
        )
    dev = {"a": {1}}
    test = {"b": {0}}
    result = assemble(input_root, output_root, order, dev, test, bos_token_id=99)
    for input_format in ("terminal", "tree", "tg"):
        assert result[input_format]["train"]["documents"] == 3
        assert result[input_format]["dev"]["documents"] == 1
        assert result[input_format]["test"]["documents"] == 1
        dev_tokens = np.load(output_root / input_format / "dev.npy")
        assert dev_tokens[0] == 99 and dev_tokens[1] == 2
        test_tokens = np.load(output_root / input_format / "test.npy")
        assert test_tokens[0] == 99 and test_tokens[1] == 4


def test_paper_manifest_covers_all_audited_model_groups():
    manifest = load_manifest(MANIFEST_PATH)
    runs = manifest["runs"]
    assert len(runs) == 27
    assert len({run["id"] for run in runs}) == 27
    assert sum(run["group"] == "bbc-100m" for run in runs) == 14
    assert sum(run["group"] == "bbc-100m-baselines" for run in runs) == 2
    assert sum(run["group"] == "bbc-500m" for run in runs) == 4
    assert sum(run["group"] == "bbc-1b-supplementary" for run in runs) == 2
    assert sum(run["group"] == "fineweb-edu-1b" for run in runs) == 5
    bbc_100m = {run["model"] for run in runs if run["group"] == "bbc-100m"}
    assert {
        "Terminal",
        "Tree",
        "TGTree",
        "TG",
        "TGNomask",
        "TGNomask-Aug",
        "Tree-NoONT",
        "Tree-Compress",
        "Tree-TripleCNT",
        "Tree-Shuffle",
        "TGNomask-Mix-TG",
        "TGTree-Mix-TG",
        "Pause-1",
        "Pause-2",
    } == bbc_100m
    assert {
        run["model"] for run in runs if run["group"] == "bbc-100m-baselines"
    } == {"TreeReg-L9", "Pushdown"}
    assert {
        run["model"] for run in runs if run["group"] == "bbc-500m"
    } == {"Terminal", "Tree", "TGTree", "TGNomask-Aug"}
    assert {
        run["model"] for run in runs if run["group"] == "bbc-1b-supplementary"
    } == {"Terminal", "Tree"}
    assert {
        run["model"] for run in runs if run["group"] == "fineweb-edu-1b"
    } == {"Terminal", "Tree", "TGTree", "Pause-1", "Pause-2"}
    for run in runs:
        assert (REPO_ROOT / run["source_config"]).is_file()


def test_every_checkpoint_source_matches_the_audited_protocol():
    from omegaconf import OmegaConf

    for run in load_manifest(MANIFEST_PATH)["runs"]:
        audit_source_config(run, OmegaConf.load(REPO_ROOT / run["source_config"]))


def test_clean_configs_are_materialized_from_checkpoint_protocol(tmp_path):
    manifest = load_manifest(MANIFEST_PATH)
    by_id = {run["id"]: run for run in manifest["runs"]}
    terminal = by_id["bbc_100m_terminal"]
    terminal_output = tmp_path / "terminal.yaml"
    materialize_config(
        terminal,
        REPO_ROOT / terminal["source_config"],
        terminal_output,
        str(REPO_ROOT),
        "test_bbc_terminal",
    )
    from olmo.config import TrainConfig

    cfg = TrainConfig.load(terminal_output, validate_paths=False)
    assert cfg.model.transformer_grammar_type == "terminal"
    assert cfg.optimizer.learning_rate == 0.005
    assert cfg.global_train_batch_size == 144
    assert cfg.load_path is None
    assert cfg.data.paths == [str(REPO_ROOT / "dataset/bbc-news/terminal/train.npy")]
    assert all("hellaswag" not in str(path) for path in cfg.data.paths)

    pause = by_id["fwedu_1b_pause2_sep"]
    pause_output = tmp_path / "pause2.yaml"
    materialize_config(
        pause,
        REPO_ROOT / pause["source_config"],
        pause_output,
        str(REPO_ROOT),
        "test_fwedu_pause2",
    )
    cfg = TrainConfig.load(pause_output, validate_paths=False)
    assert cfg.model.transformer_grammar_type == "pause2"
    assert cfg.model.pause_token_id == 151673
    assert cfg.model.max_sequence_length == 2049
    assert cfg.data.memmap_dtype == "uint32"
    assert len(cfg.data.paths) == 246


def test_fineweb_paths_use_the_historical_984_shard_partition():
    manifest = json.loads(MANIFEST_PATH.read_text())
    run = next(run for run in manifest["runs"] if run["id"] == "fwedu_1b_tree")
    paths, parse_paths = data_paths(run, "/workspace")
    assert parse_paths is None
    assert len(paths) == 246
    assert paths[0].endswith("tree/*-00(000|001|002|003)*arrow.npy")
    assert paths[-1].endswith("tree/*-00(980|981|982|983)*arrow.npy")


def test_explicit_input_validation_rejects_plain_missing_paths(tmp_path):
    run = load_manifest(MANIFEST_PATH)["runs"][0]
    output = tmp_path / "missing-data.yaml"
    materialize_config(
        run,
        REPO_ROOT / run["source_config"],
        output,
        str(tmp_path / "workspace-without-data"),
        "test_missing_inputs",
    )
    with pytest.raises(FileNotFoundError, match="data.paths"):
        validate_training_inputs(output)
