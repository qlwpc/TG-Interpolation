#!/usr/bin/env python3
"""Resolve per-checkpoint configs for the terminal-format validation campaign."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from olmo.config import TrainConfig


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "artifacts/experiment/terminal-format-validation-20260828"
CHECKPOINTS = {
    "terminal": ROOT / "saved_models/A800_models/terminal_1B/step94299-unsharded",
    "tree": ROOT / "saved_models/A800_models/tree_1B/step137217-unsharded",
    "tgtree": ROOT / "saved_models/A800_models/tgtree_1B/step143658-unsharded",
    "pause1": ROOT / "saved_models/A800_models/pause1_1B_SEP/step116061-unsharded",
    "pause2": ROOT / "saved_models/A800_models/pause2_1B_SEP/step141380-unsharded",
}


def load(path: Path) -> TrainConfig:
    return TrainConfig.load(
        path,
        overrides=[f"workspace={ROOT}"],
        validate_paths=True,
    )


def specialize(base: TrainConfig, model_name: str, suite: str) -> TrainConfig:
    checkpoint = CHECKPOINTS[model_name]
    checkpoint_cfg = load(checkpoint / "config.yaml")
    cfg = deepcopy(base)
    cfg.run_name = f"terminal_format_{suite}_{model_name}"
    cfg.load_path = str(checkpoint)
    cfg.save_folder = str(EXPERIMENT / "runs" / f"{model_name}_{suite}")
    cfg.model.transformer_grammar_type = checkpoint_cfg.model.transformer_grammar_type
    cfg.model.pause_token_id = checkpoint_cfg.model.pause_token_id
    cfg.model.max_sequence_length = checkpoint_cfg.model.max_sequence_length
    if cfg.model.d_model != checkpoint_cfg.model.d_model:
        raise ValueError(f"d_model mismatch for {model_name}")
    if cfg.model.n_layers != checkpoint_cfg.model.n_layers:
        raise ValueError(f"n_layers mismatch for {model_name}")
    return cfg


def main() -> None:
    output = EXPERIMENT / "configs"
    output.mkdir(parents=True, exist_ok=True)
    decomp = load(ROOT / "evaluation/eval_configs/terminal_format_validation_decomp.yaml")
    validation10 = load(ROOT / "evaluation/eval_configs/terminal_format_validation10.yaml")

    for model_name in ("tree", "tgtree"):
        cfg = specialize(decomp, model_name, "decomp")
        path = output / f"{model_name}_decomp.yaml"
        cfg.save(path)
        print(path)

    for model_name in ("terminal", "tree", "tgtree", "pause1", "pause2"):
        cfg = specialize(validation10, model_name, "validation10")
        path = output / f"{model_name}_validation10.yaml"
        cfg.save(path)
        print(path)


if __name__ == "__main__":
    main()
