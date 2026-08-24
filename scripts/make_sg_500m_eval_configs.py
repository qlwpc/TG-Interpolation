#!/usr/bin/env python3
"""Generate frozen 500M non-terminal/non-pause SG configs for A6000."""

from __future__ import annotations

import argparse
from pathlib import Path

from olmo.config import DataConfig, EvaluatorConfig, EvaluatorType, TrainConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "analysis-output/sg_500m_a6000_20260820"
CHECKPOINTS = {
    "tree": ROOT / "saved_models/Tree_500M/step49440-unsharded",
    "tgtree": ROOT / "saved_models/TGTree_500M/step55853-unsharded",
    "tgnomask_aug": ROOT / "saved_models/TGnomaskaug_500M/step55853-unsharded",
}


def build_config(name: str, checkpoint: Path, output_root: Path) -> Path:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    cfg = TrainConfig.load(checkpoint / "config.yaml", validate_paths=False)
    if cfg.model.d_model != 1408 or cfg.model.n_layers != 16:
        raise ValueError(f"{name} is not a 500M-family checkpoint: {checkpoint}")
    grammar = cfg.model.transformer_grammar_type or "terminal"
    if grammar == "terminal" or cfg.model.ispause:
        raise ValueError(f"excluded model family for {name}: {grammar}")

    run_name = f"sg_500m_{name}_beam300_len6"
    cfg.run_name = run_name
    cfg.workspace = str(ROOT)
    cfg.load_path = str(checkpoint)
    cfg.load_path_sharded_checkpointer = None
    cfg.try_load_latest_save = False
    cfg.save_folder = str(output_root / run_name)
    cfg.save_overwrite = True
    cfg.reset_optimizer_state = True
    cfg.reset_trainer_state = True
    cfg.max_duration = "0ep"
    cfg.stop_at = 0
    cfg.eval_on_load = True
    cfg.eval_no_save = True
    cfg.eval_subset_num_batches = -1
    cfg.wandb = None
    cfg.save_data_indices = False
    cfg.global_train_batch_size = 8  # divisible by the two A6000 ranks
    cfg.device_train_microbatch_size = 1
    cfg.device_eval_batch_size = 1
    cfg.console_log_interval = 100

    tokenizer_path = ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    cfg.tokenizer.identifier = str(tokenizer_path)
    cfg.tokenizer.vocabulary = str(tokenizer_path)
    cfg.data.paths = [str(ROOT / "dataset/bbc-news/terminal/dev.npy")]
    cfg.data.parse_tree_paths = None
    cfg.data.num_workers = 0
    cfg.data.drop_last = False
    cfg.data.pin_memory = False
    cfg.data.prefetch_factor = None
    cfg.data.persistent_workers = False
    cfg.evaluators = [
        EvaluatorConfig(
            label="syntactic_generalization",
            type=EvaluatorType.downstream,
            data=DataConfig(
                num_workers=0,
                pin_memory=False,
                prefetch_factor=None,
                persistent_workers=False,
                drop_last=False,
            ),
            device_eval_batch_size=1,
            subset_num_batches=-1,
            samples_per_sent=300,
            tree_eval_type="default",
            structure_mode="auto",
            beam_size=300,
            beam_nc_ratio=1.0,
            beam_pc=3,
            # The committed SG implementation uses the identical 6x rule.
            beam_max_len_factor=6,
        )
    ]

    output_dir = output_root / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_name}.yaml"
    cfg.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--disable-flex-attention",
        action="store_true",
        help="evaluate with the non-flex attention implementation",
    )
    args = parser.parse_args()
    for name, checkpoint in CHECKPOINTS.items():
        output = build_config(name, checkpoint, args.output_root)
        if args.disable_flex_attention:
            cfg = TrainConfig.load(output, validate_paths=False)
            cfg.model.flex_attention = False
            cfg.save(output)
        print(output)


if __name__ == "__main__":
    main()
