#!/usr/bin/env python3
"""Generate full beam-300 SG configs for the local Tree-1B checkpoints."""

from __future__ import annotations

from pathlib import Path

from olmo.config import DataConfig, EvaluatorConfig, EvaluatorType, TrainConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "analysis-output/sg_1b_rtx3090_20260822"
CHECKPOINTS = {
    "tree": ROOT / "saved_models/Tree_1B/step49440-unsharded",
    "tgtree": ROOT / "saved_models/tgtree_1B/step143658-unsharded",
}


def build_config(name: str, checkpoint: Path) -> Path:
    # The Qwen TGTree checkpoint was saved with a self-referential
    # ``workspace: ${workspace}``; supply the concrete workspace before OmegaConf
    # resolves the remaining paths.
    cfg = TrainConfig.load(
        checkpoint / "config.yaml",
        overrides=[f"workspace={ROOT}"],
        validate_paths=False,
    )
    if cfg.model.d_model != 2048 or cfg.model.n_layers != 16:
        raise ValueError(f"{name} is not a 1B-family checkpoint: {checkpoint}")

    is_qwen = cfg.model.vocab_size > 100_000
    run_name = f"sg_1b_{name}_beam300_len6"
    cfg.run_name = run_name
    cfg.workspace = str(ROOT)
    cfg.load_path = str(checkpoint)
    cfg.load_path_sharded_checkpointer = None
    cfg.try_load_latest_save = False
    cfg.save_folder = str(OUTPUT_ROOT / run_name)
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
    cfg.global_train_batch_size = 8
    cfg.device_train_microbatch_size = 1
    cfg.device_eval_batch_size = 1
    cfg.console_log_interval = 100

    tokenizer_path = (
        ROOT / "dataset/TG_QWEN3_tokenizer.json"
        if is_qwen
        else ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    )
    cfg.tokenizer.identifier = str(tokenizer_path)
    cfg.tokenizer.vocabulary = str(tokenizer_path)
    cfg.data.paths = [str(ROOT / "dataset/bbc-news/terminal/dev.npy")]
    # This is an evaluation-only placeholder loader. The local terminal stream
    # is uint16 even for the Qwen checkpoint; it is never forwarded through the
    # model because max_duration=0.
    cfg.data.memmap_dtype = "uint16"
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
            data=DataConfig(num_workers=0, pin_memory=False, prefetch_factor=None,
                            persistent_workers=False, drop_last=False),
            device_eval_batch_size=1,
            subset_num_batches=-1,
            samples_per_sent=300,
            tree_eval_type="default",
            structure_mode="auto",
            beam_size=300,
            beam_nc_ratio=1.0,
            beam_pc=3,
            beam_max_len_factor=6,
            # SGDataset itself appends ``qwen3`` for Qwen vocabularies.
            sg_dataset_path=(str(ROOT / "evaluation/SG/tokenized") if is_qwen else None),
        )
    ]
    output_dir = OUTPUT_ROOT / "configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_name}.yaml"
    cfg.save(output)
    return output


def main() -> None:
    for name, checkpoint in CHECKPOINTS.items():
        print(build_config(name, checkpoint))


if __name__ == "__main__":
    main()
