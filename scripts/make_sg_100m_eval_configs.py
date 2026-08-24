#!/usr/bin/env python3
"""Generate frozen SG evaluation configs for all non-terminal, non-pause 100M runs.

Each config is derived from the checkpoint's own saved training configuration so
model architecture and grammar settings cannot be inadvertently mixed.  The
evaluation-only settings follow ``syntax_eval_34354``: no training, no W&B,
one evaluator, a local placeholder loader, and a self-contained output path.
"""

from __future__ import annotations

from pathlib import Path

from olmo.config import DataConfig, EvaluatorConfig, EvaluatorType, TrainConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "analysis-output/sg_100m_rtx3090_20260820"

# d_model=768 / 12 layers is the project's 100M family.  Terminal and all
# pause variants are intentionally excluded by the task definition.
CHECKPOINTS = {
    "tg": ROOT / "saved_models/TG_test/step55457-unsharded",
    "tgnomask": ROOT / "saved_models/nomask_test/step55853-unsharded",
    "tgnomask_aug": ROOT / "saved_models/TGnomask_aug_pretrain/step55853-unsharded",
    "mixing": ROOT / "saved_models/TG_mix_nomask_bs240_lr0076/step69817-unsharded",
    "tree": ROOT / "saved_models/Tree_test/step49440-unsharded",
    "tgtree": ROOT / "saved_models/TGtree/step69817-unsharded",
}


def build_config(name: str, checkpoint: Path) -> Path:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    cfg = TrainConfig.load(checkpoint / "config.yaml", validate_paths=False)
    if cfg.model.d_model != 768 or cfg.model.n_layers != 12:
        raise ValueError(f"{name} is not a 100M-family checkpoint: {checkpoint}")
    grammar = cfg.model.transformer_grammar_type or "terminal"
    if grammar == "terminal" or cfg.model.ispause:
        raise ValueError(f"excluded model family for {name}: {grammar}")

    run_name = f"sg_100m_{name}_beam300_len6"
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

    tokenizer_path = ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    cfg.tokenizer.identifier = str(tokenizer_path)
    cfg.tokenizer.vocabulary = str(tokenizer_path)

    # The evaluation-only trainer still constructs its train loader.  A small
    # local terminal stream avoids stale /dev/shm paths from historical runs.
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
            # 300 is the established word-synchronous beam protocol.
            samples_per_sent=300,
            tree_eval_type="default",
            structure_mode="auto",
            beam_size=300,
            beam_nc_ratio=1.0,
            beam_pc=3,
            # Recorded in the frozen config; current SG_eval_step applies the
            # same repository-wide 6x terminal-length rule directly.
            beam_max_len_factor=6,
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
