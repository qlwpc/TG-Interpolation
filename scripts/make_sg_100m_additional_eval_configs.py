#!/usr/bin/env python3
"""Generate frozen SG configs for the remaining requested 100M checkpoints."""

from __future__ import annotations

from pathlib import Path

from olmo.config import DataConfig, EvaluatorConfig, EvaluatorType, TrainConfig


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "analysis-output/sg_100m_additional_rtx3090_20260823"

# Tree-Shuffle is terminal at inference time; it must avoid auto/beam scoring.
CHECKPOINTS = {
    "tree_noont": (ROOT / "saved_models/tree_noont/step42440-unsharded", False, "tree_noont"),
    "tree_compress": (ROOT / "saved_models/tree_compress/step45965-unsharded", False, "tree_compress"),
    "tree_shuffle": (ROOT / "saved_models/Tree_shuffle_pretrain/step49440-unsharded", True, "terminal"),
    "tgtree_mix_tg": (ROOT / "saved_models/tgtree_mix_tg_pretrain/step69817-unsharded", False, "mixing"),
    "tree_triplecnt": (ROOT / "saved_models/tree_triplecnt/step60045-unsharded", False, "tree_triplecnt"),
}


def build_config(name: str, checkpoint: Path, terminal_only: bool, grammar: str) -> Path:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    cfg = TrainConfig.load(checkpoint / "config.yaml", validate_paths=False)
    if cfg.model.d_model != 768 or cfg.model.n_layers != 12:
        raise ValueError(f"{name} is not a 100M-family checkpoint: {checkpoint}")
    if cfg.model.ispause or (not terminal_only and grammar == "terminal"):
        raise ValueError(f"excluded model family: {name}")

    suffix = "terminal_only" if terminal_only else "beam300_len6"
    run_name = f"sg_100m_{name}_{suffix}"
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
    # These variants were saved from shared training templates whose config
    # may still say ``tgtree``.  Restore the documented runtime branch.
    cfg.model.transformer_grammar_type = grammar

    tokenizer = str(ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json")
    cfg.tokenizer.identifier = tokenizer
    cfg.tokenizer.vocabulary = tokenizer
    cfg.data.paths = [str(ROOT / "dataset/bbc-news/terminal/dev.npy")]
    cfg.data.parse_tree_paths = None
    cfg.data.num_workers = 0
    cfg.data.drop_last = False
    cfg.data.pin_memory = False
    cfg.data.prefetch_factor = None
    cfg.data.persistent_workers = False

    evaluator = EvaluatorConfig(
        label="syntactic_generalization",
        type=EvaluatorType.downstream,
        data=DataConfig(num_workers=0, pin_memory=False, prefetch_factor=None,
                        persistent_workers=False, drop_last=False),
        device_eval_batch_size=1,
        subset_num_batches=-1,
        samples_per_sent=300,
        beam_size=300,
        beam_nc_ratio=1.0,
        beam_pc=3,
        beam_max_len_factor=6,
    )
    if terminal_only:
        cfg.model.transformer_grammar_type = "terminal"
        evaluator.tree_eval_type = "terminal"
        evaluator.structure_mode = "terminal"
        evaluator.beam_search = False
    else:
        evaluator.tree_eval_type = "default"
        evaluator.structure_mode = "auto"
    cfg.evaluators = [evaluator]

    output = OUTPUT_ROOT / "configs" / f"{run_name}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(output)
    return output


def main() -> None:
    for name, (checkpoint, terminal_only, grammar) in CHECKPOINTS.items():
        print(build_config(name, checkpoint, terminal_only, grammar))


if __name__ == "__main__":
    main()
