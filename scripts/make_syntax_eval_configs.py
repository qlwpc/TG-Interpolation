#!/usr/bin/env python3
"""Build reproducible BLiMP/SG configs for the step-34354 syntax checkpoints.

The generated configs distinguish the evaluation protocol explicitly:
terminal (no parse/tape), gold (BLiMP gold300), and beam (SG incremental
Pushdown beam search). TreeReg's parse-independent inference only needs the
terminal runs; its structured results are mathematically identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from olmo.config import DataConfig, EvaluatorConfig, EvaluatorType, TrainConfig


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "treereg": ROOT / "saved_models/treereg/step34354-unsharded",
    "pushdown": ROOT / "saved_models/pushdown_terminalonly/step34354-unsharded",
}


def build_config(
    model: str,
    task: str,
    mode: str,
    output_dir: Path,
    *,
    run_name: str | None = None,
    subset_num_batches: int = -1,
    pair_per_task: int | None = None,
) -> Path:
    checkpoint = CHECKPOINTS[model]
    cfg = TrainConfig.load(
        checkpoint / "config.yaml", validate_paths=False
    )

    run_name = run_name or f"{model}_{task.lower()}_{mode}_step34354"
    cfg.run_name = run_name
    cfg.workspace = str(ROOT)
    cfg.load_path = str(checkpoint)
    cfg.load_path_sharded_checkpointer = None
    cfg.try_load_latest_save = False
    cfg.save_folder = str(ROOT / "analysis-output/syntax_eval_34354" / run_name)
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
    cfg.device_eval_batch_size = 8
    cfg.console_log_interval = 100
    tokenizer_path = ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    cfg.tokenizer.identifier = str(tokenizer_path)
    cfg.tokenizer.vocabulary = str(tokenizer_path)

    # A tiny valid local dataset is still required while the test-only trainer
    # is constructed, even though max_duration=0 performs no training.
    if model == "treereg":
        cfg.data.paths = [str(ROOT / "dataset/bbc-news/parse_aligned/dev_treereg")]
    else:
        cfg.data.paths = [
            str(ROOT / "dataset/bbc-news/parse_aligned/dev_pushdown_unary_terminals")
        ]
    cfg.data.parse_tree_paths = [str(ROOT / "dataset/bbc-news/tree/dev.npy")]
    cfg.data.num_workers = 0
    cfg.data.drop_last = False
    cfg.data.pin_memory = False
    cfg.data.prefetch_factor = None
    cfg.data.persistent_workers = False

    eval_workers = 2 if task == "BLiMP" and mode == "gold" else 0
    eval_data = DataConfig(
        num_workers=eval_workers,
        pin_memory=eval_workers > 0,
        prefetch_factor=2 if eval_workers > 0 else None,
        persistent_workers=eval_workers > 0,
        drop_last=False,
    )
    evaluator = EvaluatorConfig(
        label="BLiMP" if task == "BLiMP" else "syntactic_generalization",
        type=EvaluatorType.downstream,
        data=eval_data,
        # gold300 stores exactly 300 parses per sentence. Keeping one complete
        # parse group in each batch avoids six separate forwards per sentence
        # and preserves BLiMPMetric's group boundaries. These checkpoints are
        # ~0.42 GiB and BLiMP terminal sequences are at most 100 tokens, so the
        # smoke run can validate the 24-GiB inference footprint before the full
        # multi-GPU job is released.
        device_eval_batch_size=(300 if mode == "gold" else 100)
        if task == "BLiMP"
        else 1,
        subset_num_batches=subset_num_batches,
        samples_per_sent=300 if mode == "gold" else None,
        tree_eval_type="default",
        structure_mode=mode,
        beam_search=False,
        pushdown_beam_size=300,
        pushdown_max_reduce=None,
        pair_per_task=(pair_per_task if pair_per_task is not None else 1000)
        if task == "BLiMP"
        else None,
    )
    cfg.evaluators = [evaluator]

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{run_name}.yaml"
    cfg.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analysis-output/syntax_eval_34354/configs",
    )
    args = parser.parse_args()

    jobs = [
        (("treereg", "BLiMP", "terminal"), {}),
        (("treereg", "SG", "terminal"), {}),
        (("pushdown", "BLiMP", "terminal"), {}),
        (("pushdown", "SG", "terminal"), {}),
        (("pushdown", "BLiMP", "gold"), {}),
        (("pushdown", "SG", "beam"), {}),
        (
            ("pushdown", "BLiMP", "gold"),
            {
                "run_name": "pushdown_blimp_gold300_smoke_step34354",
                "pair_per_task": 1,
            },
        ),
        (
            ("pushdown", "SG", "beam"),
            {
                "run_name": "pushdown_sg_beam300_smoke_step34354",
                "subset_num_batches": 4,
            },
        ),
    ]
    for job, kwargs in jobs:
        print(build_config(*job, output_dir=args.output_dir, **kwargs))


if __name__ == "__main__":
    main()
