"""Create pushdown_eval.yaml + treereg_eval.yaml: the base train config + the
4 downstream evaluators (SG/BLiMP/boolq/xsum) appended after the 2 docppl
evaluators, with eval_on_load + max_duration:0ep + reset states + load_path
pointing at the random-init step0 checkpoint.

Run: PYTHONPATH=. /home/wangpch/.conda/envs/LLM/bin/python tests/make_eval_configs.py
"""
import sys
from pathlib import Path
from olmo.config import TrainConfig, EvaluatorConfig, EvaluatorType

REPO = Path(__file__).resolve().parents[1]

# The 4 downstream evaluators (docppl TG-ppl-validation[-test] are already in
# the base config). xsum MUST be type=rouge (NOT downstream) so it dispatches
# to summarization_eval_step.
DOWNSTREAM = [
    ("syntactic_generalization", "downstream"),
    ("BLiMP", "downstream"),
    ("boolq", "downstream"),
    ("xsum", "rouge"),
]


def build(base_yaml, label, ckpt_path):
    cfg = TrainConfig.load(REPO / "train_configs" / base_yaml, [])
    cfg.workspace = str(REPO)
    cfg.run_name = f"{label}_eval_coldrun"
    cfg.save_folder = str(REPO / "saved_models" / f"random_init_{label}")
    cfg.save_overwrite = True
    cfg.wandb = None
    cfg.dry_run = False

    # Eval-on-load: load the ckpt, run eval(), do NOT train.
    cfg.max_duration = "0ep"
    cfg.stop_at = 0
    cfg.eval_on_load = True
    cfg.eval_no_save = True
    cfg.reset_optimizer_state = True   # eval-only: don't require optim.pt
    cfg.reset_trainer_state = True     # eval-only: don't require train.pt
    cfg.load_path = str(ckpt_path)

    # Small eval batch for CPU trial.
    cfg.device_train_microbatch_size = 2
    cfg.device_eval_batch_size = 2
    # Limit BLiMP (40M pairs) + others so the trial run is fast.
    cfg.eval_subset_num_batches = 4

    # Append the 4 downstream evaluators (skip if already present).
    existing = {e.label for e in cfg.evaluators}
    # Cap each downstream evaluator to a few batches for the trial run.
    for ev_label, ev_type in DOWNSTREAM:
        if ev_label in existing:
            continue
        cfg.evaluators.append(EvaluatorConfig(
            label=ev_label,
            type=EvaluatorType(ev_type),
            subset_num_batches=4,
        ))

    out = REPO / "train_configs" / f"{label}_eval.yaml"
    cfg.save(out)
    print(f"wrote {out}")
    print(f"  load_path = {cfg.load_path}")
    print(f"  eval_on_load = {cfg.eval_on_load}, max_duration = {cfg.max_duration}")
    print(f"  evaluators ({len(cfg.evaluators)}):")
    for e in cfg.evaluators:
        print(f"    - {e.label:28s} type={e.type}  subset={e.subset_num_batches}")
    return out


def main():
    for label, base in [("pushdown", "pushdown.yaml"), ("treereg", "treereg.yaml")]:
        ckpt = REPO / "saved_models" / f"random_init_{label}" / "step0-unsharded"
        assert (ckpt / "model.pt").is_file(), f"missing {ckpt/'model.pt'} (run make_random_init_ckpt.py first)"
        build(base, label, ckpt)


if __name__ == "__main__":
    main()
