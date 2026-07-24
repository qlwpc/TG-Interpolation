"""Generate one-evaluator eval configs for each (model × metric) pair, so that
different metrics can run on different GPUs in parallel.

Each output = the model's {label}_eval.yaml with cfg.evaluators reduced to ONLY
the evaluator(s) for that metric, every evaluator's subset_num_batches set to -1
(full scale, not the smoke-test 4), eval_on_load + max_duration:0ep + eval_no_save
+ reset states, device_eval_batch_size=8, wandb=null.

Metrics → evaluator labels:
  docppl  → TG-ppl-validation + TG-ppl-validation-test   (type lm, dev+test PPL)
  SG      → syntactic_generalization                      (type downstream)
  BLiMP   → BLiMP                                          (type downstream)
  boolq   → boolq                                          (type downstream)
  XSUM    → xsum                                           (type rouge)

Run: PYTHONPATH=. /home/wangpch/.conda/envs/LLM/bin/python tests/make_per_metric_eval_configs.py
"""
from pathlib import Path
from olmo.config import TrainConfig

REPO = Path(__file__).resolve().parents[1]

# metric name -> list of evaluator labels to keep
METRICS = {
    "docppl": ["TG-ppl-validation", "TG-ppl-validation-test"],
    "SG":     ["syntactic_generalization"],
    "BLiMP":  ["BLiMP"],
    "boolq":  ["boolq"],
    "XSUM":   ["xsum"],
}

MODELS = ["pushdown", "treereg"]


def build(label: str, metric: str, keep_labels: list[str]) -> Path:
    cfg = TrainConfig.load(REPO / "train_configs" / f"{label}_eval.yaml", [])
    cfg.workspace = str(REPO)
    cfg.run_name = f"{label}_eval_{metric}"
    cfg.save_folder = str(REPO / "analysis-output" / cfg.run_name)
    cfg.save_overwrite = True
    cfg.wandb = None
    cfg.dry_run = False

    # Eval-on-load, no training, no save.
    cfg.max_duration = "0ep"
    cfg.stop_at = 0
    cfg.eval_on_load = True
    cfg.eval_no_save = True
    cfg.reset_optimizer_state = True
    cfg.reset_trainer_state = True

    # GPU eval batch + full-scale eval (no smoke-test subset cap).
    cfg.device_train_microbatch_size = 8
    cfg.device_eval_batch_size = 8
    cfg.eval_subset_num_batches = -1

    # Keep only this metric's evaluator(s); force full-scale (subset=-1).
    keep = set(keep_labels)
    cfg.evaluators = [e for e in cfg.evaluators if e.label in keep]
    missing = keep - {e.label for e in cfg.evaluators}
    assert not missing, f"{label}/{metric}: missing evaluators {missing} in {label}_eval.yaml"
    for e in cfg.evaluators:
        e.subset_num_batches = -1   # full; override the per-evaluator smoke cap

    out_dir = REPO / "train_configs" / "eval_per_metric"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{label}_{metric}.yaml"
    cfg.save(out)
    print(f"wrote {out.relative_to(REPO)}")
    print(f"  load_path = {cfg.load_path}")
    print(f"  grammar   = {cfg.model.transformer_grammar_type}  flex={cfg.model.flex_attention}")
    print(f"  evaluators ({len(cfg.evaluators)}):")
    for e in cfg.evaluators:
        print(f"    - {e.label:30s} type={e.type}  subset={e.subset_num_batches}")
    return out


def main():
    for label in MODELS:
        for metric, keep_labels in METRICS.items():
            build(label, metric, keep_labels)
    print("\nDONE — 10 configs in train_configs/eval_per_metric/")


if __name__ == "__main__":
    main()
