"""Cold-run build: append the 5 downstream evaluators (SG/BLIMP/XSUM/BOOLQ/docppl)
to pushdown.yaml / treereg.yaml, load the resulting config, and build all 5
evaluators via build_downstream_evaluator. This mirrors exactly what the GPU
cluster would do at eval_on_load time, minus the forward pass.

Run: PYTHONPATH=. /home/wangpch/.conda/envs/LLM/bin/python tests/test_eval_5evaluators_build.py
"""
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]

# The 5 downstream evaluators. NOTE: xsum MUST be type=rouge (not downstream),
# else it dispatches to eval_step (plain LM) and crashes summarization_eval_step.
FIVE_EVALUATORS = [
    {"label": "syntactic_generalization", "type": "downstream"},
    {"label": "BLiMP", "type": "downstream"},
    {"label": "boolq", "type": "downstream"},
    {"label": "xsum", "type": "rouge"},
    # docppl (TG-ppl-validation / -test) are ALREADY in pushdown.yaml/treereg.yaml
    # as type=tg_doc -> TG_doc_eval_step. Kept from the base config.
]


def build_and_check(yaml_name, label):
    from olmo.config import TrainConfig, EvaluatorConfig, EvaluatorType
    from olmo.eval import build_downstream_evaluator
    from olmo.tokenizer import Tokenizer

    cfg = TrainConfig.load(REPO / "train_configs" / yaml_name, [])
    cfg.model.init_device = "cpu"
    cfg.workspace = str(REPO)
    cfg.wandb = None

    # Append the 4 downstream evaluators that are NOT already in the config
    # (docppl tg_doc evaluators are already present).
    existing_labels = {e.label for e in cfg.evaluators}
    for ev in FIVE_EVALUATORS:
        if ev["label"] in existing_labels:
            continue
        cfg.evaluators.append(EvaluatorConfig(
            label=ev["label"], type=EvaluatorType(ev["type"]),
            data=cfg.data,  # reuse the train data config (num_workers etc.)
        ))

    print(f"\n##### {label}: {len(cfg.evaluators)} evaluators total:")
    for e in cfg.evaluators:
        print(f"    - {e.label:30s} type={e.type}")

    tok = Tokenizer.from_train_config(cfg)
    from olmo.eval import build_evaluator
    built = 0
    for ec in cfg.evaluators:
        # docppl (TG-ppl-validation*) are type=lm in the config (no `type:` field
        # -> default lm); they route through build_evaluator's lm branch, NOT
        # build_downstream_evaluator. The 4 downstream tasks route downstream.
        if ec.type.value in ("lm",):
            ev = build_evaluator(cfg, ec, tok, torch.device("cpu"))
            n = len(ev.eval_loader.dataset)
            print(f"    [OK] {ec.label}: (lm) dataset len={n}, metric={type(ev.eval_metric).__name__}")
        else:
            ev = build_downstream_evaluator(cfg, ec, tok, torch.device("cpu"), is_unit_test=True)
            print(f"    [OK] {ec.label}: ({ec.type.value}) dataset len={len(ev.eval_loader.dataset)}, metric={type(ev.eval_metric).__name__}")
        built += 1
    print(f"    -> {built}/{len(cfg.evaluators)} evaluators built")
    return built == len(cfg.evaluators)


def main():
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    ok_all = True
    for yaml, label in [("pushdown.yaml", "pushdown"), ("treereg.yaml", "treereg")]:
        try:
            ok_all &= build_and_check(yaml, label)
        except Exception as e:
            import traceback; traceback.print_exc()
            ok_all = False
            print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    print("ALL 5 EVALUATORS BUILD OK" if ok_all else "FAILURES (see above)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
