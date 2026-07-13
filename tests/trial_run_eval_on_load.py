"""Trial-run eval-on-load for all 5 evaluators on CPU, against the random-init
pushdown/treereg checkpoints.

`scripts/train.py` hard-codes CUDA, so we can't invoke it. Instead we reproduce
the EXACT eval_on_load code path: load model.pt into OLMo(cfg) on CPU, build a
real Trainer (dataclass) with a real optim/scheduler + a tiny dummy train_loader
+ the 6 evaluators (2 docppl lm + SG/BLiMP/boolq/xsum), then call trainer.eval()
— the same method eval_on_load calls. subset_num_batches=4 keeps BLiMP/xsum fast.

This is the closest possible local trial of the GPU eval-on-load run.

Run: PYTHONPATH=. /home/wangpch/.conda/envs/LLM/bin/python tests/trial_run_eval_on_load.py
"""
import os, sys, traceback
from pathlib import Path
from dataclasses import dataclass
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = Path(__file__).resolve().parents[1]

# ---- patch: make barrier a no-op on single-proc CPU (no dist group) ----
import olmo.torch_util as _tu
_orig_barrier = _tu.barrier
def _noop_barrier(*a, **k): pass
_tu.barrier = _noop_barrier
import olmo.train as _T
_T.barrier = _noop_barrier


@dataclass
class _DummyLoader:
    """Empty train_loader stub (eval() never iterates it)."""
    dataset: object = None
    def __iter__(self): return iter([])
    def __len__(self): return 0


class _DistWrap:
    """Stand-in for DDP: exposes .module, .eval(), .train(), __call__."""
    def __init__(self, m): self.module = m; self.training = False
    def __call__(self, *a, **k): return self.module(*a, **k)
    def eval(self): self.training = False; self.module.eval(); return self
    def train(self, mode=True): self.training = mode; self.module.train(mode); return self


def run_one(label, skip_docppl=False):
    from olmo.config import TrainConfig
    from olmo.model import OLMo
    from olmo.eval import build_evaluators
    from olmo.optim import build_optimizer, build_scheduler
    from olmo.train import Trainer
    from olmo.torch_util import get_world_size, get_global_rank

    yaml = REPO / "train_configs" / f"{label}_eval.yaml"
    cfg = TrainConfig.load(yaml, [])
    cfg.model.init_device = "cpu"
    cfg.model.precision = cfg.precision
    cfg.workspace = str(REPO)
    cfg.device_train_batch_size = 1
    cfg.wandb = None

    print(f"\n##### {label}: load_path={cfg.load_path}")
    model = OLMo(cfg.model, init_params=False)
    sd = torch.load(Path(cfg.load_path) / "model.pt", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  loaded checkpoint ({sum(p.numel() for p in model.parameters()):,} params)")

    dist_model = _DistWrap(model)
    # build_optimizer calls model.named_modules() — pass the raw nn.Module.
    optim = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg)

    # Dummy iterable train_loader (eval() doesn't iterate it).
    train_loader = _DummyLoader(dataset=None)

    device = torch.device("cpu")
    evaluators = build_evaluators(cfg, device)

    if skip_docppl:
        # The TG-ppl-validation[-test] (lm) evaluators feed tree_spans to the
        # model -> pushdown's PARSED path needs flex_attention, which is
        # GPU-only (inductor rejects the depth-bias score_mod on CPU). Skip
        # them on CPU to confirm the other 4 (SG/BLiMP/boolq/xsum, which run
        # tree_spans=None) work. On a GPU node, do NOT skip.
        kept = [e for e in evaluators if e.label not in ("TG-ppl-validation", "TG-ppl-validation-test")]
        cfg.evaluators = [e for e in cfg.evaluators if e.label in {k.label for k in kept}]
        evaluators = kept
        print(f"  [skip_docppl] dropped TG-ppl-validation[-test] (flex/GPU-only); "
              f"running {len(evaluators)} evaluators")

    trainer = Trainer(
        cfg=cfg, epoch=cfg.epoch, model=model, dist_model=dist_model,
        optim=optim, scheduler=scheduler, train_loader=train_loader,
        device=device, evaluators=evaluators, indices_file=None,
    )

    print(f"  running trainer.eval() over {len(evaluators)} evaluators "
          f"(subset={cfg.eval_subset_num_batches})...")
    metrics = trainer.eval()
    print(f"  -> {len(metrics)} metric keys")
    for k in list(metrics)[:8]:
        v = metrics[k]
        print(f"     {k} = {v if not isinstance(v, float) else round(v,4)}")
    if len(metrics) > 8:
        print(f"     ... ({len(metrics)-8} more)")
    return metrics


def main():
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    no_gpu = not torch.cuda.is_available()
    results = {}
    for label in ["pushdown", "treereg"]:
        # On CPU, pushdown's docppl (parsed tree_spans -> flex) can't run
        # (inductor rejects the depth-bias score_mod). Skip docppl for pushdown
        # ONLY when no GPU. treereg's docppl uses standard attn — runs on CPU.
        skip = no_gpu and label == "pushdown"
        try:
            results[label] = run_one(label, skip_docppl=skip)
            print(f"\n  [PASS] {label}: eval() completed, {len(results[label])} metrics"
                  + (" (docppl skipped: flex is GPU-only)" if skip else ""))
        except Exception as e:
            results[label] = None
            print(f"\n  [FAIL] {label}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)
    print("\n" + "=" * 60)
    ok = [k for k, v in results.items() if v]
    print(f"PASSED: {len(ok)}/{len(results)}  ({ok})")
    sys.exit(0 if len(ok) == len(results) else 1)


if __name__ == "__main__":
    main()
