#!/usr/bin/env python
"""Local CPU cold-run test: verify the 5 downstream eval paths
(SG / BLIMP / XSUM / BOOLQ / docppl) are wired correctly for the pushdown
and treereg configs and run without runtime errors.

NO GPU is used (the dev box has no usable CUDA). The pushdown flex path is
GPU-only, so this test exercises only the paths that actually fire on CPU:
  * tree_spans=None  -> pushdown depth bias vanishes, plain causal SDPA path
                        (this is the SG/BLIMP/XSUM/BOOLQ eval path).
  * tree_spans given  -> treereg loss compute (pure tensor ops, CPU-OK);
                        pushdown parsed path needs flex (GPU) so we only
                        build the depth matrix + check shapes, NOT the full
                        flex attention (which create_block_mask rejects on CPU).

The model is built from the REAL pushdown.yaml / treereg.yaml (init_device
overridden to cpu). The 5 eval datasets are built via build_downstream_evaluator
with is_unit_test=True (no DistributedSampler), and each eval_step dispatch
branch is driven with a tiny synthetic batch.

Run: PYTHONPATH=. /home/wangpch/.conda/envs/LLM/bin/python tests/test_eval_coldrun_cpu.py
"""
import os
import sys
import traceback
from pathlib import Path

# Force CPU, disable any GPU/flex-at-runtime code paths.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_cfg(yaml_name):
    from olmo.config import TrainConfig
    yaml_path = REPO / "train_configs" / yaml_name
    cfg = TrainConfig.load(yaml_path, [])
    # Force CPU init.
    cfg.model.init_device = "cpu"
    cfg.device_eval_batch_size = 2
    cfg.workspace = str(REPO)
    # Disable wandb / eval_on_load side effects we don't want here.
    cfg.wandb = None
    return cfg


def build_model(cfg):
    from olmo.model import OLMo
    model = OLMo(cfg.model, init_params=True)
    model.eval()
    return model


def synth_batch(cfg, seq_len=64, with_tree_spans=False):
    """Minimal batch dict that model_forward + the lm/docppl eval path accept."""
    B = 2
    V = cfg.model.vocab_size
    pad = cfg.model.pad_token_id
    ids = torch.randint(0, V, (B, seq_len))
    # Make sure no pad collides with random ids (set last few to pad to test masking).
    ids[:, -4:] = pad
    # Real dataloader emits a BOOL attention_mask (collator.py:78 / parse_align.py:898).
    # The pushdown block_mask path asserts mask.dtype == bool, so match that here.
    am = (ids != pad)
    batch = {
        "input_ids": ids,
        "attention_mask": am,
        "metadata": [{"path": "synth", "label": "synth"} for _ in range(B)],
    }
    if with_tree_spans:
        # A few synthetic spans (l, r, depth-ish) in range; -1 padded.
        # spans shape (B, M, 3); use small M.
        M = 5
        spans = torch.full((B, M, 3), -1, dtype=torch.long)
        for b in range(B):
            for m in range(M):
                l = m
                r = min(m + 3, seq_len - 1)
                spans[b, m] = torch.tensor([l, r, m])
        batch["tree_spans"] = spans
        batch["tree_span_mask"] = (spans[..., 0] >= 0)
    return batch


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

PASS, FAIL = [], []

def ok(name):  PASS.append(name); print(f"  [PASS] {name}")
def bad(name, e):
    FAIL.append((name, e)); print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    traceback.print_exc(limit=4)

def section(t): print(f"\n=== {t} ===")

def test_model_forward_plain(cfg, label):
    """Forward with tree_spans=None (the SG/BLIMP/XSUM/BOOLQ + lm path)."""
    section(f"{label}: model.forward(tree_spans=None)")
    try:
        model = build_model(cfg)
        batch = synth_batch(cfg, seq_len=64, with_tree_spans=False)
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
        assert out.logits.shape[:2] == batch["input_ids"].shape, out.logits.shape
        assert out.logits.shape[-1] == cfg.model.vocab_size
        assert torch.isfinite(out.logits).all(), "non-finite logits"
        ok(f"{label} forward(tree_spans=None) -> logits {tuple(out.logits.shape)} finite")
        return model
    except Exception as e:
        bad(f"{label} forward(tree_spans=None)", e)
        return None

def test_treereg_loss(cfg, model, label):
    """compute_treereg_loss on a captured hidden state (treereg only)."""
    if cfg.model.transformer_grammar_type != "treereg":
        return
    section(f"{label}: treereg aux loss (CPU)")
    try:
        from olmo.treereg import compute_treereg_loss
        B, T = 2, 64
        d_model = cfg.model.d_model
        tr_hidden = torch.randn(B, T, d_model)
        spans = torch.full((B, 5, 3), -1, dtype=torch.long)
        for b in range(B):
            for m in range(5):
                spans[b, m] = torch.tensor([m, m + 3, m])
        span_mask = (spans[..., 0] >= 0)
        loss = compute_treereg_loss(
            tr_hidden, spans, span_mask,
            n_heads_subset=cfg.model.treereg_n_heads,
            d_head=d_model // cfg.model.n_heads,
        )
        assert torch.is_tensor(loss) and loss.dim() == 0
        assert torch.isfinite(loss), loss
        ok(f"{label} compute_treereg_loss -> {loss.item():.4f} (finite)")
    except Exception as e:
        bad(f"{label} treereg_loss", e)

def test_evaluator_build(cfg, label, eval_label, eval_type):
    """build_downstream_evaluator + is_unit_test builds dataset+metric+loader."""
    section(f"{label}: build evaluator label={eval_label!r} type={eval_type}")
    try:
        from olmo.eval import build_downstream_evaluator
        from olmo.config import EvaluatorConfig, EvaluatorType
        from olmo.tokenizer import Tokenizer
        tokenizer = Tokenizer.from_train_config(cfg)
        ec = EvaluatorConfig(label=eval_label, type=EvaluatorType(eval_type))
        ev = build_downstream_evaluator(cfg, ec, tokenizer, torch.device("cpu"), is_unit_test=True)
        n = len(ev.eval_loader.dataset)
        assert ev.eval_metric is not None
        ok(f"{label} build {eval_label!r}: dataset len={n}, metric={type(ev.eval_metric).__name__}, type={ev.type}")
        return ev, tokenizer
    except Exception as e:
        bad(f"{label} build {eval_label!r}", e)
        return None, None

def test_sg_metric_update_compute(cfg, tokenizer):
    """Drive SyntacticGeneralizationMetric.update+compute with synthetic scores,
    including inf/nan (the bug that was fixed on this branch).

    Uses a REAL task name from formula_dict ('subordination') and its REAL
    condition names so the formula eval is exercised end-to-end."""
    section("SG metric: update + compute (incl. inf/nan formula eval)")
    try:
        from olmo.eval.downstream import SyntacticGeneralizationMetric
        m = SyntacticGeneralizationMetric(metric_type="ce_loss", tree_eval_type="default")
        # formula_dict['subordination'] compares sub_no-matrix > no-sub_no-matrix
        # AND sub_matrix < no-sub_matrix. Use the real condition names + an inf.
        m.update(task="subordination",
                 score_dict={"sub_no-matrix": 1.0, "no-sub_no-matrix": 2.0,
                             "sub_matrix": float("inf"), "no-sub_matrix": 3.0})
        res = m.compute()
        assert isinstance(res, dict) and len(res) > 0, res
        ok(f"SG metric.compute -> {len(res)} keys (inf handled, no NameError)")
    except Exception as e:
        bad("SG metric update/compute", e)

class _MiniTrainer:
    """Minimal stand-in for olmo.Trainer exposing just what the real
    eval_step / SG_eval_step / eval_batch / model_forward methods need, so we
    can bind and execute the ACTUAL production methods on CPU. We do NOT
    instantiate the full Trainer (no optimizer/DDP/dataloader)."""
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.device = torch.device("cpu")
        # Wrap like DDP: dist_model.module is the raw model.
        class _Wrap:
            def __init__(s, m): s.module = m; s.training = False
            def __call__(s, *a, **k): return s.module(*a, **k)
        self.dist_model = _Wrap(model)
        self.model = model
        self.global_step = 0
        # Bind the REAL methods from the Trainer class.
        from olmo.train import Trainer
        self.model_forward = Trainer.model_forward.__get__(self)
        self.eval_batch = Trainer.eval_batch.__get__(self)
        self.eval_step = Trainer.eval_step.__get__(self)
        self.SG_eval_step = Trainer.SG_eval_step.__get__(self)
        self.get_labels = Trainer.get_labels.__get__(self)
        # loss_fn: replicate the ce-loss path (no z-loss) used by eval.
        self.loss_fn = _make_loss_fn(cfg)

    def _summon_params_ctx(self):
        import contextlib
        return contextlib.nullcontext()


def _make_loss_fn(cfg):
    """A standalone CE loss matching Trainer.loss_fn's signature for eval."""
    import torch.nn.functional as F
    def loss_fn(logits, labels, ignore_index=-100, reduction="mean", compute_z_loss=False):
        ce = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
        return ce, None
    return loss_fn


def test_treereg_forward_parsed(cfg, label):
    """treereg docppl path: forward WITH tree_spans (standard attn, no flex).
    Pushdown parsed path needs flex (GPU) — skipped, documented."""
    if cfg.model.transformer_grammar_type != "treereg":
        return
    section(f"{label}: model.forward(tree_spans=given) [docppl parsed path]")
    try:
        model = build_model(cfg)
        batch = synth_batch(cfg, seq_len=64, with_tree_spans=True)
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                tree_spans=batch["tree_spans"],
            )
        assert out.logits.shape[:2] == batch["input_ids"].shape
        assert torch.isfinite(out.logits).all()
        # treereg_hidden capture should also work (layer 6).
        if out.treereg_hidden is not None:
            ok(f"{label} forward(tree_spans=given) -> logits finite, treereg_hidden {tuple(out.treereg_hidden.shape)}")
        else:
            ok(f"{label} forward(tree_spans=given) -> logits finite (treereg_hidden None at eval)")
    except Exception as e:
        bad(f"{label} forward(tree_spans=given)", e)


def test_eval_step_dispatch(cfg, label):
    """Drive Trainer.eval_step (BoolQ/BLiMP plain-LM path) on CPU."""
    section(f"{label}: Trainer.eval_step (BoolQ/BLiMP plain-LM dispatch)")
    try:
        model = build_model(cfg)
        tr = _MiniTrainer(cfg, model)
        # Build the boolq evaluator (ICLMetric, type=downstream -> eval_step).
        from olmo.eval import build_downstream_evaluator
        from olmo.config import EvaluatorConfig, EvaluatorType
        from olmo.tokenizer import Tokenizer
        tok = Tokenizer.from_train_config(cfg)
        ec = EvaluatorConfig(label="boolq", type=EvaluatorType.downstream)
        ev = build_downstream_evaluator(cfg, ec, tok, torch.device("cpu"), is_unit_test=True)
        # Pull ONE real batch from the boolq dataset.
        dl = ev.eval_loader
        batch = next(iter(dl))
        # eval_step moves to device + calls model_forward + update_metrics.
        import olmo.train as T
        T.barrier = lambda *a, **k: None  # no-op on single proc CPU
        tr.eval_step(batch, ev)
        # compute to ensure metric aggregation runs.
        m = ev.compute_metrics()
        ok(f"{label} eval_step(boolq) -> metrics keys={list(m)[:2]}")
    except Exception as e:
        bad(f"{label} eval_step(boolq)", e)


def test_sg_eval_step_dispatch(cfg, label):
    """Drive Trainer.SG_eval_step (teacher-forced CE path) on CPU."""
    section(f"{label}: Trainer.SG_eval_step (teacher-forced CE)")
    try:
        model = build_model(cfg)
        tr = _MiniTrainer(cfg, model)
        from olmo.eval import build_downstream_evaluator
        from olmo.config import EvaluatorConfig, EvaluatorType
        from olmo.tokenizer import Tokenizer
        tok = Tokenizer.from_train_config(cfg)
        ec = EvaluatorConfig(label="syntactic_generalization", type=EvaluatorType.downstream)
        ev = build_downstream_evaluator(cfg, ec, tok, torch.device("cpu"), is_unit_test=True)
        # SG dataloader yields one case per batch (list of 4 condition dicts).
        dl = ev.eval_loader
        batch = next(iter(dl))
        # Silence the debug prints in SG_eval_step.
        import builtins
        _print = builtins.print
        builtins.print = lambda *a, **k: None
        try:
            tr.SG_eval_step(batch, ev)
        finally:
            builtins.print = _print
        m = ev.compute_metrics()
        ok(f"{label} SG_eval_step -> metrics keys={list(m)[:3]}")
    except Exception as e:
        bad(f"{label} SG_eval_step", e)


def main():
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    configs = [("pushdown", "pushdown.yaml"), ("treereg", "treereg.yaml")]

    built_evaluators = {}
    for label, yaml in configs:
        try:
            cfg = load_cfg(yaml)
            print(f"\n##### {label}: loaded {yaml} (tg_type={cfg.model.transformer_grammar_type}, "
                  f"flex={cfg.model.flex_attention}, d_model={cfg.model.d_model})")
        except Exception as e:
            bad(f"{label} load config", e)
            continue

        model = test_model_forward_plain(cfg, label)
        test_treereg_loss(cfg, model, label)
        test_treereg_forward_parsed(cfg, label)
        test_eval_step_dispatch(cfg, label)
        test_sg_eval_step_dispatch(cfg, label)

        # The 5 downstream tasks. xsum MUST use type=rouge (not downstream).
        evs = [
            ("syntactic_generalization", "downstream"),
            ("BLiMP", "downstream"),
            ("boolq", "downstream"),
            ("xsum", "rouge"),
        ]
        for el, et in evs:
            ev, tok = test_evaluator_build(cfg, label, el, et)
            if ev is not None:
                built_evaluators.setdefault(label, []).append((ev, tok, cfg, el, et))

    # SG metric standalone (the branch's fix target).
    # reuse pushdown tokenizer
    try:
        cfg = load_cfg("pushdown.yaml")
        from olmo.tokenizer import Tokenizer
        tok = Tokenizer.from_train_config(cfg)
        test_sg_metric_update_compute(cfg, tok)
    except Exception as e:
        bad("SG metric setup", e)

    # ---------------- summary ----------------
    print("\n" + "=" * 60)
    print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
    if FAIL:
        print("\nFAILURES:")
        for name, e in FAIL:
            print(f"  - {name}: {type(e).__name__}: {e}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
