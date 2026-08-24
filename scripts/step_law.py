"""Step-Law audit tool — paper Eq.(8) (Li et al. 2025, "Predictable scale").

    lr(N, D) = 1.79 * N^-0.713 * D^0.307        (peak learning rate)
    B(D)     = 0.58 * D^0.571                    (global batch size in TOKENS)

The paper states lr and batch size follow this law; the repo hard-codes the
resulting values in each train config YAML.  This script makes the formula
computable and auditable:

  * audit mode   — derive N (non-embedding params) from the model config and
                   D (training tokens) from the data paths, predict lr/B, and
                   compare against the YAML's actual values (including the
                   implied N/D that would reproduce the actual lr).
  * raw mode     — plain ``--params N --tokens D`` calculator.

Usage:
    python scripts/step_law.py --config train_configs/terminal.yaml
    python scripts/step_law.py --params 1.13e8 --tokens 1.0e10 --seq-len 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Paper Eq.(8) constants
# ---------------------------------------------------------------------------
LR_COEF = 1.79
LR_N_EXP = -0.713
LR_D_EXP = 0.307
B_COEF = 0.58
B_EXP = 0.571


def step_law_lr(N: float, D: float) -> float:
    """Peak learning rate for N non-embedding params trained on D tokens."""
    if N <= 0 or D <= 0:
        raise ValueError(f"N and D must be positive, got N={N}, D={D}")
    return LR_COEF * N**LR_N_EXP * D**LR_D_EXP


def step_law_batch_tokens(D: float) -> float:
    """Global batch size in TOKENS for a training run over D tokens."""
    if D <= 0:
        raise ValueError(f"D must be positive, got D={D}")
    return B_COEF * D**B_EXP


def solve_implied_D(lr: float, N: float) -> float:
    """D that reproduces ``lr`` under the lr law, holding N fixed."""
    return (lr / (LR_COEF * N**LR_N_EXP)) ** (1.0 / LR_D_EXP)


def solve_implied_N(lr: float, D: float) -> float:
    """N that reproduces ``lr`` under the lr law, holding D fixed."""
    return (lr / (LR_COEF * D**LR_D_EXP)) ** (1.0 / LR_N_EXP)


# ---------------------------------------------------------------------------
# N from a ModelConfig
# ---------------------------------------------------------------------------

def count_non_embedding_params(model_cfg) -> int:
    """Non-embedding parameter count (excludes .wte./.wpe.), same convention
    as ``OLMo.num_params(include_embedding=False)``.

    Builds the model with ``init_device="meta"`` (no memory, and immune to a
    caller's ``init_device: cuda``); falls back to CPU if the meta backend
    rejects some init op.
    """
    import dataclasses

    from olmo.model import OLMo

    try:
        model = OLMo(dataclasses.replace(model_cfg, init_device="meta"))
    except (RuntimeError, NotImplementedError):
        model = OLMo(dataclasses.replace(model_cfg, init_device="cpu"))
    return model.num_params(include_embedding=False)


# ---------------------------------------------------------------------------
# D from data paths
# ---------------------------------------------------------------------------

def count_data_tokens(paths: list[str], itemsize: int = 2) -> int:
    """Total token count over a list of .npy/.bin data paths (globs allowed).

    ``.npy`` files are read via mmap (header only); raw ``.bin`` files are
    assumed to be uint16 (``itemsize=2``) token streams.
    """
    import glob

    import numpy as np

    files: list[str] = []
    for pattern in paths:
        matches = sorted(glob.glob(pattern))
        if not matches and not Path(pattern).exists():
            raise FileNotFoundError(f"no data files found for: {pattern}")
        files.extend(matches or [pattern])

    total = 0
    for f in files:
        if f.endswith(".npy"):
            total += int(np.load(f, mmap_mode="r").size)
        else:
            total += Path(f).stat().st_size // itemsize
    return total


# ---------------------------------------------------------------------------
# Config audit
# ---------------------------------------------------------------------------

def audit_config(cfg, remap: list[tuple[str, str]] | None = None) -> dict:
    """Predict lr/B for a TrainConfig via the Step Law and compare to actuals.

    ``remap`` is a list of ``(old_prefix, new_prefix)`` rewrites applied to the
    config's data paths (for configs written on another machine/workspace).
    """
    N = count_non_embedding_params(cfg.model)
    paths = [str(p) for p in cfg.data.paths]
    for old, new in remap or []:
        paths = [new + p[len(old):] if p.startswith(old) else p for p in paths]
    D = count_data_tokens(paths)
    seq_len = cfg.model.max_sequence_length

    lr_pred = step_law_lr(N, D)
    batch_tokens_pred = step_law_batch_tokens(D)
    return {
        "name": cfg.run_name or Path(cfg.data.paths[0]).parent.name,
        "N": N,
        "D": D,
        "seq_len": seq_len,
        "lr_pred": lr_pred,
        "lr_actual": cfg.optimizer.learning_rate,
        "batch_tokens_pred": batch_tokens_pred,
        "batch_seq_pred": batch_tokens_pred / seq_len,
        "batch_actual": cfg.global_train_batch_size,
        "implied_D_from_lr": solve_implied_D(cfg.optimizer.learning_rate, N),
        "implied_N_from_lr": solve_implied_N(cfg.optimizer.learning_rate, D),
        "implied_D_from_batch": (cfg.global_train_batch_size * seq_len / B_COEF)
        ** (1.0 / B_EXP),
    }


def _fmt_ratio(pred: float, actual: float) -> str:
    if actual == 0:
        return "n/a"
    return f"{pred / actual:.3f}x"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", nargs="+", help="train config YAML(s) to audit")
    parser.add_argument(
        "--remap",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="rewrite data path prefix OLD→NEW (repeatable), e.g. "
        "--remap .//bbc-news=dataset/bbc-news --remap /dev/shm/dataset=dataset",
    )
    parser.add_argument("--params", type=float, help="raw mode: N non-embedding params")
    parser.add_argument("--tokens", type=float, help="raw mode: D training tokens")
    parser.add_argument("--seq-len", type=int, default=2048, help="seq len for B→seqs")
    args = parser.parse_args(argv)

    if args.config:
        from olmo.config import TrainConfig

        remap = [tuple(r.split("=", 1)) for r in args.remap]
        rows = []
        for yaml_path in args.config:
            cfg = TrainConfig.load(yaml_path, [])
            try:
                rows.append(audit_config(cfg, remap=remap))
            except FileNotFoundError as e:
                print(f"[skip] {yaml_path}: {e}", file=sys.stderr)
                continue
        if not rows:
            return 1

        header = (
            f"{'config':<28} {'N':>12} {'D':>14} {'lr pred':>10} {'lr yaml':>10} "
            f"{'ratio':>7} {'B pred':>9} {'B yaml':>7} {'ratio':>7} {'implied D(lr)':>14}"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r['name']:<28.28} {r['N']:>12,} {r['D']:>14,} "
                f"{r['lr_pred']:>10.6f} {r['lr_actual']:>10.6f} "
                f"{_fmt_ratio(r['lr_pred'], r['lr_actual']):>7} "
                f"{r['batch_seq_pred']:>9.1f} {r['batch_actual']:>7} "
                f"{_fmt_ratio(r['batch_seq_pred'], r['batch_actual']):>7} "
                f"{r['implied_D_from_lr']:>14,.3e}"
            )
        return 0

    if args.params and args.tokens:
        lr = step_law_lr(args.params, args.tokens)
        b_tokens = step_law_batch_tokens(args.tokens)
        print(f"N={args.params:.3e}  D={args.tokens:.3e}  seq_len={args.seq_len}")
        print(f"lr(N,D)      = {lr:.6f}")
        print(f"B(D)         = {b_tokens:,.0f} tokens = {b_tokens / args.seq_len:.1f} seqs")
        return 0

    parser.error("either --config YAML(s) or --params/--tokens are required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
