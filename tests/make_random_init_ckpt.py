"""Generate random-init pushdown + treereg checkpoints on CPU.

`scripts/train.py` hard-codes `torch.cuda.set_device(...)` so it can't run on
this CUDA-less dev box. But the user only needs RANDOM-INIT checkpoints to
exercise the eval path — no training. So we build OLMo(cfg) on CPU (which runs
the real `reset_parameters()` / `init_normal` weight init, the same init the
GPU run would produce), and save the state dict in the exact unsharded format
the loader expects:  <save_folder>/step0-unsharded/{model.pt, config.yaml}.

`restore_checkpoint` with `reset_optimizer_state=True, reset_trainer_state=True`
(the eval config sets both) loads ONLY model.pt, so optim.pt/train.pt are not
needed for eval-on-load.

Run: PYTHONPATH=. /home/wangpch/.conda/envs/LLM/bin/python tests/make_random_init_ckpt.py
"""
import sys, shutil
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]

CONFIGS = [
    ("pushdown", "pushdown.yaml"),
    ("treereg", "treereg.yaml"),
]


def main():
    from olmo.config import TrainConfig
    from olmo.model import OLMo

    for label, yaml in CONFIGS:
        cfg = TrainConfig.load(REPO / "train_configs" / yaml, [])
        cfg.model.init_device = "cpu"          # build on CPU
        cfg.model.precision = cfg.precision
        cfg.workspace = str(REPO)

        save_folder = REPO / "saved_models" / f"random_init_{label}"
        ckpt_dir = save_folder / "step0-unsharded"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {label}: building OLMo on CPU (d_model={cfg.model.d_model}, "
              f"n_layers={cfg.model.n_layers}, tg={cfg.model.transformer_grammar_type}) ===")
        model = OLMo(cfg.model, init_params=True)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        sd = model.state_dict()
        # Sanity: weights are real (not meta/NaN).
        w = next(iter(sd.values()))
        assert torch.isfinite(w).all(), "non-finite param"

        torch.save(sd, ckpt_dir / "model.pt")
        # Save the config the loader reads (build_sharded_checkpointer etc.).
        cfg.save(ckpt_dir / "config.yaml")
        print(f"  saved {ckpt_dir/'model.pt'}  ({n_params:,} params, "
              f"{(ckpt_dir/'model.pt').stat().st_size/1e6:.1f} MB)")
        print(f"  saved {ckpt_dir/'config.yaml'}")
    print("\nDONE. Checkpoints at saved_models/random_init_{pushdown,treereg}/step0-unsharded/")


if __name__ == "__main__":
    main()
