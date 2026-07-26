"""CPU test for the pushdown BLiMP beam-search branch.

Validates the primitive that ``Trainer.BLiMP_beam_eval_step`` calls for
``transformer_grammar_type == "pushdown"``: ``OLMo.pushdown_beam_search(tag=None)``
returns the parse-marginalized surprisal ``-log p(x)``. The branch converts this to
``log_likelihood = -surprisal`` and scatters it via ``BLiMPMetric.update_beam``.

We test the primitive directly (not the Trainer) on a handful of real BLiMP minimal
pairs from ``blimp_terminal.npy`` (task 0 = anaphor_gender_agreement): for each pair,
``surprisal(good)`` and ``surprisal(bad)`` must both be finite, and the model should
on average prefer the grammatical member (``mean surprisal_good <= mean surprisal_bad``).
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from olmo.model import OLMo

CKPT = "saved_models/pushdown/step33862-unsharded"
BLIMP_NPY = "dataset/BLiMP/tree300/blimp_terminal.npy"
BOS_ID = 50257   # <|beginoftext|> in the TG GPT-2 tokenizer (config.eos_token_id=50256 is BOS fallback)
EOS_ID = 50256
PAD_ID = 50258
N_PAIRS = 5      # task 0 pairs to average


def load_model() -> OLMo:
    from olmo.config import ModelConfig
    cfg = ModelConfig.load(os.path.join(CKPT, "config.yaml"), key="model", validate_paths=False)
    cfg.init_device = "cpu"
    model = OLMo(cfg)
    sd = torch.load(os.path.join(CKPT, "model.pt"), map_location="cpu")
    # strict=False: the checkpoint predates pushdown_attachment_head (it is
    # train-only and unused here with use_attachment_head=False), so its weights
    # are absent — leave them randomly initialized.
    missing, unexpected = model.load_state_dict(
        model._make_state_dict_compatible(sd)[0], strict=False)
    missing = [k for k in missing if "attachment_head" not in k]
    assert not missing, f"unexpected missing keys: {missing}"
    # CPU cannot run flex_attention's inductor kernel -> use the SDPA additive-mask
    # fallback (numerically equivalent). GPU eval keeps pushdown_use_flex=True.
    model.config.flex_attention = False
    model.config.pushdown_use_flex = False
    for block in model.transformer.blocks:
        if hasattr(block, "flex_attention"):
            block.flex_attention = None
        if hasattr(block, "_is_pushdown"):
            block._is_pushdown = True
    model.eval()
    return model


def strip_pad(seq: np.ndarray) -> torch.Tensor:
    """Drop trailing PAD/EOS padding, keep BOS..EOS."""
    ids = seq.tolist()
    n = len(ids)
    while n > 0 and ids[n - 1] == PAD_ID:
        n -= 1
    return torch.tensor(ids[:n], dtype=torch.long)


def surprisal(model: OLMo, seq: torch.Tensor) -> float:
    with torch.no_grad():
        return model.pushdown_beam_search(
            eval_input_ids=seq, beam_size=20, max_reduce=4,
            bos_id=BOS_ID, tag=None, use_attachment_head=False,
        )


def main() -> None:
    print("Loading pushdown model...", flush=True)
    model = load_model()

    print(f"Loading BLiMP terminal data from {BLIMP_NPY}...", flush=True)
    data = np.load(BLIMP_NPY, mmap_mode="r")   # (67 tasks, 2000 samples, 100)

    good_surps, bad_surps = [], []
    for pair in range(N_PAIRS):
        good = strip_pad(data[0, 2 * pair])      # even = good
        bad = strip_pad(data[0, 2 * pair + 1])   # odd  = bad
        sg = surprisal(model, good)
        sb = surprisal(model, bad)
        good_surps.append(sg)
        bad_surps.append(sb)
        print(f"  pair {pair}: good surprisal={sg:.3f}  bad surprisal={sb:.3f}  "
              f"{'OK' if sg <= sb else '(bad preferred)'}", flush=True)

    mg = float(np.mean(good_surps))
    mb = float(np.mean(bad_surps))
    print(f"\nmean good surprisal = {mg:.3f}", flush=True)
    print(f"mean bad  surprisal = {mb:.3f}", flush=True)

    # Assertions.
    for s in good_surps + bad_surps:
        assert np.isfinite(s), f"surprisal not finite: {s}"
    assert mg <= mb, f"model prefers bad on average (good {mg} > bad {mb})"
    print("\nPASS: all surprisals finite, good <= bad on average.", flush=True)


if __name__ == "__main__":
    main()
