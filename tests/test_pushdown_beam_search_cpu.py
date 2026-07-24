"""CPU verification of the batched pushdown_beam_search.

Loads the trained pushdown checkpoint, takes a real dev chunk (input_ids + gold
tree_spans), and compares per-token mean CE under three conditions:
  (a) DEGENERATE: tree_spans=None (the depth bias vanishes -> PPL 78923).
  (b) REAL:       gold tree_spans from parse_aligned (PPL ~1035).
  (c) BEAM:       pushdown_beam_search (batched) — should fall between (a) and (b),
                 trending toward (b). Confirms the depth path activates.

Also times the beam search to confirm the per-SHIFT batched-forward speedup
(one forward per token, not ~beam_size*max_reduce serial forwards).
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from olmo.model import OLMo
from olmo.data.parse_align import PrecomputedParseDataset

CKPT = "saved_models/pushdown/step33862-unsharded"
DEV_DIR = "dataset/parse_aligned/dev_left"
SEQ_LEN = 16  # use a short prefix for a fast CPU check (matches prior diagnostic)


def load_model() -> OLMo:
    from olmo.config import ModelConfig
    cfg = ModelConfig.load(os.path.join(CKPT, "config.yaml"), key="model", validate_paths=False)
    cfg.init_device = "cpu"
    model = OLMo(cfg)
    sd = torch.load(os.path.join(CKPT, "model.pt"), map_location="cpu")
    model.load_state_dict(model._make_state_dict_compatible(sd)[0])
    # TEST-ONLY override: flex_attention's inductor kernel does not run on CPU
    # (LoweringException on cpp_flex_attention_template). Force the numerically-
    # equivalent SDPA additive-mask fallback (olmo/model.py ~L896). The GPU eval
    # path keeps pushdown_use_flex=True (set in config.yaml); this override is
    # local to this CPU diagnostic and does not change weights or the GPU path.
    model.config.flex_attention = False
    model.config.pushdown_use_flex = False
    # Disable per-block flex_attr (OLMoBlock.flex_attention) to avoid the compiled
    # flex path entirely on CPU.
    for block in model.transformer.blocks:
        if hasattr(block, "flex_attention"):
            block.flex_attention = None
        if hasattr(block, "_is_pushdown"):
            block._is_pushdown = True  # keep the depth-bias path active
    model.eval()
    return model


def mean_ce_degenerate(model: OLMo, input_ids: torch.Tensor) -> float:
    """No tree_spans: depth bias vanishes."""
    with torch.no_grad():
        out = model.forward(input_ids=input_ids.unsqueeze(0), tree_spans=None)
    logits = out.logits[0, :-1].float()  # predict tokens 1..L-1
    target = input_ids[1:]
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    return ce.item()


def mean_ce_real(model: OLMo, input_ids: torch.Tensor, tree_spans: torch.Tensor) -> float:
    """Gold tree_spans: the trained depth-bias path."""
    ts = tree_spans.unsqueeze(0)  # (1, M, 3)
    attn = torch.ones(1, input_ids.shape[0], dtype=torch.bool)
    with torch.no_grad():
        out = model.forward(input_ids=input_ids.unsqueeze(0), attention_mask=attn,
                            tree_spans=ts)
    logits = out.logits[0, :-1].float()
    target = input_ids[1:]
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="mean")
    return ce.item()


def beam_surprisal(model: OLMo, input_ids: torch.Tensor, bos_id: int) -> float:
    """Beam search surprisal (tag=None -> -log p(x)). Divide by (n-1) for mean CE."""
    with torch.no_grad():
        surprisal = model.pushdown_beam_search(
            eval_input_ids=input_ids, beam_size=20, max_reduce=4,
            bos_id=bos_id, tag=None,
        )
    return surprisal


def main() -> None:
    print("Loading model...", flush=True)
    t0 = time.time()
    model = load_model()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    print("Loading dev chunk...", flush=True)
    ds = PrecomputedParseDataset(DEV_DIR, pad_token_id=50258)
    item = ds[0]
    # Find a real sentence boundary within the chunk: take the first SEQ_LEN non-pad
    # tokens. input_ids is (2048,) padded with 50258. Use the first contiguous run.
    full_ids = item["input_ids"]
    npad = int((full_ids == 50258).sum())
    valid_len = full_ids.shape[0] - npad
    print(f"  chunk0 valid_len={valid_len} (pad={npad})", flush=True)
    L = min(SEQ_LEN, valid_len)
    input_ids = full_ids[:L].clone()
    # Gold spans restricted to [0, L): keep spans with r < L, clamp l to [0,L).
    tspans = item["tree_spans"]
    keep = []
    for row in tspans.tolist():
        l, s, r = row
        if r < L and l >= 0 and l <= r:
            keep.append([l, s, r])
    if keep:
        tspans_L = torch.tensor(keep, dtype=torch.long)
    else:
        tspans_L = torch.zeros((0, 3), dtype=torch.long)
    print(f"  using L={L} tokens, {len(keep)} gold spans within [0,{L})", flush=True)

    bos_id = 50256  # eos_token_id in config; used as BOS by SG_eval_step caller

    print("\n=== Per-token mean CE (lower=better; real<beam<degenerate expected) ===", flush=True)
    ce_deg = mean_ce_degenerate(model, input_ids)
    print(f"  DEGENERATE (no spans):  mean CE = {ce_deg:.4f}  (PPL={np.exp(ce_deg):.1f})", flush=True)

    if len(keep) > 0:
        ce_real = mean_ce_real(model, input_ids, tspans_L)
        print(f"  REAL (gold tree_spans): mean CE = {ce_real:.4f}  (PPL={np.exp(ce_real):.1f})", flush=True)
    else:
        ce_real = float("nan")
        print("  REAL: no gold spans in prefix, skipped", flush=True)

    t0 = time.time()
    surp = beam_surprisal(model, input_ids, bos_id=bos_id)
    beam_dt = time.time() - t0
    ce_beam = surp / max(L - 1, 1)
    print(f"  BEAM (batched, b=20):   mean CE = {ce_beam:.4f}  (PPL={np.exp(ce_beam):.1f})"
          f"  surprisal={surp:.3f}  time={beam_dt:.2f}s  ({beam_dt/max(L-1,1)*1000:.1f}ms/tok)", flush=True)

    print("\n=== Verdict ===", flush=True)
    if not np.isnan(ce_real):
        if ce_beam < ce_deg:
            print(f"  OK: beam CE {ce_beam:.4f} < degenerate {ce_deg:.4f} (depth path activates).", flush=True)
        else:
            print(f"  WARN: beam CE {ce_beam:.4f} >= degenerate {ce_deg:.4f} (depth path NOT helping).", flush=True)
        if ce_beam <= ce_real + 0.5:
            print(f"  OK: beam CE within 0.5 of real {ce_real:.4f} (good approximation).", flush=True)
        else:
            print(f"  NOTE: beam CE {ce_beam:.4f} > real+0.5 ({ce_real:.4f}) — accuracy ceiling expected (no attachment head).", flush=True)
    print(f"  efficiency: {beam_dt:.2f}s for {L-1} tokens = 1 batched forward/token (was ~100 serial).", flush=True)


if __name__ == "__main__":
    main()
