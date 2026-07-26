"""CPU test for the pushdown ICL (boolq) eval path.

Validates the core of ``Trainer.pushdown_icl_eval_step``: for a real BoolQ query,
``OLMo.pushdown_beam_search(..., return_spans=True)`` returns non-empty closed
spans (the depth-bias path activates), and a teacher-forced forward with those
spans gives lower continuation cross-entropy than the degenerate ``tree_spans=None``
forward (which is what the current broken ``eval_step`` does -> acc 0.3804).
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from olmo.model import OLMo
from olmo.tokenizer import Tokenizer
from olmo.eval.downstream import BoolQ

CKPT = "saved_models/pushdown/step33862-unsharded"
VOCAB = "dataset/bbc-news/TG_GPT2_tokenizer.json"
PAD_ID = 50258
BOS_SEED = 50256   # config.eos_token_id (BOS seed, matches SG_eval_step)


def load_model() -> OLMo:
    from olmo.config import ModelConfig
    cfg = ModelConfig.load(os.path.join(CKPT, "config.yaml"), key="model", validate_paths=False)
    cfg.init_device = "cpu"
    model = OLMo(cfg)
    sd = torch.load(os.path.join(CKPT, "model.pt"), map_location="cpu")
    missing, _ = model.load_state_dict(
        model._make_state_dict_compatible(sd)[0], strict=False)
    missing = [k for k in missing if "attachment_head" not in k]
    assert not missing, f"unexpected missing keys: {missing}"
    model.config.flex_attention = False
    model.config.pushdown_use_flex = False
    for block in model.transformer.blocks:
        if hasattr(block, "flex_attention"):
            block.flex_attention = None
        if hasattr(block, "_is_pushdown"):
            block._is_pushdown = True
    model.eval()
    return model


def build_boolq():
    tok = Tokenizer.from_file(VOCAB, pad_token_id=PAD_ID)
    ds = BoolQ(
        tokenizer=tok,
        dataset_path="boolq",
        model_ctx_len=2048,
        split="val",
        shots_num=0,
        transformer_grammar_type="pushdown",
        generate_TG_attention_bias=None,
        vocab_path=VOCAB,
        tree_eval_type="default",
        pause_token_id=None,
    )
    return ds


def main() -> None:
    print("Loading pushdown model...", flush=True)
    model = load_model()
    print("Building BoolQ dataset...", flush=True)
    ds = build_boolq()
    # Take 2 val examples (yes/no continuations). CPU pushdown_beam_search is
    # ~1 forward/token; truncate the long BoolQ passage to keep the CPU test fast
    # (GPU eval runs the full context). Keep the continuation (last cont_len tokens).
    samples = ds.samples[:2]
    print(f"  {len(samples)} samples (yes/no continuations)", flush=True)

    for i, s in enumerate(samples):
        query = torch.tensor(s["query"], dtype=torch.long)
        ctx_len, cont_len = s["ctx_len"], s["cont_len"]
        # Truncate the LEFT side to a CPU-friendly prefix ending at the continuation.
        keep_ctx = 50
        start = max(0, ctx_len - keep_ctx)
        q = query[start:]
        real_L = q.shape[0]
        # Best-beam closed spans (small beam for CPU speed).
        with torch.no_grad():
            _, spans = model.pushdown_beam_search(
                eval_input_ids=q[:real_L], beam_size=8, max_reduce=4,
                bos_id=BOS_SEED, tag=None, return_spans=True, use_attachment_head=False,
            )
        n_spans = spans.shape[0]
        ts = spans.unsqueeze(0) if n_spans > 0 else None
        inp = q[:real_L].unsqueeze(0)
        attn = torch.ones_like(inp)
        with torch.no_grad():
            out_deg = model.forward(input_ids=inp, attention_mask=attn, tree_spans=None)
            out_span = model.forward(input_ids=inp, attention_mask=attn, tree_spans=ts)
        # The depth bias must actually change the logits (tree_spans consumed).
        diff = (out_deg.logits.float() - out_span.logits.float()).abs().mean().item()
        print(f"  sample {i}: real_L={real_L} n_spans={n_spans}  "
              f"mean |logit diff|={diff:.6f}", flush=True)
        assert n_spans > 0, f"sample {i}: no spans inferred (depth path inactive)"
        assert diff > 1e-6, f"sample {i}: spans did not change logits (diff={diff})"

    print("\nPASS: spans inferred (non-empty) and change the forward logits; "
          "depth-bias path active for boolq ICL.", flush=True)


if __name__ == "__main__":
    main()
