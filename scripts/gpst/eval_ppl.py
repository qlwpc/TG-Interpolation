#!/usr/bin/env python
"""Held-out perplexity for the GPST generative model (pure LM protocol).

Loads a GPST checkpoint (FastGenerativeR2D2), runs teacher-forced forward in
eval mode (dropout off) over held-out lazy-corpus windows and computes token CE
directly from the returned logits — bypassing the trainer's training-only loss
branches (parser/io losses are zeroed in eval mode by design, and ``gpt_loss``
is only computed under ``self.training``).

The CE target alignment is identical to the training-time ``gpt_loss``
(``chunk_input_ids`` with -100 padding, same ``next_token_indices`` gather),
so this measures exactly what training measured — minus dropout and with a
held-out corpus.

Example:
  python scripts/gpst/eval_ppl.py \
      --checkpoint saved_models/gpst-bbc-unsup/model.bin \
      --corpus_path corpus/bbc-test.lazy \
      --backbone olmo --batch_size 8 --num_batches 300
"""
import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONPATH", REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Same SDPA constraint as run_gpst.py: cat_input length L+1 (BOS prepended)
# trips the flash backend's multiple-of-8 assert.
import torch  # noqa: E402

torch.backends.cuda.enable_flash_sdp(False)

import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from olmo.gpst.model.model_factory import create_model  # noqa: E402
from olmo.gpst.reader.dataset import GPT2Dataset  # noqa: E402
from olmo.gpst.reader.lazy_loader import LazyLoader  # noqa: E402
from olmo.gpst.reader.data_collator import DefaultCollator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--corpus_path", required=True,
                    help="Lazy corpus dir (data + data.len.pkl) built from held-out text")
    ap.add_argument("--r2d2_config_path", default="olmo/gpst/data/en_config/r2d2_256_4_1.json")
    ap.add_argument("--gpt_config_path", default="olmo/gpst/data/gpt2-bbc/config.json")
    ap.add_argument("--backbone", choices=["gpt2", "olmo"], default="olmo",
                    help="Must match the backbone the checkpoint was trained with.")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_batches", type=int, default=300,
                    help="Eval batches; 300 x 8 x ~1900 tok ~= 4.5M tokens")
    ap.add_argument("--num_samples", type=int, default=100000,
                    help="Virtual dataset size; only the first num_batches are used")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    torch.manual_seed(args.seed)

    model = create_model("r2d2-gen-fast", args.r2d2_config_path,
                         args.gpt_config_path, backbone=args.backbone)
    state = torch.load(args.checkpoint, map_location="cpu")
    clean = {k[len("module."):] if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing or unexpected:
        print(f"missing keys: {list(missing)[:8]}{'...' if len(missing) > 8 else ''}")
        print(f"unexpected keys: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        raise SystemExit("checkpoint mismatch — aborting")
    model.to(device).eval()

    loader_ds = LazyLoader(args.corpus_path, is_array=True)
    ds = GPT2Dataset(loader_ds, num_samples=args.num_samples,
                     max_seq_len=args.max_seq_len)
    r2d2_cfg = AutoConfig.from_pretrained(args.r2d2_config_path)
    parser_max_len = getattr(r2d2_cfg, "parser_max_len", 1024)
    collator = DefaultCollator(enable_group=True, external_vocab_path=None,
                               max_seg_len=parser_max_len)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        collate_fn=collator.generative_r2d2_collate_fn_ext,
                        num_workers=1)

    tot_nll = 0.0
    tot_tok = 0
    with torch.no_grad():
        for i, inputs in enumerate(loader):
            if i >= args.num_batches:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in inputs.items()}
            out = model(**batch)
            # Eval mode gathers max_input_len+1 positions (forward line 149);
            # training gathers max_input_len. Slicing the gathered sequence back
            # to the target width reproduces the training-time gpt_loss
            # alignment exactly (slice-of-gather == gather-with-truncated-idx).
            T = out.token_targets.shape[1]
            nll = F.cross_entropy(out.logits[:, :T].float().permute(0, 2, 1),
                                  out.token_targets, ignore_index=-100,
                                  reduction="sum")
            tot_nll += float(nll)
            tot_tok += int((out.token_targets != -100).sum())
            if i % 25 == 0:
                run = math.exp(tot_nll / max(tot_tok, 1))
                print(f"batch {i:>4}: tokens={tot_tok:>9} running PPL={run:.3f}",
                      flush=True)

    ppl = math.exp(tot_nll / tot_tok)
    print("=== RESULT ===")
    print(f"tokens evaluated : {tot_tok}")
    print(f"nll per token    : {tot_nll / tot_tok:.6f}")
    print(f"perplexity       : {ppl:.4f}")


if __name__ == "__main__":
    main()
