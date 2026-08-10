#!/usr/bin/env python
"""GPST pre-training entry point (unsupervised + supervised).

Unsupervised:
  python scripts/gpst/run_gpst.py --unsupervised \
    --corpus_path corpus/mycorpus.lazy \
    --r2d2_config_path olmo/gpst/data/en_config/r2d2_256_4_1.json \
    --gpt_config_path olmo/gpst/data/gpt2-small/config.json \
    --vocab_dir olmo/gpst/data/gpt2-small --output_dir saved_models/gpst-unsup

Supervised (gold trees):
  python scripts/gpst/run_gpst.py --supervised \
    --tree_npy dataset/bbc-news/tree/train.npy \
    --tokenizer_path dataset/bbc-news/TG_GPT2_tokenizer.json \
    --r2d2_config_path olmo/gpst/data/en_config/r2d2_256_4_1.json \
    --gpt_config_path olmo/gpst/data/gpt2-small/config.json \
    --vocab_dir olmo/gpst/data/gpt2-small --output_dir saved_models/gpst-sup

Multi-GPU via torchrun:
  torchrun --nproc-per-node=8 scripts/gpst/run_gpst.py ...

OLMo backbone (instead of the default HF GPT2) — use the repo's own OLMo
block stack for the type/token transformer sub-stacks:
  python scripts/gpst/run_gpst.py --unsupervised --backbone olmo ...
"""
import argparse
import atexit
import logging
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTHONPATH", REPO)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from olmo.gpst.model.model_factory import create_model  # noqa: E402
from olmo.gpst.trainer.trainer import train, TrainConfig  # noqa: E402


def initialize_distributed(local_rank: int, device_type: str) -> bool:
    """Initialize the process group created by ``torchrun``.

    ``torchrun`` only exports rendezvous environment variables; applications
    still have to call ``init_process_group`` themselves.
    """
    if local_rank < 0 or torch.distributed.is_initialized():
        return False
    backend = "nccl" if device_type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")

    def cleanup():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    atexit.register(cleanup)
    return True


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--unsupervised", action="store_true")
    g.add_argument("--supervised", action="store_true")
    ap.add_argument("--corpus_path", default=None)
    ap.add_argument("--tree_npy", default=None)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--r2d2_config_path", required=True)
    ap.add_argument("--gpt_config_path", required=True)
    ap.add_argument("--vocab_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accumulation_steps", type=int, default=1)
    ap.add_argument("--num_samples", type=int, default=1_000_000)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--parser_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.01)
    ap.add_argument("--log_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=10000)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--gradient_checkpoint", action="store_true")
    ap.add_argument("--backbone", choices=["gpt2", "olmo"], default="gpt2",
                    help="Transformer backbone for the generative model's "
                         "type/token stacks: 'gpt2' (HF GPT2, default) or "
                         "'olmo' (OLMo block stack).")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:" + str(local_rank))
        torch.cuda.set_device(local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_distributed(local_rank, device.type)
    is_master = (not torch.distributed.is_initialized() or
                 torch.distributed.get_rank() == 0)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("gpst")
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    model = create_model('r2d2-gen-fast', args.r2d2_config_path,
                         args.gpt_config_path, gradient_checkpoint=args.gradient_checkpoint,
                         backbone=args.backbone)
    model.to(device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)

    if args.unsupervised:
        from olmo.gpst.reader.lazy_loader import LazyLoader
        from olmo.gpst.reader.dataset import GPT2Dataset
        from olmo.gpst.reader.data_collator import DefaultCollator
        assert args.corpus_path, "--corpus_path required for --unsupervised"
        loader_ds = LazyLoader(args.corpus_path, is_array=True)
        ds = GPT2Dataset(loader_ds, num_samples=args.num_samples, max_seq_len=args.max_seq_len)
        collator = DefaultCollator(enable_group=True, external_vocab_path=None)
        collate_fn = collator.generative_r2d2_collate_fn_ext
    else:
        from olmo.gpst.reader.dataset_gold import GoldTreeDataset, GoldTreeCollator
        assert args.tree_npy and args.tokenizer_path, \
            "--tree_npy and --tokenizer_path required for --supervised"
        ds = GoldTreeDataset(args.tree_npy, args.tokenizer_path,
                             max_seq_len=args.max_seq_len, num_samples=args.num_samples)
        collate_fn = GoldTreeCollator()

    sampler = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=False)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collate_fn, num_workers=1)

    cfg = TrainConfig(
        lr=args.lr, parser_lr=args.parser_lr, warmup=args.warmup,
        accumulation_steps=args.accumulation_steps, log_steps=args.log_steps,
        save_steps=args.save_steps, max_steps=args.max_steps,
    )
    train(model, loader, cfg, device, logger=log, output_dir=args.output_dir,
          is_master=is_master)


if __name__ == "__main__":
    main()
