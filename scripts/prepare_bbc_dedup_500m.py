#!/usr/bin/env python3
"""Generate paper-style 500M pretraining configs for deduplicated BBC data.

Run from the repository root using the training Python environment. Generation
only writes configs, launchers and an audit manifest; it never submits jobs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.step_law import count_non_embedding_params, step_law_lr
from scripts.submit_pause_sep_pretrain import make_batch_plan

FORMATS = {"terminal": "terminal", "tree": "tree", "tgtree": "tg", "tgnomask_aug": "tg"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stream_size(path: Path) -> int:
    import numpy as np

    stream = np.load(path, mmap_mode="r", allow_pickle=False)
    if stream.ndim != 1 or stream.dtype != np.dtype("uint16") or stream.size < 2048:
        raise ValueError(f"Expected a 1D uint16 token stream with >=2048 tokens: {path}")
    return int(stream.size)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/bbc-news-parsed-dedup")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "dataset/bbc-news-reserved-clean-v1")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/experiment/bbc_dedup_500m")
    parser.add_argument("--gpus", type=int, default=4, help="Single-node world size")
    parser.add_argument("--max-microbatch", type=int, default=4)
    args = parser.parse_args(argv)
    if args.gpus <= 0 or args.max_microbatch <= 0:
        parser.error("--gpus and --max-microbatch must be positive")
    from omegaconf import OmegaConf as om
    from olmo.config import TrainConfig
    from tokenizers import Tokenizer

    tokenizer = args.tokenizer.resolve()
    tok = Tokenizer.from_file(str(tokenizer))
    if tok.get_vocab_size() != 50320 or tok.token_to_id("<|SEP|>") != 50261:
        raise ValueError("Expected the paper BBC GPT-2 tokenizer (50320 tokens, SEP=50261)")
    tokenizer_sha = digest(tokenizer)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output-dir: {output}")
    prepared = []
    for name in dict.fromkeys(args.models):
        fmt = FORMATS[name]
        train = args.data_dir.resolve() / fmt / "train.npy"
        tokens = stream_size(train)
        receipt_path = train.with_suffix(".manifest.json")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "complete" or receipt.get("tokens") != tokens
                or receipt.get("dtype") != "uint16" or receipt.get("format") != fmt):
            raise ValueError(f"Training manifest disagrees with array: {receipt_path}")
        tokenizer_matches = receipt.get("tokenizer_sha256") == tokenizer_sha
        reference = ROOT / "artifacts/bbc_reserved_clean_20260906/remote_tokenizer.json"
        tokenizer_equivalent = tokenizer_matches or (
            reference.is_file()
            and digest(reference) == receipt.get("tokenizer_sha256")
            and json.loads(reference.read_text()) == json.loads(tokenizer.read_text())
        )
        if not tokenizer_equivalent:
            print(f"WARNING: {fmt}: training receipt tokenizer hash differs from current tokenizer; "
                  "token ID compatibility is not certified", file=sys.stderr)
        cfg = om.load(ROOT / "train_configs/tree-500M.yaml")
        cfg.run_name = f"bbc-dedup-500m-{name}"
        cfg.model.transformer_grammar_type = name
        structural = name == "tgnomask_aug"
        cfg.model.flash_attention = not structural
        cfg.model.flex_attention = structural
        cfg.wandb = None
        cfg.load_path = None
        cfg.save_overwrite = False
        cfg.save_folder = str(output / "checkpoints" / name)
        cfg.tokenizer.identifier = str(tokenizer)
        cfg.tokenizer.vocabulary = str(tokenizer)
        cfg.data.paths = [str(train)]
        cfg.data.memmap_dtype = "uint16"
        cfg.data.memmap_format = "auto"
        cfg.data.generate_attention_mask = structural
        cfg.data.generate_doc_lengths = True
        cfg.evaluators = []
        eval_paths = []
        for split in ("dev", "test"):
            path = args.eval_dir.resolve() / fmt / f"{split}.npy"
            stream_size(path)
            eval_paths.append(str(path))
            cfg.evaluators.append({"label": f"clean-{split}", "type": "lm", "data": {
                "paths": [str(path)], "memmap_dtype": "uint16", "memmap_format": "auto",
                "generate_attention_mask": structural, "generate_doc_lengths": True,
                "num_workers": 0, "drop_last": False, "persistent_workers": False}})
        model_cfg = TrainConfig.new(**om.to_container(cfg, resolve=True)).model
        params = count_non_embedding_params(model_cfg)
        batch = make_batch_plan(tokens, 2048, args.gpus, args.max_microbatch)
        cfg.optimizer.learning_rate = step_law_lr(params, tokens)
        cfg.global_train_batch_size = batch.global_batch_size
        cfg.device_train_microbatch_size = batch.microbatch_size
        cfg.device_eval_batch_size = batch.microbatch_size
        steps = (tokens // 2048) // batch.global_batch_size
        if steps <= 2000:
            raise ValueError(f"{name}: {steps} steps do not exceed the paper's 2000-step warmup")
        # Validate the fully resolved trainer schema before creating any output.
        TrainConfig.new(**om.to_container(cfg, resolve=True))
        audit = {"model": name, "input": fmt, "non_embedding_params": params,
                 "training_tokens": tokens, "learning_rate": cfg.optimizer.learning_rate,
                 "batch": asdict(batch), "estimated_full_batch_steps": steps,
                 "train_manifest": str(receipt_path), "train_manifest_sha256": digest(receipt_path),
                 "train_sha256_from_manifest": receipt.get("sha256"),
                 "tokenizer_sha256": tokenizer_sha, "eval_paths": eval_paths,
                 "training_tokenizer_sha256_from_manifest": receipt.get("tokenizer_sha256"),
                 "training_tokenizer_identity_matches": tokenizer_matches,
                 "training_tokenizer_json_equivalent": tokenizer_equivalent,
                 "training_tokenizer_reference": str(reference) if not tokenizer_matches and tokenizer_equivalent else None,
                 "validation": "Array headers, training receipt token counts, tokenizer layout and TrainConfig schema; full array hashes not recomputed",
                 "protocol_source": "camera_ready/paper.tex Appendix: Training",
                 "max_duration": "1ep"}
        prepared.append((name, cfg, audit))
    for name, cfg, audit in prepared:
        run = output / name
        run.mkdir(parents=True)
        config = run / "config.yaml"
        om.save(cfg, config, resolve=True)
        TrainConfig.load(str(config))
        audit["config_sha256"] = digest(config)
        (run / "protocol.json").write_text(json.dumps(audit, indent=2) + "\n")
        launch = run / "launch.sh"
        launch.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                          f"cd {shlex.quote(str(ROOT))}\n"
                          f"exec torchrun --standalone --nnodes=1 --nproc-per-node={args.gpus} "
                          f"scripts/train.py {shlex.quote(str(config))} \"$@\"\n")
        launch.chmod(0o755)
        print(f"{name}: D={audit['training_tokens']:,}, N={audit['non_embedding_params']:,}, "
              f"lr={audit['learning_rate']:.9g}, batch={cfg.global_train_batch_size}, "
              f"microbatch={cfg.device_train_microbatch_size}; {config}")


if __name__ == "__main__":
    main()
