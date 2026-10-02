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
import math
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.step_law import count_non_embedding_params, step_law_lr
from scripts.submit_pause_sep_pretrain import make_batch_plan

FORMATS = {"terminal": "terminal", "tree": "tree", "tgtree": "tg", "tgnomask_aug": "tg",
           "tg": "tg", "tgnomask": "tg", "tgnomask_mix_tg": "tg", "tree_noont": "tree_noont"}
DEFAULT_MODELS = ("terminal", "tree", "tgtree", "tgnomask_aug")
TYPED_MODELS = {"tg", "tgnomask", "tgnomask_mix_tg"}



def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stream_size(path: Path) -> int:
    import numpy as np

    stream = np.load(path, mmap_mode="r", allow_pickle=False)
    if stream.ndim != 1 or stream.dtype != np.dtype("uint16") or stream.size < 2048:
        raise ValueError(f"Expected a 1D uint16 token stream with >=2048 tokens: {path}")
    if path.stat().st_size != stream.offset + stream.nbytes:
        raise ValueError(f"NPY payload size mismatch: {path}")
    return int(stream.size)


def tokenizer_identity(receipt, tokenizer, reference):
    expected = receipt.get("tokenizer_sha256")
    same = expected == digest(tokenizer)
    equivalent = same or (reference.is_file() and digest(reference) == expected
                         and json.loads(reference.read_text()) == json.loads(tokenizer.read_text()))
    if not equivalent:
        raise ValueError("Tokenizer identity not certified; supply --tokenizer-reference matching the manifest")
    return same, equivalent


def eval_identity(path, fmt, split, tokenizer, reference):
    """Accept either per-split receipts or the published clean base-array manifest."""
    import numpy as np

    tokens = stream_size(path)
    receipt_path = path.with_suffix(".manifest.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "complete" or receipt.get("tokens") != tokens
                or receipt.get("dtype") != "uint16" or receipt.get("format") != fmt):
            raise ValueError(f"Evaluation manifest disagrees with array: {receipt_path}")
        expected = receipt.get("sha256")
        documents = receipt.get("documents")
        offsets_hash = receipt.get("doc_offsets_sha256")
    else:
        receipt_path = path.parent.parent / "manifest.json"
        receipt = json.loads(receipt_path.read_text())
        if receipt["outputs"][split]["tokens"][fmt] != tokens:
            raise ValueError(f"Evaluation manifest disagrees with array: {path}")
        expected = receipt["files"][f"{fmt}/{split}.npy"]
        documents = receipt["outputs"][split]["documents"]
        offsets_hash = receipt["files"][f"{fmt}/{split}_doc_offsets.npy"]
    tokenizer_identity(receipt, tokenizer, reference)
    if documents != {"dev": 1000, "test": 5000}[split] or digest(path) != expected:
        raise ValueError(f"Clean evaluation document count or hash mismatch: {path}")
    offsets_path = path.with_name(f"{split}_doc_offsets.npy")
    offsets = np.load(offsets_path, allow_pickle=False)
    if (offsets.ndim != 1 or offsets.dtype != np.dtype("uint64")
            or len(offsets) != documents + 1 or offsets[0] != 0 or offsets[-1] != tokens
            or not np.all(offsets[1:] > offsets[:-1]) or digest(offsets_path) != offsets_hash):
        raise ValueError(f"Invalid document offsets: {offsets_path}")
    return {"split": split, "format": fmt, "source_path": str(path), "tokens": tokens,
            "documents": documents, "sha256": expected, "doc_offsets_sha256": offsets_hash,
            "manifest": str(receipt_path), "manifest_sha256": digest(receipt_path)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=FORMATS, default=list(DEFAULT_MODELS))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset/bbc-news-parsed-dedup")
    parser.add_argument("--eval-dir", type=Path, default=ROOT / "dataset/bbc-news-reserved-clean-v1")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/experiment/bbc_dedup_500m")
    parser.add_argument("--tokenizer-reference", type=Path,
                        default=ROOT / "reproducibility/tokenizer-reference.json")
    parser.add_argument("--noont-eval-dir", type=Path,
                        help="Root containing tree_noont/{dev,test}.npy; defaults to --data-dir")
    parser.add_argument("--runtime-root", type=Path,
                        help="Target repository root; remap repository-relative input/output paths")
    parser.add_argument("--campaign", help="Unique run/group identity; defaults to output directory name")
    parser.add_argument("--hardware", default="unspecified", help="Hardware label for run identity")
    parser.add_argument("--wandb-entity", default="qlwpc-shanghaitech-university")
    parser.add_argument("--smoke-timeout", type=int, default=1800, help="Seconds for the short DDP/eval smoke")
    parser.add_argument("--verify-train-hash", action="store_true")
    parser.add_argument("--fixed-microbatch", type=int, help="Fixed candidate, including non-divisors; still requires GPU smoke")
    parser.add_argument("--gpus", type=int, default=4, help="Single-node world size")
    parser.add_argument("--max-microbatch", type=int, default=1)
    args = parser.parse_args(argv)
    if args.gpus <= 0 or args.max_microbatch <= 0:
        parser.error("--gpus and --max-microbatch must be positive")
    if args.smoke_timeout <= 0:
        parser.error("--smoke-timeout must be positive")
    if args.fixed_microbatch is not None and args.fixed_microbatch <= 0:
        parser.error("--fixed-microbatch must be positive")
    if args.runtime_root is not None and not args.runtime_root.is_absolute():
        parser.error("--runtime-root must be absolute")

    def runtime(path):
        # Keep logical paths (including symlinks) when transferring a unified data tree.
        path = path.absolute()
        return str(args.runtime_root / path.relative_to(ROOT)) if args.runtime_root else str(path)

    from omegaconf import OmegaConf as om
    from olmo.config import TrainConfig
    from tokenizers import Tokenizer
    from olmo.data.collator import use_typed_tg

    tokenizer = args.tokenizer.absolute()
    tok = Tokenizer.from_file(str(tokenizer))
    if (tok.get_vocab_size() != 50320 or tok.token_to_id("<|SEP|>") != 50261
            or tok.token_to_id("<|endoftext|>") != 50256 or tok.token_to_id("<|pad|>") != 50258):
        raise ValueError("Expected the paper BBC GPT-2 tokenizer (50320 tokens, SEP=50261)")
    tokenizer_sha = digest(tokenizer)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh --output-dir: {output}")
    campaign = args.campaign or output.name
    reference = args.tokenizer_reference.resolve()
    prepared = []
    eval_cache = {}
    for name in dict.fromkeys(args.models):
        fmt = FORMATS[name]
        train = args.data_dir.absolute() / fmt / "train.npy"
        tokens = stream_size(train)
        receipt_path = train.with_suffix(".manifest.json")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("status") != "complete" or receipt.get("tokens") != tokens
                or receipt.get("dtype") != "uint16" or receipt.get("format") != fmt
                or not isinstance(receipt.get("sha256"), str) or len(receipt["sha256"]) != 64):
            raise ValueError(f"Training manifest disagrees with array: {receipt_path}")
        tokenizer_matches, tokenizer_equivalent = tokenizer_identity(receipt, tokenizer, reference)
        if args.verify_train_hash and digest(train) != receipt.get("sha256"):
            raise ValueError(f"Training array hash mismatch: {train}")
        cfg = om.load(ROOT / "train_configs/templates/tree-500M.yaml")
        cfg.run_name = f"bbc-dedup-500m-{name}-{args.hardware}-{args.gpus}gpu-{campaign}"
        cfg.model.transformer_grammar_type = "mixing" if name == "tgnomask_mix_tg" else name
        typed = name in TYPED_MODELS
        structural = name == "tgnomask_aug"
        cfg.model.tg_typed_attention = typed
        cfg.model.mix_head_type = ([{"grammar_type": "tg", "n_heads": 8},
                                    {"grammar_type": "tgnomask", "n_heads": 8}]
                                   if name == "tgnomask_mix_tg" else [])
        cfg.model.flash_attention = not (structural or typed)
        cfg.model.flex_attention = structural
        cfg.wandb = dict(entity=args.wandb_entity, project="bbc-news-dedup", mode="offline",
                         name=cfg.run_name, group=campaign, rank_zero_only=True, log_interval=1,
                         log_artifacts=False, tags=["500m", "dedup", args.hardware, f"{args.gpus}gpu", name])
        cfg.update(eval_interval=3000, eval_subset_num_batches=-1, save_interval=None,
                   save_num_checkpoints_to_keep=0, save_interval_unsharded=3000,
                   save_num_unsharded_checkpoints_to_keep=1, no_pre_train_checkpoint=True,
                   max_duration="1ep", stop_at=None, stop_after=None, time_limit=158400,
                   save_data_indices=True)
        cfg.load_path = None
        cfg.save_overwrite = False
        cfg.save_folder = runtime(output / name / "checkpoints")
        cfg.tokenizer.identifier = runtime(tokenizer)
        cfg.tokenizer.vocabulary = runtime(tokenizer)
        cfg.data.paths = [runtime(train)]
        cfg.data.memmap_dtype = "uint16"
        cfg.data.memmap_format = "npy"
        cfg.data.generate_attention_mask = structural
        cfg.data.generate_doc_lengths = not typed
        cfg.data.update(num_workers=1 if typed else 4, prefetch_factor=2,
                        tg_layout_backend="native", cuda_prefetch=False)
        cfg.evaluators = []
        eval_paths = []
        eval_records = []
        for split in ("dev", "test"):
            eval_root = (args.noont_eval_dir or args.data_dir) if fmt == "tree_noont" else args.eval_dir
            path = eval_root.absolute() / fmt / f"{split}.npy"
            if path not in eval_cache:
                eval_cache[path] = eval_identity(path, fmt, split, tokenizer, reference)
            eval_records.append(dict(eval_cache[path], path=runtime(path)))
            eval_paths.append(runtime(path))
            cfg.evaluators.append({"label": f"clean-{split}", "type": "lm", "data": {
                "paths": [runtime(path)], "memmap_dtype": "uint16", "memmap_format": "npy",
                "generate_attention_mask": structural, "generate_doc_lengths": not typed,
                "tg_layout_backend": "native", "cuda_prefetch": False,
                "num_workers": 0, "drop_last": False, "persistent_workers": False}})
        model_cfg = TrainConfig.new(**om.to_container(cfg, resolve=True)).model
        params = count_non_embedding_params(model_cfg)
        batch = make_batch_plan(tokens, 2048, args.gpus, args.max_microbatch)
        cfg.optimizer.learning_rate = step_law_lr(params, tokens)
        cfg.global_train_batch_size = batch.global_batch_size
        microbatch = args.fixed_microbatch or batch.microbatch_size
        if microbatch > batch.per_device_batch_size:
            raise ValueError("Microbatch exceeds local batch; choose a smaller candidate")
        cfg.device_train_microbatch_size = microbatch
        cfg.device_eval_batch_size = microbatch
        steps = (tokens // 2048) // batch.global_batch_size
        if steps <= 2000:
            raise ValueError(f"{name}: {steps} steps do not exceed the paper's 2000-step warmup")
        # Validate the fully resolved trainer schema before creating any output.
        resolved = TrainConfig.new(**om.to_container(cfg, resolve=True))
        for data in [resolved.data, *[e.data for e in resolved.evaluators]]:
            if use_typed_tg(resolved, data) != typed:
                raise ValueError(f"Unexpected attention route for {name}")
        batch_record = asdict(batch)
        batch_record.update(microbatch_size=microbatch,
                            gradient_accumulation_steps=math.ceil(batch.per_device_batch_size / microbatch),
                            last_microbatch=((batch.per_device_batch_size - 1) % microbatch) + 1)
        audit = {"model": name, "input": fmt, "non_embedding_params": params,
                 "training_tokens": tokens, "learning_rate": cfg.optimizer.learning_rate,
                 "batch": batch_record, "estimated_full_batch_steps": steps,
                 "train_manifest": str(receipt_path), "train_manifest_sha256": digest(receipt_path),
                 "train_sha256_from_manifest": receipt.get("sha256"),
                 "tokenizer_sha256": tokenizer_sha, "eval_paths": eval_paths,
                 "eval_data_identity": eval_records,
                 "grammar": cfg.model.transformer_grammar_type,
                 "typed_attention": typed, "mix_heads": om.to_container(cfg.model.mix_head_type),
                 "runtime_root": str(args.runtime_root or ROOT),
                 "microbatch_policy": "fixed" if args.fixed_microbatch else "search_max",
                 "microbatch_status": "pending_gpu_probe", "max_microbatch": args.max_microbatch,
                 "smoke": {"timeout_seconds": args.smoke_timeout, "optimizer_steps": 3,
                           "eval_batches_per_split": 2, "world_size": args.gpus},
                 "gpu_validation": "pending: real world-size DDP training and short dev/test evaluation",
                 "train_hash_recomputed": args.verify_train_hash,
                 "train_path": runtime(train), "train_source_path": str(train),
                 "engineering": {"eval_interval": 3000, "eval_subset_num_batches": -1,
                                 "wandb_mode": "offline", "time_limit": 158400,
                                 "train_workers_per_rank": cfg.data.num_workers,
                                 "distributed_strategy": cfg.distributed_strategy,
                                 "compile": om.to_container(cfg.compile)},
                 "template_sha256": digest(ROOT / "train_configs/templates/tree-500M.yaml"),
                 "generator_sha256": digest(Path(__file__)),
                 "training_tokenizer_sha256_from_manifest": receipt.get("tokenizer_sha256"),
                 "training_tokenizer_identity_matches": tokenizer_matches,
                 "training_tokenizer_json_equivalent": tokenizer_equivalent,
                 "training_tokenizer_reference": str(reference) if not tokenizer_matches and tokenizer_equivalent else None,
                 "validation": "NPY payload/header, train receipt, tokenizer identity, full eval hashes/offsets, TrainConfig schema and train/eval typed route; GPU and remote paths not validated",
                 "protocol_source": "camera_ready/paper.tex Appendix: Training",
                 "max_duration": "1ep"}
        prepared.append((name, cfg, audit))
    for name, cfg, audit in prepared:
        run = output / name
        run.mkdir(parents=True)
        config = run / "config.yaml"
        om.save(cfg, config, resolve=True)
        om.save(cfg, run / "base.yaml", resolve=True)
        smoke = om.create(om.to_container(cfg, resolve=True))
        smoke.update(wandb=None, stop_at=3, eval_interval=3, eval_subset_num_batches=2,
                     save_interval_unsharded=None, eval_no_save=True, save_data_indices=False,
                     time_limit=None, save_folder=runtime(run / "smoke-checkpoints"))
        om.save(smoke, run / "smoke.yaml", resolve=True)
        TrainConfig.load(str(config))
        audit["config_sha256"] = digest(config)
        audit["base_config_sha256"] = digest(run / "base.yaml")
        audit["smoke_config_sha256"] = digest(run / "smoke.yaml")
        (run / "protocol.json").write_text(json.dumps(audit, indent=2) + "\n")
        for script_name, config_name in (("launch.sh", "config.yaml"), ("smoke.sh", "smoke.yaml")):
            launch = run / script_name
            timeout = f"timeout --signal=TERM --kill-after=30s {args.smoke_timeout}s " if script_name == "smoke.sh" else ""
            launch.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                              f"cd {shlex.quote(str(args.runtime_root or ROOT))}\n"
                              "export WANDB_MODE=offline\n"
                              "export TMPDIR=$(mktemp -d /tmp/tg.XXXXXXXX)\n"
                              "trap 'rm -rf -- \"$TMPDIR\"' EXIT\n"
                              f"{timeout}torchrun --standalone --nnodes=1 --nproc-per-node={args.gpus} "
                              f"scripts/train.py {shlex.quote(runtime(run / config_name))} \"$@\"\n")
            launch.chmod(0o755)
        print(f"{name}: D={audit['training_tokens']:,}, N={audit['non_embedding_params']:,}, "
              f"lr={audit['learning_rate']:.9g}, batch={cfg.global_train_batch_size}, "
              f"microbatch={cfg.device_train_microbatch_size}; {config}")


if __name__ == "__main__":
    main()
