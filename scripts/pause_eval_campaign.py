#!/usr/bin/env python3
"""Prepare, submit, verify, and collect a full Pause-model evaluation campaign.

The campaign contract is intentionally checkpoint-driven.  In particular, the
grammar type and ``pause_token_id`` are inherited from the checkpoint config and
then checked against the tokenizer before any GPU job is submitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

WORKSPACE = Path(
    os.environ.get("PAUSE_WORKSPACE", Path(__file__).resolve().parents[1])
).resolve()
SCRIPT_ROOT = Path(
    os.environ.get("PAUSE_SCRIPT_ROOT", WORKSPACE / "scripts")
).resolve()
sys.path.insert(0, str(WORKSPACE))

from olmo.config import EvaluatorConfig, EvaluatorType, TrainConfig  # noqa: E402


RUNTIME = Path(
    os.environ.get("PAUSE_RUNTIME_BIN", "/home/wangpch/.conda/envs/LLM/bin")
).resolve()
TOKENIZER = WORKSPACE / "dataset/bbc-news/TG_GPT2_tokenizer.json"
SEEDS = (6198, 13171, 31723, 42, 2026)
XSUM_PIPELINE_VERSION = 2
TASKS: dict[str, dict[str, Any]] = {
    "xsum": {
        "epochs": "3ep",
        "lr": 6e-5,
        "eval_batch": 1,
        "train_microbatch": 10,
    },
    "boolq": {
        "epochs": "5ep",
        "lr": 3e-4,
        "eval_batch": 10,
        "train_microbatch": 1,
    },
}
CANONICAL_INPUTS = {
    "tokenizer": (
        TOKENIZER,
        "94a26c15b2edbe45b08c17e20a2f5ad9485a268a7b3ac477a5bf9290ef7f0d46",
    ),
    "terminal_test": (WORKSPACE / "dataset/bbc-news/terminal/test.npy", None),
    "terminal_test_doc_index": (
        WORKSPACE / "dataset/bbc-news/terminal/test_doc_index.npy",
        None,
    ),
    "terminal_test_sent_index": (
        WORKSPACE / "dataset/bbc-news/terminal/test_sent_index.npy",
        None,
    ),
    "xsum_train": (
        WORKSPACE / "dataset/Xsum/xsum_train.txt",
        "15ba739782e8829b2c6d15ccb71898156e02798dc20b7a614d91702213f2c5ad",
    ),
    "xsum_train_summary": (
        WORKSPACE / "dataset/Xsum/xsum_train_summary.txt",
        "b0eb7e73360ce150b93115df044727bdae5bb5c5827e0b6a814856a630e12337",
    ),
    "xsum_gold_train": (
        WORKSPACE / "dataset/Xsum/gold_train_summary.jsonl",
        "89a2779dca51dc95e51473a308d1be664bf24a6e0ba561fc76d1d3f9232e8580",
    ),
    "xsum_save_ids": (
        WORKSPACE / "dataset/Xsum/save_ids.json",
        "5c72e9a790dd252f28686c1fc179f1e77d698f2415e29eea9bbd11a2718220ad",
    ),
    "xsum_test": (
        WORKSPACE / "dataset/Xsum/xsum_test.txt",
        "48ac8f6e4ac2204b47dadfd5f7a91f91959cf2a5e06710d696c2dae043541d57",
    ),
    "boolq_train": (
        WORKSPACE / "dataset/SuperGLUE/BoolQ/train.jsonl",
        "5a0cc1d6cb971a7a177b74bde27b8355de4b0f0e4d86d0a8435ec92cfeb63ba6",
    ),
    "boolq_val": (
        WORKSPACE / "dataset/SuperGLUE/BoolQ/val.jsonl",
        "0c86a5045886e5795fe9052003873f7d94b88ed3028a33007c51d99e44fd66d9",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_special_id(path: Path, token: str) -> int:
    payload = json.loads(path.read_text())
    matches = [row["id"] for row in payload.get("added_tokens", []) if row.get("content") == token]
    if len(matches) != 1:
        raise ValueError(f"expected one tokenizer entry for {token!r}, found {matches}")
    return int(matches[0])


def clean_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    if not label:
        raise ValueError("model label is empty after normalization")
    return label


def verify_inputs(
    checkpoint: Path,
    expected_pause_token_id: int = 50261,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    required = [checkpoint / "config.yaml", checkpoint / "model.pt"]
    missing = [str(path) for path in required if not path.is_file()]
    for path, _ in CANONICAL_INPUTS.values():
        if not path.is_file():
            missing.append(str(path))
    for split in ("train", "val"):
        for field in ("passage", "question"):
            path = WORKSPACE / f"dataset/SuperGLUE/BoolQ/{split}_{field}.txt"
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("missing campaign inputs:\n" + "\n".join(missing))

    cfg = TrainConfig.load(checkpoint / "config.yaml", validate_paths=False)
    grammar = cfg.model.transformer_grammar_type
    if not grammar or not grammar.startswith("pause"):
        raise ValueError(f"checkpoint grammar must be a pause variant, found {grammar!r}")
    if cfg.model.pause_token_id != expected_pause_token_id:
        raise ValueError(
            "checkpoint pause_token_id mismatch: "
            f"expected {expected_pause_token_id}, found {cfg.model.pause_token_id}"
        )
    actual_sep_id = tokenizer_special_id(TOKENIZER, "<|SEP|>")
    if actual_sep_id != expected_pause_token_id:
        raise ValueError(
            f"tokenizer maps <|SEP|> to {actual_sep_id}, expected {expected_pause_token_id}"
        )

    inputs = {}
    for name, (path, expected) in CANONICAL_INPUTS.items():
        actual = sha256_file(path) if expected is not None else None
        if expected is not None and actual != expected:
            raise ValueError(f"sha256 mismatch for {path}: expected {expected}, found {actual}")
        inputs[name] = {
            "path": str(path),
            "sha256": actual,
            "size": path.stat().st_size,
        }
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.yaml"),
        "checkpoint_model_sha256": sha256_file(checkpoint / "model.pt"),
        "grammar": grammar,
        "pause_token": "<|SEP|>",
        "pause_token_id": actual_sep_id,
        "model_max_sequence_length": cfg.model.max_sequence_length,
        "model_vocab_size": cfg.model.vocab_size,
        "inputs": inputs,
    }


def common_overrides(run_name: str, seed: int) -> list[str]:
    return [
        f"run_name={run_name}",
        f"seed={seed}",
        f"workspace={WORKSPACE}",
        f"tokenizer.identifier={TOKENIZER}",
        f"tokenizer.vocabulary={TOKENIZER}",
        f"data.paths=[{WORKSPACE / 'dataset/bbc-news/terminal/train.npy'}]",
        "data.parse_tree_paths=null",
        "data.num_workers=0",
        "data.prefetch_factor=null",
        "data.persistent_workers=false",
        "data.drop_last=false",
        "compile=null",
        "activation_checkpointing=null",
        "distributed_strategy=ddp",
        "save_data_indices=false",
        "wandb=null",
    ]


def build_base_eval_config(
    checkpoint: Path,
    task: str,
    run_name: str,
    output: Path,
) -> None:
    if task == "sg":
        evaluator = EvaluatorConfig(
            label="syntactic_generalization",
            type=EvaluatorType.downstream,
        )
    elif task == "blimp":
        evaluator = EvaluatorConfig(
            label="BLiMP",
            type=EvaluatorType.downstream,
            device_eval_batch_size=100,
        )
    elif task == "docppl":
        evaluator = EvaluatorConfig(
            label="terminal_doc_ppl",
            type=EvaluatorType.terminal_doc,
            device_eval_batch_size=1,
        )
    else:
        raise ValueError(task)
    overrides = common_overrides(run_name, 6198) + [
        "finetune_task=null",
        "max_duration=0ep",
        "stop_at=0",
        "stop_after=null",
        "eval_on_load=true",
        "eval_return=true",
        "eval_no_save=true",
        "reset_optimizer_state=true",
        "reset_trainer_state=true",
        "try_load_latest_save=false",
        "global_train_batch_size=40",
        "device_train_microbatch_size=1",
    ]
    cfg = TrainConfig.load(checkpoint / "config.yaml", overrides, validate_paths=False)
    cfg.load_path = str(checkpoint)
    cfg.save_folder = str(output.parent / "output")
    cfg.remote_save_folder = None
    cfg.evaluators = [evaluator]
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(output)


def build_finetune_config(
    checkpoint: Path,
    task: str,
    seed: int,
    run_name: str,
    output: Path,
    train_microbatch: int,
) -> None:
    settings = TASKS[task]
    overrides = common_overrides(run_name, seed) + [
        f"finetune_task={task}",
        f"optimizer.learning_rate={settings['lr']}",
        "optimizer.weight_decay=0.1",
        "scheduler.t_warmup=100",
        "scheduler.min_lr=1e-6",
        f"max_duration={settings['epochs']}",
        "global_train_batch_size=40",
        f"device_train_microbatch_size={train_microbatch}",
        "device_eval_batch_size=1",
        "precision=fp32",
        "model.flash_attention=false",
        "model.flex_attention=false",
        "eval_interval=1000000",
        "eval_on_load=false",
        "eval_return=false",
        "eval_no_save=false",
        "reset_optimizer_state=true",
        "reset_trainer_state=true",
        "try_load_latest_save=true",
        "stop_at=null",
        "stop_after=null",
        "save_interval=1000000",
        "save_interval_unsharded=1000000",
        "save_num_checkpoints_to_keep=1",
        "save_num_unsharded_checkpoints_to_keep=1",
    ]
    cfg = TrainConfig.load(checkpoint / "config.yaml", overrides, validate_paths=False)
    cfg.load_path = str(checkpoint)
    cfg.save_folder = str(output.parent / "checkpoints")
    cfg.remote_save_folder = None
    cfg.evaluators = []
    output.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(output)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_plan(
    label: str,
    checkpoint: Path,
    grammar: str,
    pause_id: int,
    xsum_train_microbatch: int,
    boolq_train_microbatch: int,
    xsum_eval_batch: int,
    boolq_eval_batch: int,
) -> str:
    return f"""# {label} full evaluation plan

## Selected route

Evaluate the retrained `{grammar}` checkpoint with SEP-backed pause insertion (`pause_token_id={pause_id}`) using the repository's canonical SG, BLiMP, terminal-document-PPL, XSum, and BoolQ protocols. Keep evaluator logic and dataset splits unchanged; parameterize orchestration so the same workflow can be reused for later Pause variants.

## Run contract

- Checkpoint: `{checkpoint}`
- Tier: main/test
- Base metrics: SG `avg`, BLiMP `overall/overall`, terminal doc `eval/downstream/terminal_doc_ppl_doc_ppl`
- XSum: 5 seeds, 3 epochs, LR 6e-5, global batch 40, device train microbatch {xsum_train_microbatch}, eval batch {xsum_eval_batch}; test ROUGE-1/2/L and R-AVG
- BoolQ: 5 seeds, 5 epochs, LR 3e-4, global batch 40, device train microbatch {boolq_train_microbatch}, eval batch {boolq_eval_batch}; zero-shot validation accuracy after finetuning
- Scheduling order: all BoolQ finetunes and evaluations must succeed before the XSum stage starts
- Seeds: {', '.join(map(str, SEEDS))}
- Dataset/split contract: full SG; full BLiMP; BBC terminal test documents; XSum filtered train and full test; BoolQ train and validation
- Stop condition: all 3 base metrics and all 10 downstream seed metrics are finite and durable
- Abandonment condition: repeated checkpoint/data-contract mismatch, non-finite weights, or an evaluator failure that changes the canonical protocol

## Hypotheses and objective

- H0: the corrected checkpoint cannot complete the canonical suite with finite, reproducible outputs across all required seeds.
- H1: the corrected checkpoint completes the canonical suite and yields finite base and downstream metrics for every required seed.
- Objective: establish a trustworthy measurement package for this SEP-based Pause model, not claim superiority over a baseline without a separate statistical comparison.

## Minimal code-change map

- `scripts/pause_eval_campaign.py`: prepare, verify, submit, and collect.
- `scripts/pause_eval/*.sh`: task runners with hash/config gates and durable markers.
- `scripts/slurm/pause_*`: reusable Slurm entry points.
- This campaign directory: frozen configs, manifests, logs, checkpoints, and results.

## Evidence ladder

- Minimum: configs validate, checkpoint health passes, and bounded smoke jobs enter each evaluator.
- Solid: all requested full metrics complete for all five seeds.
- Maximum: downstream mean/SD, per-seed audit, failure analysis, and Pause-2 reuse demonstration.
"""


def render_checklist() -> str:
    return """# Campaign checklist

- [x] Checkpoint config and model file exist
- [x] Checkpoint grammar is a Pause variant
- [x] Checkpoint `pause_token_id` equals tokenizer `<|SEP|>` id 50261
- [x] Canonical dataset hashes and required sidecars validate
- [x] Five XSum and five BoolQ seed configs are frozen
- [x] SG, BLiMP, and terminal-doc-PPL configs are frozen
- [ ] Checkpoint tensor-health report is saved and healthy
- [ ] Bounded smoke execution succeeds
- [ ] Full SG completes with finite `avg`
- [ ] Full BLiMP completes with finite `overall/overall`
- [ ] Full terminal document PPL completes with finite PPL
- [ ] Five XSum finetunes complete
- [ ] Five XSum test evaluations complete with all ROUGE metrics
- [ ] Five BoolQ finetunes complete
- [ ] Five BoolQ validation evaluations complete
- [ ] Collector reports 13/13 canonical metric records
- [ ] Final limitations and next action are recorded
"""


def environment_snapshot() -> dict[str, Any]:
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=WORKSPACE,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    try:
        import flash_attn

        flash_version = flash_attn.__version__
    except Exception as exc:  # pragma: no cover - environment-specific
        flash_version = f"unavailable:{type(exc).__name__}:{exc}"
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "flash_attn": flash_version,
        "git_sha": git_sha,
        "workspace": str(WORKSPACE),
        "runtime": str(RUNTIME),
    }


def prepare(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint.resolve()
    campaign = args.campaign_dir.resolve()
    label = clean_label(args.label or checkpoint.parent.name)
    verified = verify_inputs(checkpoint, args.expected_pause_token_id)
    campaign.mkdir(parents=True, exist_ok=True)
    existing = campaign / "campaign.json"
    if existing.is_file():
        old = json.loads(existing.read_text())
        if old["checkpoint"]["checkpoint"] != str(checkpoint):
            raise ValueError(
                f"campaign already targets {old['checkpoint']['checkpoint']}; refusing to retarget"
            )

    base_rows = []
    for index, (task, gpus) in enumerate((("sg", 2), ("blimp", 2), ("docppl", 1))):
        run_name = f"{label}_{task}"
        run_dir = campaign / "base" / run_name
        config = run_dir / "config.yaml"
        build_base_eval_config(checkpoint, task, run_name, config)
        base_rows.append(
            {
                "index": index,
                "run_name": run_name,
                "task": task,
                "checkpoint": str(checkpoint),
                "config": str(config),
                "run_dir": str(run_dir),
                "gpus": gpus,
            }
        )

    finetune_rows = []
    for task in ("xsum", "boolq"):
        for seed in args.seeds:
            index = len(finetune_rows)
            run_name = f"{label}_{task}_seed{seed}"
            run_dir = campaign / "runs" / run_name
            config = run_dir / "train_config.yaml"
            build_finetune_config(
                checkpoint,
                task,
                seed,
                run_name,
                config,
                args.xsum_train_microbatch
                if task == "xsum"
                else args.boolq_train_microbatch,
            )
            eval_batch = (
                args.xsum_eval_batch if task == "xsum" else args.boolq_eval_batch
            )
            finetune_rows.append(
                {
                    "index": index,
                    "run_name": run_name,
                    "task": task,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "config": str(config),
                    "run_dir": str(run_dir),
                    "save": str(run_dir / "checkpoints"),
                    "eval_batch": eval_batch,
                }
            )

    write_tsv(campaign / "base_runs.tsv", base_rows)
    write_tsv(campaign / "finetune_runs.tsv", finetune_rows)
    resolved_tasks = {
        task: {
            **settings,
            "train_microbatch": (
                args.xsum_train_microbatch
                if task == "xsum"
                else args.boolq_train_microbatch
            ),
            "eval_batch": (
                args.xsum_eval_batch if task == "xsum" else args.boolq_eval_batch
            ),
        }
        for task, settings in TASKS.items()
    }
    manifest = {
        "schema_version": 1,
        "xsum_pipeline_version": XSUM_PIPELINE_VERSION,
        "label": label,
        "checkpoint": verified,
        "seeds": list(args.seeds),
        "tasks": resolved_tasks,
        "train_microbatch": {
            "xsum": args.xsum_train_microbatch,
            "boolq": args.boolq_train_microbatch,
        },
        "eval_batch": {
            "xsum": args.xsum_eval_batch,
            "boolq": args.boolq_eval_batch,
        },
        "base_runs": base_rows,
        "finetune_runs": finetune_rows,
        "environment": environment_snapshot(),
        "commands": {
            "prepare": " ".join(map(str, sys.argv)),
            "verify": f"{sys.executable} {Path(__file__).resolve()} verify --campaign-dir {campaign}",
            "submit": f"{sys.executable} {Path(__file__).resolve()} submit --campaign-dir {campaign}",
            "collect": f"{sys.executable} {Path(__file__).resolve()} collect --campaign-dir {campaign}",
        },
    }
    (campaign / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (campaign / "environment.json").write_text(
        json.dumps(manifest["environment"], indent=2) + "\n"
    )
    (campaign / "PLAN.md").write_text(
        render_plan(
            label,
            checkpoint,
            verified["grammar"],
            verified["pause_token_id"],
            args.xsum_train_microbatch,
            args.boolq_train_microbatch,
            args.xsum_eval_batch,
            args.boolq_eval_batch,
        )
    )
    checklist = campaign / "CHECKLIST.md"
    if not checklist.exists():
        checklist.write_text(render_checklist())
    (campaign / "slurm").mkdir(exist_ok=True)
    print(
        f"prepared {len(base_rows)} base evaluations and {len(finetune_rows)} finetune/eval runs "
        f"in {campaign}"
    )


def load_campaign(path: Path) -> dict[str, Any]:
    campaign = path.resolve()
    manifest_path = campaign / "campaign.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["campaign_dir"] = str(campaign)
    return manifest


def verify_campaign(args: argparse.Namespace) -> None:
    manifest = load_campaign(args.campaign_dir)
    verified = verify_inputs(
        Path(manifest["checkpoint"]["checkpoint"]),
        int(manifest["checkpoint"]["pause_token_id"]),
    )
    for key in ("checkpoint_config_sha256", "checkpoint_model_sha256", "grammar", "pause_token_id"):
        if verified[key] != manifest["checkpoint"][key]:
            raise ValueError(
                f"campaign checkpoint drift for {key}: {manifest['checkpoint'][key]!r} -> {verified[key]!r}"
            )
    for row in manifest["base_runs"] + manifest["finetune_runs"]:
        cfg = TrainConfig.load(row["config"], validate_paths=False)
        if cfg.model.pause_token_id != verified["pause_token_id"]:
            raise ValueError(f"pause_token_id drift in {row['config']}")
        if cfg.model.transformer_grammar_type != verified["grammar"]:
            raise ValueError(f"grammar drift in {row['config']}")
    print(
        f"verified campaign={manifest['campaign_dir']} grammar={verified['grammar']} "
        f"pause_token_id={verified['pause_token_id']} configs="
        f"{len(manifest['base_runs']) + len(manifest['finetune_runs'])}"
    )


def submit_one(command: list[str], dry_run: bool) -> int | None:
    print("SUBMIT", " ".join(command))
    if dry_run:
        return None
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"sbatch failed with exit code {exc.returncode}: "
            f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
        ) from exc
    match = re.search(r"(\d+)", result.stdout)
    if match is None:
        raise RuntimeError(f"could not parse sbatch output: {result.stdout!r}")
    return int(match.group(1))


def scheduler_options(args: argparse.Namespace, gpus: int | None = None) -> list[str]:
    options = []
    if args.partition:
        options.append(f"--partition={args.partition}")
    if args.account:
        options.append(f"--account={args.account}")
    if args.qos:
        options.append(f"--qos={args.qos}")
    if gpus is not None:
        gres = f"gpu:{args.gpu_type}:{gpus}" if args.gpu_type else f"gpu:{gpus}"
        options.append(f"--gres={gres}")
    return options


def job_export(campaign: Path, **extra: str | int) -> str:
    values: dict[str, str | int] = {
        "PAUSE_CAMPAIGN_DIR": campaign,
        "PAUSE_WORKSPACE": WORKSPACE,
        "PAUSE_SCRIPT_ROOT": SCRIPT_ROOT,
        "PAUSE_RUNTIME_BIN": RUNTIME,
        **extra,
    }
    for key, value in values.items():
        if "," in str(value):
            raise ValueError(f"Slurm export value for {key} contains a comma: {value}")
    return "--export=ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def submit(args: argparse.Namespace) -> None:
    verify_campaign(argparse.Namespace(campaign_dir=args.campaign_dir))
    manifest = load_campaign(args.campaign_dir)
    campaign = Path(manifest["campaign_dir"])
    jobs_path = campaign / "jobs.json"
    partial_path = campaign / "jobs.partial.json"
    if jobs_path.exists() and not args.resubmit:
        raise FileExistsError(f"{jobs_path} exists; pass --resubmit to create another attempt")
    if args.resume_partial:
        if not partial_path.is_file():
            raise FileNotFoundError(
                f"{partial_path} does not exist; cannot resume partial submission"
            )
        jobs: dict[str, Any] = json.loads(partial_path.read_text())
    else:
        if partial_path.exists() and not args.resubmit:
            raise FileExistsError(
                f"{partial_path} exists; inspect it and pass --resume-partial"
            )
        jobs = {
            "smoke": None,
            "base": [],
            "finetune": {},
            "eval": {},
            "collect": None,
        }

    def persist_partial() -> None:
        if args.dry_run:
            return
        jobs["last_updated_at"] = datetime.now().astimezone().isoformat()
        partial_path.write_text(json.dumps(jobs, indent=2) + "\n")

    smoke_command = [
        "sbatch",
        "--parsable",
        *scheduler_options(args, gpus=4),
        f"--job-name={manifest['label']}-smoke",
        f"--output={campaign / 'slurm/smoke-%j.out'}",
        job_export(campaign, PAUSE_SMOKE_TRAIN_STEPS=args.smoke_train_steps),
        str(SCRIPT_ROOT / "slurm/pause_campaign_smoke.sbatch"),
    ]
    if jobs.get("smoke") and jobs["smoke"].get("job_id") is not None:
        smoke_id = int(jobs["smoke"]["job_id"])
        print(f"RESUME smoke job {smoke_id}")
    else:
        smoke_id = submit_one(smoke_command, args.dry_run)
        jobs["smoke"] = {"job_id": smoke_id, "command": smoke_command}
        persist_partial()
    base_sbatch = SCRIPT_ROOT / "slurm/pause_base_eval.sbatch"
    existing_base = {row["task"]: row for row in jobs.get("base", [])}
    for row in manifest["base_runs"]:
        output = campaign / "slurm" / f"base-{row['task']}-%j.out"
        command = [
            "sbatch",
            "--parsable",
            *scheduler_options(args, gpus=int(row["gpus"])),
            f"--job-name={manifest['label']}-{row['task']}",
            f"--output={output}",
            job_export(campaign, BASE_INDEX=row["index"]),
        ]
        if smoke_id is not None:
            command.append(f"--dependency=afterok:{smoke_id}")
        command.append(str(base_sbatch))
        if (
            row["task"] in existing_base
            and existing_base[row["task"]].get("job_id") is not None
        ):
            job_id = int(existing_base[row["task"]]["job_id"])
            print(f"RESUME base {row['task']} job {job_id}")
        else:
            job_record = {
                "task": row["task"],
                "job_id": submit_one(command, args.dry_run),
                "command": command,
            }
            jobs["base"].append(job_record)
            existing_base[row["task"]] = job_record
            persist_partial()

    task_indices = {
        task: sorted(
            int(row["index"])
            for row in manifest["finetune_runs"]
            if row["task"] == task
        )
        for task in ("boolq", "xsum")
    }
    for task, indices in task_indices.items():
        if len(indices) != 5 or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"expected five contiguous {task} indices, found {indices}")

    if args.packed_downstream:
        jobs.setdefault("packed", {})
        previous_packed_id = smoke_id
        for task in ("boolq", "xsum"):
            packed_command = [
                "sbatch",
                "--parsable",
                *scheduler_options(args, gpus=args.packed_gpus),
                f"--job-name={manifest['label']}-{task}-packed",
                f"--output={campaign / f'slurm/packed-{task}-%j.out'}",
                job_export(
                    campaign,
                    PAUSE_PACKED_TASK=task,
                    PAUSE_PACKED_PARALLEL=args.packed_parallel,
                    PAUSE_GPUS_PER_RUN=args.gpus_per_run,
                ),
            ]
            if previous_packed_id is not None:
                packed_command.append(f"--dependency=afterok:{previous_packed_id}")
            packed_command.append(
                str(SCRIPT_ROOT / "slurm/pause_downstream_packed.sbatch")
            )
            existing_packed = jobs["packed"].get(task)
            if existing_packed and existing_packed.get("job_id") is not None:
                packed_id = int(existing_packed["job_id"])
                print(f"RESUME packed {task} job {packed_id}")
            else:
                packed_id = submit_one(packed_command, args.dry_run)
                jobs["packed"][task] = {
                    "job_id": packed_id,
                    "command": packed_command,
                }
                persist_partial()
            previous_packed_id = packed_id
    else:
        previous_eval_id = smoke_id
        for task in ("boolq", "xsum"):
            indices = task_indices[task]
            array = f"{indices[0]}-{indices[-1]}%{args.max_parallel}"
            train_command = [
                "sbatch",
                "--parsable",
                *scheduler_options(args, gpus=4),
                f"--array={array}",
                f"--job-name={manifest['label']}-{task}-ft",
                f"--output={campaign / f'slurm/train-{task}-%A_%a.out'}",
                job_export(campaign),
            ]
            if previous_eval_id is not None:
                train_command.append(f"--dependency=afterok:{previous_eval_id}")
            train_command.append(
                str(SCRIPT_ROOT / "slurm/pause_finetune_array.sbatch")
            )
            existing_train = jobs.get("finetune", {}).get(task)
            if existing_train and existing_train.get("job_id") is not None:
                train_id = int(existing_train["job_id"])
                print(f"RESUME finetune {task} job {train_id}")
            else:
                train_id = submit_one(train_command, args.dry_run)
                jobs["finetune"][task] = {
                    "job_id": train_id,
                    "command": train_command,
                }
                persist_partial()

            eval_command = [
                "sbatch",
                "--parsable",
                *scheduler_options(args, gpus=4),
                f"--array={array}",
                f"--job-name={manifest['label']}-{task}-eval",
                f"--output={campaign / f'slurm/eval-{task}-%A_%a.out'}",
                job_export(campaign),
            ]
            if train_id is not None:
                eval_command.append(f"--dependency=aftercorr:{train_id}")
            eval_command.append(
                str(SCRIPT_ROOT / "slurm/pause_finetune_eval_array.sbatch")
            )
            existing_eval = jobs.get("eval", {}).get(task)
            if existing_eval and existing_eval.get("job_id") is not None:
                eval_id = int(existing_eval["job_id"])
                print(f"RESUME eval {task} job {eval_id}")
            else:
                eval_id = submit_one(eval_command, args.dry_run)
                jobs["eval"][task] = {
                    "job_id": eval_id,
                    "command": eval_command,
                }
                persist_partial()
            previous_eval_id = eval_id

    collect_command = [
        "sbatch",
        "--parsable",
        *scheduler_options(args),
        f"--job-name={manifest['label']}-collect",
        f"--output={campaign / 'slurm/collect-%j.out'}",
        job_export(campaign),
    ]
    dependency_ids = [
        row["job_id"] for row in jobs["base"] if row["job_id"] is not None
    ]
    if args.packed_downstream:
        packed_xsum = jobs.get("packed", {}).get("xsum")
        if packed_xsum and packed_xsum.get("job_id") is not None:
            dependency_ids.append(packed_xsum["job_id"])
    else:
        dependency_ids.extend(
            row["job_id"]
            for row in jobs["eval"].values()
            if row["job_id"] is not None
        )
    dependency_ids = list(dict.fromkeys(dependency_ids))
    if dependency_ids:
        collect_command.append(
            "--dependency=afterok:" + ":".join(map(str, dependency_ids))
        )
    collect_command.append(str(SCRIPT_ROOT / "slurm/pause_collect.sbatch"))
    if jobs.get("collect") and jobs["collect"].get("job_id") is not None:
        collect_id = int(jobs["collect"]["job_id"])
        print(f"RESUME collect job {collect_id}")
    else:
        collect_id = submit_one(collect_command, args.dry_run)
        jobs["collect"] = {"job_id": collect_id, "command": collect_command}
        persist_partial()
    jobs["submitted_at"] = datetime.now().astimezone().isoformat()
    jobs["dry_run"] = args.dry_run
    output_path = campaign / ("jobs.dry_run.json" if args.dry_run else "jobs.json")
    output_path.write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"wrote {output_path}")


def last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def parse_base_metric(task: str, text: str) -> dict[str, float | None]:
    if task == "sg":
        return {"sg_avg": finite_or_none(last_float(r"^\s+avg=([0-9.eE+-]+)$", text))}
    if task == "blimp":
        return {
            "blimp_accuracy": finite_or_none(
                last_float(r"^\s+overall/overall=([0-9.eE+-]+)$", text)
            )
        }
    if task == "docppl":
        return {
            "terminal_doc_ppl": finite_or_none(
                last_float(
                    r"^\s+eval/downstream/terminal_doc_ppl_doc_ppl=([0-9.eE+-]+)$",
                    text,
                )
            )
        }
    raise ValueError(task)


def parse_downstream_metric(task: str, text: str) -> dict[str, float | None]:
    if task == "boolq":
        return {
            "boolq_accuracy": finite_or_none(
                last_float(r"eval/downstream/boolq_acc__=([0-9.eE+-]+)", text)
            )
        }
    metrics = {
        key: finite_or_none(last_float(rf"^\s+(?:xsum/)?{pattern}=([0-9.eE+-]+)$", text))
        for key, pattern in (
            ("rouge1", "rouge1"),
            ("rouge2", "rouge2"),
            ("rougeL", "rougeL"),
            ("r_avg", "R-AVG"),
        )
    }
    return metrics


def canonical_value(row: dict[str, Any]) -> float | None:
    metrics = row["metrics"]
    if row["kind"] == "base":
        return next(iter(metrics.values()))
    return metrics.get("r_avg") if row["task"] == "xsum" else metrics.get("boolq_accuracy")


def record_complete(row: dict[str, Any]) -> bool:
    if not row["done"] or canonical_value(row) is None:
        return False
    if row["kind"] == "base":
        return True
    if not row["train_done"]:
        return False
    required = (
        ("rouge1", "rouge2", "rougeL", "r_avg")
        if row["task"] == "xsum"
        else ("boolq_accuracy",)
    )
    return all(row["metrics"].get(key) is not None for key in required)


def render_report(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    base = [row for row in rows if row["kind"] == "base"]
    downstream = [row for row in rows if row["kind"] == "downstream"]
    complete = sum(record_complete(row) for row in rows)
    lines = [
        f"# {manifest['label']} full evaluation results",
        "",
        f"- Checkpoint: `{manifest['checkpoint']['checkpoint']}`",
        f"- Grammar / pause token: `{manifest['checkpoint']['grammar']}` / "
        f"`{manifest['checkpoint']['pause_token']}={manifest['checkpoint']['pause_token_id']}`",
        f"- Canonical metric records: {complete}/{len(rows)}",
        "",
        "## Base evaluations",
        "",
        "| Task | Metric | Done marker | Log |",
        "|---|---:|---:|---|",
    ]
    for row in base:
        value = canonical_value(row)
        lines.append(
            f"| {row['task']} | {'—' if value is None else f'{value:.6f}'} | "
            f"{row['done']} | `{row['log_path']}` |"
        )
    for task, title, key in (
        ("xsum", "XSum test R-AVG", "r_avg"),
        ("boolq", "BoolQ validation accuracy", "boolq_accuracy"),
    ):
        task_rows = sorted(
            [row for row in downstream if row["task"] == task], key=lambda row: row["seed"]
        )
        values = [row["metrics"].get(key) for row in task_rows]
        finite = [value for value in values if value is not None]
        summary = "—"
        if finite:
            sd = statistics.stdev(finite) if len(finite) > 1 else 0.0
            summary = f"{statistics.mean(finite):.6f} ± {sd:.6f}"
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Seed | Metric | Train done | Eval done | Log |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for row in task_rows:
            value = row["metrics"].get(key)
            lines.append(
                f"| {row['seed']} | {'—' if value is None else f'{value:.6f}'} | "
                f"{row['train_done']} | {row['done']} | `{row['log_path']}` |"
            )
        lines.extend(["", f"Mean ± sample SD: **{summary}**"])
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            (
                "All requested canonical metrics are present and finite."
                if complete == len(rows)
                else "Campaign is incomplete; missing metrics remain visible in the tables above."
            ),
            "",
            "## Limitations and next action",
            "",
            f"This package establishes the requested {manifest['checkpoint']['grammar']} "
            "measurements; it does not by itself establish superiority over another model. "
            "Use the same frozen protocol for comparison models and perform a separate "
            "paired/statistical comparison across the five shared seeds.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_checklist(campaign: Path, rows: list[dict[str, Any]]) -> None:
    checklist = campaign / "CHECKLIST.md"
    if not checklist.is_file():
        return
    smoke_done = (campaign / "smoke/SMOKE_DONE").is_file()
    base_by_task = {
        row["task"]: record_complete(row) for row in rows if row["kind"] == "base"
    }
    downstream = [row for row in rows if row["kind"] == "downstream"]
    xsum = [row for row in downstream if row["task"] == "xsum"]
    boolq = [row for row in downstream if row["task"] == "boolq"]
    all_complete = len(rows) == 13 and all(record_complete(row) for row in rows)
    conditions = {
        "Checkpoint tensor-health report is saved and healthy": bool(
            json.loads((campaign / "checkpoint_health.json").read_text()).get("healthy")
        )
        if (campaign / "checkpoint_health.json").is_file()
        else False,
        "Bounded smoke execution succeeds": smoke_done,
        "Full SG completes with finite `avg`": base_by_task.get("sg", False),
        "Full BLiMP completes with finite `overall/overall`": base_by_task.get(
            "blimp", False
        ),
        "Full terminal document PPL completes with finite PPL": base_by_task.get(
            "docppl", False
        ),
        "Five XSum finetunes complete": len(xsum) == 5
        and all(row["train_done"] for row in xsum),
        "Five XSum test evaluations complete with all ROUGE metrics": len(xsum) == 5
        and all(record_complete(row) for row in xsum),
        "Five BoolQ finetunes complete": len(boolq) == 5
        and all(row["train_done"] for row in boolq),
        "Five BoolQ validation evaluations complete": len(boolq) == 5
        and all(record_complete(row) for row in boolq),
        "Collector reports 13/13 canonical metric records": all_complete,
        "Final limitations and next action are recorded": all_complete,
    }
    text = checklist.read_text()
    for label, done in conditions.items():
        state = "x" if done else " "
        text = re.sub(
            rf"^- \[[ x]\] {re.escape(label)}$",
            f"- [{state}] {label}",
            text,
            flags=re.MULTILINE,
        )
    checklist.write_text(text)


def collect(args: argparse.Namespace) -> None:
    manifest = load_campaign(args.campaign_dir)
    campaign = Path(manifest["campaign_dir"])
    rows: list[dict[str, Any]] = []
    for run in manifest["base_runs"]:
        run_dir = Path(run["run_dir"])
        log = run_dir / "eval.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        rows.append(
            {
                "kind": "base",
                "task": run["task"],
                "run_name": run["run_name"],
                "seed": None,
                "done": (run_dir / "EVAL_DONE").is_file(),
                "log_path": str(log),
                "metrics": parse_base_metric(run["task"], text),
            }
        )
    for run in manifest["finetune_runs"]:
        run_dir = Path(run["run_dir"])
        log = run_dir / "eval.log"
        text = log.read_text(errors="replace") if log.is_file() else ""
        rows.append(
            {
                "kind": "downstream",
                "task": run["task"],
                "run_name": run["run_name"],
                "seed": run["seed"],
                "train_done": (run_dir / "TRAIN_DONE").is_file(),
                "done": (run_dir / "EVAL_DONE").is_file(),
                "log_path": str(log),
                "metrics": parse_downstream_metric(run["task"], text),
            }
        )
    (campaign / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    csv_path = campaign / "results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kind",
                "task",
                "run_name",
                "seed",
                "done",
                "train_done",
                "canonical_metric",
                "log_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "kind": row["kind"],
                    "task": row["task"],
                    "run_name": row["run_name"],
                    "seed": row["seed"],
                    "done": row["done"],
                    "train_done": row.get("train_done", ""),
                    "canonical_metric": canonical_value(row),
                    "log_path": row["log_path"],
                }
            )
    (campaign / "REPORT.md").write_text(render_report(manifest, rows))
    refresh_checklist(campaign, rows)
    complete = sum(record_complete(row) for row in rows)
    print(f"collected {complete}/{len(rows)} canonical metric records in {campaign}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--campaign-dir", type=Path, required=True)
    prepare_parser.add_argument("--label")
    prepare_parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    prepare_parser.add_argument("--expected-pause-token-id", type=int, default=50261)
    prepare_parser.add_argument("--xsum-train-microbatch", type=int, default=10)
    prepare_parser.add_argument("--boolq-train-microbatch", type=int, default=1)
    prepare_parser.add_argument("--xsum-eval-batch", type=int, default=1)
    prepare_parser.add_argument("--boolq-eval-batch", type=int, default=10)
    prepare_parser.set_defaults(func=prepare)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--campaign-dir", type=Path, required=True)
    verify_parser.set_defaults(func=verify_campaign)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--campaign-dir", type=Path, required=True)
    submit_parser.add_argument("--max-parallel", type=int, default=2)
    submit_parser.add_argument("--partition")
    submit_parser.add_argument("--account")
    submit_parser.add_argument("--qos")
    submit_parser.add_argument("--gpu-type")
    submit_parser.add_argument("--smoke-train-steps", type=int, default=1)
    submit_parser.add_argument("--packed-downstream", action="store_true")
    submit_parser.add_argument("--packed-parallel", type=int, default=2)
    submit_parser.add_argument("--packed-gpus", type=int, default=8)
    submit_parser.add_argument("--gpus-per-run", type=int, default=4)
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument("--resubmit", action="store_true")
    submit_parser.add_argument("--resume-partial", action="store_true")
    submit_parser.set_defaults(func=submit)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--campaign-dir", type=Path, required=True)
    collect_parser.set_defaults(func=collect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in (
        "xsum_train_microbatch",
        "boolq_train_microbatch",
        "xsum_eval_batch",
        "boolq_eval_batch",
    ):
        if getattr(args, name, 1) <= 0:
            raise ValueError(f"{name.replace('_', ' ')} must be positive")
    if getattr(args, "max_parallel", 1) <= 0:
        raise ValueError("max parallel must be positive")
    if getattr(args, "smoke_train_steps", 1) <= 0:
        raise ValueError("smoke train steps must be positive")
    if getattr(args, "packed_parallel", 1) <= 0:
        raise ValueError("packed parallel must be positive")
    if getattr(args, "packed_gpus", 1) <= 0 or getattr(args, "gpus_per_run", 1) <= 0:
        raise ValueError("packed GPUs and GPUs per run must be positive")
    if (
        getattr(args, "packed_downstream", False)
        and args.packed_parallel * args.gpus_per_run > args.packed_gpus
    ):
        raise ValueError("packed parallel * GPUs per run exceeds packed GPU allocation")
    if hasattr(args, "seeds") and len(args.seeds) != 5:
        raise ValueError("the canonical campaign requires exactly five seeds")
    args.func(args)


if __name__ == "__main__":
    main()
