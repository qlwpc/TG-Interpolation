#!/usr/bin/env python3
"""Generate and submit reproducible 100M Pause-1/Pause-2 SEP pretraining.

The generated runs use the BBC terminal stream and insert ``<|SEP|>`` online.
Peak learning rate and global batch size are derived from paper Eq. (8):

    eta(N, D) = 1.79 * N^-0.713 * D^0.307
    B(D)       = 0.58 * D^0.571

``D`` includes online pause positions. ``B(D)`` is converted from tokens to
sequences, then rounded to the closest multiple of the requested GPU count.
The per-device microbatch is chosen as the largest divisor of the per-device
batch not exceeding ``--max-microbatch``. This is important because the train
entrypoint derives gradient accumulation with integer division.

Dry-run (default):
    python scripts/submit_pause_sep_pretrain.py

Submit Pause-1 first and queue Pause-2 after it succeeds:
    python scripts/submit_pause_sep_pretrain.py --submit --account ACCOUNT

The command is idempotent with respect to the campaign's ``jobs.json``. Use
``--resubmit`` only when a new attempt is intentionally required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.step_law import (  # noqa: E402
    count_non_embedding_params,
    step_law_batch_tokens,
    step_law_lr,
)


@dataclass(frozen=True)
class PauseVariant:
    name: str
    pause_count: int
    sequence_length: int


@dataclass(frozen=True)
class BatchPlan:
    raw_batch_tokens: float
    raw_batch_sequences: float
    global_batch_size: int
    gpu_count: int
    per_device_batch_size: int
    microbatch_size: int
    gradient_accumulation_steps: int


VARIANTS = {
    "pause1": PauseVariant("pause1", pause_count=1, sequence_length=2048),
    # The repository's published Pause-2 protocol uses 683 real positions,
    # expanded to 683 * 3 = 2049 positions. Preserve that comparison contract.
    "pause2": PauseVariant("pause2", pause_count=2, sequence_length=2049),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_token_stream(path: Path, itemsize: int = 2) -> int:
    """Count a 1-D NumPy or headerless raw token stream without loading it."""
    if itemsize <= 0:
        raise ValueError(f"itemsize must be positive, got {itemsize}")

    from olmo.memmap_utils import inspect_memmap_file

    dtype = f"uint{itemsize * 8}"
    return inspect_memmap_file(path, dtype, "auto").element_count


def round_to_multiple(value: float, multiple: int) -> int:
    """Round positive ``value`` to the nearest positive integer multiple."""
    if value <= 0 or multiple <= 0:
        raise ValueError(f"value and multiple must be positive, got {value}, {multiple}")
    # Explicit half-up rounding avoids Python's banker-rounding ambiguity.
    return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)


def choose_microbatch(per_device_batch: int, max_microbatch: int) -> int:
    """Choose the largest exact divisor under a memory-oriented upper bound."""
    if per_device_batch <= 0 or max_microbatch <= 0:
        raise ValueError("per-device batch and max microbatch must be positive")
    candidates = [
        candidate
        for candidate in range(1, min(per_device_batch, max_microbatch) + 1)
        if per_device_batch % candidate == 0
    ]
    return max(candidates)


def make_batch_plan(
    dataset_tokens: int,
    sequence_length: int,
    gpu_count: int,
    max_microbatch: int,
    microbatch_override: int | None = None,
) -> BatchPlan:
    raw_tokens = step_law_batch_tokens(dataset_tokens)
    raw_sequences = raw_tokens / sequence_length
    global_batch = round_to_multiple(raw_sequences, gpu_count)
    per_device = global_batch // gpu_count
    microbatch = (
        microbatch_override
        if microbatch_override is not None
        else choose_microbatch(per_device, max_microbatch)
    )
    if microbatch <= 0 or per_device % microbatch:
        raise ValueError(
            f"microbatch={microbatch} must divide per-device batch={per_device} exactly"
        )
    return BatchPlan(
        raw_batch_tokens=raw_tokens,
        raw_batch_sequences=raw_sequences,
        global_batch_size=global_batch,
        gpu_count=gpu_count,
        per_device_batch_size=per_device,
        microbatch_size=microbatch,
        gradient_accumulation_steps=per_device // microbatch,
    )


def run_checked(command: Sequence[str], *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        rendered = shlex.join(str(item) for item in command)
        raise RuntimeError(
            f"command failed ({exc.returncode}): {rendered}\n"
            f"stdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc
    return result.stdout.strip()


def remote_bash(host: str, command: str) -> str:
    return run_checked(
        ["ssh", "-o", "BatchMode=yes", host, f"bash -lc {shlex.quote(command)}"]
    )


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def tokenizer_special_id(tokenizer_path: Path, token: str) -> int:
    raw = json.loads(tokenizer_path.read_text())
    matches = [
        int(item["id"])
        for item in raw.get("added_tokens", [])
        if item.get("content") == token
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {token!r} in {tokenizer_path}, got {matches}")
    return matches[0]


def derive_non_embedding_params(base_config: Path) -> int:
    from olmo.config import TrainConfig

    cfg = TrainConfig.load(base_config, validate_paths=False)
    return count_non_embedding_params(cfg.model)


def generate_config(
    *,
    base_config: Path,
    output: Path,
    variant: PauseVariant,
    run_name: str,
    remote_workspace: str,
    pause_token_id: int,
    learning_rate: float,
    batch_plan: BatchPlan,
) -> None:
    from omegaconf import OmegaConf as om
    from olmo.config import TrainConfig

    cfg = om.load(base_config)

    def put(key: str, value: Any) -> None:
        om.update(cfg, key, value, merge=False, force_add=True)

    terminal_root = f"{remote_workspace}/dataset/bbc-news/terminal"
    tokenizer = f"{remote_workspace}/dataset/bbc-news/TG_GPT2_tokenizer.json"
    save_folder = f"{remote_workspace}/saved_models/{run_name}"

    put("run_name", run_name)
    put("workspace", remote_workspace)
    put("seed", 6198)
    put("model.transformer_grammar_type", variant.name)
    put("model.pause_token_id", pause_token_id)
    put("model.max_sequence_length", variant.sequence_length)
    put("model.flash_attention", True)
    # Causal Pause models must take the FlashAttention-2 path, not construct a
    # structural FlexAttention block mask.
    put("model.flex_attention", False)
    put("model.attention_dropout", 0.0)
    put("optimizer.learning_rate", learning_rate)
    put("scheduler.t_warmup", 2000)
    put("scheduler.min_lr", 1.0e-5)
    put("scheduler.t_constant", 0)
    put("global_train_batch_size", batch_plan.global_batch_size)
    put("device_train_batch_size", batch_plan.per_device_batch_size)
    put("device_train_microbatch_size", batch_plan.microbatch_size)
    put("device_train_grad_accum", batch_plan.gradient_accumulation_steps)
    put("device_eval_batch_size", batch_plan.microbatch_size)
    put("precision", "amp_bf16")
    put("compile", None)
    put("max_duration", "1ep")
    put("stop_at", 2_000_000_000)
    put("data.paths", [f"{terminal_root}/train.npy"])
    put("data.memmap_dtype", "uint16")
    put("data.generate_attention_mask", False)
    put("data.num_workers", 4)
    put("data.prefetch_factor", 8)
    put("data.persistent_workers", True)
    put("tokenizer.identifier", tokenizer)
    put("tokenizer.vocabulary", tokenizer)
    put("save_folder", save_folder)
    put("remote_save_folder", None)
    put("load_path", None)
    put("try_load_latest_save", False)
    put("restore_dataloader", True)
    put("save_overwrite", False)
    put("save_interval_unsharded", 5000)
    put("save_num_unsharded_checkpoints_to_keep", 2)
    put("save_data_indices", True)
    put("eval_interval", 5000)
    put("eval_subset_num_batches", -1)
    put("wandb.name", run_name)
    put("wandb.group", "pause-sep-100m-step-law")
    put("wandb.mode", "offline")

    evaluator_paths = [f"{terminal_root}/dev.npy", f"{terminal_root}/test.npy"]
    for index, eval_path in enumerate(evaluator_paths):
        put(f"evaluators.{index}.data.paths", [eval_path])
        put(f"evaluators.{index}.data.memmap_dtype", "uint16")
        put(f"evaluators.{index}.data.generate_doc_lengths", True)

    output.parent.mkdir(parents=True, exist_ok=True)
    om.save(cfg, output)
    resolved = TrainConfig.load(output, validate_paths=False)
    if resolved.model.pause_token_id != pause_token_id:
        raise AssertionError("generated pause_token_id did not survive config loading")
    if not resolved.model.flash_attention or resolved.model.flex_attention:
        raise AssertionError("generated config does not force causal FlashAttention")
    if resolved.global_train_batch_size % batch_plan.gpu_count:
        raise AssertionError("global batch is not divisible by GPU count")
    if batch_plan.per_device_batch_size % resolved.device_train_microbatch_size:
        raise AssertionError("microbatch does not divide per-device batch")


def generate_sbatch(
    *,
    output: Path,
    config_path_remote: str,
    log_path_remote: str,
    run_name: str,
    remote_workspace: str,
    runtime_bin: str,
    pause_token_id: int,
    account: str | None,
    partition: str,
    gpu_count: int,
    walltime: str,
) -> None:
    account_directive = f"#SBATCH --account={account}\n" if account else ""
    text = f"""#!/usr/bin/env bash
#SBATCH --job-name={run_name}
#SBATCH --partition={partition}
{account_directive.rstrip()}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={gpu_count * 8}
#SBATCH --mem=400G
#SBATCH --gres=gpu:{gpu_count}
#SBATCH --time={walltime}
#SBATCH --output={log_path_remote}
#SBATCH --open-mode=append

set -euo pipefail

workspace={shlex.quote(remote_workspace)}
export PATH={shlex.quote(runtime_bin)}:$PATH
export PYTHONPATH="$workspace${{PYTHONPATH:+:$PYTHONPATH}}"
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$workspace"
config_path={shlex.quote(config_path_remote)}
echo "RUN_NAME={run_name}"
echo "HOST=$(hostname) JOB_ID=${{SLURM_JOB_ID}} GPUS=${{SLURM_GPUS_ON_NODE:-unknown}}"
nvidia-smi

# Hard gate: a missing/incompatible extension must fail the job rather than
# silently falling back to PyTorch SDPA. This runs on the allocated compute node.
python -c 'import flash_attn; from flash_attn import flash_attn_func; assert flash_attn_func is not None; print("FLASH_ATTN_OK", flash_attn.__version__)'
TRAIN_CONFIG="$config_path" EXPECTED_PAUSE_ID={pause_token_id} python -c 'import os; from olmo.config import TrainConfig; c=TrainConfig.load(os.environ["TRAIN_CONFIG"], validate_paths=True); assert c.model.flash_attention and not c.model.flex_attention; assert c.model.pause_token_id == int(os.environ["EXPECTED_PAUSE_ID"]); print("CONFIG_OK", c.run_name, c.global_train_batch_size, c.device_train_microbatch_size)'

exec torchrun --standalone --nnodes=1 --nproc-per-node={gpu_count} \\
  scripts/train.py "$config_path"
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    output.chmod(0o755)


def parse_job_id(output: str) -> int:
    match = re.match(r"\s*(\d+)", output)
    if match is None:
        raise RuntimeError(f"could not parse Slurm job id from {output!r}")
    return int(match.group(1))


def sync_campaign_files(
    host: str,
    remote_campaign: str,
    files: Iterable[tuple[Path, str]],
) -> None:
    remote_bash(host, f"mkdir -p {shlex.quote(remote_campaign)}")
    for local_path, relative_remote in files:
        remote_target = f"{host}:{remote_campaign}/{relative_remote}"
        remote_bash(
            host,
            f"mkdir -p {shlex.quote(str(Path(remote_campaign) / Path(relative_remote).parent))}",
        )
        run_checked(["rsync", "-a", str(local_path), remote_target])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "train_configs/terminal.yaml")
    parser.add_argument("--terminal-data", type=Path, default=REPO_ROOT / "dataset/bbc-news/terminal/train.npy")
    parser.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json")
    parser.add_argument("--campaign-dir", type=Path, default=REPO_ROOT / "artifacts/experiment/pause_sep_100m_sist_20260828")
    parser.add_argument("--remote-host", default="SIST")
    parser.add_argument("--remote-workspace", default="/public/home/wangpch/TG-Interpolation")
    parser.add_argument("--runtime-bin", default="/public/home/wangpch/venvs/LLM-sm120/bin")
    parser.add_argument("--partition", default="ShangHAI")
    parser.add_argument(
        "--account",
        help="Slurm account authorized for the selected partition (cluster-specific).",
    )
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--pause-token-id", type=int, default=50261)
    parser.add_argument("--max-microbatch", type=int, default=17)
    parser.add_argument("--pause1-microbatch", type=int)
    parser.add_argument("--pause2-microbatch", type=int)
    parser.add_argument("--walltime", default="3-00:00:00")
    parser.add_argument("--first", choices=sorted(VARIANTS), default="pause1")
    parser.add_argument("--dependency-mode", choices=["afterok", "afterany"], default="afterok")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resubmit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gpus <= 0:
        raise ValueError("--gpus must be positive")
    campaign = args.campaign_dir.resolve()
    campaign.mkdir(parents=True, exist_ok=True)

    actual_pause_id = tokenizer_special_id(args.tokenizer.resolve(), "<|SEP|>")
    if actual_pause_id != args.pause_token_id:
        raise ValueError(
            f"requested pause id {args.pause_token_id}, tokenizer maps <|SEP|> to {actual_pause_id}"
        )
    terminal_tokens = count_token_stream(args.terminal_data.resolve(), itemsize=2)
    non_embedding_params = derive_non_embedding_params(args.base_config.resolve())

    remote_campaign = f"{args.remote_workspace}/artifacts/experiment/{campaign.name}"
    plans: dict[str, dict[str, Any]] = {}
    synced: list[tuple[Path, str]] = []
    overrides = {
        "pause1": args.pause1_microbatch,
        "pause2": args.pause2_microbatch,
    }
    for key, variant in VARIANTS.items():
        expanded_tokens = terminal_tokens * (variant.pause_count + 1)
        lr_raw = step_law_lr(non_embedding_params, expanded_tokens)
        lr_config = round(lr_raw, 6)
        batch = make_batch_plan(
            expanded_tokens,
            variant.sequence_length,
            args.gpus,
            args.max_microbatch,
            overrides[key],
        )
        run_name = f"pretrain_{key}_100M_SEP50261_steplaw"
        run_dir = campaign / "runs" / run_name
        config = run_dir / "config.yaml"
        sbatch = run_dir / "submit.sbatch"
        relative_dir = f"runs/{run_name}"
        remote_run_dir = f"{remote_campaign}/{relative_dir}"
        generate_config(
            base_config=args.base_config.resolve(),
            output=config,
            variant=variant,
            run_name=run_name,
            remote_workspace=args.remote_workspace,
            pause_token_id=args.pause_token_id,
            learning_rate=lr_config,
            batch_plan=batch,
        )
        generate_sbatch(
            output=sbatch,
            config_path_remote=f"{remote_run_dir}/config.yaml",
            log_path_remote=f"{remote_run_dir}/logs/slurm-%x-%j.out",
            run_name=run_name,
            remote_workspace=args.remote_workspace,
            runtime_bin=args.runtime_bin,
            pause_token_id=args.pause_token_id,
            account=args.account,
            partition=args.partition,
            gpu_count=args.gpus,
            walltime=args.walltime,
        )
        plans[key] = {
            "variant": asdict(variant),
            "run_name": run_name,
            "terminal_tokens": terminal_tokens,
            "expanded_training_tokens_D": expanded_tokens,
            "non_embedding_parameters_N": non_embedding_params,
            "step_law_learning_rate_raw": lr_raw,
            "learning_rate_config": lr_config,
            "batch": asdict(batch),
            "pause_token_id": args.pause_token_id,
            "config": str(config),
            "config_sha256": sha256_file(config),
            "sbatch": str(sbatch),
            "sbatch_sha256": sha256_file(sbatch),
            "remote_run_dir": remote_run_dir,
            "remote_save_folder": f"{args.remote_workspace}/saved_models/{run_name}",
        }
        synced.extend([(config, f"{relative_dir}/config.yaml"), (sbatch, f"{relative_dir}/submit.sbatch")])

    manifest_path = campaign / "run_manifest.json"
    manifest = {
        "run_id": campaign.name,
        "status": "generated" if not args.submit else "submitting",
        "generated_at": utc_now(),
        "paper_equation": {
            "learning_rate": "1.79 * N^-0.713 * D^0.307",
            "batch_tokens": "0.58 * D^0.571",
            "learning_rate_rounding_decimals": 6,
        },
        "dataset": {
            "local_path": str(args.terminal_data.resolve()),
            "remote_path": f"{args.remote_workspace}/dataset/bbc-news/terminal/train.npy",
            "bytes": args.terminal_data.stat().st_size,
            "tokens_uint16": terminal_tokens,
            "token_count_method": "format-aware NumPy/raw header inspection",
            "sha256_omitted_reason": "20 GB stream; byte count and remote stat are checked before submission",
        },
        "tokenizer": {
            "path": str(args.tokenizer.resolve()),
            "sha256": sha256_file(args.tokenizer.resolve()),
            "sep_token": "<|SEP|>",
            "sep_token_id": actual_pause_id,
        },
        "scheduler": {
            "host": args.remote_host,
            "partition": args.partition,
            "account": args.account,
            "gpu_count": args.gpus,
            "runtime_bin": args.runtime_bin,
            "dependency_mode": args.dependency_mode,
            "first": args.first,
            "walltime": args.walltime,
        },
        "variants": plans,
        "invocation": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
    }
    save_json(manifest_path, manifest)
    synced.append((manifest_path, "run_manifest.json"))
    source_files = [
        (Path(__file__).resolve(), "source/submit_pause_sep_pretrain.py"),
        (args.base_config.resolve(), "source/terminal.yaml"),
        (REPO_ROOT / "scripts/step_law.py", "source/step_law.py"),
        (REPO_ROOT / "tests/test_submit_pause_sep_pretrain.py", "source/test_submit_pause_sep_pretrain.py"),
        (campaign / "PLAN.md", "PLAN.md"),
        (campaign / "CHECKLIST.md", "CHECKLIST.md"),
    ]
    synced.extend(source_files)

    jobs_path = campaign / "jobs.json"
    jobs: dict[str, Any] = json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    if not args.submit:
        for key in ["pause1", "pause2"]:
            plan = plans[key]
            batch = plan["batch"]
            print(
                f"{key}: lr={plan['learning_rate_config']:.6f} "
                f"global={batch['global_batch_size']} "
                f"device={batch['per_device_batch_size']} "
                f"micro={batch['microbatch_size']} "
                f"accum={batch['gradient_accumulation_steps']}"
            )
        print(f"dry-run only; generated {manifest_path}")
        return 0

    existing = [key for key in VARIANTS if jobs.get(key, {}).get("job_id")]
    if existing and not args.resubmit:
        raise RuntimeError(
            f"campaign already records submitted jobs for {existing}; use --resubmit for a new attempt"
        )

    # Sync only campaign-owned files; never modify the remote dirty worktree.
    sync_campaign_files(args.remote_host, remote_campaign, synced)
    remote_log_dirs = " ".join(
        shlex.quote(plan["remote_run_dir"] + "/logs") for plan in plans.values()
    )
    remote_bash(args.remote_host, f"mkdir -p {remote_log_dirs}")
    expected_data_bytes = args.terminal_data.stat().st_size
    expected_tokenizer_sha = sha256_file(args.tokenizer.resolve())
    remote_snapshot = remote_bash(
        args.remote_host,
        " && ".join(
            [
                f"cd {shlex.quote(args.remote_workspace)}",
                "export PATH=/opt/gridview/slurm/bin:$PATH",
                "git rev-parse HEAD",
                "git status --short",
                "sha256sum olmo/config.py olmo/model.py olmo/data/util.py olmo/data/memmap_dataset.py olmo/train.py scripts/train.py",
                f"test $(stat -c '%s' {shlex.quote(args.remote_workspace + '/dataset/bbc-news/terminal/train.npy')}) -eq {expected_data_bytes}",
                f"test $(sha256sum {shlex.quote(args.remote_workspace + '/dataset/bbc-news/TG_GPT2_tokenizer.json')} | cut -d' ' -f1) = {expected_tokenizer_sha}",
                f"test -x {shlex.quote(args.runtime_bin + '/torchrun')}",
                "echo REMOTE_PREFLIGHT_OK",
            ]
        ),
    )
    (campaign / "remote_preflight.txt").write_text(remote_snapshot + "\n")
    remote_patch = remote_bash(
        args.remote_host,
        f"cd {shlex.quote(args.remote_workspace)}; git diff --binary -- "
        "olmo/config.py olmo/model.py olmo/data/util.py olmo/data/memmap_dataset.py "
        "olmo/train.py scripts/train.py",
    )
    (campaign / "remote_training_code.patch").write_text(remote_patch + "\n")
    synced_snapshot = [
        (campaign / "remote_preflight.txt", "remote_preflight.txt"),
        (campaign / "remote_training_code.patch", "remote_training_code.patch"),
    ]
    sync_campaign_files(args.remote_host, remote_campaign, synced_snapshot)

    order = [args.first, "pause2" if args.first == "pause1" else "pause1"]
    first_key, second_key = order
    first = plans[first_key]
    first_output = remote_bash(
        args.remote_host,
        f"export PATH=/opt/gridview/slurm/bin:$PATH; sbatch --parsable {shlex.quote(first['remote_run_dir'] + '/submit.sbatch')}",
    )
    first_id = parse_job_id(first_output)
    jobs[first_key] = {
        "job_id": first_id,
        "submitted_at": utc_now(),
        "dependency": None,
        "run_name": first["run_name"],
    }
    save_json(jobs_path, jobs)

    second = plans[second_key]
    second_output = remote_bash(
        args.remote_host,
        "export PATH=/opt/gridview/slurm/bin:$PATH; "
        f"sbatch --parsable --dependency={args.dependency_mode}:{first_id} "
        f"{shlex.quote(second['remote_run_dir'] + '/submit.sbatch')}",
    )
    second_id = parse_job_id(second_output)
    jobs[second_key] = {
        "job_id": second_id,
        "submitted_at": utc_now(),
        "dependency": f"{args.dependency_mode}:{first_id}",
        "run_name": second["run_name"],
    }
    save_json(jobs_path, jobs)
    manifest["status"] = "queued"
    manifest["submitted_at"] = utc_now()
    manifest["jobs"] = jobs
    save_json(manifest_path, manifest)
    sync_campaign_files(
        args.remote_host,
        remote_campaign,
        [(jobs_path, "jobs.json"), (manifest_path, "run_manifest.json")],
    )

    queue = remote_bash(
        args.remote_host,
        "export PATH=/opt/gridview/slurm/bin:$PATH; "
        f"squeue -h -j {first_id},{second_id} -o '%i|%j|%P|%T|%R|%b'",
    )
    (campaign / "queue_after_submit.txt").write_text(queue + "\n")
    sync_campaign_files(
        args.remote_host,
        remote_campaign,
        [(campaign / "queue_after_submit.txt", "queue_after_submit.txt")],
    )
    print(f"{first_key}: job {first_id}")
    print(f"{second_key}: job {second_id} dependency={args.dependency_mode}:{first_id}")
    print(queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
