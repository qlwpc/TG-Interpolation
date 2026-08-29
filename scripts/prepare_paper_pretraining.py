#!/usr/bin/env python3
"""Materialize clean pretraining campaigns for every model in the paper.

The historical checkpoint ``config.yaml`` is the architecture/optimizer source
of truth.  ``train_configs/paper_pretraining_manifest.json`` records the fields
audited in ``EXPERIMENT_REPRODUCTION_RECORD.md`` and maps every model to its
input representation.  This program verifies the two sources agree, copies the
checkpoint config, and replaces only run-local state and canonical data paths.

It never submits jobs.  Each generated run contains a resolved ``config.yaml``,
an executable ``launch.sh``, and a compact ``protocol.json`` provenance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "train_configs/paper_pretraining_manifest.json"
SCALE_SHAPES = {
    "100M": (768, 12, 12),
    "500M": (1408, 16, 16),
    "1B": (2048, 16, 16),
}
PAPER_GPU_COUNTS = {"100M": 4, "500M": 4, "1B": 64}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema: {raw.get('schema_version')}")
    runs = raw.get("runs", [])
    ids = [run["id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("paper pretraining manifest contains duplicate run ids")
    for run in runs:
        marker = f"/step{run['final_step']}-"
        if marker not in run["source_config"]:
            raise ValueError(
                f"{run['id']}: final_step does not identify its source checkpoint"
            )
    return raw


def fineweb_edu_patterns(workspace: str, input_format: str) -> list[str]:
    """The 984 source shards, grouped exactly as the 246 historical configs."""
    root = f"{workspace}/dataset/fineweb-edu-v2/{input_format}"
    return [
        f"{root}/*-00({i:03d}|{i + 1:03d}|{i + 2:03d}|{i + 3:03d})*arrow.npy"
        for i in range(0, 984, 4)
    ]


def data_paths(run: dict[str, Any], workspace: str) -> tuple[list[str], list[str] | None]:
    input_format = run["input"]
    if run["corpus"] == "fineweb-edu":
        return fineweb_edu_patterns(workspace, input_format), None

    root = f"{workspace}/dataset/bbc-news"
    if input_format == "treereg":
        return [f"{root}/parse_aligned/train_treereg"], [f"{root}/tree/train.npy"]
    if input_format == "pushdown":
        return [f"{root}/parse_aligned/train_pushdown_unary_terminals"], [
            f"{root}/tree/train.npy"
        ]
    return [f"{root}/{input_format}/train.npy"], None


def tokenizer_path(run: dict[str, Any], workspace: str) -> str:
    if run["corpus"] == "fineweb-edu":
        return f"{workspace}/dataset/TG_QWEN3_tokenizer.json"
    return f"{workspace}/dataset/bbc-news/TG_GPT2_tokenizer.json"


def _source_value(cfg: Any, dotted: str) -> Any:
    from omegaconf import OmegaConf as om

    return om.select(cfg, dotted, default=None)


def audit_source_config(run: dict[str, Any], source_cfg: Any) -> None:
    """Fail if a checkpoint config drifts from the manually audited record."""
    expected_shape = SCALE_SHAPES[run["scale"]]
    actual_shape = tuple(
        int(_source_value(source_cfg, key))
        for key in ("model.d_model", "model.n_layers", "model.n_heads")
    )
    if actual_shape != expected_shape:
        raise ValueError(f"{run['id']}: shape {actual_shape} != {expected_shape}")

    token_layout = (
        {
            "model.vocab_size": 50320,
            "model.embedding_size": 50320,
            "model.eos_token_id": 50256,
            "model.pad_token_id": 50258,
        }
        if run["corpus"] == "bbc"
        else {
            "model.vocab_size": 151732,
            "model.embedding_size": 151732,
            "model.eos_token_id": 151643,
            "model.pad_token_id": 151670,
        }
    )

    comparisons = {
        **token_layout,
        "optimizer.learning_rate": run["learning_rate"],
        "optimizer.weight_decay": 0.1,
        "optimizer.betas": [0.9, 0.95],
        "optimizer.eps": 1.0e-8,
        "scheduler.t_warmup": 2000,
        "scheduler.min_lr": 1.0e-5,
        "max_grad_norm": 1.0,
        "global_train_batch_size": run["global_batch"],
        "device_train_microbatch_size": run["microbatch"],
        "data.memmap_dtype": run["dtype"],
        "seed": 6198,
        "precision": "amp_bf16",
        "max_duration": "1ep",
    }
    for key, expected in comparisons.items():
        actual = _source_value(source_cfg, key)
        if actual != expected:
            raise ValueError(f"{run['id']}: source {key}={actual!r}, expected {expected!r}")

    actual_grammar = _source_value(source_cfg, "model.transformer_grammar_type")
    # The oldest terminal checkpoint predates the required explicit key.
    if actual_grammar is None and run["grammar"] == "terminal":
        actual_grammar = "terminal"
    if actual_grammar != run["grammar"]:
        raise ValueError(
            f"{run['id']}: source grammar={actual_grammar!r}, expected {run['grammar']!r}"
        )

    if "sequence_length" in run:
        actual = _source_value(source_cfg, "model.max_sequence_length")
        if actual != run["sequence_length"]:
            raise ValueError(f"{run['id']}: sequence length {actual} is not audited value")
    elif _source_value(source_cfg, "model.max_sequence_length") != 2048:
        raise ValueError(f"{run['id']}: expected the common 2048-token sequence length")
    if "pause_token_id" in run:
        actual = _source_value(source_cfg, "model.pause_token_id")
        if actual != run["pause_token_id"]:
            raise ValueError(f"{run['id']}: pause id {actual} is not audited value")
    if "mix_heads" in run:
        actual = [
            [item["grammar_type"], int(item["n_heads"])]
            for item in _source_value(source_cfg, "model.mix_head_type")
        ]
        if actual != run["mix_heads"]:
            raise ValueError(f"{run['id']}: mixing heads {actual} != {run['mix_heads']}")


def _bbc_evaluators(run: dict[str, Any], workspace: str) -> list[dict[str, Any]]:
    root = f"{workspace}/dataset/bbc-news"
    input_format = run["input"]
    if input_format == "treereg":
        eval_root = f"{root}/parse_aligned"
        names = ("dev_treereg", "test_treereg")
        tree_paths = (f"{root}/tree/dev.npy", f"{root}/tree/test.npy")
        attention = True
    elif input_format == "pushdown":
        eval_root = f"{root}/parse_aligned"
        names = ("dev_pushdown_unary_terminals", "test_pushdown_unary_terminals")
        tree_paths = (f"{root}/tree/dev.npy", f"{root}/tree/test.npy")
        attention = True
    else:
        eval_root = f"{root}/{input_format}"
        names = ("dev.npy", "test.npy")
        tree_paths = (None, None)
        attention = run["grammar"] in {"tg", "tgnomask", "tgnomask_aug", "mixing"}

    evaluators = []
    for index, (name, tree_path) in enumerate(zip(names, tree_paths)):
        path = f"{eval_root}/{name}"
        evaluators.append(
            {
                "label": "TG-ppl-validation" if index == 0 else "TG-ppl-validation-test",
                "type": "lm",
                "data": {
                    "paths": [path],
                    "parse_tree_paths": [tree_path] if tree_path else None,
                    "memmap_dtype": run["dtype"],
                    "memmap_format": "auto",
                    "generate_attention_mask": attention,
                    "generate_doc_lengths": True,
                    "num_workers": 0,
                    "drop_last": False,
                    "persistent_workers": False,
                },
            }
        )
    return evaluators


def materialize_config(
    run: dict[str, Any], source: Path, output: Path, workspace: str, run_name: str
) -> None:
    from omegaconf import OmegaConf as om

    cfg = om.load(source)
    audit_source_config(run, cfg)

    def put(key: str, value: Any) -> None:
        om.update(cfg, key, value, merge=False, force_add=True)

    paths, parse_tree_paths = data_paths(run, workspace)
    tok = tokenizer_path(run, workspace)
    put("workspace", workspace)
    put("run_name", run_name)
    put("seed", 6198)
    put("model.transformer_grammar_type", run["grammar"])
    put("optimizer.learning_rate", run["learning_rate"])
    put("global_train_batch_size", run["global_batch"])
    put("device_train_batch_size", None)
    put("device_train_microbatch_size", run["microbatch"])
    put("device_train_grad_accum", None)
    put("precision", "amp_bf16")
    put("max_duration", "1ep")
    put("data.paths", paths)
    put("data.parse_tree_paths", parse_tree_paths)
    put("data.memmap_dtype", run["dtype"])
    put("data.memmap_format", "auto")
    put("tokenizer.identifier", tok)
    put("tokenizer.vocabulary", tok)
    put("save_folder", f"{workspace}/saved_models/reproduction/{run_name}")
    put("remote_save_folder", None)
    put("load_path", None)
    put("load_path_sharded_checkpointer", None)
    put("try_load_latest_save", False)
    put("reset_optimizer_state", False)
    put("reset_trainer_state", False)
    put("eval_on_load", False)
    put("eval_no_save", False)
    put("save_overwrite", False)
    put("evaluators", [] if run["corpus"] == "fineweb-edu" else _bbc_evaluators(run, workspace))
    if _source_value(cfg, "wandb") is not None:
        put("wandb.name", run_name)
        put("wandb.group", "paper-pretraining-reproduction")
        put("wandb.mode", "offline")

    output.parent.mkdir(parents=True, exist_ok=True)
    om.save(cfg, output)

    # Structured loading catches unknown or stale config fields.  Path checks
    # are an explicit later gate because campaign generation is a dry run.
    from olmo.config import TrainConfig

    resolved = TrainConfig.load(output, validate_paths=False)
    if resolved.run_name != run_name or resolved.data.paths != paths:
        raise AssertionError(f"{run['id']}: generated config did not resolve as requested")
    if resolved.model.transformer_grammar_type != run["grammar"]:
        raise AssertionError(f"{run['id']}: grammar changed during legacy migration")


def validate_training_inputs(config: Path) -> dict[str, Any]:
    """Resolve a generated config and require all local training inputs.

    ``TrainConfig.load(validate_paths=True)`` only checks paths written through
    OmegaConf path resolvers. Historical checkpoint configs store
    ``data.paths`` as plain strings, so they need this explicit launch gate.
    """
    from olmo.config import TrainConfig

    cfg = TrainConfig.load(config, validate_paths=False)

    def require_patterns(label: str, patterns: Sequence[str] | None) -> list[str]:
        matches: list[str] = []
        for pattern in patterns or []:
            current = sorted(glob(str(pattern)))
            if not current:
                raise FileNotFoundError(f"{label}: {pattern} matched no local input")
            matches.extend(current)
        return matches

    data_matches = require_patterns("data.paths", cfg.data.paths)
    parse_matches = require_patterns(
        "data.parse_tree_paths", getattr(cfg.data, "parse_tree_paths", None)
    )
    evaluator_matches = 0
    for evaluator in cfg.evaluators or []:
        evaluator_matches += len(
            require_patterns(
                f"evaluator[{evaluator.label}].data.paths", evaluator.data.paths
            )
        )
        evaluator_matches += len(
            require_patterns(
                f"evaluator[{evaluator.label}].data.parse_tree_paths",
                getattr(evaluator.data, "parse_tree_paths", None),
            )
        )

    tokenizer_paths = {
        str(value)
        for value in (cfg.tokenizer.identifier, cfg.tokenizer.vocabulary)
        if value
    }
    for path in tokenizer_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"tokenizer input does not exist: {path}")

    return {
        "run_name": cfg.run_name,
        "grammar": cfg.model.transformer_grammar_type,
        "data_files": len(data_matches),
        "parse_tree_inputs": len(parse_matches),
        "evaluator_inputs": evaluator_matches,
        "tokenizer_files": len(tokenizer_paths),
    }


def generate_launch(
    output: Path, config: Path, workspace: str, gpu_count: int, run: dict[str, Any]
) -> None:
    default_nproc = min(gpu_count, 8)
    default_nnodes = gpu_count // default_nproc
    text = f"""#!/usr/bin/env bash
set -euo pipefail

cd {shlex.quote(workspace)}
export PYTHONPATH={shlex.quote(workspace)}${{PYTHONPATH:+:$PYTHONPATH}}
export WANDB_MODE=${{WANDB_MODE:-offline}}
export CONFIG_PATH={shlex.quote(str(config))}

NNODES=${{NNODES:-{default_nnodes}}}
NPROC_PER_NODE=${{NPROC_PER_NODE:-{default_nproc}}}
NODE_RANK=${{NODE_RANK:-0}}
MASTER_ADDR=${{MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${{MASTER_PORT:-29500}}
if (( NNODES * NPROC_PER_NODE != {gpu_count} )); then
  echo "expected the paper world size {gpu_count}, got $NNODES*$NPROC_PER_NODE" >&2
  exit 2
fi
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" ]]; then
  echo "set MASTER_ADDR and run this launcher once per scheduler node" >&2
  exit 2
fi
python scripts/prepare_paper_pretraining.py --validate-config "$CONFIG_PATH"

exec torchrun \\
  --nnodes="$NNODES" \\
  --nproc-per-node="$NPROC_PER_NODE" \\
  --node-rank="$NODE_RANK" \\
  --master-addr="$MASTER_ADDR" \\
  --master-port="$MASTER_PORT" \\
  scripts/train.py "$CONFIG_PATH"
"""
    output.write_text(text)
    output.chmod(0o755)


def select_runs(
    runs: Iterable[dict[str, Any]], groups: Sequence[str], models: Sequence[str]
) -> list[dict[str, Any]]:
    groups_set = set(groups)
    models_set = set(models)
    selected = [
        run
        for run in runs
        if (not groups_set or "all" in groups_set or run["group"] in groups_set)
        and (not models_set or run["id"] in models_set)
    ]
    if not selected:
        raise ValueError("no paper pretraining runs matched --groups/--models")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--workspace", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/experiment/paper_pretraining_reproduction",
    )
    parser.add_argument("--groups", nargs="*", default=["all"])
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--list", action="store_true", help="list audited runs and exit")
    parser.add_argument(
        "--validate-config",
        type=Path,
        help="validate one generated config's schema and every local input, then exit",
    )
    parser.add_argument(
        "--validate-data",
        action="store_true",
        help="after generation, require every configured tokenizer/data path to exist",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_config is not None:
        result = validate_training_inputs(args.validate_config.resolve())
        print("CONFIG_AND_INPUTS_OK", json.dumps(result, sort_keys=True))
        return 0
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    selected = select_runs(manifest["runs"], args.groups, args.models)
    if args.list:
        for run in selected:
            print(
                f"{run['id']:<34} {run['group']:<23} "
                f"{run['model']:<18} {run['input']}"
            )
        return 0

    workspace_path = args.workspace.resolve()
    workspace = str(workspace_path)
    campaign = args.campaign_dir.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    for run in selected:
        source = workspace_path / run["source_config"]
        if not source.is_file():
            raise FileNotFoundError(f"{run['id']}: missing source config {source}")
        run_name = f"repro_{run['id']}_seed6198"
        run_dir = campaign / "runs" / run["id"]
        config = run_dir / "config.yaml"
        launch = run_dir / "launch.sh"
        materialize_config(run, source, config, workspace, run_name)
        gpu_count = PAPER_GPU_COUNTS[run["scale"]]
        generate_launch(launch, config, workspace, gpu_count, run)
        protocol = {
            **run,
            "run_name": run_name,
            "paper_gpu_count": gpu_count,
            "source_config_sha256": sha256_file(source),
            "generated_config_sha256": sha256_file(config),
            "config": str(config),
            "launch": str(launch),
            "generated_at": utc_now(),
        }
        save_json(run_dir / "protocol.json", protocol)
        if args.validate_data:
            validate_training_inputs(config)
        generated.append(protocol)

    campaign_manifest = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "authority": manifest["authority"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "workspace": workspace,
        "data_validated": bool(args.validate_data),
        "runs": generated,
        "invocation": [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
    }
    output_manifest = campaign / "run_manifest.json"
    save_json(output_manifest, campaign_manifest)
    print(f"generated {len(generated)} paper pretraining runs -> {output_manifest}")
    if not args.validate_data:
        print("data paths were not checked; rerun with --validate-data before launch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
