#!/usr/bin/env python3
"""Reproducible stage runner for the BBC and FineWeb-Edu pretraining corpora.

The full corpora are too large for one opaque command.  This entry point keeps
each resumable stage explicit and makes the array-task mapping deterministic.
Use ``plan`` first, then run ``download``/``parse`` as scheduler arrays,
``tokenize`` once over completed parsed shards, and BBC-only
``make-split-indices``/``assemble``/``variants``/``baselines`` as needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_FILE = Path(__file__).with_name("bbc_configs.txt")
CORPORA = ("bbc", "fineweb-edu")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_bbc_configs(path: Path = CONFIG_FILE) -> list[str]:
    configs = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(configs) != len(set(configs)):
        raise ValueError(f"duplicate BBC config in {path}")
    return configs


def fineweb_group(index: int) -> str:
    if not 0 <= index < 246:
        raise IndexError(f"FineWeb-Edu task index must be 0..245, got {index}")
    first = index * 4
    return f".*-00({first:03d}|{first + 1:03d}|{first + 2:03d}|{first + 3:03d}).*arrow"


def validate_fineweb_arrow_shards(paths: Sequence[Path]) -> None:
    if len(paths) != 984:
        raise ValueError(
            f"FineWeb-Edu sample-100BT must expose 984 Arrow shards, got {len(paths)}"
        )
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("FineWeb-Edu Arrow staging contains duplicate filenames")
    for index in range(246):
        pattern = re.compile(fineweb_group(index))
        matches = [name for name in names if pattern.match(name)]
        if len(matches) != 4:
            raise ValueError(
                f"FineWeb-Edu task {index} must match four Arrow shards, got {len(matches)}"
            )


def task_index(value: int | None) -> int:
    if value is not None:
        return value
    env = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env is None:
        raise ValueError("pass --task-index or set SLURM_ARRAY_TASK_ID")
    return int(env)


def run_checked(command: Sequence[str]) -> None:
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def defaults(corpus: str) -> dict[str, Path]:
    if corpus == "bbc":
        return {
            "raw": REPO_ROOT / "dataset/bbc-news-raw",
            "parsed": REPO_ROOT / "dataset/bbc-news-parsed",
            "tokenized": REPO_ROOT / "dataset/bbc-news-shards",
            "final": REPO_ROOT / "dataset/bbc-news",
            "tokenizer": REPO_ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json",
        }
    return {
        "raw": REPO_ROOT / "dataset/fineweb-edu-100BT-arrow",
        "parsed": REPO_ROOT / "dataset/fineweb-edu-parsed",
        "tokenized": REPO_ROOT / "dataset/fineweb-edu-v2",
        "final": REPO_ROOT / "dataset/fineweb-edu-v2",
        "tokenizer": REPO_ROOT / "dataset/TG_QWEN3_tokenizer.json",
    }


def command_plan(corpus: str, paths: dict[str, Path]) -> dict[str, Any]:
    module = "datatools.parse_pretrain_data.build_pretrain_data"
    common = [sys.executable, "-m", module]
    tasks = len(read_bbc_configs()) if corpus == "bbc" else 246
    stages: list[dict[str, Any]] = [
        {
            "stage": "tokenizer",
            "array": None,
            "command": [*common, "tokenizer", "--corpus", corpus],
        },
        {
            "stage": "download",
            "array": f"0-{tasks - 1}" if corpus == "bbc" else None,
            "command": [*common, "download", "--corpus", corpus, "--task-index", "$TASK_ID"]
            if corpus == "bbc"
            else [*common, "download", "--corpus", corpus],
        },
        {
            "stage": "parse",
            "array": f"0-{tasks - 1}",
            "command": [*common, "parse", "--corpus", corpus, "--task-index", "$TASK_ID"],
        },
        {
            "stage": "tokenize",
            "array": None,
            "command": [*common, "tokenize", "--corpus", corpus],
        },
    ]
    if corpus == "bbc":
        stages.extend(
            [
                {
                    "stage": "make-split-indices",
                    "array": None,
                    "command": [*common, "make-split-indices", "--corpus", corpus],
                    "note": (
                        "deterministic reconstruction; released index JSON is required "
                        "for byte-identical historical splits"
                    ),
                },
                {
                    "stage": "assemble",
                    "array": None,
                    "command": [*common, "assemble", "--corpus", corpus],
                },
                {
                    "stage": "variants",
                    "array": None,
                    "command": [*common, "variants", "--corpus", corpus],
                },
                {
                    "stage": "baselines",
                    "array": None,
                    "command": [*common, "baselines", "--corpus", corpus],
                },
            ]
        )
    stages.append(
        {
            "stage": "validate",
            "array": None,
            "command": [*common, "validate", "--corpus", corpus],
        }
    )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "corpus": corpus,
        "paths": {key: str(value) for key, value in paths.items()},
        "task_count": tasks,
        "stages": stages,
    }


def stage_fineweb_arrow_cache(raw_dir: Path, copy_files: bool = False) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train"
    )
    cache_files = [Path(item["filename"]).resolve() for item in dataset.cache_files]
    if not cache_files:
        raise RuntimeError("datasets returned no FineWeb-Edu cache files")
    validate_fineweb_arrow_shards(cache_files)
    raw_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for source in cache_files:
        destination = raw_dir / source.name
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink() and destination.resolve() != source:
                raise FileExistsError(f"staged file points elsewhere: {destination}")
            if not destination.is_symlink() and (
                not copy_files or destination.stat().st_size != source.stat().st_size
            ):
                raise FileExistsError(f"staged file does not match source: {destination}")
        elif copy_files:
            shutil.copy2(source, destination)
        else:
            destination.symlink_to(source)
        staged.append(
            {"path": str(destination), "source": str(source), "bytes": source.stat().st_size}
        )
    manifest = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "staging": "copy" if copy_files else "symlink",
        "cache_files": staged,
        "generated_at": utc_now(),
    }
    (raw_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def allocate_counts(counts: dict[str, int], total: int) -> dict[str, int]:
    available = sum(counts.values())
    if total < 0 or total > available:
        raise ValueError(f"requested split size {total} outside corpus size {available}")
    exact = {key: value * total / available for key, value in counts.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    ranking = sorted(counts, key=lambda key: (exact[key] - allocation[key], key), reverse=True)
    for key in ranking[:remaining]:
        allocation[key] += 1
    return allocation


def make_split_indices(
    parsed_dir: Path, order: Sequence[str], dev_size: int, test_size: int, seed: int
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    counts: dict[str, int] = {}
    for stem in order:
        path = parsed_dir / f"{stem}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            counts[stem] = sum(1 for _ in handle)
    test_counts = allocate_counts(counts, test_size)
    remaining_counts = {key: value - test_counts[key] for key, value in counts.items()}
    dev_counts = allocate_counts(remaining_counts, dev_size)
    rng = random.Random(seed)
    dev: dict[str, list[int]] = {}
    test: dict[str, list[int]] = {}
    for stem in order:
        selected = rng.sample(range(counts[stem]), test_counts[stem] + dev_counts[stem])
        test[stem] = sorted(selected[: test_counts[stem]])
        dev[stem] = sorted(selected[test_counts[stem] :])
    return dev, test


def validate_tokens(corpus: str, tokenized: Path, tokenizer_path: Path) -> dict[str, Any]:
    import numpy as np
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    expected_dtype = np.dtype("uint16" if tokenizer.get_vocab_size() < 65536 else "uint32")
    expected = len(read_bbc_configs()) if corpus == "bbc" else 246
    result: dict[str, Any] = {
        "corpus": corpus,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "expected_dtype": str(expected_dtype),
        "formats": {},
    }
    for input_format in ("terminal", "tree", "tg"):
        files = sorted((tokenized / input_format).glob("*.npy"))
        if len(files) != expected:
            raise ValueError(
                f"{input_format}: expected {expected} shards, found {len(files)} in "
                f"{tokenized / input_format}"
            )
        tokens = 0
        for path in files:
            array = np.load(path, mmap_mode="r")
            if array.ndim != 1 or array.dtype != expected_dtype:
                raise ValueError(f"invalid token shard {path}: {array.shape} {array.dtype}")
            tokens += int(array.size)
        result["formats"][input_format] = {"shards": len(files), "tokens": tokens}
    return result


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--corpus", choices=CORPORA, required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--parsed-dir", type=Path)
    parser.add_argument("--tokenized-dir", type=Path)
    parser.add_argument("--final-dir", type=Path)
    parser.add_argument("--tokenizer", type=Path)


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    values = defaults(args.corpus)
    for argument, key in (
        ("raw_dir", "raw"), ("parsed_dir", "parsed"),
        ("tokenized_dir", "tokenized"), ("final_dir", "final"),
        ("tokenizer", "tokenizer"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            values[key] = value.resolve()
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)

    for name in ("plan", "tokenizer", "download", "parse", "tokenize", "validate"):
        sub = subparsers.add_parser(name)
        add_common(sub)
    subparsers.choices["plan"].add_argument("--output", type=Path)
    subparsers.choices["download"].add_argument("--task-index", type=int)
    subparsers.choices["download"].add_argument("--copy-cache-files", action="store_true")
    subparsers.choices["parse"].add_argument("--task-index", type=int)
    subparsers.choices["parse"].add_argument("--max-docs", type=int)
    subparsers.choices["parse"].add_argument("--skip-deps", action="store_true")
    subparsers.choices["tokenize"].add_argument("--jobs", type=int, default=16)
    subparsers.choices["validate"].add_argument("--output", type=Path)

    split = subparsers.add_parser("make-split-indices")
    add_common(split)
    split.add_argument("--dev-size", type=int, default=5000)
    split.add_argument("--test-size", type=int, default=5000)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--output-dir", type=Path)

    assemble_parser = subparsers.add_parser("assemble")
    add_common(assemble_parser)
    assemble_parser.add_argument("--dev-index", type=Path)
    assemble_parser.add_argument("--test-index", type=Path)
    assemble_parser.add_argument("--overwrite", action="store_true")

    variants = subparsers.add_parser("variants")
    add_common(variants)
    variants.add_argument("--jobs", type=int, default=1)
    variants.add_argument("--overwrite", action="store_true")

    baselines = subparsers.add_parser("baselines")
    add_common(baselines)
    baselines.add_argument("--workers", type=int, default=0)
    baselines.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = resolved_paths(args)
    if args.stage == "plan":
        plan = command_plan(args.corpus, paths)
        output = args.output or REPO_ROOT / f"artifacts/data/{args.corpus}_build_plan.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.corpus} data build plan -> {output}")
        return 0

    if args.stage == "tokenizer":
        model = "gpt2" if args.corpus == "bbc" else "qwen3"
        run_checked(
            [sys.executable, "-m", "datatools.parse_pretrain_data.get_TG_tokenizer",
             "--model-name", model, "--output", str(paths["tokenizer"])]
        )
        return 0

    if args.stage == "download":
        if args.corpus == "bbc":
            configs = read_bbc_configs()
            index = task_index(args.task_index)
            if not 0 <= index < len(configs):
                raise IndexError(f"BBC task index must be 0..{len(configs) - 1}, got {index}")
            from datatools.parse_pretrain_data.setup_parse_deps import fetch_bbc_shard

            fetch_bbc_shard(configs[index], paths["raw"])
        else:
            stage_fineweb_arrow_cache(paths["raw"], copy_files=args.copy_cache_files)
        return 0

    if args.stage == "parse":
        index = task_index(args.task_index)
        if args.corpus == "bbc":
            configs = read_bbc_configs()
            if not 0 <= index < len(configs):
                raise IndexError(f"BBC task index must be 0..{len(configs) - 1}, got {index}")
            config = configs[index]
            files = sorted((paths["raw"] / config).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no downloaded parquet for {config}")
            command = [
                sys.executable, "-m", "datatools.parse_pretrain_data.benepar_parse",
                "--data_files", ",".join(str(path) for path in files),
                "--output-dir", str(paths["parsed"]),
            ]
        else:
            group = fineweb_group(index)
            command = [
                sys.executable, "-m", "datatools.parse_pretrain_data.parse_input",
                "--input_list", f"finewebedu{group}",
                "--fineweb-edu-arrow-dir", str(paths["raw"]),
                "--output-dir", str(paths["parsed"]),
            ]
        if args.max_docs is not None and args.corpus == "bbc":
            command.extend(["--max-docs", str(args.max_docs)])
        if args.skip_deps:
            command.append("--skip-deps")
        run_checked(command)
        return 0

    if args.stage == "tokenize":
        run_checked(
            [
                sys.executable, "-m", "datatools.parse_pretrain_data.convert_TG_and_tokenize",
                "--input_dir", str(paths["parsed"]),
                "--tokenizer", str(paths["tokenizer"]),
                "--output_dir", str(paths["tokenized"]),
                "--jobs", str(args.jobs),
            ]
        )
        return 0

    if args.stage == "make-split-indices":
        if args.corpus != "bbc":
            raise ValueError("FineWeb-Edu pretraining consumes all 246 tokenized shard groups")
        output_dir = (args.output_dir or paths["final"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        dev, test = make_split_indices(
            paths["parsed"], read_bbc_configs(), args.dev_size, args.test_size, args.seed
        )
        (output_dir / "dev_index.json").write_text(json.dumps(dev, sort_keys=True) + "\n")
        (output_dir / "test_index.json").write_text(json.dumps(test, sort_keys=True) + "\n")
        metadata = {
            "seed": args.seed,
            "dev_documents": sum(map(len, dev.values())),
            "test_documents": sum(map(len, test.values())),
            "historical_byte_identical": False,
            "note": (
                "Use the released dev_index.json/test_index.json for byte-identical "
                "paper streams."
            ),
        }
        (output_dir / "split_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        print(f"wrote deterministic split indices -> {output_dir}")
        return 0

    if args.stage == "assemble":
        if args.corpus != "bbc":
            raise ValueError("FineWeb-Edu remains sharded and does not use BBC split assembly")
        dev_index = args.dev_index or paths["final"] / "dev_index.json"
        test_index = args.test_index or paths["final"] / "test_index.json"
        command = [
            sys.executable, "-m", "datatools.parse_pretrain_data.assemble_streams",
            "--input-root", str(paths["tokenized"]),
            "--output-root", str(paths["final"]),
            "--tokenizer", str(paths["tokenizer"]),
            "--dev-index", str(dev_index),
            "--test-index", str(test_index),
        ]
        if args.overwrite:
            command.append("--overwrite")
        run_checked(command)
        return 0

    if args.stage == "variants":
        if args.corpus != "bbc":
            raise ValueError("tree ablation variants are built only for BBC experiments")
        for variant in ("noont", "compress", "triplecnt"):
            output_dir = paths["final"] / f"tree_{variant}"
            for split in ("train", "dev", "test"):
                output = output_dir / f"{split}.npy"
                if output.exists() and not args.overwrite:
                    print(f"skip existing {output}")
                    continue
                run_checked(
                    [
                        sys.executable, "-m", "datatools.parse_pretrain_data.make_tree_variant",
                        "--input", str(paths["final"] / "tree" / f"{split}.npy"),
                        "--output-dir", str(output_dir),
                        "--variant", variant,
                        "--tokenizer", str(paths["tokenizer"]),
                        "--workers", str(args.jobs),
                    ]
                )
        return 0

    if args.stage == "baselines":
        if args.corpus != "bbc":
            raise ValueError("TreeReg/Pushdown baselines use BBC parse-aligned streams")
        for split in args.splits:
            run_checked(
                [
                    sys.executable, "scripts/precompute_treereg.py",
                    "--split", split,
                    "--tree-dir", str(paths["final"] / "tree"),
                    "--tokenizer", str(paths["tokenizer"]),
                    "--out-dir", str(paths["final"] / "parse_aligned" / f"{split}_treereg"),
                    "--workers", str(args.workers),
                ]
            )
            run_checked(
                [
                    sys.executable, "scripts/precompute_pushdown_unary.py",
                    "--split", split,
                    "--tree-dir", str(paths["final"] / "tree"),
                    "--tokenizer", str(paths["tokenizer"]),
                    "--out-dir", str(
                        paths["final"] / "parse_aligned" / f"{split}_pushdown_unary_terminals"
                    ),
                    "--direction", "right",
                    "--sentence-format", "terminals",
                    "--workers", str(args.workers),
                ]
            )
        return 0

    if args.stage == "validate":
        result = validate_tokens(args.corpus, paths["tokenized"], paths["tokenizer"])
        output = args.output or paths["tokenized"] / "validation_manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
