#!/usr/bin/env python3
"""Audit native-validation counts and fixed-shot disjointness for the 10-task run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import datasets


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strings(value: Any, key: str | None = None) -> Iterable[str]:
    ignored = {"id", "source_id", "label", "answer", "answerkey", "gold"}
    if isinstance(value, dict):
        for child_key in sorted(value):
            yield from strings(value[child_key], child_key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from strings(child, key)
    elif isinstance(value, str) and (key or "").lower() not in ignored:
        yield value


def text_digest(record: dict[str, Any]) -> str:
    text = " ".join(strings(record)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_by_id(records, field, selected):
    # Match the evaluator's get_shots implementations: later duplicate IDs
    # replace earlier records, then the fixed ID order selects exactly N shots.
    by_id = {record[field]: record for record in records}
    missing = [value for value in selected if value not in by_id]
    if missing:
        raise ValueError(f"missing fixed shot IDs for {field}: {missing}")
    return [by_id[value] for value in selected]


def task_record(name, split, validation, shots, paths, expected_eval=None):
    validation_hashes = {text_digest(record) for record in validation}
    shot_hashes = {text_digest(record) for record in shots}
    overlap = validation_hashes & shot_hashes
    eval_count = len(validation)
    if name == "mmlu":
        shot_keys = {
            (
                record["subject"], record["question"],
                tuple(record["choices"]), record["answer"],
            )
            for record in shots
        }
        validation = [
            record for record in validation
            if (
                record["subject"], record["question"],
                tuple(record["choices"]), record["answer"],
            ) not in shot_keys
        ]
        eval_count = len(validation)
        # For MMLU the shots intentionally originate in validation, but are
        # removed before scoring. The final eval/shot overlap must be zero.
        validation_hashes = {text_digest(record) for record in validation}
        overlap = validation_hashes & shot_hashes
    if overlap:
        raise ValueError(f"{name}: {len(overlap)} shot/eval text overlaps")
    if expected_eval is not None and eval_count != expected_eval:
        raise ValueError(f"{name}: expected {expected_eval}, found {eval_count}")
    return {
        "task": name,
        "split": split,
        "validation_source_count": len(validation) + (len(shots) if name == "mmlu" else 0),
        "shot_count": len(shots),
        "evaluation_count": eval_count,
        "shot_eval_text_overlap": 0,
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []

    boolq_train = read_jsonl(ROOT / "dataset/SuperGLUE/BoolQ/train.jsonl")
    boolq_val = read_jsonl(ROOT / "dataset/SuperGLUE/BoolQ/val.jsonl")
    records.append(task_record(
        "boolq", "val", boolq_val, [boolq_train[i] for i in [0, 1, 7]],
        [ROOT / "dataset/SuperGLUE/BoolQ/val.jsonl",
         ROOT / "dataset/SuperGLUE/BoolQ/val_passage.txt",
         ROOT / "dataset/SuperGLUE/BoolQ/val_question.txt"], 3270,
    ))

    hs_train = read_jsonl(ROOT / "dataset/hellaswag/hellaswag_train.jsonl")
    hs_val = read_jsonl(ROOT / "dataset/hellaswag/hellaswag_val.jsonl")
    hs_ids = ["wikihow~19", "wikihow~66", "activitynet~v_-2dxp-mv2zo",
              "activitynet~v_-Xl95IW5H_s", "wikihow~62"]
    records.append(task_record(
        "hellaswag", "val", hs_val, select_by_id(hs_train, "source_id", hs_ids),
        [ROOT / "dataset/hellaswag/hellaswag_val.jsonl",
         ROOT / "dataset/hellaswag/hellaswag_val.txt"], 10042,
    ))

    wg_train = list(datasets.load_from_disk(str(ROOT / "dataset/winogrande/train")))
    wg_val = list(datasets.load_from_disk(str(ROOT / "dataset/winogrande/val")))
    records.append(task_record(
        "winogrande", "val", wg_val,
        [wg_train[i] for i in [19875, 26035, 2302, 13568, 7412]],
        [ROOT / "dataset/winogrande/val/state.json",
         ROOT / "dataset/winogrande/val/data-00000-of-00001.arrow",
         ROOT / "dataset/winogrande/winogrande_val.txt"], 1267,
    ))

    simple_specs = [
        ("piqa", "validation", "dataset/piqa/piqa", [9, 3, 4], None, 1838),
        ("social_iqa", "validation", "dataset/social_i_qa/social_i_qa", [0, 8, 1], None, 1954),
    ]
    for name, split, stem, indices, _, expected in simple_specs:
        train = read_jsonl(ROOT / f"{stem}_train.jsonl")
        validation = read_jsonl(ROOT / f"{stem}_{split}.jsonl")
        records.append(task_record(
            name, split, validation, [train[i] for i in indices],
            [ROOT / f"{stem}_{split}.jsonl", ROOT / f"{stem}_{split}.txt"], expected,
        ))

    id_specs = [
        ("arc_easy", "dataset/ai2_arc/ARC-Easy/arc_easy",
         "id", ["MCAS_2007_8_5189", "Mercury_SC_401169", "MCAS_2004_8_27"], 570),
        ("arc_challenge", "dataset/ai2_arc/ARC-Challenge/arc_challenge",
         "id", ["Mercury_SC_415702", "MCAS_2009_5_6516", "Mercury_7233695"], 299),
        ("openbook_qa", "dataset/openbookqa/openbookqa",
         "id", ["7-584", "7-870", "9-732", "9-782", "8-72"], 500),
        ("commonsense_qa", "dataset/commonsense_qa/commonsense_qa",
         "id", ["61fe6e879ff18686d7552425a36344c8",
                "02e821a3e53cb320790950aab4489e85",
                "23505889b94e880c3e89cff4ba119860"], 1221),
    ]
    for name, stem, id_field, ids, expected in id_specs:
        train = read_jsonl(ROOT / f"{stem}_train.jsonl")
        validation = read_jsonl(ROOT / f"{stem}_validation.jsonl")
        records.append(task_record(
            name, "validation", validation, select_by_id(train, id_field, ids),
            [ROOT / f"{stem}_validation.jsonl", ROOT / f"{stem}_validation.txt"], expected,
        ))

    mmlu_val = list(datasets.load_from_disk(str(ROOT / "dataset/mmlu/validation")))
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in mmlu_val:
        by_subject.setdefault(row["subject"], []).append(row)
    mmlu_shots = [row for subject in sorted(by_subject) for row in by_subject[subject][:5]]
    records.append(task_record(
        "mmlu", "validation_holdout", mmlu_val, mmlu_shots,
        [ROOT / "dataset/mmlu/validation/state.json",
         ROOT / "dataset/mmlu/validation/data-00000-of-00001.arrow",
         ROOT / "dataset/mmlu/mmluvalidation.txt"], 1246,
    ))

    order = ["winogrande", "hellaswag", "boolq", "mmlu", "openbook_qa",
             "commonsense_qa", "social_iqa", "arc_easy", "arc_challenge", "piqa"]
    by_name = {record["task"]: record for record in records}
    manifest = {
        "protocol": "terminal-format-validation-20260828",
        "validation_10_tasks": order,
        "validation_8_decomposition_tasks": [
            task for task in order if task not in {"boolq", "mmlu"}
        ],
        "mmlu_redux": {
            "included": False,
            "reason": "local all split is corrected MMLU test data; no native validation",
        },
        "tasks": [by_name[name] for name in order],
        "validation_10_evaluation_count": sum(by_name[name]["evaluation_count"] for name in order),
        "validation_8_evaluation_count": sum(
            by_name[name]["evaluation_count"]
            for name in order if name not in {"boolq", "mmlu"}
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "validation_10": manifest["validation_10_evaluation_count"],
        "validation_8": manifest["validation_8_evaluation_count"],
    }))


if __name__ == "__main__":
    main()
