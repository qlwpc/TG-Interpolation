#!/usr/bin/env python3
"""Collect step-34354 BLiMP/SG scores into JSON and Markdown."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis-output/syntax_eval_34354"
LOG_DIR = BASE / "logs"

RUNS = {
    ("treereg", "BLiMP", "terminal"): "treereg_blimp_terminal_step34354_*.out",
    ("treereg", "SG", "terminal"): "treereg_sg_terminal_step34354_*.out",
    ("pushdown", "BLiMP", "terminal"): "pushdown_blimp_terminal_step34354_*.out",
    ("pushdown", "SG", "terminal"): "pushdown_sg_terminal_step34354_*.out",
    ("pushdown", "BLiMP", "gold300"): "pushdown_blimp_gold300_step34354_*.out",
    ("pushdown", "SG", "beam300"): "pushdown_sg_beam300_step34354_*.out",
}


def score_from_log(path: Path, task: str) -> float:
    text = path.read_text(errors="replace")
    if not re.search(r"exit=0\b", text):
        raise RuntimeError(f"run did not finish successfully: {path}")
    key = "overall/overall" if task == "BLiMP" else "avg"
    values = re.findall(rf"^\s*{re.escape(key)}=([-+0-9.eE]+)\s*$", text, re.M)
    if not values:
        raise RuntimeError(f"missing {key} in {path}")
    return float(values[-1])


def main() -> None:
    results: dict[str, dict[str, dict[str, object]]] = {}
    for (model, task, mode), pattern in RUNS.items():
        matches = sorted(LOG_DIR.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no log matches {pattern}")
        log_path = matches[-1]
        results.setdefault(model, {}).setdefault(task, {})[mode] = {
            "score": score_from_log(log_path, task),
            "log": str(log_path.relative_to(ROOT)),
        }

    # TreeReg's SCIN term is a training-only auxiliary loss. Its forward logits
    # do not consume tree_spans, so fixed-K gold/beam marginalization adds only a
    # sentence-independent constant and cannot change BLiMP or SG decisions.
    results["treereg"]["BLiMP"]["gold300"] = {
        **results["treereg"]["BLiMP"]["terminal"],
        "derived_equivalent": True,
    }
    results["treereg"]["SG"]["beam300"] = {
        **results["treereg"]["SG"]["terminal"],
        "derived_equivalent": True,
    }

    payload = {
        "checkpoint_step": 34354,
        "results": results,
        "notes": {
            "terminal": "teacher-forced terminal tokens; no gold parse or stack tape",
            "gold300": "BLiMP marginalization over 300 supplied parses",
            "beam300": "SG incremental parse marginalization with beam size 300",
            "treereg_equivalence": (
                "checkpoint-level validation found exact elementwise equality "
                "between TreeReg logits with tree_spans=None and with gold spans"
            ),
        },
    }
    json_path = BASE / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    rows = [
        ("TreeReg", "BLiMP", "terminal", results["treereg"]["BLiMP"]["terminal"]["score"]),
        ("TreeReg", "BLiMP", "gold300（等价）", results["treereg"]["BLiMP"]["gold300"]["score"]),
        ("TreeReg", "SG", "terminal", results["treereg"]["SG"]["terminal"]["score"]),
        ("TreeReg", "SG", "beam300（等价）", results["treereg"]["SG"]["beam300"]["score"]),
        ("Pushdown", "BLiMP", "terminal", results["pushdown"]["BLiMP"]["terminal"]["score"]),
        ("Pushdown", "BLiMP", "gold300", results["pushdown"]["BLiMP"]["gold300"]["score"]),
        ("Pushdown", "SG", "terminal", results["pushdown"]["SG"]["terminal"]["score"]),
        ("Pushdown", "SG", "beam300", results["pushdown"]["SG"]["beam300"]["score"]),
    ]
    lines = [
        "# TreeReg / Pushdown step34354 句法评测",
        "",
        "| 模型 | 测评 | 协议 | 得分 |",
        "|---|---|---|---:|",
    ]
    lines.extend(f"| {m} | {t} | {p} | {s:.4f} |" for m, t, p, s in rows)
    lines.extend(
        [
            "",
            "TreeReg 的树正则只在训练损失中使用；推理 logits 不读取树。",
            "因此其 gold300/beam300 行与 terminal 行严格等价，并非额外运行。",
            "",
        ]
    )
    (BASE / "RESULTS.md").write_text("\n".join(lines))
    print(json_path)
    print(BASE / "RESULTS.md")


if __name__ == "__main__":
    main()
