#!/usr/bin/env python3
"""Extract rank-local XSum predictions and collapse statistics from a train log."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


EVAL_RECORD = re.compile(
    r"<New Passage>: (?P<body>.*?)(?=\n(?:<New Passage>: |\d{4}-\d{2}-\d{2} .*?\[eval_step=))",
    flags=re.DOTALL,
)
RANKED_EVAL_RECORD = re.compile(
    r"^[^\n]*?XSUM_PREDICTION\s+\[global_rank=(?P<rank>\d+)\]\s+"
    r"<New Passage>:\s*(?P<body>.*?)"
    r"(?=^\d{4}-\d{2}-\d{2}[^\n]*?(?:XSUM_PREDICTION|\[eval_step=)|\Z)",
    flags=re.MULTILINE | re.DOTALL,
)
PROMPT = "Summarize the above article in 1 sentence ."
SPECIAL_TOKEN = re.compile(r"<\|(?:beginoftext|endoftext|padding)\|>")


def normalize_prediction(text: str) -> str:
    text = SPECIAL_TOKEN.sub("", text)
    return " ".join(text.split())


def parse_record(body: str) -> tuple[str, str]:
    marker = body.rfind(PROMPT)
    if marker < 0:
        marker = body.rfind("\n<(S><(VP> Summarize")
        prediction_start = body.find("\n", marker + 1) if marker >= 0 else -1
        if marker < 0 or prediction_start < 0:
            raise ValueError("XSum prompt marker is missing from an evaluation record")
        article = body[:marker]
        prediction = body[prediction_start + 1 :]
    else:
        article = body[:marker]
        prediction = body[marker + len(PROMPT) :]
    return article.strip(), normalize_prediction(prediction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    starts = [match.start() for match in re.finditer(r"Running evaluation for 'xsum(?:_valid)?'", text)]
    if not starts:
        raise SystemExit("No XSum evaluation section found")
    eval_start = starts[-1]
    eval_text = text[eval_start:]

    records = []
    malformed_records = 0
    ranked_matches = list(RANKED_EVAL_RECORD.finditer(eval_text))
    matches = ranked_matches or list(EVAL_RECORD.finditer(eval_text))
    for source_order, match in enumerate(matches):
        try:
            article, prediction = parse_record(match.group("body"))
        except ValueError:
            malformed_records += 1
            continue
        record = {
            "source_order": source_order,
            "prediction": prediction,
            "article_prefix": " ".join(article.split())[:300],
        }
        if ranked_matches:
            record["global_rank"] = int(match.group("rank"))
        records.append(record)

    counts = Counter(record["prediction"] for record in records)
    metric_patterns = {
        "rouge1": r"^\s+(?:xsum/)?rouge1=([0-9.eE+-]+)$",
        "rouge2": r"^\s+(?:xsum/)?rouge2=([0-9.eE+-]+)$",
        "rougeL": r"^\s+(?:xsum/)?rougeL=([0-9.eE+-]+)$",
        "R-AVG": r"^\s+(?:xsum/)?R-AVG=([0-9.eE+-]+)$",
    }
    metrics = {}
    for name, pattern in metric_patterns.items():
        matches = re.findall(pattern, eval_text, flags=re.MULTILINE)
        if matches:
            metrics[name] = float(matches[-1])

    summary = {
        "source_log": str(args.log.resolve()),
        "world_size": args.world_size,
        "num_predictions": len(records),
        "malformed_records": malformed_records,
        "num_unique": len(counts),
        "unique_fraction": len(counts) / len(records) if records else None,
        "rank_counts": dict(sorted(Counter(record.get("global_rank") for record in records).items()))
        if ranked_matches else None,
        "top_predictions": [
            {"prediction": prediction, "count": count, "fraction": count / len(records)}
            for prediction, count in counts.most_common(20)
        ],
        "metrics": metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions_all_ranks.jsonl").open("w") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
