"""Convert GPT2-aligned SG tags to QWEN3-aligned tags.

Reads JSON files from ``evaluation/SG/tokenized/`` (GPT2-aligned tags),
maps the critical-region tag masks to QWEN3 token positions via prefix-text
alignment, and writes the converted files to ``evaluation/SG/tokenized/qwen3/``.

Usage::

    python evaluation/SG/convert_tags_to_qwen3.py
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from olmo.tokenizer import Tokenizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

GPT2_TOKENIZER_PATH = "dataset/bbc-news/TG_GPT2_tokenizer.json"
QWEN3_TOKENIZER_PATH = "dataset/TG_QWEN3_tokenizer.json"
INPUT_DIR = "evaluation/SG/tokenized"
OUTPUT_DIR = "evaluation/SG/tokenized/qwen3"


def convert_tag(
    sent_input: str,
    gpt2_tag: list,
    gpt2_tok: Tokenizer,
    qwen3_tok: Tokenizer,
) -> list | None:
    """Convert a GPT2-aligned binary tag mask to QWEN3 token positions.

    The critical region text is recovered by decoding the GPT2 tokens
    where ``tag == 1``.  The same text span is then located in the QWEN3
    tokenisation by re-encoding the prefix (tokens before the first tagged
    token) and the prefix + critical region.

    Returns the QWEN3-aligned tag list, or ``None`` if there is no tagged
    region or the GPT2 token count mismatches the tag length.
    """
    text = " " + sent_input
    gpt2_tokens = gpt2_tok.encode(text, add_special_tokens=False)

    if len(gpt2_tag) != len(gpt2_tokens):
        log.warning(
            "Tag length mismatch for input %r: tag_len=%d, gpt2_len=%d",
            sent_input[:80],
            len(gpt2_tag),
            len(gpt2_tokens),
        )
        return None

    # Locate the contiguous block of 1s in the GPT2 tag mask.
    first_tagged: int | None = None
    last_tagged: int | None = None
    for i, x in enumerate(gpt2_tag):
        if x == 1:
            if first_tagged is None:
                first_tagged = i
            last_tagged = i

    if first_tagged is None:
        return None  # no critical region

    # Recover the prefix and prefix+critical text from GPT2 decoding.
    prefix_text = gpt2_tok.decode(gpt2_tokens[:first_tagged])
    prefix_plus_critical_text = gpt2_tok.decode(gpt2_tokens[: last_tagged + 1])

    # Locate the same span in the QWEN3 tokenisation.
    qwen3_tokens = qwen3_tok.encode(text, add_special_tokens=False)
    qwen3_prefix = qwen3_tok.encode(prefix_text, add_special_tokens=False)
    qwen3_full = qwen3_tok.encode(prefix_plus_critical_text, add_special_tokens=False)

    qwen3_start = len(qwen3_prefix)
    qwen3_end = len(qwen3_full)

    qwen3_tag = [0] * len(qwen3_tokens)
    for i in range(qwen3_start, qwen3_end):
        qwen3_tag[i] = 1

    return qwen3_tag


def main() -> None:
    gpt2_tok = Tokenizer.from_file(GPT2_TOKENIZER_PATH)
    qwen3_tok = Tokenizer.from_file(QWEN3_TOKENIZER_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    task_files = sorted(
        f for f in os.listdir(INPUT_DIR) if f.endswith(".json")
    )

    total_converted = 0
    total_skipped = 0
    for task_file in task_files:
        in_path = os.path.join(INPUT_DIR, task_file)
        out_path = os.path.join(OUTPUT_DIR, task_file)

        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        converted = 0
        skipped = 0
        for case in data["data"]:
            for sent in case:
                qwen3_tag = convert_tag(
                    sent["input"], sent["tag"][0], gpt2_tok, qwen3_tok
                )
                if qwen3_tag is not None:
                    sent["tag"] = [qwen3_tag]
                    converted += 1
                else:
                    skipped += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        total_converted += converted
        total_skipped += skipped
        log.info(
            "  %-35s  %4d converted  %4d skipped", task_file, converted, skipped
        )

    log.info(
        "Done. %d converted, %d skipped → %s",
        total_converted,
        total_skipped,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
