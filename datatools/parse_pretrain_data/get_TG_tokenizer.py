"""Build the GPT-2/Qwen3 tokenizer with the paper's structural tokens."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from tokenizers import AddedToken, Tokenizer

MODEL_NAMES = {
    "gpt2": "gpt2",
    "qwen3": "Qwen/Qwen3-0.6B",
    "olmo-1B": "allenai/OLMo-1B-0724-hf",
}
PAPER_LAYOUTS = {
    "gpt2": {"vocab_size": 50320, "sep_token_id": 50261},
    "qwen3": {"vocab_size": 151732, "sep_token_id": 151673},
}
NON_TERMINALS = (
    "ADJP", "ADVP", "CONJP", "FRAG", "INTJ", "LST", "NAC", "NML", "NP",
    "PP", "PRN", "PRT", "QP", "RRC", "S", "SBAR", "SBARQ", "SINV", "SQ",
    "UCP", "VP", "WHADJP", "WHADVP", "WHNP", "WHPP", "X",
)


def token(content: str, *, special: bool) -> AddedToken:
    return AddedToken(
        content,
        single_word=False,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=special,
    )


def build_tokenizer(model_name: str) -> Tokenizer:
    tokenizer = Tokenizer.from_pretrained(MODEL_NAMES[model_name])
    tokenizer.add_special_tokens(
        [token("<|beginoftext|>", special=True), token("<|pad|>", special=True)]
    )
    tokenizer.add_special_tokens(
        [
            token("<|SUM|>", special=True),
            token("<|CLS|>", special=True),
            token("<|SEP|>", special=True),
        ]
    )
    tokenizer.add_tokens(
        [
            token(name, special=False)
            for name in ("-LRB-", "-RRB-", "-LCB-", "-RCB-", "-LSB-", "-RSB-")
        ]
    )
    for non_terminal in NON_TERMINALS:
        tokenizer.add_special_tokens([token(f"<({non_terminal}>", special=True)])
    for non_terminal in NON_TERMINALS:
        tokenizer.add_special_tokens([token(f"<{non_terminal})>", special=True)])
    return tokenizer


def validate_paper_layout(model_name: str, tokenizer: Tokenizer) -> None:
    expected = PAPER_LAYOUTS.get(model_name)
    if expected is None:
        return
    actual = {
        "vocab_size": tokenizer.get_vocab_size(),
        "sep_token_id": tokenizer.token_to_id("<|SEP|>"),
    }
    if actual != expected:
        raise ValueError(
            f"{model_name} tokenizer layout drifted: {actual}, expected {expected}"
        )


def default_output(model_name: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    if model_name == "gpt2":
        return root / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    return root / f"dataset/TG_{model_name.upper()}_tokenizer.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name", "--model_name", dest="model_name",
        choices=sorted(MODEL_NAMES), default="gpt2",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    output = (args.output or default_output(args.model_name)).resolve()
    if output.exists() and not args.overwrite:
        tokenizer = Tokenizer.from_file(str(output))
        validate_paper_layout(args.model_name, tokenizer)
        print(f"reused existing {args.model_name} tokenizer -> {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(args.model_name)
    validate_paper_layout(args.model_name, tokenizer)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        tokenizer.save(str(temporary))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"saved {args.model_name} tokenizer ({tokenizer.get_vocab_size()} tokens, "
        f"SEP={tokenizer.token_to_id('<|SEP|>')}) -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
