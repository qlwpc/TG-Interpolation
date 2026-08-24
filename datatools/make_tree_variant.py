"""Generate tree-linearization variant token streams from a LIN1 (tree) .npy.

Generalizes the dev-only prototype ``datatools/process_bbc.py`` into a
parameterized, streaming generator for the paper's causal-attention variants
(Table 1/3):

===============  ==============  ==================================================
variant          paper name      transform (over the LIN1 ``tree/*.npy`` stream)
===============  ==============  ==================================================
``noont``        Tree-NoONT      drop every opening non-terminal (ONT)
``compress``     Tree-Compress   collapse each run of closing NTs into ``<X)>``
``triplecnt``    Tree-TripleCNT  repeat every closing NT 3x (LIN3)
===============  ==============  ==================================================

Differences from the prototype: no hardcoded paths, handles ``train``-size
files via mmap streaming (two passes: length scan, then write), chunk
boundaries are aligned to closing-NT run starts so ``compress`` is exact,
optional multi-process chunk filling, and NT ranges are derived from the
tokenizer JSON instead of the compiled ``tg_mask`` module.

Note: only the flat token stream is transformed.  Sentence/document index
files (``*_sent_index.npy`` etc.) are NOT regenerated.

Usage::

    python -m datatools.make_tree_variant \
        --input dataset/bbc-news/tree/dev.npy \
        --output-dir dataset/bbc-news/tree_noont \
        --variant noont \
        --tokenizer dataset/bbc-news/TG_GPT2_tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VARIANTS = ("noont", "compress", "triplecnt")

# Default chunk: 2^28 tokens = 512 MB of uint16 — bounds RAM per worker.
DEFAULT_CHUNK_TOKENS = 2**28


# ---------------------------------------------------------------------------
# Non-terminal id ranges
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NTRanges:
    """Half-open id ranges of opening/closing non-terminal tokens."""

    open_lo: int
    open_hi: int  # exclusive
    close_lo: int
    close_hi: int  # exclusive
    generic_close: int  # id of the label-less merged closing token ``<X)>``

    def is_opening(self, arr: np.ndarray) -> np.ndarray:
        return (arr >= self.open_lo) & (arr < self.open_hi)

    def is_closing(self, arr: np.ndarray) -> np.ndarray:
        return (arr >= self.close_lo) & (arr < self.close_hi)


def nt_ranges_from_tokenizer(tokenizer_json: str | Path) -> NTRanges:
    """Derive NT id ranges from a TG tokenizer JSON's ``added_tokens``.

    Opening NTs are named ``<(S>``, closing NTs ``S)>``; the generic merged
    closing token is ``<X)>``.
    """
    with open(tokenizer_json) as f:
        data = json.load(f)

    added = data.get("added_tokens", [])
    open_ids = [t["id"] for t in added if t["content"].startswith("<(")]
    close_ids = [t["id"] for t in added if t["content"].endswith(")>")]
    if not open_ids or not close_ids:
        raise ValueError(f"no NT tokens found in {tokenizer_json}")

    generic = [t["id"] for t in added if t["content"] == "<X)>"]
    return NTRanges(
        open_lo=min(open_ids),
        open_hi=max(open_ids) + 1,
        close_lo=min(close_ids),
        close_hi=max(close_ids) + 1,
        generic_close=generic[0] if generic else max(close_ids),
    )


# ---------------------------------------------------------------------------
# Core transforms (pure, vectorized, per chunk)
# ---------------------------------------------------------------------------

def transform_tokens(
    tokens: np.ndarray,
    ranges: NTRanges,
    variant: str,
    fixed_token: int,
) -> np.ndarray:
    """Apply one variant to a chunk of LIN1 tokens.

    For ``compress`` the chunk must not start mid-run (guaranteed by
    :func:`chunk_bounds`); otherwise the split run would merge twice.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    if tokens.size == 0:
        return tokens

    if variant == "noont":
        return tokens[~ranges.is_opening(tokens)]

    is_close = ranges.is_closing(tokens)

    if variant == "triplecnt":
        return np.repeat(tokens, np.where(is_close, 3, 1))

    # compress: keep a closing NT iff it starts its run, then relabel it
    keep = ~is_close
    keep[0] = True
    keep[1:] |= ~is_close[:-1]
    out = tokens[keep].copy()
    out[ranges.is_closing(out)] = fixed_token
    return out


def chunk_bounds(tokens, chunk_size: int, ranges: NTRanges):
    """Yield ``(start, end)`` chunks that never split a closing-NT run.

    A boundary that lands mid-run is moved back to the run start; if the run
    begins at (or spans) ``start`` — i.e. it is longer than ``chunk_size`` —
    the boundary instead extends forward past the run end.  Either way the
    next chunk starts at a run edge, which keeps ``compress`` exact.

    ``tokens`` may be an ndarray or an mmap'd array; only scalar indexing is
    used for boundary adjustment, so nothing large is materialized.
    """
    n = len(tokens)

    def _is_close(i: int) -> bool:
        return bool(ranges.is_closing(tokens[i : i + 1])[0])

    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n and _is_close(end) and _is_close(end - 1):
            # boundary splits a closing run: move it to a run edge
            while end > start and _is_close(end - 1):
                end -= 1  # back to the run start
            if end == start:
                # run is longer than chunk_size: extend forward past its end
                end = min(start + chunk_size, n)
                while end < n and _is_close(end):
                    end += 1
        yield start, end
        start = end


# ---------------------------------------------------------------------------
# File-level streaming transform
# ---------------------------------------------------------------------------

def _process_chunk(
    src_path: str,
    start: int,
    end: int,
    out_path: str | None,
    out_start: int,
    ranges: NTRanges,
    variant: str,
    fixed_token: int,
) -> int:
    """Transform one chunk; write into the output memmap if given, else
    return the output length (length-scan mode)."""
    src = np.load(src_path, mmap_mode="r")
    out = transform_tokens(np.asarray(src[start:end]), ranges, variant, fixed_token)
    if out_path is not None:
        dst = np.load(out_path, mmap_mode="r+")
        dst[out_start : out_start + out.size] = out
        dst.flush()
    return int(out.size)


def transform_file(
    src: str | Path,
    dst: str | Path,
    ranges: NTRanges,
    variant: str,
    fixed_token: int,
    chunk_size: int = DEFAULT_CHUNK_TOKENS,
    workers: int = 1,
) -> Path:
    """Transform one .npy stream into ``dst`` (created, parent dirs included).

    Two passes: (1) per-chunk output lengths so the destination .npy can be
    pre-allocated; (2) transform + write.  With ``workers > 1`` both passes
    fan out over chunks; workers write disjoint slices of the destination
    memmap, so no locking is needed.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_mmap = np.load(src, mmap_mode="r")
    bounds = list(chunk_bounds(src_mmap, chunk_size, ranges))

    def _lengths() -> list[int]:
        if workers <= 1:
            return [
                _process_chunk(str(src), s, e, None, 0, ranges, variant, fixed_token)
                for s, e in bounds
            ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(_process_chunk, str(src), s, e, None, 0, ranges, variant, fixed_token)
                for s, e in bounds
            ]
            return [f.result() for f in futs]

    lengths = _lengths()
    out_len = int(sum(lengths))

    # pre-allocate destination .npy with the exact output shape
    header = np.lib.format.open_memmap(dst, mode="w+", dtype=src_mmap.dtype, shape=(out_len,))
    header.flush()
    del header

    offsets = []
    pos = 0
    for (s, e), length in zip(bounds, lengths):
        offsets.append((s, e, pos))
        pos += length

    if workers <= 1:
        for s, e, out_start in offsets:
            _process_chunk(str(src), s, e, str(dst), out_start, ranges, variant, fixed_token)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    _process_chunk, str(src), s, e, str(dst), out_start, ranges, variant, fixed_token
                )
                for s, e, out_start in offsets
            ]
            for f in futs:
                f.result()

    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", nargs="+", required=True, help="LIN1 tree .npy stream(s)")
    parser.add_argument("--output-dir", required=True, help="e.g. dataset/bbc-news/tree_noont")
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--tokenizer", required=True, help="TG tokenizer JSON (derives NT ranges)")
    parser.add_argument(
        "--fixed-token", type=int, default=None,
        help="generic closing id for compress (default: <X)> from tokenizer)",
    )
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    ranges = nt_ranges_from_tokenizer(args.tokenizer)
    fixed_token = args.fixed_token if args.fixed_token is not None else ranges.generic_close

    out_dir = Path(args.output_dir)
    for src in args.input:
        dst = transform_file(
            src,
            out_dir / Path(src).name,
            ranges,
            args.variant,
            fixed_token,
            chunk_size=args.chunk_tokens,
            workers=args.workers,
        )
        n_in = np.load(src, mmap_mode="r").size
        n_out = np.load(dst, mmap_mode="r").size
        print(f"{src} -> {dst}  ({n_in:,} -> {n_out:,} tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
