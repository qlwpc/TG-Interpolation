from nltk import Tree
import argparse
from tokenizers import Tokenizer
import re
import numpy as np
import os
from joblib import Parallel, delayed
import tempfile
from contextlib import ExitStack
from pathlib import Path

from datatools.parse_pretrain_data.pipeline_io import atomic_json, sha256_file
from datatools.parse_pretrain_data.shard_integrity import (
    FORMATS, TOKENIZATION_PROTOCOL, receipt_path, verify_receipt,
)

def _is_qwen3_style_tokenizer(tokenizer) -> bool:
    """Check if tokenizer has bracket mapping enabled.

    Reads the ``use_bracket_mapping`` flag from the tokenizer object
    if available, defaults to False otherwise (safe for GPT-2).
    """
    if hasattr(tokenizer, 'use_bracket_mapping'):
        return bool(tokenizer.use_bracket_mapping)
    return False


def pformat_flat(self, nodesep="", parens="()", quotes=False,
                 use_bracket_mapping=True):
    childstrs = []
    for child in self:
        if isinstance(child, Tree):
            childstrs.append(pformat_flat(child, nodesep, parens, quotes,
                                          use_bracket_mapping=use_bracket_mapping))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            if use_bracket_mapping:
                mapping = {
                    "-LRB-": "(",
                    "-RRB-": ")",
                    "-LCB-": "{",
                    "-RCB-": "}",
                    "-LSB-": "[",
                    "-RSB-": "]",
                    "Ċ": "\n",
                }
                out = mapping[child] if child in mapping else child
            else:
                out = child
            return " " + out
        else:
            childstrs.append(repr(child))
    if isinstance(self._label, str):
        if self._label == "qlwpcRegen":
            return "".join(childstrs)
        else:
            return "<{}{}{}>{}<{}{}>".format(
                parens[0],
                self._label,
                nodesep,
                "".join(childstrs),
                self._label,
                parens[1],
            )


def convert_TG_format(input: str, use_bracket_mapping: bool = True, *, strict: bool = False) -> str:
    line = "(qlwpcRegen " + input.strip() + ")"
    try:
        tree = Tree.fromstring(line, remove_empty_top_bracketing=False)
        if strict and (not tree.leaves() or any(not isinstance(child, Tree) for child in tree)):
            raise ValueError("empty or non-tree parsed document")
        outputstr = pformat_flat(tree, use_bracket_mapping=use_bracket_mapping)
        if strict and not outputstr:
            raise ValueError("parsed document produced an empty tree stream")
    except Exception as e:
        if strict:
            raise ValueError("invalid parsed document") from e
        print("error occurs when processing data: ")
        print(line)
        print(e)
        outputstr = ""
    return outputstr

def encode_tree_document(text, tokenizer, vocab, dtype):
    """Encode one parsed document with explicit, format-stable boundaries."""
    token_ids = [vocab.bos, *tokenizer.encode(text).ids, vocab.eos]
    return np.asarray(token_ids, dtype=dtype)

def tokenize_shard(source: Path, output_root: Path, tokenizer, vocab,
                   tokenizer_sha256: str, *, overwrite: bool = False) -> dict:
    """Stream one document at a time, then publish three arrays and a receipt.

    A receipt is the completion marker. Interrupted or legacy outputs without
    one are never silently reused. Scratch space is bounded by one shard, not
    by corpus size; RAM is bounded by a single document plus the copy buffer.
    """
    dtype = np.dtype("uint16" if tokenizer.get_vocab_size() < 65536 else "uint32")
    source_hash = sha256_file(source)
    outputs = {fmt: output_root / fmt / f"{source.stem}.npy" for fmt in FORMATS}
    receipt = receipt_path(output_root, source.stem)
    if not overwrite and (receipt.exists() or any(p.exists() for p in outputs.values())):
        result = verify_receipt(output_root, source.stem, tokenizer_sha256, source_hash)
        print(f"verified existing shard: {source.stem}", flush=True)
        return result
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(FORMATS, 0)
    documents = 0
    with tempfile.TemporaryDirectory(prefix=f".tokenize-{source.stem}-", dir=output_root) as scratch_name:
        scratch = Path(scratch_name)
        with ExitStack() as stack:
            handles = {fmt: stack.enter_context((scratch / f"{fmt}.bin").open("wb")) for fmt in FORMATS}
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    try:
                        if not line.endswith("\n"):
                            raise ValueError("unterminated parsed row (possibly interrupted parser)")
                        text = convert_TG_format(line, strict=True)
                        tree = encode_tree_document(text, tokenizer, vocab, dtype)
                        arrays = {"tree": tree,
                                  "terminal": vocab.convert_treenpy_to_terminal(tree),
                                  "tg": vocab.convert_treenpy_to_TG(tree)}
                        for fmt, array in arrays.items():
                            if (array.ndim != 1 or array.size < 2 or array[0] != vocab.bos or
                                    array[-1] != vocab.eos or np.count_nonzero(array == vocab.bos) != 1 or
                                    np.count_nonzero(array == vocab.eos) != 1):
                                raise ValueError(f"invalid or embedded document boundaries in {fmt}")
                            array.astype(dtype, copy=False).tofile(handles[fmt])
                            counts[fmt] += int(array.size)
                        documents += 1
                    except Exception as exc:
                        raise ValueError(f"{source}:{line_number}: tokenization failed; no rows may be skipped") from exc
        if not documents:
            raise ValueError(f"{source}: empty parsed shard")
        if sha256_file(source) != source_hash:
            raise ValueError(f"parsed source changed while tokenizing: {source}")
        result = {"schema_version": 1, "protocol": TOKENIZATION_PROTOCOL,
                  "source": str(source.resolve()), "source_sha256": source_hash,
                  "tokenizer_sha256": tokenizer_sha256, "dtype": str(dtype),
                  "documents": documents, "formats": {}}
        # Stage all three standard NPY files before replacing any final output.
        for fmt in FORMATS:
            staged = scratch / f"{fmt}.npy"
            dest = np.lib.format.open_memmap(staged, mode="w+", dtype=dtype, shape=(counts[fmt],))
            raw = np.memmap(scratch / f"{fmt}.bin", mode="r", dtype=dtype)
            for start in range(0, counts[fmt], 4 * 1024 * 1024):
                dest[start:start + 4 * 1024 * 1024] = raw[start:start + 4 * 1024 * 1024]
            dest.flush()
            del dest, raw
            result["formats"][fmt] = {"tokens": counts[fmt], "sha256": sha256_file(staged)}
        # Invalidate an old receipt only once replacement starts. On failure,
        # validation refuses the incomplete group instead of marking it done.
        receipt.unlink(missing_ok=True)
        for fmt, output in outputs.items():
            os.replace(scratch / f"{fmt}.npy", output)
        atomic_json(receipt, result)
    print(f"tokenized {source.stem}: {documents} documents", flush=True)
    return result


def _tokenize_worker(source: Path, output: Path, tokenizer_path: Path, overwrite: bool):
    from olmo.data.tg_mask import SentencepieceVocab

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = SentencepieceVocab.from_vocab_file(str(tokenizer_path))
    return tokenize_shard(source, output, tokenizer, vocab, sha256_file(tokenizer_path), overwrite=overwrite)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Strict, resumable, memory-bounded three-format tokenization")
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="parallel shards; each worker needs scratch disk for one shard")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    sources = sorted(args.input_dir.glob("*.txt"))
    if not sources:
        raise FileNotFoundError(f"no parsed .txt shards in {args.input_dir}")
    Parallel(n_jobs=args.jobs)(delayed(_tokenize_worker)(source, args.output_dir, args.tokenizer, args.overwrite)
                              for source in sources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
