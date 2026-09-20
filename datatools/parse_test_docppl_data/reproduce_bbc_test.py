#!/usr/bin/env python3
"""Reproduce BBC dev/test streams and audit archived test versions.

The ``extract`` command is stdlib-only and can run through ``ssh ... python -``.
It emits selected raw rows as JSONL, preserving the released JSON index order.
See datatools/parse_test_docppl_data/README.md for reconstruction commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def load_indices(dev_path, test_path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate shard {key}")
            result[key] = value
        return result

    indices = [json.loads(p.read_text(), object_pairs_hook=unique)
               for p in (dev_path, test_path)]
    for index in indices:
        for shard, rows in index.items():
            if Path(shard).name != shard or shard in (".", ".."):
                raise ValueError(f"invalid shard name {shard}")
            if (not isinstance(rows, list) or
                    any(type(i) is not int or i < 0 for i in rows) or
                    len(rows) != len(set(rows))):
                raise ValueError(f"invalid/duplicate row indices in {shard}")
    dev, test = indices
    for shard in set(dev) | set(test):
        if set(dev.get(shard, [])) & set(test.get(shard, [])):
            raise ValueError(f"dev/test overlap in {shard}")
    return dev, test


def extract(args):
    dev, test = load_indices(args.dev_index, args.test_index)
    split = getattr(args, "split", "test")
    index = {"dev": dev, "test": test}[split]
    # Fail before emitting any rows if a required source is missing.
    for shard in index:
        if not (args.parsed_dir / f"{shard}.txt").is_file():
            raise FileNotFoundError(args.parsed_dir / f"{shard}.txt")
    emit({"type": "header", "schema_version": 1, "split": split,
          "parsed_dir": str(args.parsed_dir.resolve()),
          "dev_sha256": sha256_file(args.dev_index),
          "test_sha256": sha256_file(args.test_index),
          "documents": sum(map(len, index.values())),
          "order": f"{split} JSON shard insertion order, then list order"})
    doc_id = 0
    for shard, selection in index.items():
        source = args.parsed_dir / f"{shard}.txt"
        before = source.stat()
        selected = set(selection)
        required = set(dev.get(shard, [])) | set(test.get(shard, []))
        max_required = max(required, default=-1)
        found = {}
        scanned = 0
        with source.open("rb") as handle:
            for row, raw in enumerate(handle):
                scanned = row + 1
                if row in selected:
                    if not raw.endswith(b"\n"):
                        raise ValueError(f"unterminated parsed row: {source}:{row+1}")
                    found[row] = raw
                if required and row >= max_required:
                    break
        if required and scanned <= max(required):
            raise IndexError(f"out-of-range split index: {source}, scanned {scanned}")
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"source changed during extraction: {source}")
        for row in selection:
            raw = found[row]
            emit({"type": "document", "doc_id": doc_id, "shard": shard,
                  "source_row": row, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                  "parsed": raw.decode("utf-8")})
            doc_id += 1
        emit({"type": "source", "shard": shard, "size_bytes": before.st_size,
              "mtime_ns": before.st_mtime_ns, "rows_scanned": scanned,
              "selected_documents": len(selection)})
        print(f"extracted {shard}: {len(selection)} documents ({doc_id} total)",
              file=sys.stderr, flush=True)
    emit({"type": "complete", "documents": doc_id})


def selection_split(path):
    with path.open(encoding="utf-8") as handle:
        header = json.loads(next(handle))
    if header.get("type") != "header" or header.get("schema_version") != 1:
        raise ValueError("invalid extraction header")
    # Extraction files from the original test-only script have no split field.
    split = header.get("split", "test")
    if split not in ("dev", "test"):
        raise ValueError(f"invalid extraction split: {split}")
    return split


def selected_rows(path, expected_split=None):
    header = None
    count = 0
    complete = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if complete:
                raise ValueError("records after extraction completion marker")
            if record["type"] == "header":
                if header is not None or count:
                    raise ValueError("duplicate/misplaced extraction header")
                header = record
                split = record.get("split", "test")
                if split not in ("dev", "test") or (expected_split and split != expected_split):
                    raise ValueError(f"extraction split {split} does not match {expected_split}")
            elif record["type"] == "document":
                if header is None or record["doc_id"] != count:
                    raise ValueError("missing header or nonsequential document ID")
                raw = record["parsed"].encode("utf-8")
                if hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
                    raise ValueError(f"parsed row hash mismatch at {count}")
                yield record
                count += 1
            elif record["type"] == "complete":
                if header is None or count != header["documents"] or count != record["documents"]:
                    raise ValueError("incomplete extracted corpus")
                complete = True
            elif record["type"] != "source":
                raise ValueError(f"unknown extraction record {record['type']}")
    if not complete:
        raise ValueError("missing extraction completion marker")


def legacy_format(parsed, vocab):
    """Explicit compatibility with the archived GPT-2 stream's serialization.

    PTB brackets are bare added tokens, newline preterminals become ' \n',
    known NTs use added tokens, and unknown labels retain historical literals.
    This policy is validated against the archived arrays, not inferred from
    whichever converter happens to be installed at execution time.
    """
    from nltk import Tree

    tree = Tree.fromstring("(qlwpcRegen " + parsed.strip() + ")")

    def render(node):
        if not isinstance(node, Tree) or not node:
            raise ValueError("empty/non-tree parsed node")
        if isinstance(node[0], str):
            if len(node) != 1:
                raise ValueError("preterminal must have exactly one leaf")
            leaf = node[0]
            return " " + ("\n" if leaf == "Ċ" else leaf)
        content = "".join(render(child) for child in node)
        label = node.label()
        if label == "qlwpcRegen":
            return content
        if f"<({label}>" in vocab and f"<{label})>" in vocab:
            return f"<({label}>{content}<{label})>"
        return f" ({label}{content} {label})"

    return render(tree)


def legacy_tokenizer(tokenizer):
    """Restore the six PTB AddedToken rules from datatools/TG_GPT2_tokenizer.json.

    Their IDs are unchanged. Single-word matching matters for leaves such as
    '8-RRB-' and '-LRB-Nigerian'; lstrip matters for '-RRB-:'.
    """
    from tokenizers import Tokenizer

    config = json.loads(tokenizer.to_str())
    for token in config["added_tokens"]:
        if 50262 <= token["id"] <= 50267:
            token.update(single_word=True, lstrip=True, normalized=True)
    return Tokenizer.from_str(json.dumps(config))


def normalize_adj_labels(parsed):
    """Map ADJ constituents to ADJP before encoding; preserve all text leaves."""
    from nltk import Tree

    document = Tree.fromstring("(qlwpcRegen " + parsed.strip() + ")")
    for node in document.subtrees():
        if node.label() == "ADJ" and any(isinstance(child, Tree) for child in node):
            node.set_label("ADJP")
    return " ".join(child.pformat(margin=sys.maxsize) for child in document)


def build_raw(args):
    import numpy as np
    from tokenizers import Tokenizer

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from datatools.parse_pretrain_data.convert_TG_and_tokenize import convert_TG_format

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    validate_tokenizer(tokenizer)
    if args.encoding == "legacy":
        tokenizer = legacy_tokenizer(tokenizer)
    if args.output_root.exists():
        raise FileExistsError(f"use a new output directory: {args.output_root}")
    split = selection_split(args.selected)
    if getattr(args, "split", None) and args.split != split:
        raise ValueError(f"requested split {args.split} differs from extraction split {split}")
    # Validate the completion marker and all row hashes before publication.
    rows = list(selected_rows(args.selected, expected_split=split))
    tree_parts = []
    vocab = tokenizer.get_vocab()
    normalize_adj = getattr(args, "normalize_adj", False)
    for row in rows:
        parsed = normalize_adj_labels(row["parsed"]) if normalize_adj else row["parsed"]
        text = (convert_TG_format(parsed, strict=True) if args.encoding == "pipeline"
                else legacy_format(parsed, vocab))
        tree_parts.append(np.asarray([50257, *tokenizer.encode(text).ids, 50256], dtype=np.uint16))
    tree = np.concatenate(tree_parts)
    opens = [i for t, i in vocab.items() if t.startswith("<(") and t.endswith(">")]
    closes = [i for t, i in vocab.items() if t.startswith("<") and t.endswith(")>")]
    nt = np.isin(tree, opens + closes)
    close = np.isin(tree, closes)
    arrays = {"tree": tree, "terminal": tree[~nt], "tg": np.repeat(tree, 1 + close)}
    args.output_root.mkdir(parents=True)
    manifest = {"split": split, "encoding": args.encoding, "selected_sha256": sha256_file(args.selected),
                "label_normalization": {"ADJ": "ADJP"} if normalize_adj else {},
                "tokenizer_sha256": sha256_file(args.tokenizer),
                "effective_tokenizer_sha256": hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
                "converter_sha256": sha256_file(repo / "datatools/parse_pretrain_data/convert_TG_and_tokenize.py"),
                "reproduction_script_sha256": sha256_file(Path(__file__)),
                "documents": len(rows), "outputs": {}}
    for fmt, array in arrays.items():
        target = args.output_root / fmt / f"{split}.npy"
        target.parent.mkdir()
        np.save(target, array)
        manifest["outputs"][fmt] = {"tokens": len(array), "sha256": sha256_file(target)}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def validate_tokenizer(tokenizer):
    expected = {"<|beginoftext|>": 50257, "<|endoftext|>": 50256, "<|pad|>": 50258}
    if tokenizer.get_vocab_size() >= 65536 or any(tokenizer.token_to_id(t) != i for t, i in expected.items()):
        raise ValueError("this reproduction recipe requires the BBC GPT-2 tokenizer layout")


def content_with_ids(block, opens, closes):
    import numpy as np

    opening, closing = np.isin(block, opens), np.isin(block, closes)
    depth = np.cumsum(opening.astype(np.int32) - closing.astype(np.int32))
    if len(depth) and (depth[-1] or depth.min() < 0):
        raise ValueError("unbalanced tree brackets")
    content_mask = (depth > 0) & ~opening & ~closing
    if np.any(np.isin(block[content_mask], [50256, 50257, 50258])):
        raise ValueError("document special token inside a tree")
    starts = np.flatnonzero(opening & (depth == 1))
    ends = np.flatnonzero(closing & (depth == 0)) + 1
    sentences = [np.asarray(block[s:e][content_mask[s:e]]) for s, e in zip(starts, ends)]
    if any(len(s) == 0 for s in sentences):
        raise ValueError("empty tree terminals")
    return np.asarray(block[content_mask]), sentences


def read_documents(path, opens, closes):
    import numpy as np

    stream = np.load(path, mmap_mode="r", allow_pickle=False)
    if stream.ndim != 1 or stream.dtype != np.uint16:
        raise ValueError(f"expected flat uint16 stream: {path}")
    starts, ends = np.flatnonzero(stream == 50257), np.flatnonzero(stream == 50256) + 1
    if (not len(starts) or len(starts) != len(ends) or starts[0] != 0 or
            ends[-1] != len(stream) or not np.array_equal(starts[1:], ends[:-1]) or
            np.any(ends <= starts)):
        raise ValueError(f"invalid document boundaries: {path}")
    return [make_document(stream[s:e], opens, closes) for s, e in zip(starts, ends)]


def make_document(block, opens, closes):
    import numpy as np

    terminals, sentences = content_with_ids(block, opens, closes)
    return {"tree": np.asarray(block), "terminals": terminals, "sentences": sentences,
            "hash": hashlib.sha256(terminals.astype("<u2", copy=False).tobytes()).hexdigest()}


def read_ppl(path, opens, closes):
    import numpy as np

    stream = np.load(path / "tree_300.npy", mmap_mode="r", allow_pickle=False)
    lengths = np.load(path / "tree_sent_index.npy", mmap_mode="r", allow_pickle=False)
    counts = np.load(path / "tree_doc_index.npy", allow_pickle=False)
    if lengths.ndim != 1 or lengths.size % 300 or counts.ndim != 1 or np.any(counts == 0):
        raise ValueError("invalid tree300 sentence/document index shape")
    lengths = lengths.reshape(-1, 300)
    if np.any(lengths == 0) or int(counts.sum()) != len(lengths):
        raise ValueError("empty candidate or sentence/document index disagreement")
    offsets = np.concatenate(([0], np.cumsum(lengths.sum(axis=1, dtype=np.int64))))
    if offsets[-1] != len(stream):
        raise ValueError("candidate lengths do not cover tree_300.npy")
    docs = []
    cursor = 0
    for count in counts:
        blocks = []
        for sentence in range(cursor, cursor + int(count)):
            start = int(offsets[sentence])
            block = stream[start:start + int(lengths[sentence, 0])]
            _, roots = content_with_ids(block, opens, closes)
            if len(roots) != 1:
                raise ValueError(f"candidate zero has {len(roots)} roots at sentence {sentence}")
            blocks.append(block)
        docs.append(make_document(np.concatenate(blocks), opens, closes))
        cursor += int(count)
    return docs


def difference(left, right, tokenizer):
    import difflib

    a, b = left["terminals"].tolist(), right["terminals"].tolist()
    opcodes = []
    for tag, i, j, k, l in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        opcodes.append({"operation": tag, "left_span": [i, j], "right_span": [k, l],
                        "left_ids": a[i:j], "right_ids": b[k:l],
                        "left_text": tokenizer.decode(a[i:j]),
                        "right_text": tokenizer.decode(b[k:l])})
    return {"left_tokens": len(a), "right_tokens": len(b),
            "left_sentences": len(left["sentences"]), "right_sentences": len(right["sentences"]),
            "left_preview": tokenizer.decode(a[:60]), "right_preview": tokenizer.decode(b[:60]),
            "token_edits": opcodes}


def compare_documents(left, right, tokenizer, *, offset=None):
    import numpy as np

    lookup = defaultdict(list)
    for i, doc in enumerate(left):
        lookup[doc["hash"]].append(i)
    votes = Counter(i - j for j, doc in enumerate(right) for i in lookup[doc["hash"]])
    if offset is None:
        if not votes:
            raise ValueError("no exact terminal matches from which to infer an offset")
        offset = votes.most_common(1)[0][0]
    result = {"left_documents": len(left), "right_documents": len(right),
              "offset_left_minus_right": offset, "offset_votes": votes.most_common(5),
              "terminal_equal_documents": 0, "sentence_equal_documents": 0,
              "full_terminal_projection_equal_documents": 0,
              "tree_equal_documents": 0, "differences": []}
    nt_ids = [i for t, i in tokenizer.get_vocab().items()
              if t.startswith("<(") or t.endswith(")>")]
    for j, rdoc in enumerate(right):
        i = j + offset
        if not 0 <= i < len(left):
            raise ValueError(f"mapped document outside source: {i}")
        ldoc = left[i]
        # This second definition retains BOS/EOS, external whitespace and any
        # external ordinary leaves, exactly as terminal/test.npy does.
        result["full_terminal_projection_equal_documents"] += int(np.array_equal(
            ldoc["tree"][~np.isin(ldoc["tree"], nt_ids)],
            rdoc["tree"][~np.isin(rdoc["tree"], nt_ids)]))
        result["tree_equal_documents"] += int(np.array_equal(ldoc["tree"], rdoc["tree"]))
        result["sentence_equal_documents"] += int(
            len(ldoc["sentences"]) == len(rdoc["sentences"]) and
            all(np.array_equal(a, b) for a, b in zip(ldoc["sentences"], rdoc["sentences"])))
        if np.array_equal(ldoc["terminals"], rdoc["terminals"]):
            result["terminal_equal_documents"] += 1
        else:
            result["differences"].append({"left_doc": i, "right_doc": j,
                "exact_matches_elsewhere_in_left": lookup[rdoc["hash"]],
                **difference(ldoc, rdoc, tokenizer)})
    return result


def audit(args):
    from tokenizers import Tokenizer

    if args.output_root.exists():
        raise FileExistsError(f"use a new output directory: {args.output_root}")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    validate_tokenizer(tokenizer)
    vocab = tokenizer.get_vocab()
    opens = [i for t, i in vocab.items() if t.startswith("<(") and t.endswith(">")]
    closes = [i for t, i in vocab.items() if t.startswith("<") and t.endswith(")>")]
    paths = {"old": args.bbc_root / "tree/test.pre_testppl_alignment.npy",
             "current": args.bbc_root / "tree/test.npy"}
    if args.raw_root:
        paths["raw"] = args.raw_root / "tree/test.npy"
    if args.legacy_root:
        paths["legacy"] = args.legacy_root / "tree/test.npy"
    corpora = {}
    inputs = {"tokenizer_sha256": sha256_file(args.tokenizer)}
    for name, path in paths.items():
        print(f"reading {name}: {path}", flush=True)
        corpora[name] = read_documents(path, opens, closes)
        inputs[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    print(f"reading candidate zero: {args.ppl_dir}", flush=True)
    corpora["ppl"] = read_ppl(args.ppl_dir, opens, closes)
    inputs["ppl"] = {}
    for filename in ("tree_300.npy", "tree_sent_index.npy", "tree_doc_index.npy"):
        path = args.ppl_dir / filename
        inputs["ppl"][filename] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    results = {"claim_type": "computed", "inputs": inputs, "corpora": {}, "comparisons": {}}
    for name, docs in corpora.items():
        results["corpora"][name] = {"documents": len(docs),
            "sentences": sum(len(d["sentences"]) for d in docs),
            "content_terminal_tokens": sum(len(d["terminals"]) for d in docs),
            "serialized_tree_tokens": sum(len(d["tree"]) for d in docs)}
    pairs = [("old", "current"), ("current", "ppl")]
    if "raw" in corpora:
        pairs += [("raw", "old"), ("raw", "current")]
    if "legacy" in corpora:
        pairs += [("legacy", "old"), ("legacy", "current")]
    for left, right in pairs:
        result = compare_documents(corpora[left], corpora[right], tokenizer)
        results["comparisons"][f"{left}_vs_{right}"] = result
        print(f"{left} vs {right}: {result['terminal_equal_documents']}/{len(corpora[right])} "
              f"terminal-equal, offset {result['offset_left_minus_right']}", flush=True)
    args.output_root.mkdir(parents=True)
    (args.output_root / "comparison.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    with (args.output_root / "document_fingerprints.jsonl").open("w") as handle:
        for name, docs in corpora.items():
            for i, doc in enumerate(docs):
                handle.write(json.dumps({"corpus": name, "doc_id": i,
                    "terminal_sha256": doc["hash"], "terminal_tokens": len(doc["terminals"]),
                    "sentence_terminal_lengths": list(map(len, doc["sentences"]))}) + "\n")


def build_current(args):
    """Use canonical candidate zero including its external whitespace."""
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from datatools.parse_test_docppl_data.build_aligned_test import build

    if args.output_root.exists():
        raise FileExistsError(f"use a new output directory: {args.output_root}")
    build(argparse.Namespace(input_dir=args.ppl_dir, tokenizer=args.tokenizer,
                             output_root=args.output_root, overwrite=False))


def verify_raw(args):
    """Verify a dev/test rebuild against archived arrays, with durable results."""
    import numpy as np

    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    manifest = json.loads((args.rebuilt_root / "manifest.json").read_text())
    if manifest.get("split", "test") != args.split:
        raise ValueError("rebuilt manifest split does not match requested verification")
    report = {"claim_type": "computed", "split": args.split,
              "rebuild_manifest": str((args.rebuilt_root / "manifest.json").resolve()),
              "documents": manifest["documents"], "files": {}, "passed": True}
    for fmt in ("tree", "terminal", "tg"):
        actual = args.rebuilt_root / fmt / f"{args.split}.npy"
        expected = args.reference_root / fmt / f"{args.split}.npy"
        rebuilt_docs = split_tree_stream(actual)
        reference_docs = split_tree_stream(expected)
        differences = [i for i in range(max(len(rebuilt_docs), len(reference_docs)))
                       if i >= len(rebuilt_docs) or i >= len(reference_docs)
                       or not np.array_equal(rebuilt_docs[i], reference_docs[i])]
        digest, expected_digest = sha256_file(actual), sha256_file(expected)
        passed = (digest == expected_digest == manifest["outputs"][fmt]["sha256"]
                  and len(rebuilt_docs) == len(reference_docs) == manifest["documents"]
                  and not differences)
        report["files"][fmt] = {"rebuilt": str(actual.resolve()), "reference": str(expected.resolve()),
            "tokens": sum(map(len, rebuilt_docs)), "reference_tokens": sum(map(len, reference_docs)),
            "sha256": digest, "reference_sha256": expected_digest,
            "rebuilt_documents": len(rebuilt_docs), "reference_documents": len(reference_docs),
            "different_document_ids": differences, "passed": passed}
        report["passed"] &= passed
        print(f"{fmt}/{args.split}.npy: {'PASS' if passed else 'FAIL'}, "
              f"{len(rebuilt_docs)} documents, {len(differences)} differing documents", flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    if not report["passed"]:
        raise ValueError(f"rebuild differs from reference; see {args.output_json}")


def split_tree_stream(path):
    import numpy as np

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    starts = np.flatnonzero(array == 50257)
    ends = np.flatnonzero(array == 50256) + 1
    if (array.ndim != 1 or array.dtype != np.uint16 or not len(starts) or
            len(starts) != len(ends) or starts[0] != 0 or ends[-1] != len(array) or
            not np.array_equal(starts[1:], ends[:-1]) or np.any(ends <= starts)):
        raise ValueError(f"invalid BBC tree stream: {path}")
    return [array[a:b] for a, b in zip(starts, ends)]


def apply_edits(source, edits):
    """Edits use immutable source offsets, never offsets after previous edits."""
    import numpy as np

    pieces = []
    cursor = 0
    for edit in edits:
        start, end = edit["source_span"]
        values = edit["insert_ids"]
        if not cursor <= start <= end <= len(source):
            raise ValueError("overlapping/out-of-range recipe edits")
        if any(type(i) is not int or not 0 <= i <= 65535 for i in values):
            raise ValueError("invalid recipe token ID")
        pieces.extend((source[cursor:start], np.asarray(values, dtype=np.uint16)))
        cursor = end
    pieces.append(source[cursor:])
    return np.concatenate(pieces)


def freeze_recipe(args):
    """Freeze a transparent delta; content replacements remain explicit data."""
    import difflib
    import numpy as np

    if args.recipe.exists():
        raise FileExistsError(args.recipe)
    source_path = args.legacy_root / "tree/test.npy"
    source = split_tree_stream(source_path)
    target = split_tree_stream(args.target_root / "tree/test.npy")
    if len(source) - len(target) != args.offset:
        raise ValueError("recipe offset must select the complete source suffix")
    patches = []
    for j, values in enumerate(target):
        original = source[j + args.offset]
        if np.array_equal(original, values):
            continue
        edits = [{"source_span": [a, b], "insert_ids": values[c:d].tolist()}
                 for tag, a, b, c, d in difflib.SequenceMatcher(
                     None, original.tolist(), values.tolist(), autojunk=False).get_opcodes()
                 if tag != "equal"]
        if not np.array_equal(apply_edits(original, edits), values):
            raise AssertionError(f"recipe replay failed at document {j}")
        patches.append({"target_doc": j, "source_doc": j + args.offset, "edits": edits})
    expected = {}
    for fmt in ("tree", "terminal", "tg"):
        expected[fmt] = {name: sha256_file(args.target_root / fmt / name)
                        for name in ("test.npy", "test_doc_index.npy", "test_sent_index.npy")}
    recipe = {"schema_version": 1, "source_encoding": "legacy",
              "source_tree_sha256": sha256_file(source_path),
              "target_documents": len(target), "offset": args.offset,
              "provenance": "Observed delta to canonical tree300 candidate zero; not an inferred cleaning rule",
              "patches": patches, "expected_sha256": expected,
              "tree_sentence_lengths": np.load(args.target_root / "tree/test_sent_index.npy").tolist(),
              "document_sentence_counts": np.load(args.target_root / "tree/test_doc_index.npy").tolist()}
    args.recipe.parent.mkdir(parents=True, exist_ok=True)
    with args.recipe.open("x") as handle:
        json.dump(recipe, handle, separators=(",", ":"))
        handle.write("\n")
    print(f"froze {len(patches)} document patches: {args.recipe}")


def build_recipe(args):
    import numpy as np
    from tokenizers import Tokenizer

    recipe = json.loads(args.recipe.read_text())
    if recipe["schema_version"] != 1 or recipe["source_encoding"] != "legacy":
        raise ValueError("unsupported BBC recipe")
    source_path = args.legacy_root / "tree/test.npy"
    if sha256_file(source_path) != recipe["source_tree_sha256"]:
        raise ValueError("source tree fingerprint differs from frozen recipe")
    if args.output_root.exists():
        raise FileExistsError(f"use a new output directory: {args.output_root}")
    parts = split_tree_stream(source_path)[recipe["offset"]:]
    if len(parts) != recipe["target_documents"]:
        raise ValueError("recipe document count mismatch")
    seen = set()
    for patch in recipe["patches"]:
        j = patch["target_doc"]
        if j in seen or not 0 <= j < len(parts) or patch["source_doc"] != j + recipe["offset"]:
            raise ValueError("invalid/duplicate recipe document mapping")
        seen.add(j)
        parts[j] = apply_edits(parts[j], patch["edits"])
    tree = np.concatenate(parts)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    validate_tokenizer(tokenizer)
    vocab = tokenizer.get_vocab()
    nt_ids = [i for t, i in vocab.items() if t.startswith("<(") or t.endswith(")>")]
    close_ids = [i for t, i in vocab.items() if t.endswith(")>")]
    terminal_mask, close_mask = ~np.isin(tree, nt_ids), np.isin(tree, close_ids)
    arrays = {"tree": tree, "terminal": tree[terminal_mask], "tg": np.repeat(tree, 1 + close_mask)}
    tree_lengths = np.asarray(recipe["tree_sentence_lengths"], dtype=np.uint32)
    counts = np.asarray(recipe["document_sentence_counts"], dtype=np.uint32)
    if int(tree_lengths.sum()) != len(tree) or int(counts.sum()) != len(tree_lengths):
        raise ValueError("recipe sentence/document lengths do not cover output")
    starts = np.concatenate(([0], np.cumsum(tree_lengths, dtype=np.int64)))[:-1]
    lengths = {"tree": tree_lengths,
               "terminal": np.add.reduceat(terminal_mask, starts, dtype=np.uint32),
               "tg": tree_lengths + np.add.reduceat(close_mask, starts, dtype=np.uint32)}
    # Stage all nine outputs and check their recorded file hashes before publish.
    import os
    import tempfile
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bbc-recipe-", dir=args.output_root.parent) as name:
        staged = Path(name) / "result"
        staged.mkdir()
        for fmt, values in arrays.items():
            (staged / fmt).mkdir()
            for filename, array in (("test.npy", values), ("test_sent_index.npy", lengths[fmt]),
                                    ("test_doc_index.npy", counts)):
                path = staged / fmt / filename
                np.save(path, array)
                if sha256_file(path) != recipe["expected_sha256"][fmt][filename]:
                    raise ValueError(f"frozen recipe output hash mismatch: {fmt}/{filename}")
        (staged / "manifest.json").write_text(json.dumps({"recipe_sha256": sha256_file(args.recipe),
            "source_tree_sha256": recipe["source_tree_sha256"], "verified_outputs": recipe["expected_sha256"]}, indent=2) + "\n")
        os.rename(staged, args.output_root)
    print(f"verified all nine output SHA-256 fingerprints: {args.output_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("extract", help="emit indexed dev or test raw rows to stdout")
    p.add_argument("--split", choices=("dev", "test"), default="test")
    p.add_argument("--parsed-dir", type=Path, required=True)
    p.add_argument("--dev-index", type=Path, required=True)
    p.add_argument("--test-index", type=Path, required=True)
    p.set_defaults(func=extract)
    p = commands.add_parser("build-raw", help="apply the current pretraining converter to selected rows")
    p.add_argument("--split", choices=("dev", "test"), help="assert the split recorded in selected JSONL")
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("dataset/bbc-news/TG_GPT2_tokenizer.json"))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--encoding", choices=("pipeline", "legacy"), default="pipeline")
    p.add_argument("--normalize-adj", action="store_true",
                   help="map ADJ constituents to ADJP before encoding (changes the historical version)")
    p.set_defaults(func=build_raw)
    p = commands.add_parser("verify-raw", help="verify all three dev/test arrays against archived files")
    p.add_argument("--split", choices=("dev", "test"), required=True)
    p.add_argument("--rebuilt-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.set_defaults(func=verify_raw)
    p = commands.add_parser("audit", help="compare in-root BPE terminals and sentence boundaries")
    p.add_argument("--bbc-root", type=Path, default=Path("dataset/bbc-news"))
    p.add_argument("--ppl-dir", type=Path, default=Path("dataset/bbc-news/testppl/tree300"))
    p.add_argument("--tokenizer", type=Path, default=Path("dataset/bbc-news/TG_GPT2_tokenizer.json"))
    p.add_argument("--raw-root", type=Path)
    p.add_argument("--legacy-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.set_defaults(func=audit)
    p = commands.add_parser("build-current", help="reproduce current Tree/terminal/TG from canonical tree300")
    p.add_argument("--ppl-dir", type=Path, default=Path("dataset/bbc-news/testppl/tree300"))
    p.add_argument("--tokenizer", type=Path, default=Path("dataset/bbc-news/TG_GPT2_tokenizer.json"))
    p.add_argument("--output-root", type=Path, required=True)
    p.set_defaults(func=build_current)
    p = commands.add_parser("freeze-recipe", help="freeze a raw-to-current delta for standalone reuse")
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--target-root", type=Path, required=True)
    p.add_argument("--offset", type=int, default=59)
    p.add_argument("--recipe", type=Path, required=True)
    p.set_defaults(func=freeze_recipe)
    p = commands.add_parser("build-recipe", help="reproduce current test from legacy raw rebuild + frozen delta")
    p.add_argument("--legacy-root", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=Path("dataset/bbc-news/TG_GPT2_tokenizer.json"))
    p.add_argument("--output-root", type=Path, required=True)
    p.set_defaults(func=build_recipe)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
