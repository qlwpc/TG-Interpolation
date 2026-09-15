"""Regression checks for the archived BBC test reconstruction contract."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from datatools.parse_test_docppl_data.reproduce_bbc_test import (
    apply_edits, build_raw, extract, legacy_format, legacy_tokenizer,
    normalize_adj_labels, selected_rows, selection_split, sha256_file, verify_raw,
)


def test_extract_keeps_index_order_and_checks_completion(tmp_path, capsys):
    (tmp_path / "shard.txt").write_text("(NN zero)\n(NN one)\n(NN two)\n")
    dev, test = tmp_path / "dev.json", tmp_path / "test.json"
    dev.write_text(json.dumps({"shard": [1]}))
    test.write_text(json.dumps({"shard": [2, 0]}))
    extract(argparse.Namespace(parsed_dir=tmp_path, dev_index=dev, test_index=test))
    output = capsys.readouterr().out
    selected = tmp_path / "selected.jsonl"
    selected.write_text(output)
    rows = list(selected_rows(selected))
    assert [r["source_row"] for r in rows] == [2, 0]
    assert [r["parsed"] for r in rows] == ["(NN two)\n", "(NN zero)\n"]
    selected.write_text("\n".join(output.splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="completion"):
        list(selected_rows(selected))


def test_extract_rejects_dev_overlap_before_output(tmp_path, capsys):
    dev, test = tmp_path / "dev.json", tmp_path / "test.json"
    dev.write_text('{"shard": [2]}')
    test.write_text('{"shard": [2]}')
    with pytest.raises(ValueError, match="overlap"):
        extract(argparse.Namespace(parsed_dir=tmp_path, dev_index=dev, test_index=test))
    assert not capsys.readouterr().out


@pytest.mark.parametrize("split, expected", [("dev", [2, 0]), ("test", [1])])
def test_split_selection_and_wrong_split_rejection(tmp_path, capsys, split, expected):
    (tmp_path / "shard.txt").write_text("(NN zero)\n(NN one)\n(NN two)\n")
    dev, test = tmp_path / "dev.json", tmp_path / "test.json"
    dev.write_text('{"shard": [2, 0]}')
    test.write_text('{"shard": [1]}')
    extract(argparse.Namespace(parsed_dir=tmp_path, dev_index=dev, test_index=test, split=split))
    selected = tmp_path / "selected.jsonl"
    selected.write_text(capsys.readouterr().out)
    assert selection_split(selected) == split
    assert [r["source_row"] for r in selected_rows(selected, expected_split=split)] == expected
    with pytest.raises(ValueError, match="does not match"):
        list(selected_rows(selected, expected_split="test" if split == "dev" else "dev"))


def test_dev_build_uses_dev_filename_and_no_test_recipe(tmp_path, capsys):
    tokenizer = Path(__file__).resolve().parents[1] / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    if not tokenizer.exists():
        pytest.skip("local BBC tokenizer is not shipped in a clean checkout")
    (tmp_path / "shard.txt").write_text("(S (NN zero))\n(S (NN one))\n(S (NN two))\n")
    dev, test = tmp_path / "dev.json", tmp_path / "test.json"
    dev.write_text('{"shard": [2, 0]}')
    test.write_text('{"shard": [1]}')
    extract(argparse.Namespace(parsed_dir=tmp_path, dev_index=dev, test_index=test, split="dev"))
    selected = tmp_path / "selected.jsonl"
    selected.write_text(capsys.readouterr().out)
    output = tmp_path / "rebuilt"
    args = argparse.Namespace(selected=selected, tokenizer=tokenizer, output_root=output,
                              split="dev", encoding="legacy")
    build_raw(args)
    for fmt in ("tree", "terminal", "tg"):
        assert not (output / fmt / "test.npy").exists()
        array = np.load(output / fmt / "dev.npy")
        assert np.count_nonzero(array == 50257) == np.count_nonzero(array == 50256) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["split"] == "dev" and manifest["documents"] == 2
    args.split, args.output_root = "test", tmp_path / "wrong"
    with pytest.raises(ValueError, match="differs from extraction"):
        build_raw(args)
    assert not args.output_root.exists()


def test_verify_raw_detects_changed_content_with_unchanged_document_count(tmp_path):
    rebuilt, reference = tmp_path / "rebuilt", tmp_path / "reference"
    manifest = {"split": "dev", "documents": 1, "outputs": {}}
    for fmt in ("tree", "terminal", "tg"):
        for root in (rebuilt, reference):
            (root / fmt).mkdir(parents=True)
            np.save(root / fmt / "dev.npy", np.asarray([50257, 100, 50256], dtype=np.uint16))
        manifest["outputs"][fmt] = {"sha256": sha256_file(rebuilt / fmt / "dev.npy")}
    (rebuilt / "manifest.json").write_text(json.dumps(manifest))
    args = argparse.Namespace(split="dev", rebuilt_root=rebuilt, reference_root=reference,
                              output_json=tmp_path / "pass.json")
    verify_raw(args)
    assert json.loads(args.output_json.read_text())["passed"]
    np.save(rebuilt / "terminal/dev.npy", np.asarray([50257, 101, 50256], dtype=np.uint16))
    args.output_json = tmp_path / "fail.json"
    with pytest.raises(ValueError, match="differs from reference"):
        verify_raw(args)
    report = json.loads(args.output_json.read_text())
    assert not report["passed"]
    assert report["files"]["terminal"]["different_document_ids"] == [0]


def test_recipe_offsets_are_relative_to_unedited_source():
    original = np.asarray([10, 11, 12, 13, 14], dtype=np.uint16)
    edits = [{"source_span": [1, 2], "insert_ids": [21, 22, 23]},
             {"source_span": [4, 5], "insert_ids": []}]
    assert apply_edits(original, edits).tolist() == [10, 21, 22, 23, 12, 13]
    with pytest.raises(ValueError, match="overlapping"):
        apply_edits(original, edits[::-1])


def test_legacy_bracket_word_boundaries_and_unknown_labels():
    from tokenizers import Tokenizer

    path = Path(__file__).resolve().parents[1] / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    if not path.exists():
        pytest.skip("local BBC tokenizer is not shipped in a clean checkout")
    modern = Tokenizer.from_file(str(path))
    legacy = legacy_tokenizer(modern)
    # These contexts distinguish whole-word matching from substring replacement.
    assert legacy.encode(" 8-RRB-").ids == [807, 12, 21095, 33, 12]
    assert legacy.encode(" -RRB-:").ids == [50263, 25]
    assert modern.encode(" -RRB-:").ids == [220, 50263, 25]
    rendered = legacy_format("(S (ADJ (JJ worth) (NP (DT the) (NN wait)))) (Ċ Ċ)", legacy.get_vocab())
    assert " (ADJ worth" in rendered and " ADJ)" in rendered
    assert rendered.endswith(" \n")


def test_adj_normalization_preserves_words_and_emits_real_nonterminals():
    from nltk import Tree
    from tokenizers import Tokenizer

    parsed = "(S (ADJ (JJ worth) (NP (DT the) (NN wait))) (NN ADJ)) (Ċ Ċ)"
    normalized = normalize_adj_labels(parsed)
    before = Tree.fromstring("(ROOT " + parsed + ")")
    after = Tree.fromstring("(ROOT " + normalized + ")")
    assert before.leaves() == after.leaves()
    assert after[0][0].label() == "ADJP"
    assert after[0][1][0] == "ADJ"  # Literal article text must not be replaced.
    assert normalize_adj_labels(normalized) == normalized
    path = Path(__file__).resolve().parents[1] / "dataset/bbc-news/TG_GPT2_tokenizer.json"
    if not path.exists():
        pytest.skip("local BBC tokenizer is not shipped in a clean checkout")
    tokenizer = legacy_tokenizer(Tokenizer.from_file(str(path)))
    ids = tokenizer.encode(legacy_format(normalized, tokenizer.get_vocab())).ids
    assert ids.count(50268) == ids.count(50294) == 1


def test_builder_accepts_normalized_tree300_and_preserves_whitespace(tmp_path):
    from datatools.parse_test_docppl_data.build_aligned_test import build

    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"model": {"vocab": {
        "<|beginoftext|>": 50257, "<|endoftext|>": 50256, "<|pad|>": 50258,
        "<(S>": 50268, "<S)>": 50269}}, "added_tokens": []}))
    source = tmp_path / "ppl"
    source.mkdir()
    records = [[50257, 50268, 100, 50269, 220, 198],
               [50268, 101, 50269, 50256],
               [50257, 50268, 102, 50269, 50256]]
    np.save(source / "tree_300.npy", np.concatenate([
        np.tile(np.asarray(r, dtype=np.uint16), 300) for r in records]))
    np.save(source / "tree_sent_index.npy", np.repeat(
        np.asarray(list(map(len, records)), dtype=np.uint16), 300))
    np.save(source / "tree_doc_index.npy", np.asarray([2, 1], dtype=np.uint32))
    output = tmp_path / "rebuilt"
    build(argparse.Namespace(input_dir=source, tokenizer=tokenizer, output_root=output, overwrite=False))
    assert np.load(output / "tree/test.npy").tolist() == sum(records, [])
    assert np.load(output / "terminal/test.npy").tolist() == [50257, 100, 220, 198, 101, 50256, 50257, 102, 50256]
    assert np.load(output / "tree/test_doc_index.npy").tolist() == [2, 1]
