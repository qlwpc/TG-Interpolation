"""Offline release-pipeline regressions; no corpus downloads or training."""

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from nltk import Tree
from tokenizers import Tokenizer, models, pre_tokenizers

from datatools.parse_pretrain_data import build_pretrain_data as build
from datatools.parse_pretrain_data.assemble_streams import FORMATS, assemble, main as assemble_main
from datatools.parse_pretrain_data.convert_TG_and_tokenize import tokenize_shard
from datatools.parse_pretrain_data.parse_integrity import checked_parse_trees, parsed_row_count, write_parse_receipt
from datatools.parse_pretrain_data.pipeline_io import load_index, sha256_file, validate_split_indices


@pytest.fixture
def tiny_tokenizer(tmp_path):
    tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, "alpha": 1, "beta": 2, "gamma": 3, "delta": 4}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(["<|beginoftext|>", "<|endoftext|>", "<|pad|>", "<(S>", "<(NP>", "<S)>", "<NP)>"])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    opening = {tokenizer.token_to_id(t) for t in ("<(S>", "<(NP>")}
    closing = {tokenizer.token_to_id(t) for t in ("<S)>", "<NP)>")}
    # Same transforms as SentencepieceVocab, without requiring a compiled .so
    # for the source-only regression suite.
    vocab = SimpleNamespace(
        bos=tokenizer.token_to_id("<|beginoftext|>"), eos=tokenizer.token_to_id("<|endoftext|>"),
        convert_treenpy_to_terminal=lambda a: a[~np.isin(a, list(opening | closing))],
        convert_treenpy_to_TG=lambda a: np.repeat(a, [2 if int(i) in closing else 1 for i in a]),
    )
    return tokenizer, vocab, path


def parsed_shard(path, words=("alpha", "beta", "gamma", "delta")):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"(S (NP (NN {word})))\n" for word in words))
    return path


def raw_shards(tmp_path, counts=None):
    root = tmp_path / "shards"
    for fmt in FORMATS:
        (root / fmt).mkdir(parents=True)
        for stem in ("a", "b"):
            n = (counts or {}).get((fmt, stem), 4)
            np.save(root / fmt / f"{stem}.npy", np.array([v for i in range(n) for v in (99, i + 1, 98)], dtype=np.uint16))
    return root


@pytest.mark.parametrize("payload", ['{"a":[1,1]}', '{"a":[true]}', '{"a":[1.0]}',
                                     '{"a":["1"]}', '{"a":[-1]}', '{"a":[],"a":[]}', '[]'])
def test_split_json_is_strict(tmp_path, payload):
    path = tmp_path / "index.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        load_index(path)


def test_released_indices_are_pinned_and_train_only_keys_explicit():
    result = validate_split_indices(build.RELEASED_SPLIT_DIR / "dev_index.json",
                                    build.RELEASED_SPLIT_DIR / "test_index.json",
                                    build.read_bbc_configs(), released=True)
    assert result["dev"]["documents"] == 4980
    assert result["test"]["documents"] == 5025
    assert result["train_only_shards"] == build.read_bbc_configs()[-5:]


def test_released_pin_detects_order_change(tmp_path):
    path = tmp_path / "dev.json"
    index = load_index(build.RELEASED_SPLIT_DIR / "dev_index.json")
    for values in index.values():
        values.reverse()
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="differs from the released"):
        validate_split_indices(path, build.RELEASED_SPLIT_DIR / "test_index.json", build.read_bbc_configs(), released=True)


def test_assembly_preserves_unsorted_index_order_and_no_sentinel(tmp_path):
    root = raw_shards(tmp_path)
    result = assemble(root, tmp_path / "final", ["a", "b"], {"a": [3, 1]}, {"a": [2]}, 99)
    for fmt in FORMATS:
        assert np.load(tmp_path / "final" / fmt / "dev.npy").tolist() == [99, 4, 98, 99, 2, 98]
        assert result[fmt]["train"]["documents"] == 5  # a[0] + all of unindexed b
        assert result[fmt]["dev"]["documents"] == 2
        assert result[fmt]["test"]["documents"] == 1


@pytest.mark.parametrize("dev,test,error", [({"a": [4]}, {}, IndexError), ({"a": [1]}, {"a": [1]}, ValueError),
                                           ({"unknown": [0]}, {}, ValueError), ({"a": [1, 1]}, {}, ValueError)])
def test_assembly_invalid_indices_write_nothing(tmp_path, dev, test, error):
    root = raw_shards(tmp_path)
    with pytest.raises(error):
        assemble(root, tmp_path / "final", ["a", "b"], dev, test, 99)
    assert not (tmp_path / "final").exists()


def test_assembly_checks_per_shard_alignment_before_overwriting(tmp_path):
    root = raw_shards(tmp_path, {("tg", "a"): 3, ("tg", "b"): 5})  # totals still equal!
    output = tmp_path / "final/terminal/train.npy"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="not aligned"):
        assemble(root, tmp_path / "final", ["a", "b"], {}, {}, 99, overwrite=True)
    assert output.read_bytes() == b"keep me"
    assert not (tmp_path / "final/tree").exists()


def test_existing_late_target_does_not_allow_early_writes(tmp_path):
    root = raw_shards(tmp_path)
    late = tmp_path / "final/tg/test.npy"
    late.parent.mkdir(parents=True)
    late.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        assemble(root, tmp_path / "final", ["a", "b"], {}, {}, 99)
    assert not (tmp_path / "final/terminal").exists()


def test_tokenize_assemble_validate_end_to_end(tmp_path, tiny_tokenizer, monkeypatch):
    tokenizer, vocab, tokenizer_path = tiny_tokenizer
    root = tmp_path / "tokens"
    source = parsed_shard(tmp_path / "parsed/a.txt")
    receipt = tokenize_shard(source, root, tokenizer, vocab, sha256_file(tokenizer_path))
    assert receipt["documents"] == 4
    # Verify memory-bounded writer produces exactly the legacy transforms on
    # valid input, including explicit BOS/EOS and doubled closing NTs.
    from datatools.parse_pretrain_data.convert_TG_and_tokenize import convert_TG_format, encode_tree_document
    trees = [encode_tree_document(convert_TG_format(row), tokenizer, vocab, "uint16")
             for row in source.read_text().splitlines()]
    np.testing.assert_array_equal(np.load(root / "tree/a.npy"), np.concatenate(trees))
    np.testing.assert_array_equal(np.load(root / "terminal/a.npy"), vocab.convert_treenpy_to_terminal(np.concatenate(trees)))
    np.testing.assert_array_equal(np.load(root / "tg/a.npy"), vocab.convert_treenpy_to_TG(np.concatenate(trees)))
    monkeypatch.setattr(build, "expected_stems", lambda corpus: ["a"])
    import datatools.parse_pretrain_data.get_TG_tokenizer as token_module
    monkeypatch.setattr(token_module, "validate_paper_layout", lambda *args: None)
    assert build.validate_tokens("bbc", root, tokenizer_path)["formats"]["tg"]["documents"] == 4
    dev, test, order = (tmp_path / name for name in ("dev.json", "test.json", "order.txt"))
    dev.write_text('{"a": [3, 1]}')
    test.write_text('{"a": [2]}')
    order.write_text("a\n")
    final = tmp_path / "final"
    assert assemble_main(["--input-root", str(root), "--output-root", str(final), "--tokenizer", str(tokenizer_path),
                          "--dev-index", str(dev), "--test-index", str(test), "--shard-order", str(order)]) == 0
    checked = build.validate_assembled(final, tokenizer_path)
    assert checked["formats"]["terminal"]["dev"]["documents"] == 2
    np.save(final / "tg/test.npy", np.array([1], dtype=np.uint16))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        build.validate_assembled(final, tokenizer_path)


@pytest.mark.parametrize("damage", ["source", "tokenizer", "output", "receipt"])
def test_tokenization_resume_checks_identity(tmp_path, tiny_tokenizer, damage):
    tokenizer, vocab, path = tiny_tokenizer
    source = parsed_shard(tmp_path / "a.txt")
    root = tmp_path / "tokens"
    token_hash = sha256_file(path)
    tokenize_shard(source, root, tokenizer, vocab, token_hash)
    out = root / "tree/a.npy"
    previous_mtime = out.stat().st_mtime_ns
    tokenize_shard(source, root, tokenizer, vocab, token_hash)
    assert out.stat().st_mtime_ns == previous_mtime
    if damage == "source":
        parsed_shard(source, ["beta"])
    elif damage == "tokenizer":
        token_hash = "changed"
    elif damage == "output":
        out.write_bytes(b"broken npy")
    else:
        (root / "manifests/a.json").unlink()
    with pytest.raises(ValueError):
        tokenize_shard(source, root, tokenizer, vocab, token_hash)
    tokenize_shard(source, root, tokenizer, vocab, token_hash, overwrite=True)
    assert np.load(out).ndim == 1


@pytest.mark.parametrize("bad_line", ["\n", "(S (NN broken)\n", "not a tree\n", "(S (NN tail))"])
def test_bad_rows_never_shift_document_indices_or_publish_outputs(tmp_path, tiny_tokenizer, bad_line):
    tokenizer, vocab, path = tiny_tokenizer
    source = parsed_shard(tmp_path / "a.txt", ["alpha"])
    source.write_text(source.read_text() + bad_line)
    root = tmp_path / "tokens"
    with pytest.raises(ValueError, match=r"a.txt:2:.*no rows may be skipped"):
        tokenize_shard(source, root, tokenizer, vocab, sha256_file(path))
    assert not list(root.rglob("*.npy"))
    assert not list(root.glob(".tokenize-*"))


def test_failed_overwrite_keeps_previous_valid_shard(tmp_path, tiny_tokenizer):
    tokenizer, vocab, path = tiny_tokenizer
    source = parsed_shard(tmp_path / "a.txt")
    root = tmp_path / "tokens"
    tokenize_shard(source, root, tokenizer, vocab, sha256_file(path))
    before = {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}
    source.write_text("broken\n")
    with pytest.raises(ValueError):
        tokenize_shard(source, root, tokenizer, vocab, sha256_file(path), overwrite=True)
    assert before == {p: sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_plan_never_resamples_and_preserves_custom_paths(tmp_path):
    paths = {key: tmp_path / f"custom {key}" for key in build.defaults("bbc")}
    plan = build.command_plan("bbc", paths)
    assert "make-split-indices" not in [s["stage"] for s in plan["stages"]]
    assert "validate-splits" in [s["stage"] for s in plan["stages"]]
    for stage in plan["stages"]:
        for path in paths.values():
            assert str(path) in stage["command"]
            assert str(path) in shlex.split(stage["shell_command"])
    fineweb = build.command_plan("fineweb-edu", paths)
    assert sum(stage["stage"] == "validate" for stage in fineweb["stages"]) == 1


def test_new_split_requires_separate_explicit_destination(tmp_path):
    with pytest.raises(SystemExit):
        build.main(["make-split-indices", "--corpus", "bbc"])
    with pytest.raises(ValueError, match="separate output directory"):
        build.main(["make-split-indices", "--corpus", "bbc", "--output-dir", str(build.RELEASED_SPLIT_DIR)])
    (tmp_path / "test_index.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        build.main(["make-split-indices", "--corpus", "bbc", "--output-dir", str(tmp_path)])
    assert (tmp_path / "test_index.json").read_text() == "preserve"


def test_partial_parse_is_not_ready_for_tokenization(tmp_path, monkeypatch):
    source = parsed_shard(tmp_path / "a.txt")
    monkeypatch.setattr(build, "expected_stems", lambda corpus: ["a"])
    write_parse_receipt(source, 4, "benepar_en3_large", complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        build.validate_parsed_shards("bbc", tmp_path)
    write_parse_receipt(source, 4, "benepar_en3_large", complete=True)
    build.validate_parsed_shards("bbc", tmp_path)
    source.write_text(source.read_text() + "(S (NN extra))\n")
    with pytest.raises(ValueError, match="fingerprint changed"):
        build.validate_parsed_shards("bbc", tmp_path)


def test_benepar_standard_and_topk_keep_full_sentence():
    tree = Tree.fromstring("(S (NP (NN alpha)) (VP (VB beta)))")
    inputs = [SimpleNamespace(words=["alpha", "beta"])]
    assert checked_parse_trees([tree], inputs) == [tree]
    assert checked_parse_trees([[tree, tree]], inputs) == [tree]
    with pytest.raises(ValueError, match="trees for"):
        checked_parse_trees([], inputs)
    with pytest.raises(ValueError, match="terminals"):
        checked_parse_trees([tree[0]], inputs)
    # Benepar escapes brackets even inside a word, not only standalone tokens.
    escaped = Tree.fromstring("(S (NN alpha-LRB-beta-RRB-))")
    assert checked_parse_trees([escaped], [SimpleNamespace(words=["alpha(beta)"])]) == [escaped]


def test_empty_holdouts_are_standard_npy_files(tmp_path):
    root = raw_shards(tmp_path)
    result = assemble(root, tmp_path / "final", ["a", "b"], {}, {}, 99)
    assert result["tree"]["train"]["documents"] == 8
    for fmt in FORMATS:
        for split in ("dev", "test"):
            assert np.load(tmp_path / "final" / fmt / f"{split}.npy").shape == (0,)


def test_existing_tokenizer_is_reused_without_network(tmp_path, tiny_tokenizer, monkeypatch):
    from datatools.parse_pretrain_data import get_TG_tokenizer as module

    _, _, path = tiny_tokenizer
    monkeypatch.setattr(module, "validate_paper_layout", lambda *a: None)
    def unexpected_download(*args):
        raise AssertionError("should not download an existing tokenizer")
    monkeypatch.setattr(module, "build_tokenizer", unexpected_download)
    before = sha256_file(path)
    assert module.main(["--model-name", "gpt2", "--output", str(path)]) == 0
    assert sha256_file(path) == before


def test_parser_refuses_partial_last_row(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("(S (NN alpha))\n(partial")
    with pytest.raises(ValueError, match="incomplete"):
        parsed_row_count(path)


def test_fineweb_max_docs_is_forwarded(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(build, "run_checked", commands.append)
    assert build.main(["parse", "--corpus", "fineweb-edu", "--task-index", "0", "--max-docs", "3", "--skip-deps"]) == 0
    assert commands[0][commands[0].index("--max-docs") + 1] == "3"


def test_exact_shard_names_required_even_when_count_matches(tmp_path):
    (tmp_path / "wrong.npy").write_bytes(b"")
    with pytest.raises(ValueError, match="shard names mismatch"):
        build.check_shard_names(tmp_path, ["right"], ".npy")


def test_available_paper_tokenizers_stream_through_compiled_vocab(tmp_path):
    extension = pytest.importorskip("olmo.data.tg_mask")
    source = parsed_shard(tmp_path / "a.txt")
    tested = 0
    for corpus, dtype in (("bbc", "uint16"), ("fineweb-edu", "uint32")):
        path = build.defaults(corpus)["tokenizer"]
        if not path.is_file():
            continue
        tokenizer = Tokenizer.from_file(str(path))
        vocab = extension.SentencepieceVocab.from_vocab_file(str(path))
        result = tokenize_shard(source, tmp_path / corpus, tokenizer, vocab, sha256_file(path))
        assert result["dtype"] == dtype
        assert result["documents"] == 4
        tested += 1
    if not tested:
        pytest.skip("paper tokenizers are not available; synthetic tests still run")


@pytest.mark.parametrize("broken", [True, False])
def test_download_only_publishes_complete_files(tmp_path, monkeypatch, broken):
    import requests
    from datatools.parse_pretrain_data.setup_parse_deps import fetch_hf_file

    class Response:
        headers = {"content-length": "6"}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, **kwargs):
            yield b"abc"
            if not broken:
                yield b"def"

    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response())
    output = tmp_path / "shard.parquet"
    output.write_bytes(b"old valid file")
    if broken:
        with pytest.raises(RuntimeError, match="incomplete download"):
            fetch_hf_file("repo", "shard.parquet", output)
        assert output.read_bytes() == b"old valid file"
    else:
        fetch_hf_file("repo", "shard.parquet", output)
        assert output.read_bytes() == b"abcdef"
    assert not list(tmp_path.glob("*.part"))
