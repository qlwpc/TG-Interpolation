"""Cross-script contract tests for the datatools parse/tokenize pipeline.

Covers the interfaces between:
  * benepar_parse.py / parse_input.py  (produce one bracket tree per line)
  * convert_TG_and_tokenize.py         (consumes those lines)
  * setup_parse_deps.py                (bootstraps models/tokenizers)

Runs fully offline; benepar models are expected in ~/nltk_data/models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATATOOLS = str(REPO / "datatools" / "parse_pretrain_data")

# A benepar-style parsed line (same shape benepar_parse.py writes):
# newline-only sentences appear as "(Ċ Ċ)", PTB escapes appear as -LRB- etc.
SAMPLE_LINE = (
    "(Ċ Ċ) (S (NP (DT The) (NN cat)) (VP (VBZ sits) (PP (IN on) (NP (DT the) (NN mat)))) (. .)) (Ċ Ċ)"
)
SAMPLE_LINE_LRB = (
    "(S (NP (DT A) (NN bracket)) (PRN (-LRB- -LRB-) (NP (JJ plain)) (-RRB- -RRB-)) (. .))"
)


def _import_parse_input():
    if DATATOOLS not in sys.path:
        sys.path.insert(0, DATATOOLS)
    import parse_input

    return parse_input


# ---------------------------------------------------------------------------
# convert_TG_and_tokenize: parse-script output must be consumable input
# ---------------------------------------------------------------------------

class TestConvertTGFormatContract:
    def test_output_round_trips_through_nltk(self):
        from datatools.parse_pretrain_data.convert_TG_and_tokenize import convert_TG_format
        from nltk import Tree

        for line in (SAMPLE_LINE, SAMPLE_LINE_LRB):
            out = convert_TG_format(line)
            assert out, f"convert_TG_format returned empty for {line!r}"
            # the tokenizer script re-wraps with the root label before parsing
            Tree.fromstring("(qlwpcRegen " + out + ")")

    def test_lrb_mapping_applied(self):
        from datatools.parse_pretrain_data.convert_TG_and_tokenize import convert_TG_format

        out = convert_TG_format(SAMPLE_LINE_LRB)
        assert "-LRB-" not in out
        assert "(" in out and ")" in out

    def test_newline_wrapper_preserved(self):
        from datatools.parse_pretrain_data.convert_TG_and_tokenize import convert_TG_format

        out = convert_TG_format(SAMPLE_LINE)
        # (Ċ Ċ) wrappers are mapped by pformat_flat to a space + newline
        # (Ċ → \n), which the tokenizer then encodes as tokens [220, 198].
        # The literal "(Ċ Ċ)" string never survives conversion.
        assert out.startswith(" \n")
        assert out.endswith(" \n")

    def test_tokenized_document_has_explicit_bos_and_eos(self):
        from datatools.parse_pretrain_data.convert_TG_and_tokenize import (
            convert_TG_format,
            encode_tree_document,
        )
        from olmo.data.tg_mask import SentencepieceVocab
        from tokenizers import Tokenizer

        tokenizer_path = REPO / "dataset/bbc-news/TG_GPT2_tokenizer.json"
        if not tokenizer_path.is_file():
            pytest.skip("paper GPT-2 tokenizer is not available")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        vocab = SentencepieceVocab.from_vocab_file(str(tokenizer_path))
        tokens = encode_tree_document(
            convert_TG_format(SAMPLE_LINE), tokenizer, vocab, "uint16"
        )
        assert tokens[0] == vocab.bos
        assert tokens[-1] == vocab.eos


# ---------------------------------------------------------------------------
# setup_parse_deps: deterministic checks (no installs)
# ---------------------------------------------------------------------------

class TestSetupParseDeps:
    def test_benepar_model_dir_finds_cached_model(self):
        from datatools.parse_pretrain_data.setup_parse_deps import benepar_model_dir

        found = benepar_model_dir("benepar_en3")
        if found is None:
            pytest.skip("benepar_en3 not cached on this machine")
        assert found.is_dir()
        assert found.name == "benepar_en3"

    def test_benepar_model_dir_missing_returns_none(self):
        from datatools.parse_pretrain_data.setup_parse_deps import benepar_model_dir

        assert benepar_model_dir("benepar_zzz_nonexistent") is None

    def test_nltk_data_dirs_include_legacy_path(self):
        from datatools.parse_pretrain_data.setup_parse_deps import nltk_data_dirs

        dirs = [str(d) for d in nltk_data_dirs()]
        assert any("2024233198" in d for d in dirs)
        assert any(d.endswith("nltk_data") for d in dirs)

    def test_spacy_model_ok_false_for_bogus(self):
        from datatools.parse_pretrain_data.setup_parse_deps import spacy_model_ok

        assert spacy_model_ok("definitely_not_a_spacy_model") is False

    def test_check_mode_reports_missing_without_installing(self, capsys):
        from datatools.parse_pretrain_data.setup_parse_deps import main

        rc = main(["--check"])
        out = capsys.readouterr().out
        assert "benepar:benepar_en3" in out
        assert rc in (0, 1)


# ---------------------------------------------------------------------------
# parse_input: offline import safety (lazy t5-small)
# ---------------------------------------------------------------------------

class TestParseInputOfflineImport:
    def test_import_does_not_load_tokenizer(self):
        """``import parse_input`` must work offline (t5-small lazy-loaded);
        the tokenizer is only loaded on first split_list_limit call."""
        parse_input = _import_parse_input()

        assert parse_input._tokenizer is None
        assert callable(parse_input.split_text_into_sents)
        assert callable(parse_input.split_list_limit)
        assert callable(parse_input.process_doc_into_maxlen)

    def test_split_long_sentence_punctuation_priority(self):
        from parse_input import split_long_sentence

        # period inside the first max_len window → split right after it
        tokens = ["word"] * 50 + ["."] + ["tail"] * 55
        parts = split_long_sentence(tokens, max_len=100)
        assert all(len(p) <= 100 for p in parts)
        assert sum(len(p) for p in parts) == len(tokens)
        assert parts[0][-1] == "."

    def test_split_long_sentence_punct_outside_window_hard_splits(self):
        """Punctuation is only searched within the first max_len tokens; a
        period beyond the window cannot anchor the first split."""
        from parse_input import split_long_sentence

        tokens = ["word"] * 300 + ["."] + ["tail"] * 5
        parts = split_long_sentence(tokens, max_len=100)
        assert all(len(p) <= 100 for p in parts)
        assert sum(len(p) for p in parts) == len(tokens)
        assert parts[0] == ["word"] * 100  # hard split, no punct in window

    def test_split_long_sentence_hard_split_no_punct(self):
        from parse_input import split_long_sentence

        tokens = ["w"] * 250
        parts = split_long_sentence(tokens, max_len=100)
        assert [len(p) for p in parts] == [100, 100, 50]
