"""
Tests for bracket mapping behavior in pformat_flat / convert_TG_format.
Qwen3-style tokenizers should map -LRB- → (, GPT-2 tokenizers should not.
"""

from unittest.mock import MagicMock, patch
from nltk import Tree

from olmo.data.util import (
    pformat_flat,
    encode_TG_string,
    _get_bracket_mapping_from_tokenizer,
)


# ---------------------------------------------------------------------------
# _get_bracket_mapping_from_tokenizer
# ---------------------------------------------------------------------------

class TestGetBracketMappingFromTokenizer:
    def test_tokenizer_with_flag_true(self):
        """Tokenizer with use_bracket_mapping=True → return True."""
        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = True
        assert _get_bracket_mapping_from_tokenizer(mock_tok) is True

    def test_tokenizer_with_flag_false(self):
        """Tokenizer with use_bracket_mapping=False → return False."""
        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = False
        assert _get_bracket_mapping_from_tokenizer(mock_tok) is False

    def test_tokenizer_without_flag_defaults_false(self):
        """Tokenizer without use_bracket_mapping attr → default False."""
        mock_tok = MagicMock(spec=[])  # no attributes at all
        assert _get_bracket_mapping_from_tokenizer(mock_tok) is False

    def test_tokenizer_with_flag_none(self):
        """None flag → False."""
        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = None
        assert _get_bracket_mapping_from_tokenizer(mock_tok) is False


# ---------------------------------------------------------------------------
# pformat_flat with use_bracket_mapping
# ---------------------------------------------------------------------------

class TestPformatFlatBracketMapping:
    @staticmethod
    def _make_tree_with_leaf(leaf_value: str):
        """Build an NLTK Tree programmatically (avoids fromstring parsing issues
        with '-' in leaf values)."""
        # Tree('qlwpcRegen', [Tree('S', [Tree('NP', [leaf_value])])])
        return Tree("qlwpcRegen", [Tree("S", [Tree("NP", [leaf_value])])])

    def test_mapping_enabled_maps_lrb(self):
        """With mapping enabled: -LRB- → (."""
        tree = self._make_tree_with_leaf("-LRB-")
        result = pformat_flat(tree, use_bracket_mapping=True)
        assert "(" in result
        assert "-LRB-" not in result

    def test_mapping_disabled_keeps_lrb(self):
        """Without mapping: -LRB- stays as-is."""
        tree = self._make_tree_with_leaf("-LRB-")
        result = pformat_flat(tree, use_bracket_mapping=False)
        assert "-LRB-" in result

    def test_mapping_enabled_maps_rrb(self):
        tree = self._make_tree_with_leaf("-RRB-")
        result = pformat_flat(tree, use_bracket_mapping=True)
        assert ")" in result
        assert "-RRB-" not in result

    def test_mapping_disabled_keeps_rrb(self):
        tree = self._make_tree_with_leaf("-RRB-")
        result = pformat_flat(tree, use_bracket_mapping=False)
        assert "-RRB-" in result

    def test_non_bracket_text_unchanged(self):
        """Normal words are identical regardless of mapping."""
        tree = self._make_tree_with_leaf("hello")
        r1 = pformat_flat(tree, use_bracket_mapping=True)
        r2 = pformat_flat(tree, use_bracket_mapping=False)
        assert r1 == r2

    def test_mapping_param_threaded_to_children(self):
        """use_bracket_mapping is passed recursively to child subtrees."""
        # Build a tree with a child that's also a Tree
        inner = Tree("VP", [Tree("NN", ["-LRB-"])])
        tree = Tree("qlwpcRegen", [Tree("S", [Tree("NP", ["the"]), inner])])
        result = pformat_flat(tree, use_bracket_mapping=True)
        assert "(" in result
        assert "-LRB-" not in result
        result_no = pformat_flat(tree, use_bracket_mapping=False)
        assert "-LRB-" in result_no


# ---------------------------------------------------------------------------
# convert_TG_format with use_bracket_mapping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# convert_TG_format — verify use_bracket_mapping is passed to pformat_flat
# ---------------------------------------------------------------------------

class TestConvertTGFormatFlagThreading:
    """Verify convert_TG_format passes use_bracket_mapping to pformat_flat."""

    def test_flag_true_passed(self):
        import olmo.data.util as util_mod

        with patch("olmo.data.util.pformat_flat") as mock_pf:
            mock_pf.return_value = "output"
            result = util_mod.convert_TG_format(
                "(S (NP test))", use_bracket_mapping=True
            )
            assert mock_pf.call_args[1]["use_bracket_mapping"] is True

    def test_flag_false_passed(self):
        import olmo.data.util as util_mod

        with patch("olmo.data.util.pformat_flat") as mock_pf:
            mock_pf.return_value = "output"
            result = util_mod.convert_TG_format(
                "(S (NP test))", use_bracket_mapping=False
            )
            assert mock_pf.call_args[1]["use_bracket_mapping"] is False

    def test_default_is_true_for_backward_compat(self):
        import olmo.data.util as util_mod

        with patch("olmo.data.util.pformat_flat") as mock_pf:
            mock_pf.return_value = "output"
            result = util_mod.convert_TG_format("(S (NP test))")
            assert mock_pf.call_args[1]["use_bracket_mapping"] is True


# ---------------------------------------------------------------------------
# encode_TG_string auto-detection
# ---------------------------------------------------------------------------

class TestEncodeTGString:
    """Tests for encode_TG_string bracket mapping flag logic."""

    def test_gpt2_tokenizer_disables_mapping(self):
        import olmo.data.util as util_mod

        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = False
        mock_tok.encode.return_value = [1, 2, 3]

        with patch("olmo.data.util.convert_TG_format") as mock_conv:
            mock_conv.return_value = "formatted"
            encode_TG_string(mock_tok, "(S (NP test))",
                             string_with_POS_tags=True)

        assert mock_conv.call_args[1]["use_bracket_mapping"] is False

    def test_qwen3_tokenizer_enables_mapping(self):
        import olmo.data.util as util_mod

        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = True
        mock_tok.encode.return_value = [1, 2, 3]

        with patch("olmo.data.util.convert_TG_format") as mock_conv:
            mock_conv.return_value = "formatted"
            encode_TG_string(mock_tok, "(S (NP test))",
                             string_with_POS_tags=True)

        assert mock_conv.call_args[1]["use_bracket_mapping"] is True

    def test_explicit_override_false(self):
        import olmo.data.util as util_mod

        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = True
        mock_tok.encode.return_value = [1, 2]

        with patch("olmo.data.util.convert_TG_format") as mock_conv:
            mock_conv.return_value = "formatted"
            encode_TG_string(mock_tok, "(S (NP test))",
                             string_with_POS_tags=True,
                             use_bracket_mapping=False)

        assert mock_conv.call_args[1]["use_bracket_mapping"] is False

    def test_explicit_override_true(self):
        import olmo.data.util as util_mod

        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = False
        mock_tok.encode.return_value = [1, 2]

        with patch("olmo.data.util.convert_TG_format") as mock_conv:
            mock_conv.return_value = "formatted"
            encode_TG_string(mock_tok, "(S (NP test))",
                             string_with_POS_tags=True,
                             use_bracket_mapping=True)

        assert mock_conv.call_args[1]["use_bracket_mapping"] is True

    def test_string_without_pos_tags_no_conversion(self):
        """string_with_POS_tags=False skips convert_TG_format entirely."""
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2]
        encode_TG_string(mock_tok, "-LRB- raw -RRB-",
                         string_with_POS_tags=False)
        call_arg = mock_tok.encode.call_args[0][0]
        assert "-LRB- raw -RRB-" == call_arg


# ---------------------------------------------------------------------------
# Edge cases for datatools copy
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TokenizerConfig integration
# ---------------------------------------------------------------------------

def _offline_hf_tokenizer():
    """Minimal WordLevel tokenizer so tests run without network access."""
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"<pad>": 0, "(": 1, ")": 2, "-LRB-": 3, "Cats": 4, "chase": 5, "mice": 6}
    tok = HFTokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    tok.pre_tokenizer = Whitespace()
    return tok


class TestTokenizerConfigIntegration:
    def test_default_is_false(self):
        from olmo.config import TokenizerConfig

        cfg = TokenizerConfig()
        assert cfg.use_bracket_mapping is False

    def test_qwen3_config_true(self):
        from olmo.config import TokenizerConfig

        cfg = TokenizerConfig(use_bracket_mapping=True)
        assert cfg.use_bracket_mapping is True

    def test_flag_stored_on_tokenizer(self):
        from olmo.tokenizer import Tokenizer as OlmoTokenizer

        base = _offline_hf_tokenizer()
        tok = OlmoTokenizer(base, eos_token_id=50256, use_bracket_mapping=True)
        assert tok.use_bracket_mapping is True

    def test_flag_default_false_on_tokenizer(self):
        from olmo.tokenizer import Tokenizer as OlmoTokenizer

        base = _offline_hf_tokenizer()
        tok = OlmoTokenizer(base, eos_token_id=50256)
        assert tok.use_bracket_mapping is False


# ---------------------------------------------------------------------------
# Datatools copy verification
# ---------------------------------------------------------------------------
    """Verify datatools/convert_TG_and_tokenize.py has same behavior."""

    def test_datatools_pformat_flat_has_parameter(self):
        from datatools.parse_pretrain_data.convert_TG_and_tokenize import pformat_flat as dt_pformat

        tree = Tree.fromstring("(qlwpcRegen (S (NP -LRB-)))")
        sig = dt_pformat.__code__.co_varnames
        assert "use_bracket_mapping" in sig, \
            "datatools copy should accept use_bracket_mapping parameter"

    def test_datatools_convert_tg_format_has_parameter(self):
        from datatools.parse_pretrain_data.convert_TG_and_tokenize import convert_TG_format as dt_convert

        sig = dt_convert.__code__.co_varnames
        assert "use_bracket_mapping" in sig, \
            "datatools convert_TG_format should accept use_bracket_mapping parameter"

    def test_datatools_has_helper(self):
        from datatools.parse_pretrain_data.convert_TG_and_tokenize import _is_qwen3_style_tokenizer as dt_check

        mock_tok = MagicMock()
        mock_tok.use_bracket_mapping = False
        assert dt_check(mock_tok) is False
