import pytest

from diagnostics.audit_bbc_raw_duplicates import fast_legacy_format, legacy_format, summarize_groups


@pytest.mark.parametrize('parsed', [
    '(S (NP (NNP Alice)) (VP (VBZ runs)) (. .)) (Ċ Ċ)',
    '(S (ADJ (JJ good)) (. !))',
    '(S (NP (NN report)) (VP (VBD) (VB ran)))',
    '(S (NP) (. .))',
    '(NP (NN -LRB-) (NN café) (NN -RRB-))',
    '(X bare words)',
    '',
])
def test_fast_formatter_matches_historical_tree_formatter(parsed):
    vocab = {t: i for i, t in enumerate(['<(S>', '<S)>', '<(NP>', '<NP)>', '<(VP>', '<VP)>'])}
    assert fast_legacy_format(parsed, vocab) == legacy_format(parsed, vocab)


def test_empty_unknown_label_keeps_two_spaces():
    assert fast_legacy_format('(VBD)', {}) == ' (VBD  VBD)'


def test_duplicate_definitions_and_filtered_shard_masks():
    groups = {
        b'a': [b'', 4, 3, 1, 0, 0, 0, 0, 0b101, None],
        b'b': [b'', 2, 2, 0, 0, 0, 0, 0, 0b010, None],
        b'c': [b'', 1, 0, 1, 0, 0, 0, 0, 0b100, None],
    }
    all_rows = summarize_groups(groups, 7, 1)
    assert all_rows['unique_terminal_documents'] == 3
    assert all_rows['extra_duplicate_copies'] == 4
    assert all_rows['documents_in_duplicate_groups'] == 6
    assert all_rows['groups_present_in_multiple_shards'] == 1
    historic = summarize_groups(groups, 5, 2, 0b011)
    assert historic['extra_duplicate_copies'] == 3
    assert historic['groups_present_in_multiple_shards'] == 0
    reserved = summarize_groups(groups, 2, 3, 0b100)
    assert reserved['extra_duplicate_copies'] == 0
