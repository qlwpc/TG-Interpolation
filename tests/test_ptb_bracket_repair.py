"""Regressions for native bracket text and policy-bound shard rebuilding."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from nltk import Tree
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from types import SimpleNamespace

from datatools.ptb_brackets import normalize_ptb_brackets, NATIVE_BRACKET_POLICY
from datatools.parse_pretrain_data.convert_TG_and_tokenize import (
    encode_tree_document, pformat_flat, resolve_bracket_mapping, tokenize_shard,
)
from scripts.campaigns.fineweb_formula_filter import main as filter_main


@pytest.mark.parametrize('leaf,expected', [
    ('alpha-LRB-beta-RRB-', 'alpha(beta)'),
    ('P-LRB-1+r/n-RRB-nt', 'P(1+r/n)nt'),
    ('-LSB-x-RSB--LCB-y-RCB-', '[x]{y}'),
    ('-LRB-n=370-RRB-', '(n=370)'),
    ('-LRB-', '('),
])
def test_offline_and_runtime_restore_embedded_brackets_without_new_spaces(leaf, expected):
    from olmo.data.util import pformat_flat as runtime_format
    tree = Tree('S', [Tree('NN', [leaf])])
    for format_tree in (pformat_flat, runtime_format):
        assert format_tree(tree, use_bracket_mapping=True) == f'<(S> {expected}<S)>'
        assert format_tree(tree, use_bracket_mapping=False) == f'<(S> {leaf}<S)>'
    assert normalize_ptb_brackets(normalize_ptb_brackets(leaf)) == expected


def test_tree_labels_are_not_normalized():
    tree = Tree('S', [Tree('-LRB-', ['-LRB-']), Tree('NN', ['alpha-LRB-beta-RRB-'])])
    # The preterminal label is omitted, as before. A non-preterminal label must
    # stay structural and must not be rewritten by a global replace on a tree.
    labelled = Tree('-LRB-', [tree])
    assert pformat_flat(labelled) == '<(-LRB-><(S> ( alpha(beta)<S)><-LRB-)>'


@pytest.mark.parametrize('size,sep,expected', [(151732, 151673, True), (50320, 50261, False)])
def test_bare_tokenizer_layout_routes_policy_without_python_flag(size, sep, expected):
    tokenizer = SimpleNamespace(get_vocab_size=lambda: size, token_to_id=lambda _: sep)
    assert resolve_bracket_mapping(tokenizer) is expected


def test_unknown_layout_requires_explicit_policy():
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0}, unk_token='<unk>'))
    with pytest.raises(ValueError, match='unregistered'):
        resolve_bracket_mapping(tokenizer)
    assert resolve_bracket_mapping(tokenizer, 'native')
    assert not resolve_bracket_mapping(tokenizer, 'ptb')


@pytest.mark.parametrize('single_word', [True, False])
def test_structural_ids_cannot_leak_into_terminal_text(single_word):
    tok = Tokenizer(models.WordLevel({'<unk>': 0, 'A': 1}, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens([AddedToken(marker, single_word=single_word, special=True)
                            for marker in ['<|beginoftext|>', '<|endoftext|>', '<(NP>', '<NP)>']])
    vocab = SimpleNamespace(bos=tok.token_to_id('<|beginoftext|>'), eos=tok.token_to_id('<|endoftext|>'))
    text = '<(NP> A<NP)>'
    ids = encode_tree_document(text, tok, vocab, 'uint16').tolist()
    assert ids == [vocab.bos, tok.token_to_id('<(NP>'), 1, tok.token_to_id('<NP)>'), vocab.eos]
    if not single_word:
        assert ids == [vocab.bos, *tok.encode(text).ids, vocab.eos]


def test_terminal_text_and_resume_are_bound_to_bracket_policy(tmp_path):
    tok = Tokenizer(models.WordLevel({'<unk>': 0, 'alpha(beta)': 1, 'alpha-LRB-beta-RRB-': 2}, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(['<|beginoftext|>', '<|endoftext|>', '<(S>', '<S)>'])
    tok.use_bracket_mapping = True
    op, cl = tok.token_to_id('<(S>'), tok.token_to_id('<S)>')
    vocab = SimpleNamespace(bos=tok.token_to_id('<|beginoftext|>'), eos=tok.token_to_id('<|endoftext|>'),
        convert_treenpy_to_terminal=lambda a: a[~np.isin(a, [op, cl])],
        convert_treenpy_to_TG=lambda a: np.repeat(a, [2 if int(i) == cl else 1 for i in a]))
    source = tmp_path/'a.txt'
    source.write_text('(S (NN alpha-LRB-beta-RRB-))\n')
    root = tmp_path/'tokens'
    receipt = tokenize_shard(source, root, tok, vocab, 'tokenizer-hash')
    assert receipt['bracket_policy'] == NATIVE_BRACKET_POLICY
    assert np.load(root/'terminal/a.npy').tolist() == [vocab.bos, 1, vocab.eos]
    with pytest.raises(ValueError, match='normalization policy'):
        tokenize_shard(source, root, tok, vocab, 'tokenizer-hash', use_bracket_mapping=False)
    manifest = root/'manifests/a.json'
    legacy = json.loads(manifest.read_text())
    legacy.pop('bracket_policy')
    manifest.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match='normalization policy'):
        tokenize_shard(source, root, tok, vocab, 'tokenizer-hash', use_bracket_mapping=True)
    tokenize_shard(source, root, tok, vocab, 'tokenizer-hash', use_bracket_mapping=False, overwrite=True)
    assert np.load(root/'terminal/a.npy').tolist() == [vocab.bos, 2, vocab.eos]


def test_candidate_export_preserves_source_identity_and_moves_evidence_to_clean_text(tmp_path):
    originals = [
        {'sample_id': '0', 'text': 'The study -LRB-n = 370-RRB- reported outcomes.'},
        {'sample_id': '1', 'text': 'The formula A = P-LRB-1 + r / n-RRB-nt uses compound interest.'},
    ]
    for r in originals:
        r.update(tokens=100, start_token=20, end_token=120, file='/source/a.npy', token_bytes_sha256='source-hash')
    source = tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in originals))
    for policy in ['native-brackets', 'source']:
        output = tmp_path/policy
        filter_main(['--input', str(source), '--output', str(output), '--text-policy', policy])
        records = [json.loads(line) for path in output.glob('*.jsonl') for line in path.open()]
        assert len(records) == len(originals)
        for r in records:
            original = originals[int(r['sample_id'])]
            assert r['text'] == (normalize_ptb_brackets(original['text']) if policy == 'native-brackets' else original['text'])
            assert r.get('source_text', r['text']) == original['text']
            for key in ['tokens', 'start_token', 'end_token', 'token_bytes_sha256']:
                assert r[key] == original[key]
            assert r['text_normalization']['source_text_sha256'] == hashlib.sha256(original['text'].encode()).hexdigest()
            assert r['formula_filter']['decision'] == ('keep' if r['sample_id'] == '0' else 'drop')
            for signal in r['formula_filter']['signals']:
                assert r['text'][signal['start_char']:signal['end_char']][:180] == signal['matched']
