from functools import lru_cache

import numpy as np
import pytest

from datatools.parse_test_docppl_data.labeled_topk import LabeledKBest, serialize_candidate


def brute(scores):
    @lru_cache(None)
    def enumerate_cell(state, left, right):
        real = state == 'R' or (state == 'B' and left != right)
        labels = range(1 if real else 0, scores.shape[-1])
        forests = [(0., ())]
        if left != right:
            forests = []
            for split in range(left, right):
                for sa, a in enumerate_cell('A', left, split):
                    for sb, b in enumerate_cell('B', split + 1, right):
                        forests.append((sa + sb, a + b))
        return tuple((s + scores[left, right, label], spans + (((left, right, label),) if label else ()))
                     for label in labels for s, spans in forests)
    return sorted(enumerate_cell('R', 0, len(scores) - 1), key=lambda x: -x[0])


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_exact_kbest_against_exhaustive_grammar(n):
    scores = np.random.default_rng(571 + n).normal(size=(n, n, 3))
    expected = brute(scores)
    actual = LabeledKBest(scores).topk(300)
    assert len(actual) == min(300, len(expected))
    assert len({spans for _, spans in actual}) == len(actual)
    np.testing.assert_allclose([s for s, _ in actual], [s for s, _ in expected[:300]], atol=1e-12)
    assert [t for _, t in actual] == [t for _, t in expected[:300]]


def test_alias_normalization_ranks_unique_trees_by_max_source_score():
    scores = np.asarray([[[0., 2., 3., 1.]]])
    actual = LabeledKBest(scores, {0: '', 1: 'ADJ', 2: 'ADJP', 3: 'NP'}).topk()
    assert actual == [(3., ((0, 0, 2),)), (1., ((0, 0, 3),))]


def test_serialization_preserves_original_ids_outer_tokens_and_unary_chains():
    spans = ((0, 0, 1), (1, 2, 2), (0, 2, 3))
    pairs = {1: ((100, 101), (111, 110)), 2: ((102,), (112,)), 3: ((103,), (113,))}
    result = serialize_candidate(spans, ((10, 11), (12,), (13, 14)), pairs, (1,), (198, 2))
    assert result.tolist() == [1, 103, 100, 101, 10, 11, 111, 110, 102, 12, 13, 14, 112, 113, 198, 2]


def test_chunk_finalize_and_independent_audit_reject_corruption(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from datatools.reserved_clean.build import encode_document
    from datatools.parse_test_docppl_data.reproduce_bbc_test import legacy_tokenizer
    from datatools.parse_test_docppl_data.generate_native_topk import CanonicalPPLCorpus, _sha256_path
    from datatools.parse_test_docppl_data.generate_labeled_topk import decode, finalize, sources
    from datatools.parse_test_docppl_data.audit_labeled_topk import run, audit_block
    from datatools.parse_test_docppl_data.labeled_topk import label_token_pairs
    root = Path(__file__).resolve().parents[1]
    tokenizer_path = root / 'dataset/bbc-news/TG_GPT2_tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    arrays, bounds, _ = encode_document('(S (NP (NN Alice)) (VP (VB runs))) (S (NN wow))', legacy_tokenizer(tokenizer))
    canonical = tmp_path / 'canonical'
    canonical.mkdir()
    for name, data in [('tree', arrays['tree']), ('sentence_offsets', bounds),
                       ('document_sentence_counts', np.asarray([2], dtype=np.uint32))]:
        np.save(canonical / (name + '.npy'), data)
    (canonical / 'manifest.json').write_text(json.dumps(dict(format='canonical-tree-sentences-v1',
        tree_path='tree.npy', document_count=1, sentence_count=2,
        **{name + '_sha256': _sha256_path(canonical / (name + '.npy'))
           for name in ('tree', 'sentence_offsets', 'document_sentence_counts')})))
    corpus = CanonicalPPLCorpus(canonical, tokenizer_path)
    labels = {0: '', 1: 'S', 2: 'NP', 3: 'VP'}
    pairs = label_token_pairs(labels, tokenizer)
    results = []
    for i in range(2):
        row = corpus.sentence_input(i)
        chart = np.random.default_rng(i).normal(size=(len(row.words), len(row.words), 4))
        results.append(decode((row, chart, labels, pairs)))
    out = tmp_path / 'tree300'
    shard = out / 'shard-00000-of-00001'
    shard.mkdir(parents=True)
    block = dict(first=0, last=2, tokens=np.concatenate([x[0] for x in results]),
        lengths=np.stack([x[1] for x in results]), scores=np.stack([x[2] for x in results]),
        valid_counts=np.asarray([x[3] for x in results], dtype=np.uint16))
    np.savez(shard / 'block-00000000-00000002.npz', **block)
    args = SimpleNamespace(ppl_dir=canonical, test_tree=canonical / 'tree.npy', tokenizer=tokenizer_path,
                           num_shards=1, output=out, model='synthetic')
    (shard / 'sources.json').write_text(json.dumps(sources(args, corpus)))
    (shard / 'run.json').write_text(json.dumps(dict(status='complete', max_sentences=None,
                                                   sentence_start=0, sentence_end=2)))
    finalize(args)
    audit_args = SimpleNamespace(canonical=canonical, tokenizer=tokenizer_path, input=out,
                                output=tmp_path / 'audit.json', smoke=False)
    run(audit_args)
    assert json.loads(audit_args.output.read_text())['sentences'] == 2
    assert int(np.load(out / 'valid_counts.npy')[-1]) == 3
    corrupted = block['tokens'].copy()
    closing_ids = {t for _, close in pairs.values() for t in close}
    pos = next(i for i, t in enumerate(corrupted) if int(t) in closing_ids)
    corrupted[pos] = next(t for t in closing_ids if t != corrupted[pos])
    with pytest.raises(ValueError, match='label mismatch'):
        audit_block(corrupted, block['lengths'], block['scores'], block['valid_counts'], corpus, 0)
