import json
import math

import pytest

from scripts.merge_reserved_docppl import merge_documents


def make_results(path):
    for doc, sent, tokens in [(0, 0, 2), (1, 1, 3)]:
        row = dict(document_id=doc, fingerprint='frozen', sentence_count=1,
                   terminal_count=tokens, total_nll=2.0 * tokens,
                   non_candidate0_count=int(doc == 1), job_id=str(doc),
                   sentences=[dict(sentence_id=sent, nll=2.0 * tokens, terminal_count=tokens,
                                   valid_candidates=122, selected_candidate=doc)])
        (path / f'document_{doc:05d}.json').write_text(json.dumps(row))


def test_merge_is_token_weighted_and_validates_exact_coverage(tmp_path):
    make_results(tmp_path)
    result = merge_documents(tmp_path, [1, 1], [2, 3])
    assert result['docppl'] == pytest.approx(math.exp(2))
    assert result['non_candidate0_ratio'] == 0.5
    (tmp_path / 'document_00001.json').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        merge_documents(tmp_path, [1, 1], [2, 3])


def test_merge_requires_the_frozen_logical_candidate_counts(tmp_path):
    make_results(tmp_path)
    assert merge_documents(tmp_path, [1, 1], [2, 3], [122, 122])['documents'] == 2
    with pytest.raises(ValueError, match='logical counts'):
        merge_documents(tmp_path, [1, 1], [2, 3], [300, 122])


@pytest.mark.parametrize('field,value,error', [
    ('fingerprint', 'old_history', 'fingerprints'),
    ('terminal_count', 4, 'denominator'),
    ('document_id', 0, 'identity'),
    ('total_nll', float('nan'), 'NLL'),
])
def test_merge_rejects_corrupt_or_mixed_runs(tmp_path, field, value, error):
    make_results(tmp_path)
    path = tmp_path / 'document_00001.json'
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match=error):
        merge_documents(tmp_path, [1, 1], [2, 3])
