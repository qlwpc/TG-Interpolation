import math

import numpy as np
import pytest

from scripts.projected_candidate_utils import deduplicate_samples, stable_unique_indices


def test_exact_stable_identity_and_distinct_probability_mass():
    first, inverse = stable_unique_indices([[1, 2], [1, 3], [1, 2], [1, 3], [1, 4]])
    assert first.tolist() == [0, 1, 4]
    assert inverse.tolist() == [0, 1, 0, 1, 2]
    losses = np.array([2., 3., 2., 3., 4.])
    unique_nll = -math.log(np.exp(-losses[first]).sum())
    original_nll = -math.log(np.exp(-losses).sum())
    assert unique_nll > original_nll
    assert unique_nll == pytest.approx(-math.log(math.exp(-2)+math.exp(-3)+math.exp(-4)))


def test_identity_is_complete_unpadded_sequence_and_integer_width_independent():
    first, inverse = stable_unique_indices([
        np.array([1, 2], dtype=np.uint16), np.array([1, 2], dtype=np.int64),
        np.array([1, 2, 0]), np.array([2, 1])])
    assert first.tolist() == [0, 2, 3]
    assert inverse.tolist() == [0, 0, 1, 2]


def test_deduplication_preserves_first_metadata_and_rejects_different_targets():
    samples = [dict(input_ids=[1, 2], label_mask=[False, True], index=i) for i in (7, 9)]
    unique, first, inverse = deduplicate_samples(samples)
    assert unique[0] is samples[0]
    assert unique[0]['index'] == 7
    samples[1]['label_mask'] = [True, True]
    with pytest.raises(ValueError, match='target mask'):
        deduplicate_samples(samples)


def test_actual_noont_and_compress_collisions():
    from pathlib import Path
    from olmo.data.tg_mask import SentencepieceVocab
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[1]
    path = root/'dataset/bbc-news/TG_GPT2_tokenizer.json'
    if not path.is_file():
        path = root/'reproducibility/tokenizer-reference.json'
    vocab = SentencepieceVocab.from_vocab_file(str(path))
    tokens = Tokenizer.from_file(str(path)).get_vocab()
    def tree(parts):
        return np.array([tokens.get(t, int(t) if t.isdigit() else -1) for t in parts.split()], dtype=np.int64)
    # NoONT retains labels on closes; moving an opening produces a collision.
    a = tree('<(S> <(NP> 1 <(VP> 2 <VP)> 3 <NP)> <S)>')
    b = tree('<(S> <(NP> <(VP> 1 2 <VP)> 3 <NP)> <S)>')
    assert stable_unique_indices([a, b])[0].tolist() == [0, 1]
    assert stable_unique_indices([vocab.convert_treenpy_to_noont(x) for x in (a, b)])[0].tolist() == [0]
    c = tree('<(S> <(NP> 1 <(VP> 2 <(PP> 3 4 <PP)> 5 <VP)> <NP)> 6 <S)>')
    d = tree('<(S> <(NP> 1 <(VP> 2 <(PP> 3 4 <PP)> <VP)> 5 <NP)> 6 <S)>')
    assert stable_unique_indices([c, d])[0].tolist() == [0, 1]
    assert stable_unique_indices([vocab.convert_treenpy_to_compress(x) for x in (c, d)])[0].tolist() == [0]
