"""CPU parity for candidate-0 caches, window slides and document restarts."""
from types import SimpleNamespace

import pytest
import torch

from olmo.config import ModelConfig
from olmo.model import OLMo
from olmo.eval import pushdown_document_ppl as ppl
from olmo.pushdown import compute_depth_matrix_gpu, compute_depth_rows_gpu


@pytest.fixture
def tiny_model():
    torch.manual_seed(77)
    config = ModelConfig(d_model=32, n_heads=4, n_layers=2, mlp_hidden_size=64,
                         vocab_size=32, embedding_size=32, max_sequence_length=32,
                         transformer_grammar_type="pushdown", pushdown_use_flex=False,
                         pushdown_use_attachment_head_inference=True,
                         bos_token_id=0, eos_token_id=1, pad_token_id=2,
                         attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0,
                         rope=True, flash_attention=False, init_device="cpu")
    return OLMo(config).eval()


def candidates():
    tokens = (0, 10, 11, 12, 1)
    return (ppl._native_candidate(tokens, ((1, 1, 2), (1, 2, 3)), (1, 4)),
            ppl._native_candidate(tokens, ((2, 2, 3), (1, 1, 3)), (1, 4)))


class TinyCorpus(ppl.NativePushdownTopKCorpus):
    def __init__(self):
        self.vocab = SimpleNamespace(bos=0)
        self.samples_per_sentence = 2
        self.rows = [(0, candidates()), (0, candidates()), (0, candidates()),
                     (0, candidates()), (1, candidates()), (1, candidates())]

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


@pytest.mark.parametrize("normalization", ["stack_legal", "sentence_causal"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("weight_tying", [False, True])
def test_cached_scores_match_full_prefix(tiny_model, normalization, joint, weight_tying):
    model = tiny_model
    if not weight_tying:
        model.config.weight_tying = False
        model.transformer.ff_out = torch.nn.Linear(32, 32, bias=False)
    first = candidates()
    _, cache = ppl.score_pushdown_native_candidates(model, (), first, "cpu", joint, normalization,
                                                   return_candidate0_cache=True)
    following = tuple(ppl._drop_leading_bos(c, 0) for c in first)
    full = ppl.score_pushdown_native_candidates(model, (first[0],), following, "cpu", joint, normalization)
    cached, next_cache = ppl.score_pushdown_native_candidates(model, (first[0],), following, "cpu", joint, normalization,
                                                            prefix_cache=cache, return_candidate0_cache=True)
    for key in ("token_nll", "attachment_nll", "joint_nll"):
        torch.testing.assert_close(getattr(cached, key), getattr(full, key), atol=2e-5, rtol=1e-6)
    assert next_cache.context == (first[0], following[0])
    # Slicing without cloning would retain the entire candidate batch storage.
    for key, value in next_cache.key_values:
        assert key.untyped_storage().nbytes() == key.numel() * key.element_size()
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    with pytest.raises(ValueError, match="context"):
        ppl.score_pushdown_native_candidates(model, (), following, "cpu", prefix_cache=cache)


@pytest.mark.parametrize("normalization", ["stack_legal", "sentence_causal"])
def test_document_window_slides_and_resume_match_full_prefix(tiny_model, normalization):
    rows = []
    corpus = TinyCorpus()
    options = dict(device="cpu", eval_batch_size=2, max_sequence_length=10,
                   attachment_normalization=normalization)
    cached = ppl.evaluate_pushdown_document_ppl(tiny_model, corpus, document_complete=lambda _, r: rows.append(r), **options)
    full = ppl.evaluate_pushdown_document_ppl(tiny_model, corpus, use_kv_cache=False, **options)
    assert cached.kv_cache_hits > 0
    assert cached.kv_cache_rebuilds > 0
    for field in ("legacy_log_likelihood", "uniform_mixture_log_likelihood", "token_only_log_likelihood"):
        assert getattr(cached, field) == pytest.approx(getattr(full, field), abs=1e-4)
        assert sum(r[field] for r in rows) == pytest.approx(getattr(cached, field))
    resumed_rows = []
    resumed = ppl.evaluate_pushdown_document_ppl(tiny_model, corpus, completed_document_ids={0},
                                                document_complete=lambda _, r: resumed_rows.append(r), **options)
    assert [r["document_id"] for r in resumed_rows] == [1]
    assert resumed.sentence_count == 2
    assert resumed.document_count == 1
    assert resumed.terminal_count == rows[1]["terminal_count"]
    assert resumed.legacy_log_likelihood == pytest.approx(rows[1]["legacy_log_likelihood"])
    empty = ppl.evaluate_pushdown_document_ppl(tiny_model, corpus, completed_document_ids={0, 1}, **options)
    assert empty.sentence_count == empty.document_count == empty.terminal_count == 0


def test_oom_retries_same_candidates_without_duplicate_documents(tiny_model, monkeypatch):
    scorer = ppl.score_pushdown_native_candidates
    sizes = []
    def limited(model, prefix, candidates, *args, **kwargs):
        sizes.append(len(candidates))
        if len(candidates) > 1:
            raise torch.OutOfMemoryError("synthetic candidate-batch OOM")
        return scorer(model, prefix, candidates, *args, **kwargs)
    monkeypatch.setattr(ppl, "score_pushdown_native_candidates", limited)
    rows = []
    result = ppl.evaluate_pushdown_document_ppl(tiny_model, TinyCorpus(), "cpu", eval_batch_size=2,
                                               document_complete=lambda _, r: rows.append(r))
    assert 1 in sizes and 2 in sizes
    assert len(rows) == result.document_count == 2
    assert result.candidate_slots == 12


@pytest.mark.parametrize("start,end", [(0, 7), (1, 3), (3, 7), (6, 7)])
def test_current_depth_range_implementation_covers_remote_multirow_optimization(start, end):
    spans = torch.tensor([[[0, 1, 3], [1, 2, 2], [4, 4, 6], [-1, -1, -1]],
                          [[0, 2, 6], [2, 3, 5], [3, 3, 3], [3, 3, 3]]])
    torch.testing.assert_close(compute_depth_rows_gpu(spans, 7, start, end),
                               compute_depth_matrix_gpu(spans, 7)[:, start:end])


@pytest.mark.parametrize("binary", [False, True])
def test_best_cache_retains_nonzero_winner_and_scores_next_sentence(tiny_model, binary):
    if binary:
        from olmo.eval.gpst_binary_pushdown_document_ppl import score_gpst_binary_pushdown_candidates as score
        def run(prefix, trees, **kwargs):
            return score(tiny_model, prefix, trees, "cpu", **kwargs)
    else:
        def run(prefix, trees, **kwargs):
            result = ppl.score_pushdown_native_candidates(tiny_model, prefix, trees, "cpu", **kwargs)
            return result if isinstance(result, tuple) else (result, None)
    trees = candidates()
    raw, _ = run((), trees)
    # Put the actual less likely tree in slot zero, so the cache must select row 1.
    order = raw.joint_nll.argsort(descending=True).tolist()
    trees = tuple(trees[i] for i in order)
    scores, cache = run((), trees, return_best_cache=True)
    assert scores.joint_nll[1] < scores.joint_nll[0]
    assert cache.context == (trees[1],)
    following = tuple(ppl._drop_leading_bos(c, 0) for c in trees)
    cached, _ = run(cache.context, following, prefix_cache=cache)
    full, _ = run(cache.context, following)
    torch.testing.assert_close(cached.joint_nll, full.joint_nll, atol=2e-5, rtol=1e-6)
    for key, value in cache.key_values:
        assert key.untyped_storage().nbytes() == key.numel() * key.element_size()
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_document_selection_matches_greedy_full_prefix_and_microbatching(tiny_model):
    corpus = TinyCorpus()
    corpus.rows = [(doc, tuple(reversed(trees))) for doc, trees in corpus.rows]
    prefix = ()
    previous_doc = None
    nonzero = 0
    for doc, trees in corpus:
        if doc != previous_doc:
            prefix = ()
        else:
            trees = tuple(ppl._drop_leading_bos(c, 0) for c in trees)
        context = ppl._trim_prefix(prefix, trees[0], 10)
        scores = ppl.score_pushdown_native_candidates(tiny_model, context, trees, "cpu")
        selected = int(scores.joint_nll.argmin())
        nonzero += int(selected != 0)
        prefix = context + (trees[selected],)
        previous_doc = doc
    assert nonzero > 0
    for batch_size in (1, 2):
        rows = []
        result = ppl.evaluate_pushdown_document_ppl(tiny_model, corpus, "cpu", eval_batch_size=batch_size,
                                                   max_sequence_length=10,
                                                   document_complete=lambda _, row: rows.append(row))
        assert result.prefix_policy == "model_best"
        assert result.non_candidate0_count == nonzero
        assert result.non_candidate0_ratio == pytest.approx(nonzero / len(corpus))
        assert sum(row["non_candidate0_count"] for row in rows) == nonzero
