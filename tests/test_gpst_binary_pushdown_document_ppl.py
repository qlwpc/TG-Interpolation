from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from olmo.config import (
    ActivationType,
    BlockType,
    InitFnType,
    LayerNormType,
    ModelConfig,
)
from olmo.data.parse_align import binarize_tree, collapse_unary_tree, tree_spans
from olmo.eval.gpst_binary_pushdown_document_ppl import (
    _binary_attachment_actions,
    _build_prefix_cache,
    _v1_attachment_nll_from_hidden,
    _v2_attachment_nll_from_hidden,
    evaluate_gpst_binary_pushdown_document_ppl,
    gpst_merge_orders_to_pushdown_spans,
    gpst_merge_orders_to_spliced_bpe_spans,
    right_binarize_native_nary_spans,
    right_binarize_native_nary_spans_with_word_atoms,
    score_gpst_binary_pushdown_candidates,
)
from olmo.eval.pushdown_document_ppl import (
    PushdownCandidateScores,
    PushdownGoldCandidate,
    _attachment_nll_from_logits,
    _native_candidate,
    _trim_prefix,
)
from olmo.model import OLMo
from olmo.pushdown import compute_depth_matrix_gpu, compute_depth_rows_gpu
from scripts.merge_gpst_binary_pushdown_document_ppl import merge, validate_full_bbc


def _tiny_model() -> OLMo:
    config = ModelConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        mlp_hidden_size=32,
        vocab_size=32,
        embedding_size=32,
        max_sequence_length=32,
        block_type=BlockType.sequential,
        layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu,
        init_fn=InitFnType.normal,
        init_std=0.02,
        init_device="cpu",
        rope=True,
        flash_attention=False,
        flex_attention=False,
        pushdown_use_flex=False,
        transformer_grammar_type="pushdown",
        pushdown_max_depth=8,
        weight_tying=True,
        bos_token_id=7,
        eos_token_id=8,
        pad_token_id=9,
    )
    return OLMo(config).eval()


def test_merge_orders_convert_to_exact_postorder_binary_spans():
    orders = np.asarray([[0, 2, 1], [2, 1, 0]], dtype=np.int16)
    spans = gpst_merge_orders_to_pushdown_spans(
        orders, content_start=1, content_length=4
    )
    assert spans.tolist() == [
        [[1, 1, 2], [3, 3, 4], [1, 2, 4]],
        [[3, 3, 4], [2, 2, 4], [1, 1, 4]],
    ]
    with pytest.raises(ValueError, match="permutation"):
        gpst_merge_orders_to_pushdown_spans(
            np.asarray([[0, 0, 1]]), content_start=0, content_length=4
        )
    with pytest.raises(ValueError, match="outside"):
        gpst_merge_orders_to_pushdown_spans(
            np.asarray([[-1, 1, 2]]),
            content_start=0,
            content_length=4,
            validate=False,
        )
    with pytest.raises(ValueError, match="full-sentence root"):
        gpst_merge_orders_to_pushdown_spans(
            np.asarray([[0, 0, 0]]),
            content_start=0,
            content_length=4,
            validate=False,
        )


def test_binary_attachment_fast_path_matches_reference_stack_derivation():
    orders = np.asarray([[0, 2, 1], [2, 1, 0]], dtype=np.int16)
    spans = gpst_merge_orders_to_pushdown_spans(
        orders, content_start=1, content_length=4
    )
    targets, legal = _binary_attachment_actions(spans, 6, 1, 5)
    sentence_ids = torch.tensor([[-1, 0, 0, 0, 0, -1]]).expand(2, -1)
    from olmo.attachment import derive_gold_attachment_actions

    reference_targets, reference_legal = derive_gold_attachment_actions(
        torch.from_numpy(spans), sentence_ids
    )
    assert targets.tolist() == reference_targets.tolist()
    assert legal == [[tuple(keys) for keys in row] for row in reference_legal]


def test_native_nary_spliced_bpe_right_binarization_deduplicates():
    # These two distinct n-ary structures both right-binarize to
    # token0 + (token1 + token2), so the binary latent support keeps one row.
    nary = np.full((2, 2, 3), -1, dtype=np.int64)
    nary[0, 0] = (1, 3, 3)
    nary[1, 0] = (2, 3, 3)
    nary[1, 1] = (1, 3, 3)
    binary, sources = right_binarize_native_nary_spans(
        nary, np.asarray([1, 2]), content_left=1, content_right=4
    )
    assert sources.tolist() == [0]
    assert binary.tolist() == [[[1, 1, 3], [2, 2, 3]]]


def test_gpst_word_topology_can_be_reexpanded_with_spliced_bpe_rule():
    # Three recovered words have BPE lengths 2, 1, 2. The stored direct axis
    # first closes its fixed word atoms (gaps 0 and 3), then forms the word tree
    # w0 + (w1 + w2) at cross-word gaps 2 and 1.
    stored_orders = np.asarray([[0, 3, 2, 1]], dtype=np.int64)
    spliced, sources = gpst_merge_orders_to_spliced_bpe_spans(
        stored_orders,
        np.asarray([0, 2, 3]),
        content_left=0,
        content_right=5,
    )
    assert sources.tolist() == [0]
    assert {tuple(span) for span in spliced[0]} == {
        (0, 0, 4),
        (1, 1, 4),
        (2, 2, 4),
        (3, 3, 4),
    }
    stored = gpst_merge_orders_to_pushdown_spans(
        stored_orders, content_start=0, content_length=5
    )
    assert {tuple(span) for span in stored[0]} != {tuple(span) for span in spliced[0]}

    # With one BPE per word, re-expansion is topology-identical.
    single_bpe_orders = np.asarray([[1, 0]], dtype=np.int64)
    reexpanded, _ = gpst_merge_orders_to_spliced_bpe_spans(
        single_bpe_orders,
        np.asarray([0, 1, 2]),
        content_left=0,
        content_right=3,
    )
    direct = gpst_merge_orders_to_pushdown_spans(
        single_bpe_orders, content_start=0, content_length=3
    )
    assert {tuple(span) for span in reexpanded[0]} == {
        tuple(span) for span in direct[0]
    }

    # A nested four-token tree: ((0,1),2,3) becomes
    # ((0,1),(2,3)) under right binarization of the three root children.
    nested = np.full((1, 2, 3), -1, dtype=np.int64)
    nested[0, 0] = (1, 2, 2)
    nested[0, 1] = (1, 4, 4)
    binary, sources = right_binarize_native_nary_spans(
        nested, np.asarray([2]), content_left=1, content_right=5
    )
    assert sources.tolist() == [0]
    assert {tuple(span) for span in binary[0]} == {
        (1, 1, 2),
        (3, 3, 4),
        (1, 2, 4),
    }

    # One parser word split into three BPE terminals has no real n-ary spans;
    # both representations reduce to the same right-recursive tree.
    empty = np.empty((1, 0, 3), dtype=np.int64)
    binary, sources = right_binarize_native_nary_spans(
        empty, np.asarray([0]), content_left=1, content_right=4
    )
    assert sources.tolist() == [0]
    assert binary.tolist() == [[[1, 1, 3], [2, 2, 3]]]


def test_native_nary_word_atoms_exactly_match_direct_training_representation():
    # Three words have BPE lengths 2, 1, 2 and word tree w0 + (w1 + w2).
    # The native axis stores only its two real word constituents in BPE bounds.
    nary = np.asarray([[[2, 4, 4], [0, 4, 4]]], dtype=np.int64)
    fixed, sources = right_binarize_native_nary_spans_with_word_atoms(
        nary,
        np.asarray([2]),
        np.asarray([0, 2, 3]),
        content_left=0,
        content_right=5,
    )
    direct = gpst_merge_orders_to_pushdown_spans(
        np.asarray([[0, 3, 2, 1]], dtype=np.int64),
        content_start=0,
        content_length=5,
    )
    assert sources.tolist() == [0]
    assert {tuple(span) for span in fixed[0]} == {
        tuple(span) for span in direct[0]
    }
    assert {tuple(span) for span in fixed[0]} == {
        (0, 0, 1),
        (3, 3, 4),
        (2, 2, 4),
        (0, 1, 4),
    }


def test_stored_direct_adapter_matches_actual_parse_preprocessing_functions():
    tree = (
        "S",
        [
            ("W0", [10, 11]),
            ("W1", [12]),
            ("W2", [13, 14, 15]),
        ],
    )
    preprocessed = binarize_tree(collapse_unary_tree(tree), direction="right")
    leaves, spans = tree_spans(preprocessed)
    actual_pushdown_spans = {span for span in spans if span[0] < span[2]}
    direct = gpst_merge_orders_to_pushdown_spans(
        np.asarray([[0, 4, 3, 2, 1]], dtype=np.int64),
        content_start=0,
        content_length=6,
    )
    assert leaves == [10, 11, 12, 13, 14, 15]
    assert actual_pushdown_spans == {tuple(span) for span in direct[0]}


def test_cached_depth_rows_equal_full_matrix_for_multi_token_suffix():
    spans = torch.tensor(
        [
            [[0, 0, 2], [1, 1, 2], [3, 4, 6], [-1, -1, -1]],
            [[0, 2, 6], [2, 3, 5], [3, 3, 3], [5, 5, 6]],
        ],
        dtype=torch.long,
    )
    full = compute_depth_matrix_gpu(spans, 7)
    assert torch.equal(compute_depth_rows_gpu(spans, 7, 3, 7), full[:, 3:7])
    assert torch.equal(compute_depth_rows_gpu(spans, 7, 6, 7), full[:, 6:7])


def test_context_truncation_keeps_only_complete_sentence_suffix():
    prefix = (
        _candidate((1, 2, 3), 0),
        _candidate((4, 5, 6, 7), 0),
        _candidate((8, 9), 0),
    )
    current = _candidate((10, 11, 12), 0)
    assert _trim_prefix(prefix, current, 8) == (prefix[-1],)
    with pytest.raises(ValueError, match="one sentence"):
        _trim_prefix(prefix, _candidate(tuple(range(9)), 0), 8)


def test_sparse_v1_hidden_scoring_matches_dense_mask_then_softmax():
    torch.manual_seed(17)
    model = _tiny_model()
    batch, total, prefix = 2, 5, 2
    hidden = torch.randn(batch, total, model.config.d_model)
    full_ids = torch.tensor([[7, 1, 2, 3, 4], [7, 5, 6, 7, 8]])
    current_ids = full_ids[:, prefix:]
    targets = torch.tensor([[2, 2, 3], [2, 3, 4]])
    legal = torch.tensor(
        [
            [[2, -1, -1], [3, 2, -1], [4, 3, 2]],
            [[2, -1, -1], [3, 2, -1], [4, 3, -1]],
        ]
    )
    sparse = _v1_attachment_nll_from_hidden(
        model, hidden, current_ids, targets, legal, prefix
    )
    dense_logits = model.pushdown_attachment_head(
        hidden,
        full_ids,
        model.transformer.wte.weight,
        query_range=(prefix, total),
    )
    dense_mask = torch.zeros_like(dense_logits, dtype=torch.bool)
    for b in range(batch):
        for q in range(total - prefix):
            keys = legal[b, q]
            dense_mask[b, q, keys[keys >= 0]] = True
    dense = _attachment_nll_from_logits(
        dense_logits, targets, dense_mask, "stack_legal"
    )
    assert torch.allclose(sparse, dense, atol=1e-6, rtol=1e-6)


def test_sentence_causal_v2_hidden_scoring_matches_dense_head():
    torch.manual_seed(19)
    model = _tiny_model()
    batch, total, prefix = 2, 6, 2
    hidden = torch.randn(batch, total, model.config.d_model)
    full_ids = torch.tensor([[7, 1, 2, 3, 4, 8], [7, 5, 6, 7, 8, 9]])
    current_ids = full_ids[:, prefix:]
    current_sids = torch.tensor([0, 0, 0, -1])
    full_sids = torch.tensor([-1, -1, 0, 0, 0, -1]).expand(batch, -1)
    targets = torch.tensor([[2, 2, 3, -100], [2, 3, 3, -100]])
    legal = torch.tensor(
        [
            [[2, -1, -1], [3, 2, -1], [4, 3, 2], [-1, -1, -1]],
            [[2, -1, -1], [3, 2, -1], [4, 3, -1], [-1, -1, -1]],
        ]
    )
    sparse = _v2_attachment_nll_from_hidden(
        model, hidden, current_ids, current_sids, targets, legal, prefix
    )
    dense_logits = model.pushdown_attachment_head(
        hidden,
        full_ids,
        model.transformer.wte.weight,
        sentence_ids=full_sids,
        query_range=(prefix, total),
    )
    dense_mask = torch.zeros_like(dense_logits, dtype=torch.bool)
    for b in range(batch):
        for q in range(total - prefix):
            keys = legal[b, q]
            dense_mask[b, q, keys[keys >= 0]] = True
    dense = _attachment_nll_from_logits(
        dense_logits, targets, dense_mask, "sentence_causal"
    )
    assert torch.allclose(sparse, dense, atol=1e-6, rtol=1e-6)


def test_kv_cached_candidate_scores_match_full_prefix_reference():
    torch.manual_seed(23)
    model = _tiny_model()
    history = _native_candidate((7, 1, 2), ((1, 1, 2),), (1, 3))
    left = _native_candidate((3, 4, 5), ((0, 0, 1), (0, 1, 2)), (0, 3))
    right = _native_candidate((3, 4, 5), ((1, 1, 2), (0, 0, 2)), (0, 3))
    candidates = (left, right)
    full, _ = score_gpst_binary_pushdown_candidates(
        model, (history,), candidates, "cpu"
    )
    cache = _build_prefix_cache(model, (history,), "cpu")
    cached, _ = score_gpst_binary_pushdown_candidates(
        model, (history,), candidates, "cpu", prefix_cache=cache
    )
    for field in ("joint_nll", "token_nll", "attachment_nll"):
        assert torch.allclose(
            getattr(cached, field), getattr(full, field), atol=2e-5, rtol=2e-5
        )


def _candidate(tokens: tuple[int, ...], marker: int) -> PushdownGoldCandidate:
    length = len(tokens)
    return PushdownGoldCandidate(
        tokens=tokens,
        spans=((marker, marker, marker),),
        sentence_ids=tuple(-1 for _ in range(length)),
        attachment_targets=tuple(-1 for _ in range(length)),
        legal_attachment_targets=tuple(() for _ in range(length)),
    )


class _RaggedCorpus:
    vocab = SimpleNamespace(bos=7)
    samples_per_sentence = 300

    def __init__(self) -> None:
        self.rows = (
            (0, (_candidate((7, 1, 2), 0), _candidate((7, 1, 2), 1))),
            (0, (_candidate((3, 4), 0),)),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


class _FakeModel:
    pushdown_attachment_head = object()

    def eval(self):
        return self


class _TinyBinaryCorpus:
    vocab = SimpleNamespace(bos=7)
    samples_per_sentence = 300

    def __init__(self) -> None:
        first = _native_candidate((7, 1, 2), ((1, 1, 2),), (1, 3))
        second_left = _native_candidate((3, 4, 5), ((0, 0, 1), (0, 1, 2)), (0, 3))
        second_right = _native_candidate((3, 4, 5), ((1, 1, 2), (0, 0, 2)), (0, 3))
        self.rows = ((0, (first,)), (0, (second_left, second_right)))

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def test_end_to_end_tiny_model_produces_both_finite_metrics():
    torch.manual_seed(29)
    result = evaluate_gpst_binary_pushdown_document_ppl(
        _tiny_model(),
        _TinyBinaryCorpus(),
        "cpu",
        eval_batch_size=2,
        max_sequence_length=16,
        prefetch_sentences=0,
    )
    assert math.isfinite(result.joint_document_perplexity_v1)
    assert math.isfinite(result.candidate0_structured_terminal_perplexity)
    assert result.terminal_count == 5
    assert result.valid_candidate_count == 3
    assert result.kv_cache_hits == 1


def test_end_to_end_v2_uses_training_normalization_and_versioned_fields():
    torch.manual_seed(31)
    result = evaluate_gpst_binary_pushdown_document_ppl(
        _tiny_model(),
        _TinyBinaryCorpus(),
        "cpu",
        eval_batch_size=2,
        max_sequence_length=16,
        prefetch_sentences=0,
        attachment_normalization="sentence_causal",
    )
    payload = result.as_dict()
    assert result.protocol_version == 2
    assert result.attachment_normalization == "sentence_causal"
    assert "joint_log_likelihood_v2" in payload
    assert "joint_document_perplexity_v2" in payload
    assert "joint_log_likelihood_v1" not in payload
    assert math.isfinite(payload["joint_document_perplexity_v2"])


def test_metric_uses_valid_k_sum_and_candidate0_terminal_only(monkeypatch):
    seen_prefixes = []

    def fake_score(
        _model,
        prefix,
        candidates,
        _device,
        prefix_cache=None,
        return_candidate0_cache=False,
        attachment_normalization="stack_legal",
    ):
        del prefix_cache, return_candidate0_cache, attachment_normalization
        seen_prefixes.append(tuple(prefix))
        if candidates[0].tokens[0] == 7:
            token = torch.tensor([1.0, 3.0], dtype=torch.float64)
            attachment = torch.tensor([0.5, 0.5], dtype=torch.float64)
        else:
            token = torch.tensor([2.0], dtype=torch.float64)
            attachment = torch.tensor([0.25], dtype=torch.float64)
        return PushdownCandidateScores(token + attachment, token, attachment), None

    monkeypatch.setattr(
        "olmo.eval.gpst_binary_pushdown_document_ppl."
        "score_gpst_binary_pushdown_candidates",
        fake_score,
    )
    result = evaluate_gpst_binary_pushdown_document_ppl(
        _FakeModel(),
        _RaggedCorpus(),
        "cpu",
        use_kv_cache=False,
        prefetch_sentences=0,
        eval_batch_size=300,
    )
    expected_joint_ll = (
        torch.logsumexp(-torch.tensor([1.5, 3.5], dtype=torch.float64), 0).item() - 2.25
    )
    assert result.joint_log_likelihood_v1 == pytest.approx(expected_joint_ll)
    assert result.candidate0_terminal_log_likelihood == pytest.approx(-3.0)
    assert result.joint_document_perplexity_v1 == pytest.approx(
        math.exp(-expected_joint_ll / 4)
    )
    assert result.candidate0_structured_terminal_perplexity == pytest.approx(
        math.exp(3.0 / 4)
    )
    assert result.valid_candidate_count == 3
    assert result.model_candidate_forwards == 3
    assert result.candidate_slots == 600
    assert result.terminal_count == 4
    assert seen_prefixes[0] == ()
    assert seen_prefixes[1] == (_RaggedCorpus().rows[0][1][0],)


def test_evaluator_rejects_nonfinite_candidate_with_location(monkeypatch):
    calls = 0

    def fake_score(
        _model,
        _prefix,
        candidates,
        _device,
        prefix_cache=None,
        return_candidate0_cache=False,
        attachment_normalization="stack_legal",
    ):
        nonlocal calls
        calls += 1
        del prefix_cache, return_candidate0_cache, attachment_normalization
        token = torch.zeros(len(candidates), dtype=torch.float64)
        attachment = torch.zeros_like(token)
        token[-1] = math.nan
        return PushdownCandidateScores(token + attachment, token, attachment), None

    monkeypatch.setattr(
        "olmo.eval.gpst_binary_pushdown_document_ppl."
        "score_gpst_binary_pushdown_candidates",
        fake_score,
    )
    with pytest.raises(
        FloatingPointError,
        match=r"document_id=0.*document_sentence_index=0.*candidate_index",
    ):
        evaluate_gpst_binary_pushdown_document_ppl(
            _FakeModel(),
            _RaggedCorpus(),
            "cpu",
            use_kv_cache=False,
            prefetch_sentences=0,
            eval_batch_size=2,
        )
    # batch 2 -> batch 1; the persistent bad candidate remains non-finite and
    # is ultimately rejected rather than skipped.
    assert calls >= 2


def test_evaluator_retries_transient_nonfinite_at_smaller_microbatch(monkeypatch):
    calls = 0

    def fake_score(
        _model,
        _prefix,
        candidates,
        _device,
        prefix_cache=None,
        return_candidate0_cache=False,
        attachment_normalization="stack_legal",
    ):
        nonlocal calls
        calls += 1
        del prefix_cache, return_candidate0_cache, attachment_normalization
        token = torch.ones(len(candidates), dtype=torch.float64)
        attachment = torch.zeros_like(token)
        if calls == 1:
            token[-1] = math.nan
        return PushdownCandidateScores(token + attachment, token, attachment), None

    monkeypatch.setattr(
        "olmo.eval.gpst_binary_pushdown_document_ppl."
        "score_gpst_binary_pushdown_candidates",
        fake_score,
    )
    result = evaluate_gpst_binary_pushdown_document_ppl(
        _FakeModel(),
        _RaggedCorpus(),
        "cpu",
        use_kv_cache=False,
        prefetch_sentences=0,
        eval_batch_size=2,
    )
    assert result.nonfinite_retries == 1
    assert math.isfinite(result.joint_document_perplexity_v1)


def test_shard_merge_sums_likelihoods_and_rejects_overlap():
    contract = {
        "protocol_version": 1,
        "structure_source": "v2_gpst_strict_binary_to_pushdown",
        "prefix_policy": "candidate0",
        "context_truncation": "left_drop_complete_sentences",
        "attachment_normalization": "stack_legal",
        "candidate_aggregation": "valid_unique_truncated_joint_sum",
        "divide_by_candidate_count": False,
        "ppl_denominator": "terminal_count",
        "max_sequence_length": 2048,
        "max_candidates_per_sentence": 300,
        "checkpoint_model_sha256": "checkpoint-sha256",
        "native_manifest_sha256": "manifest-sha256",
        "tokenizer_sha256": "tokenizer-sha256",
        "checkpoint": "/checkpoint",
        "native_data": "/native-data",
        "tokenizer_path": "/tokenizer.json",
        "sentence_count": 2,
        "document_count": 1,
        "valid_candidate_count": 17,
        "candidate_slots": 600,
        "model_candidate_forwards": 17,
        "kv_cache_hits": 1,
        "kv_cache_rebuilds": 0,
        "oom_retries": 0,
        "terminal_count": 10,
        "joint_log_likelihood_v1": -20.0,
        "candidate0_terminal_log_likelihood": -15.0,
    }
    left = {**contract, "start_document": 0, "end_document": 1}
    right = {**contract, "start_document": 1, "end_document": 2}
    result = merge([right, left])
    assert result["terminal_count"] == 20
    assert result["valid_candidate_count"] == 34
    assert result["joint_document_perplexity_v1"] == pytest.approx(math.exp(2.0))
    with pytest.raises(ValueError, match="overlapping"):
        merge([left, left])
    with pytest.raises(ValueError, match="non-contiguous"):
        merge([left, {**right, "start_document": 2, "end_document": 3}])
    with pytest.raises(ValueError, match="checkpoint_model_sha256"):
        merge([left, {**right, "checkpoint_model_sha256": "other"}])
    with pytest.raises(ValueError, match="model_candidate_forwards"):
        merge([{**left, "model_candidate_forwards": 16}])
    with pytest.raises(ValueError, match="non-finite joint_log_likelihood_v1"):
        merge([{**left, "joint_log_likelihood_v1": math.nan}])
    with pytest.raises(
        ValueError, match="non-finite candidate0_terminal_log_likelihood"
    ):
        merge([{**left, "candidate0_terminal_log_likelihood": math.inf}])
    with pytest.raises(ValueError, match="full BBC corpus invariants"):
        validate_full_bbc(result)
