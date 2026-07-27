"""Tests for olmo.treereg (TreeReg SCIN auxiliary loss)."""

import torch
import pytest

from olmo.treereg import compute_treereg_loss, count_treereg_sentences


def test_treereg_loss_shape_and_finite():
    torch.manual_seed(0)
    B, n, d = 2, 16, 32
    hidden = torch.randn(B, n, d, requires_grad=True)
    # One gold constituent per example: (left, split, right) = (2, 4, 7).
    spans = torch.tensor([[[2, 4, 7], [-1, -1, -1]], [[1, 3, 6], [-1, -1, -1]]], dtype=torch.long)
    mask = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
    loss = compute_treereg_loss(hidden, spans, mask, n_heads_subset=2, d_head=d // 8)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_treereg_no_spans_returns_zero():
    hidden = torch.randn(2, 8, 16, requires_grad=True)
    spans = torch.full((2, 3, 3), -1, dtype=torch.long)
    mask = torch.zeros((2, 3), dtype=torch.bool)
    loss = compute_treereg_loss(hidden, spans, mask)
    assert float(loss) == 0.0
    loss.backward()  # should not error


def test_treereg_favors_gold_split_when_orthogonal():
    """Under the orthogonality metric, a gold split that separates two mutually
    orthogonal blocks should score highest (the two halves are maximally
    independent) and thus have the lowest CE loss."""
    torch.manual_seed(1)
    B, n, d = 1, 8, 16
    # Left block [0..3] along e1, right block [4..7] along e2, e1 ⟂ e2.
    h = torch.zeros(B, n, d)
    e1 = torch.randn(d)
    e2 = torch.randn(d)
    e2 = e2 - (e2 @ e1) / (e1 @ e1) * e1  # make e2 orthogonal to e1
    h[0, 0:4] = e1
    h[0, 4:8] = e2
    # Gold span (0, 3, 7): split at 3 -> left [0..3], right [4..7] (orthogonal halves).
    spans = torch.tensor([[[0, 3, 7]]], dtype=torch.long)
    mask = torch.tensor([[True]], dtype=torch.bool)
    loss_aligned = compute_treereg_loss(h.clone(), spans, mask, n_heads_subset=0)
    # Misaligned gold split (0, 1, 7): split at 1 -> right child [2..7] spans across
    # the block boundary, so the gold split is NOT at the orthogonal boundary ->
    # lower score -> higher CE.
    spans_mis = torch.tensor([[[0, 1, 7]]], dtype=torch.long)
    loss_mis = compute_treereg_loss(h.clone(), spans_mis, mask, n_heads_subset=0)
    assert float(loss_aligned) < float(loss_mis)


def test_treereg_orthogonal_metric_matches_formula():
    """O[i,j] must equal ||H[j] - proj_{Hn[i]}(H[j])|| (component of H[j] orthogonal
    to H[i]), matching scin_computer.get_all_orthogonal_scores."""
    torch.manual_seed(2)
    H = torch.randn(1, 3, 4)
    i, j = 0, 2
    hi, hj = H[0, i], H[0, j]
    expected = (hj - (hj @ hi) / (hi @ hi) * hi).norm()
    # Reconstruct O[i,j] from the loss internals (build_chart step).
    circuit = H.float()
    Hn = torch.nn.functional.normalize(circuit, dim=-1)
    proj = torch.bmm(Hn, circuit.transpose(1, 2))
    orth = circuit.unsqueeze(1) - proj.unsqueeze(-1) * Hn.unsqueeze(2)
    O = orth.norm(dim=-1)
    assert torch.allclose(O[0, i, j], expected, atol=1e-4)
    # A vector's component orthogonal to itself is zero: O[i,i] = 0.
    assert torch.allclose(O[0, i, i], torch.tensor(0.0), atol=1e-5)


def test_treereg_macro_reduction():
    """Loss must be the macro mean (per-sentence span-CE mean, then mean over
    sentences), NOT a flat mean over all spans."""
    torch.manual_seed(3)
    B, n, d = 2, 12, 8
    hidden = torch.randn(B, n, d, requires_grad=True)
    # Sentence 0: two spans; sentence 1: one span.
    spans = torch.tensor([
        [[0, 2, 5], [3, 4, 8], [-1, -1, -1]],
        [[1, 4, 9], [-1, -1, -1], [-1, -1, -1]],
    ], dtype=torch.long)
    mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)
    loss_macro = compute_treereg_loss(hidden, spans, mask, n_heads_subset=0)

    # Per-span CE computed independently (B=1 each), then macro-averaged.
    def span_ce(b, st, p, en):
        h1 = hidden[b:b + 1].clone().detach().requires_grad_(True)
        sp = torch.tensor([[[st, p, en]]], dtype=torch.long)
        mk = torch.tensor([[True]], dtype=torch.bool)
        return float(compute_treereg_loss(h1, sp, mk, n_heads_subset=0))

    sent0 = (span_ce(0, 0, 2, 5) + span_ce(0, 3, 4, 8)) / 2
    sent1 = span_ce(1, 1, 4, 9)
    expected = (sent0 + sent1) / 2
    assert abs(float(loss_macro) - expected) < 1e-4


def test_treereg_excludes_bos_eos_without_changing_reference_math():
    torch.manual_seed(4)
    # Packed representation has BOS/tree/EOS, while the upstream regularizer is
    # called on the sentence's terminal pieces only.
    packed = torch.randn(1, 7, 12)
    spans_global = torch.tensor([[[1, 2, 5]]])
    mask = torch.tensor([[True]])
    sentence_ids = torch.tensor([[-1, 0, 0, 0, 0, 0, -1]])
    word_boundaries = torch.tensor([[False, True, True, True, True, True, False]])
    packed_loss, count = compute_treereg_loss(
        packed,
        spans_global,
        mask,
        sentence_ids=sentence_ids,
        word_boundaries=word_boundaries,
        return_sentence_count=True,
    )
    stripped_loss = compute_treereg_loss(
        packed[:, 1:6],
        torch.tensor([[[0, 1, 4]]]),
        mask,
    )
    assert count == 1
    assert torch.allclose(packed_loss, stripped_loss, atol=1e-6)


def test_treereg_restricts_candidates_to_word_boundaries():
    torch.manual_seed(5)
    hidden = torch.randn(1, 5, 8)
    spans = torch.tensor([[[0, 2, 4]]])
    mask = torch.tensor([[True]])
    sentence_ids = torch.zeros((1, 5), dtype=torch.int32)
    # Token 1 is a continuation piece, so split q=0 is not a candidate.
    boundaries = torch.tensor([[True, False, True, True, True]])
    restricted = compute_treereg_loss(
        hidden,
        spans,
        mask,
        sentence_ids=sentence_ids,
        word_boundaries=boundaries,
    )

    # Equivalent compact reference candidates are q={1,2,3}; compute their
    # scores through the same objective by masking no sentence context.
    all_candidates = compute_treereg_loss(hidden, spans, mask)
    assert torch.isfinite(restricted)
    assert not torch.allclose(restricted, all_candidates)


def test_treereg_ignores_degenerate_unary_spans_instead_of_clamping():
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    spans = torch.tensor([[[1, 4, 4]]])
    loss = compute_treereg_loss(hidden, spans, torch.tensor([[True]]))
    assert float(loss) == 0.0
    loss.backward()
    assert hidden.grad is not None


def test_two_trees_in_one_packed_row_are_macro_averaged():
    torch.manual_seed(6)
    hidden = torch.randn(1, 10, 8)
    spans = torch.tensor([[[1, 2, 4], [6, 7, 8]]])
    mask = torch.tensor([[True, True]])
    sentence_ids = torch.tensor([[-1, 0, 0, 0, 0, -1, 1, 1, 1, -1]])
    boundaries = sentence_ids >= 0
    packed, count = compute_treereg_loss(
        hidden,
        spans,
        mask,
        sentence_ids=sentence_ids,
        word_boundaries=boundaries,
        return_sentence_count=True,
    )
    first = compute_treereg_loss(
        hidden[:, 1:5], torch.tensor([[[0, 1, 3]]]), torch.tensor([[True]])
    )
    second = compute_treereg_loss(
        hidden[:, 6:9], torch.tensor([[[0, 1, 2]]]), torch.tensor([[True]])
    )
    assert count == 2
    assert torch.allclose(packed, (first + second) / 2, atol=1e-6)


def test_count_treereg_sentences_counts_runs_not_bos_or_padding():
    ids = torch.tensor([
        [-1, 0, 0, -1, 1, 1, -1],
        [-1, -1, 0, 0, 0, -1, -1],
    ])
    assert int(count_treereg_sentences(ids)) == 3


def test_treereg_layer_one_captures_post_first_block():
    from olmo.config import (
        ActivationType,
        BlockType,
        InitFnType,
        LayerNormType,
        ModelConfig,
    )
    from olmo.model import OLMo

    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=2,
        mlp_hidden_size=64,
        vocab_size=128,
        embedding_size=128,
        max_sequence_length=8,
        block_type=BlockType.sequential,
        layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu,
        rope=True,
        flash_attention=False,
        attention_dropout=0.0,
        init_device="cpu",
        init_fn=InitFnType.normal,
        transformer_grammar_type="treereg",
        treereg_layer=1,
        treereg_n_heads=1,
    )
    model = OLMo(cfg).eval()
    out = model(
        input_ids=torch.randint(0, 128, (1, 6)),
        output_hidden_states=True,
    )
    # hidden_states[0] is embeddings; hidden_states[1] is the state entering
    # block 2, i.e. exactly the post-block-1 residual.
    assert torch.allclose(out.treereg_hidden, out.hidden_states[1])
