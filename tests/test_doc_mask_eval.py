"""GPU tests for document-level masking in pushdown/treereg LM eval (faithful PPL).

These guard the doc_lens + doc-mask path added so that eval PPL conditions each
token only on its own document within a multi-document 2048-token chunk:

* pushdown: the flex ``_pushdown_mask_mod`` gains a ``doc_id[b,q]==doc_id[b,kv]``
  clause (the depth bias is already doc-local — spans never cross an EOS doc
  boundary). Eval-only (``past_key_values is None``).
* treereg: the ``OLMo.forward`` else-branch skips building a causal
  ``attention_bias`` when ``doc_lens``/``max_doc_lens`` are set, so
  ``OLMoBlock._scaled_dot_product_attention`` takes the ``flash_attn_varlen_func``
  doc-mask branch (causality + doc boundary + pad all handled by varlen).

GPU-only: ``create_block_mask``/``flex_attention`` and ``flash_attn_varlen_func``
require CUDA. Skipped when ``torch.cuda.is_available()`` is False.
"""

import pytest
import torch

from olmo.config import (
    ModelConfig,
    BlockType,
    LayerNormType,
    ActivationType,
    InitFnType,
)


def _make_cfg(grammar_type: str, flex: bool) -> ModelConfig:
    return ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        mlp_ratio=4,
        mlp_hidden_size=256,
        vocab_size=50320,
        embedding_size=50320,
        max_sequence_length=32,
        block_type=BlockType.sequential,
        layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu,
        rope=True,
        flash_attention=True,
        flex_attention=flex,
        attention_dropout=0.0,
        init_device="cpu",
        init_fn=InitFnType.normal,
        init_std=0.02,
        transformer_grammar_type=grammar_type,
        pushdown_max_depth=16,
        pushdown_use_flex=flex,
        weight_tying=True,
        eos_token_id=50256,
        pad_token_id=50258,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")
def test_pushdown_doc_mask_blocks_cross_doc_attention():
    """With doc_lens set, a token in doc1 must NOT attend to doc0 keys. We assert
    this by comparing the doc-masked forward to the single-doc (no doc_lens)
    forward: doc1 positions must differ (cross-doc attention removed), while the
    very first token of doc0 (position 0, no past) is unaffected."""
    from olmo.model import OLMo

    torch.manual_seed(0)
    m = OLMo(_make_cfg("pushdown", flex=True)).cuda().eval()

    B, n = 1, 16
    EOS = 50256
    # Two documents: [doc0: 0..6 (EOS at 6)] [doc1: 7..15].
    input_ids = torch.randint(0, 50000, (B, n), device="cuda")
    input_ids[0, 6] = EOS
    attn = torch.ones_like(input_ids, dtype=torch.bool)
    # Spans strictly within doc0 / doc1 (doc-local depth bias).
    spans = torch.tensor([[[0, 2, 5], [7, 9, 12]]], dtype=torch.long, device="cuda")

    # Single-doc baseline (current behavior): no doc_lens -> whole chunk is one doc.
    # Doc-masked: doc_lens = [7, 9] (doc0 = positions 0..6 incl EOS, doc1 = 7..15).
    doc_lens = torch.tensor([[7, 9, 0, 0]], dtype=torch.long, device="cuda")
    max_doc_lens = [9]

    with torch.no_grad():
        out_single = m(input_ids=input_ids, attention_mask=attn,
                       tree_spans=spans).logits
        out_doc = m(input_ids=input_ids, attention_mask=attn,
                    tree_spans=spans, doc_lens=doc_lens,
                    max_doc_lens=max_doc_lens).logits

    # doc1 positions (>=7) must differ: cross-doc attention to doc0 was removed.
    diff_doc1 = (out_doc[:, 7:, :] - out_single[:, 7:, :]).abs().max().item()
    assert diff_doc1 > 1e-5, (
        f"doc1 positions should change with doc masking (cross-doc attention cut); "
        f"got max diff {diff_doc1}"
    )
    # Position 0 (start of doc0, no past tokens) is unaffected by doc masking.
    diff_p0 = (out_doc[:, 0:1, :] - out_single[:, 0:1, :]).abs().max().item()
    assert diff_p0 < 1e-4, (
        f"position 0 (doc0 start, no past) should be unchanged; got {diff_p0}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention needs CUDA")
def test_treereg_doc_mask_runs_and_changes_ppl():
    """treereg eval: with flex_attention=false + doc_lens, the forward must take
    the flash_attn_varlen_func doc-mask branch (no assert failure) and produce a
    different PPL vs the no-doc-mask baseline (cross-doc attention cut)."""
    from olmo.model import OLMo

    torch.manual_seed(3)
    m = OLMo(_make_cfg("treereg", flex=False)).cuda().eval()

    B, n = 1, 16
    EOS = 50256
    input_ids = torch.randint(0, 50000, (B, n), device="cuda")
    input_ids[0, 6] = EOS
    attn = torch.ones_like(input_ids, dtype=torch.bool)
    spans = torch.tensor([[[0, 2, 5], [7, 9, 12]]], dtype=torch.long, device="cuda")
    doc_lens = torch.tensor([[7, 9, 0, 0]], dtype=torch.long, device="cuda")
    max_doc_lens = [9]

    def _ce(logits):
        # next-token CE over non-pad positions (labels = shifted input_ids).
        labels = input_ids[:, 1:]
        logits = logits[:, :-1, :]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=50258, reduction="mean")
        return loss.item()

    # flash_attn_varlen_func (the treereg doc-mask branch) requires fp16/bf16;
    # run under bf16 autocast, matching the real eval path (eval_batch wraps the
    # forward in torch.autocast("cuda", dtype=autocast_precision)).
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_single = m(input_ids=input_ids, attention_mask=attn,
                       tree_spans=spans).logits
        out_doc = m(input_ids=input_ids, attention_mask=attn,
                    tree_spans=spans, doc_lens=doc_lens,
                    max_doc_lens=max_doc_lens).logits

    # Must run without asserting (flash_attn_varlen_func path reached cleanly).
    assert out_doc.isfinite().all(), "treereg doc-mask forward produced NaN/inf"
    # PPL must differ: doc masking changes doc1 conditioning.
    assert abs(_ce(out_doc) - _ce(out_single)) > 1e-5, (
        "treereg doc masking should change the CE vs no-doc-mask baseline"
    )
