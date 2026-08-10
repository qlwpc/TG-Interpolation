"""Phase 2 test: the full generative model (FastGenerativeR2D2) runs forward +
backward on CPU, producing finite struct/non-struct losses and correctly-shaped
action logits (B, L, 2).

Uses DefaultCollator to build a realistic batch (sentence-split chunking), then
FastGenerativeR2D2.forward in train mode (which computes parser_loss,
inside_outside_loss, gpt_loss, action_loss).
"""
from __future__ import annotations

import torch

import olmo.gpst  # noqa: F401

_R2D2_CFG = "olmo/gpst/data/en_config/r2d2_256_4_1.json"
_GPT_CFG = "olmo/gpst/data/gpt2-small/config.json"


def _build_model():
    from olmo.gpst.model.model_factory import create_model
    return create_model('r2d2-gen-fast', _R2D2_CFG, _GPT_CFG)


def _make_batch():
    """Two short documents with sentence splits (collator chunks by sentence)."""
    from olmo.gpst.reader.data_collator import DefaultCollator
    items = [
        {"text": [10, 20, 30, 40, 50], "sentence_splits": [3, 5]},
        {"text": [60, 70, 80, 90], "sentence_splits": [2, 4]},
    ]
    collator = DefaultCollator(enable_group=True, external_vocab_path=None)
    batch = collator.generative_r2d2_collate_fn_ext(items)
    return batch


def _to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def test_generative_forward_finite():
    torch.manual_seed(0)
    model = _build_model().to(torch.device("cpu"))
    model.train()
    batch = _to_device(_make_batch(), torch.device("cpu"))
    result = model(coeff=1.0, temperature=1.0, **batch)

    assert torch.isfinite(result.struct_loss), result.struct_loss
    assert torch.isfinite(result.non_struct_loss), result.non_struct_loss
    assert result.action_logits is not None
    assert result.action_logits.shape[-1] == 2
    assert result.logits is not None
    assert result.logits.shape[0] == result.action_logits.shape[0]
    assert torch.isfinite(result.gpt_loss), result.gpt_loss
    assert torch.isfinite(result.action_loss), result.action_loss
    assert torch.isfinite(result.parser_loss), result.parser_loss
    assert torch.isfinite(result.inside_outside_loss), result.inside_outside_loss


def test_generative_backward_runs():
    from olmo.gpst.model.weighted_sum_func import WeightedSumFunc
    torch.manual_seed(0)
    model = _build_model().to(torch.device("cpu"))
    model.train()
    batch = _to_device(_make_batch(), torch.device("cpu"))

    # Mimic the trainer's two-backward pattern (hard-EM):
    # 1) struct loss with a_ij grad enabled
    WeightedSumFunc.a_ij_require_grad = True
    result1 = model(coeff=1.0, temperature=1.0, **batch)
    result1.struct_loss.backward(retain_graph=True)
    # 2) non-struct loss with a_ij grad disabled (gradient stop)
    WeightedSumFunc.a_ij_require_grad = False
    result2 = model(coeff=1.0, temperature=1.0, **batch)
    result2.non_struct_loss.backward()

    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    assert has_grad


def test_generation_gpt2_stack_has_no_extra_position_embedding():
    """The deep token stack consumes gathered type states without a second WPE."""
    from transformers import AutoConfig
    from olmo.gpst.model.gpt2_flash_attn import GPT2Model

    cfg = AutoConfig.from_pretrained(_GPT_CFG)
    cfg.n_layer = 1
    model = GPT2Model(cfg, no_embedding=True, no_extra_embedding=True).eval()
    x = torch.randn(1, 4, cfg.n_embd)
    pos_a = torch.zeros(1, 4, dtype=torch.long)
    pos_b = torch.ones(1, 4, dtype=torch.long)

    with torch.no_grad():
        out_a = model(inputs_embeds=x, position_ids=pos_a).last_hidden_state
        out_b = model(inputs_embeds=x, position_ids=pos_b).last_hidden_state

    assert torch.allclose(out_a, out_b, atol=0.0, rtol=0.0)
