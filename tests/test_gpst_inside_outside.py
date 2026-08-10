"""Phase 1 test: the composition model (InsideOutsideModule) runs forward on
CPU, producing finite losses and correctly-shaped outputs.

Drives ``InsideOutsideModule.forward`` with a tiny 2-sentence batch and the
small r2d2 config (hidden_size=256, window_size=2). Verifies:
- E-step produces a parse tree + inside/outside representations;
- ``parser_loss`` (L_p) and ``inside_outside_loss`` (L_ae) are finite;
- ``ldr_repr`` (surrogate inputs for the generative model) has the expected
  leading dims (group_size, seq_len, hidden).
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import torch

import olmo.gpst  # noqa: F401

_CFG_PATH = os.path.join(os.path.dirname(olmo.gpst.__file__),
                         "data", "en_config", "r2d2_256_4_1.json")


def _load_config():
    cfg = json.load(open(_CFG_PATH))
    # keep the full vocab (50264) so special-token ids (bos=50256, reduce=50257,
    # cls=101, ...) are valid embedding indices; the synthetic input ids
    # (10..50) are well within range.
    return cfg


def _make_module():
    from olmo.gpst.model.r2d2_insideoutside import InsideOutsideModule
    class _Cfg:  # noqa: D401
        pass
    c = _Cfg()
    for k, v in _load_config().items():
        setattr(c, k, v)
    c.parser_chunked = True
    c.use_gumbel = False
    c.ldr_detach = False
    return InsideOutsideModule(c)


def _make_batch(module, device):
    """Two sentences padded to a common length 3 (the batch max real length).
    No sentence chunking (group=sent). The parser/embeddings receive *clean* ids
    (no -100 sentinel); -100 is only used on the generation target side by
    FastGenerativeR2D2, which sanitizes ids via
    ``torch.where(chunk_input_ids == -100, 0, chunk_input_ids)`` before calling
    the composition module.
    """
    # sentence 1 has 3 tokens, sentence 2 has 2 -> pad sentence 2 with 0 (mask 0)
    chunk = torch.tensor([[10, 20, 30], [40, 50, 0]],
                         dtype=torch.long, device=device)
    chunk_masks = torch.tensor([[1, 1, 1], [1, 1, 0]],
                               dtype=torch.long, device=device)
    input_ids = chunk.clone()
    masks = (chunk_masks != 0).long()
    group_ids = np.array([0, 1], dtype=np.int32)
    r2d2_embeddings = torch.randn(chunk.shape[0], chunk.shape[1],
                                   module.input_dim, device=device)
    return dict(
        chunk_input_ids=chunk,
        chunk_masks=chunk_masks,
        input_ids=input_ids,
        masks=masks,
        r2d2_embeddings=r2d2_embeddings,
        group_ids=group_ids,
        max_input_len=3,
    )


def test_inside_outside_forward_shapes_and_finite():
    torch.manual_seed(0)
    module = _make_module().to(torch.device("cpu"))
    module.train()
    batch = _make_batch(module, torch.device("cpu"))
    ctx, flatten_ids, ldr_repr, position_ids, tgt_ids, token_indices, ext_ids, \
        split_targets, l_height = module(
            batch["chunk_input_ids"], batch["chunk_masks"], batch["input_ids"],
            batch["masks"], batch["r2d2_embeddings"], batch["group_ids"], batch["max_input_len"],
            atom_spans=None, coeff=1.0, temperature=1.0)

    assert ldr_repr.dim() == 3
    assert ldr_repr.shape[0] == 2
    assert ldr_repr.shape[-1] == module.input_dim
    assert torch.isfinite(ldr_repr).all()

    # parser_loss (L_p) finite
    parser_loss = module.parser_loss(ctx)
    assert torch.isfinite(parser_loss), parser_loss
    # outside pass produces finite outside representations (used for L_ae)
    outside_repr = module.outside_embeddings(ctx)
    assert torch.isfinite(outside_repr).all(), outside_repr

    assert split_targets.shape[0] == 2
    assert torch.isfinite(l_height), l_height


def test_inside_outside_backward_runs():
    torch.manual_seed(0)
    module = _make_module().to(torch.device("cpu"))
    module.train()
    batch = _make_batch(module, torch.device("cpu"))
    ctx, flatten_ids, ldr_repr, *_ = module(
        batch["chunk_input_ids"], batch["chunk_masks"], batch["input_ids"],
        batch["masks"], batch["r2d2_embeddings"], batch["group_ids"], batch["max_input_len"],
        atom_spans=None, coeff=1.0, temperature=1.0)
    parser_loss = module.parser_loss(ctx)
    outside_repr = module.outside_embeddings(ctx)
    # include ldr_repr in the graph so gradients flow back to the composition fn
    loss = parser_loss + ldr_repr.sum() * 0.0 + outside_repr.sum() * 0.0
    loss.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for n, p in module.named_parameters() if 'parser' in n)
    assert has_grad


def test_unsupervised_parser_runs_once_per_forward():
    """The sampled merge order and parser scores must come from one parse pass."""
    torch.manual_seed(0)
    module = _make_module().to(torch.device("cpu"))
    module.train()
    batch = _make_batch(module, torch.device("cpu"))

    with patch.object(module.parser, "forward", wraps=module.parser.forward) as parser_forward:
        module(
            batch["chunk_input_ids"], batch["chunk_masks"], batch["input_ids"],
            batch["masks"], batch["r2d2_embeddings"], batch["group_ids"],
            batch["max_input_len"], atom_spans=None, coeff=1.0, temperature=1.0,
        )

    assert parser_forward.call_count == 1
