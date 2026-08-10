"""Phase 4 test: the hard-EM trainer runs 2 steps on CPU in both unsupervised
and supervised modes, with finite loss and the gradient-stop toggle working.
"""
from __future__ import annotations

import torch
import numpy as np

import olmo.gpst  # noqa: F401
from olmo.gpst.trainer.trainer import train, TrainConfig
from olmo.gpst.model.weighted_sum_func import WeightedSumFunc

_R2D2_CFG = "olmo/gpst/data/en_config/r2d2_256_4_1.json"
_GPT_CFG = "olmo/gpst/data/gpt2-small/config.json"


def _build_model():
    from olmo.gpst.model.model_factory import create_model
    return create_model('r2d2-gen-fast', _R2D2_CFG, _GPT_CFG)


def _unsup_batch():
    from olmo.gpst.reader.data_collator import DefaultCollator
    items = [
        {"text": [10, 20, 30, 40, 50], "sentence_splits": [3, 5]},
        {"text": [60, 70, 80, 90], "sentence_splits": [2, 4]},
    ]
    c = DefaultCollator(enable_group=True, external_vocab_path=None)
    return c.generative_r2d2_collate_fn_ext(items)


def _sup_batch():
    from olmo.gpst.reader.dataset_gold import GoldTreeCollator
    items = [
        {"text": np.array([10, 20, 30, 40, 50], dtype=np.int32),
         "sentence_splits": [5], "merge_orders": np.array([0, 3, 1], dtype=np.int32)},
        {"text": np.array([60, 70, 80, 90], dtype=np.int32),
         "sentence_splits": [4], "merge_orders": np.array([0, 2, 1], dtype=np.int32)},
    ]
    return GoldTreeCollator()(items)


def _loader(batch):
    class _L:
        def __iter__(self):
            yield batch
            yield batch
        def __len__(self):
            return 2
    return _L()


def _to_dev(batch, dev):
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(dev)
    return batch


def test_trainer_unsupervised_cpu():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = _build_model().to(dev)
    batch = _to_dev(_unsup_batch(), dev)
    cfg = TrainConfig(lr=5e-5, parser_lr=1e-3, accumulation_steps=1,
                      log_steps=1, save_steps=100000, amp=False, max_steps=2)
    m = train(model, _loader(batch), cfg, dev)
    assert torch.isfinite(torch.tensor(m["total_loss"]))
    assert WeightedSumFunc.a_ij_require_grad is False


def test_trainer_supervised_cpu():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = _build_model().to(dev)
    batch = _to_dev(_sup_batch(), dev)
    assert "merge_orders" in batch
    cfg = TrainConfig(lr=5e-5, parser_lr=1e-3, accumulation_steps=1,
                      log_steps=1, save_steps=100000, amp=False, max_steps=2)
    m = train(model, _loader(batch), cfg, dev)
    assert torch.isfinite(torch.tensor(m["total_loss"]))
    assert torch.isfinite(torch.tensor(m["parser_loss"]))


def test_grad_stop_toggle():
    """a_ij_require_grad toggles True (struct) -> False (non-struct) per step."""
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = _build_model().to(dev)
    batch = _to_dev(_unsup_batch(), dev)
    seen = []
    orig = WeightedSumFunc.a_ij_require_grad
    WeightedSumFunc.a_ij_require_grad = True
    r = model(coeff=1.0, temperature=1.0, **batch)
    r.struct_loss.backward(retain_graph=True)
    seen.append(WeightedSumFunc.a_ij_require_grad)
    WeightedSumFunc.a_ij_require_grad = False
    r2 = model(coeff=1.0, temperature=1.0, **batch)
    r2.non_struct_loss.backward()
    seen.append(WeightedSumFunc.a_ij_require_grad)
    WeightedSumFunc.a_ij_require_grad = orig
    assert seen == [True, False], seen
