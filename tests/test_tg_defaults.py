"""Default CPU C++ metadata, routing boundaries and native layout transport."""

import copy
import pickle
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from olmo.attention_kernels import TGLayout, TGNoMaskLayout, MixTGLayout, build_tgnomask_layout
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset, build_eval_dataloader
from olmo.data.collator import DataCollator, use_typed_tg
from olmo.data.prefetch import CUDABatchPrefetcher
from olmo.data.tg_mask import SentencepieceVocab
from olmo.model import OLMo
from olmo.torch_util import move_to_device
from olmo.train import Trainer
from test_tg_typed import model_config, reference, tokens, VOCAB

GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA allocation")
GRAMMARS = ["tg", "tgnomask", "tgnomaskaug", "mixing"]
GROUPS = (("tg", 1), ("tgnomaskaug", 2), ("tgnomask", 1))


def config(grammar):
    cfg = TrainConfig(
        model=model_config(GROUPS if grammar == "mixing" else (), transformer_grammar_type=grammar)
    )
    cfg.tokenizer.vocabulary = VOCAB
    return cfg


@pytest.mark.parametrize("augmented", [False, True])
@pytest.mark.parametrize("n", [1, 17, 129, 2048])
def test_nomask_native_python_packed_and_sliced(augmented, n):
    vocab = SentencepieceVocab.from_vocab_file(VOCAB)
    ids = tokens(n, batch=3)
    ids[0, -min(7, n) :] = vocab.pad
    native = build_tgnomask_layout(ids, vocab, augmented=augmented)
    python = build_tgnomask_layout(ids, vocab, augmented=augmented, backend="python", pack=False)
    for candidate in (native, pickle.loads(pickle.dumps(native)), native.slice_batch(1, 3).pack()):
        ref = python.slice_batch(1, 3) if candidate.shape[0] == 2 else python
        assert candidate.augmented == augmented
        for f in fields(ref):
            if isinstance(getattr(ref, f.name), torch.Tensor):
                assert torch.equal(getattr(candidate, f.name), getattr(ref, f.name)), f.name
        assert len({t.untyped_storage().data_ptr() for t in candidate._tensors()}) == 2
    assert not hasattr(native, "fwd_keys")


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_default_dataset_collator_eval_and_microbatch(grammar, tmp_path):
    cfg = config(grammar)
    cfg.model.max_sequence_length = 17
    ids = tokens(17, batch=3)
    path = tmp_path / "tokens.npy"
    np.save(path, ids.numpy().astype("uint16").reshape(-1))
    cfg.data.paths = [str(path)]
    assert cfg.model.tg_typed_attention is None
    assert cfg.data.tg_layout_backend == "native"
    dataset = build_memmap_dataset(cfg, cfg.data)
    assert "attention_bias" not in dataset[0]
    collator = DataCollator.from_train_config(cfg)
    batch = collator([dataset[i] for i in range(3)])
    assert collator.tg_typed_attention
    assert "attention_bias" not in batch
    assert isinstance(
        batch["tg_layout"], {"tg": TGLayout, "mixing": MixTGLayout}.get(grammar, TGNoMaskLayout)
    )
    mask, label = reference(ids, GROUPS if grammar == "mixing" else ((grammar, 1),))
    assert torch.equal(batch["tg_layout"].dense_mask(), mask)
    assert torch.equal(batch["label_mask"], label & (ids != cfg.model.pad_token_id))
    trainer = SimpleNamespace(cfg=SimpleNamespace(device_train_microbatch_size=2))
    micro = Trainer.split_batch(trainer, move_to_device(batch, torch.device("cpu")))
    assert torch.equal(torch.cat([b["tg_layout"].dense_mask() for b in micro]), mask)
    # Eval must use its own DataConfig for both halves of the routing decision.
    eval_data = copy.deepcopy(cfg.data)
    eval_data.tg_layout_backend = "python"
    loader = build_eval_dataloader(cfg, eval_data, batch_size=2, shuffle=False)
    assert loader.collate_fn.tg_layout_backend == "python"
    assert "tg_layout" in next(iter(loader))
    eval_data.generate_attention_mask = True
    loader = build_eval_dataloader(cfg, eval_data, batch_size=2, shuffle=False)
    dense_batch = next(iter(loader))
    assert "tg_layout" not in dense_batch
    assert "attention_bias" in dense_batch and "attention_mask" in dense_batch
    cfg.model.tg_typed_attention = False
    dataset = build_memmap_dataset(cfg, cfg.data)
    batch = DataCollator.from_train_config(cfg)([dataset[0]])
    assert "tg_layout" not in batch and "attention_bias" in batch
    assert torch.equal(batch["attention_bias"], mask[:1])


@pytest.mark.parametrize(
    "change",
    ["dropout", "alibi", "gqa", "head_dim", "block", "mask", "doc", "finetune", "proximal", "causal"],
)
def test_auto_preserves_unsupported_paths_and_explicit_true_rejects(change):
    cfg = config("tg")
    if change == "dropout":
        cfg.model.attention_dropout = 0.1
    elif change == "alibi":
        cfg.model.alibi = True
    elif change == "gqa":
        cfg.model.n_kv_heads = 2
    elif change == "head_dim":
        cfg.model.d_model = 32
    elif change == "block":
        cfg.model.block_type = "llama"
    elif change == "mask":
        cfg.data.generate_attention_mask = True
    elif change == "doc":
        cfg.data.generate_doc_lengths = True
    elif change == "finetune":
        cfg.finetune_task = "boolq"
    elif change == "proximal":
        cfg.model.transformer_grammar_type = "tgproximal"
    elif change == "causal":
        cfg.model.transformer_grammar_type = "terminal"
    assert not use_typed_tg(cfg)
    cfg.model.tg_typed_attention = True
    with pytest.raises(ValueError, match="tg_typed_attention requires"):
        use_typed_tg(cfg)


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_default_layout_cpu_model_matches_dense(grammar):
    cfg = config(grammar)
    ids = tokens(17)
    batch = DataCollator.from_train_config(cfg)(list(ids))
    mask, _ = reference(ids, GROUPS if grammar == "mixing" else ((grammar, 1),))
    torch.manual_seed(420)
    model = OLMo(cfg.model).eval()
    with torch.no_grad():
        actual = model(ids, tg_layout=batch["tg_layout"]).logits
        expected = model(ids, attention_bias=mask).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@GPU
@pytest.mark.parametrize("grammar", GRAMMARS)
@pytest.mark.parametrize("workers", [0, 2])
def test_default_native_worker_to_model_backward(grammar, workers, monkeypatch):
    cfg = config(grammar)
    collator = DataCollator.from_train_config(cfg)
    ids = tokens(127)
    ids[0, -5:] = collator.vocab.pad
    mask, label = reference(ids, GROUPS if grammar == "mixing" else ((grammar, 1),))
    loader = DataLoader(
        list(ids) * 2,
        batch_size=2,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    cpu = next(iter(loader))
    layout = cpu["tg_layout"]
    members = (layout.tg, layout.aug) if isinstance(layout, MixTGLayout) else (layout,)
    assert all(t.is_pinned() for part in members for t in part._tensors())
    torch.manual_seed(312)
    model = OLMo(cfg.model).cuda().to(torch.bfloat16).train()
    calls = []
    import olmo.model as model_module

    entry = {"tg": "tg_attention", "mixing": "mix_tg_attention"}.get(grammar, "tgnomask_attention")
    original = getattr(model_module, entry)

    def counted(*args, **kwargs):
        calls.append(entry)
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, entry, counted)

    def run(batch, typed):
        model.zero_grad(set_to_none=True)
        logits = model(
            batch["input_ids"],
            **({"tg_layout": batch["tg_layout"]} if typed else {"attention_bias": mask.cuda()}),
        ).logits
        target = batch["input_ids"][:, 1:].reshape(-1)
        keep = label[:, 1:].reshape(-1).cuda() & (target != collator.vocab.pad)
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]), target, reduction="none"
        )[keep].mean()
        loss.backward()
        return (
            logits.detach(),
            loss.detach(),
            [p.grad.detach().clone() for p in model.parameters() if p.grad is not None],
        )

    # The GPU path must consume metadata directly, never reconstruct NxN masks.
    def no_dense(*args):
        raise AssertionError("CUDA typed attention constructed a dense mask")

    for cls in (TGLayout, TGNoMaskLayout, MixTGLayout):
        monkeypatch.setattr(cls, "dense_mask", no_dense)
    with CUDABatchPrefetcher(loader, "cuda") as batches:
        for batch in batches:
            actual, loss, grads = run(batch, True)
            expected, ref_loss, ref_grads = run(batch, False)
            torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.04)
            torch.testing.assert_close(loss, ref_loss, atol=0.015, rtol=0.04)
            for g, r in zip(grads, ref_grads):
                assert g.isfinite().all()
                torch.testing.assert_close(g, r, atol=0.015, rtol=0.04)
    assert len(calls) == 2 * cfg.model.n_layers


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_layout_transfer_does_not_leave_tensor_reference_cycles(grammar):
    import gc
    import weakref

    layout = DataCollator.from_train_config(config(grammar))(list(tokens(17)))["tg_layout"]
    layout = layout.tg if isinstance(layout, MixTGLayout) else layout
    # Deferred tensor destruction can abort a forked worker after the parent
    # has used CUDA. Owners must be freed without requiring cyclic GC.
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        moved = layout._map_buffers(lambda t: t.clone())
        owners = [weakref.ref(t) for t in moved._buffers]
        del moved
        assert all(ref() is None for ref in owners)
    finally:
        if enabled:
            gc.enable()


@GPU
@pytest.mark.parametrize("augmented", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_long_nomask_optimized_prefix(augmented, dtype):
    """Exercise the D64/N>=512 prefix tuning used by full-length training."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from olmo.attention_kernels import tgnomask_attention

    vocab = SentencepieceVocab.from_vocab_file(VOCAB)
    ids = tokens(2048)
    ids[0, -7:] = vocab.pad
    layout = build_tgnomask_layout(ids, vocab, "cuda", augmented=augmented)
    mask, _ = reference(ids, (("tgnomaskaug" if augmented else "tgnomask", 1),))
    torch.manual_seed(189)
    inputs = [torch.randn(2, 2, 2048, 64, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    upstream = torch.randn_like(inputs[0])
    actual = tgnomask_attention(*inputs, layout)
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*refs, attn_mask=mask.cuda())
    actual.backward(upstream)
    expected.backward(upstream)
    tolerance = 0.012 if dtype == torch.bfloat16 else 0.002
    for value, ref in [(actual, expected)] + [(x.grad, r.grad) for x, r in zip(inputs, refs)]:
        assert value.isfinite().all()
        error = (value.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-10)
        assert error < tolerance, float(error)
        torch.testing.assert_close(value, ref, atol=0.035 if dtype == torch.bfloat16 else 0.006, rtol=0.04)
