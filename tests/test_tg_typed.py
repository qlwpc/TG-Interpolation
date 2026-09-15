"""Exact TG/mixed semantics, CUDA gradients, strides and model integration."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from olmo.data.tg_mask import SentencepieceVocab, TG_attention_bias, KProximal_TG_attention_bias
from olmo.attention_kernels.tg_attention import build_tg_layout, build_mix_tg_layout, tg_attention, mix_tg_attention

ROOT = Path(__file__).resolve().parents[1]
VOCAB = str(ROOT / 'dataset/bbc-news/TG_GPT2_tokenizer.json')
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Slurm CUDA allocation')


@pytest.fixture(scope='module')
def vocab():
    return SentencepieceVocab.from_vocab_file(VOCAB)


def tokens(n, start=13, batch=2):
    data = np.load(ROOT / 'dataset/bbc-news/tg/test.npy', mmap_mode='r')
    return torch.stack([torch.from_numpy(np.array(data[start+i*2048:start+i*2048+n], dtype=np.int64)) for i in range(batch)])


def reference(ids, groups=(('tg', 1),)):
    rows, labels = [], []
    n = ids.shape[1]
    for row in ids:
        heads = []
        label = None
        for kind, count in groups:
            if kind == 'tgtree':
                mask = torch.ones(n, n, dtype=torch.bool).tril()
            else:
                gen = TG_attention_bias(VOCAB, max(2048, n)) if kind == 'tg' else KProximal_TG_attention_bias(VOCAB, max(2048, n), max(2048, n), False)
                mask, label = gen(row.cpu())
            heads.append(mask.expand(count, n, n))
        rows.append(torch.cat(heads))
        labels.append(label if label is not None else torch.ones(n, dtype=torch.bool))
    return torch.stack(rows), torch.stack(labels)


@pytest.mark.parametrize('n', [1, 2, 17, 127, 128, 129, 511, 2048])
@pytest.mark.parametrize('tile', [16, 32, 64])
def test_real_layout(vocab, n, tile):
    ids = tokens(n)
    ids[0, -min(7, n):] = vocab.pad
    layout = build_tg_layout(ids, vocab, q_tile=tile, k_tile=tile)
    mask, label = reference(ids)
    assert torch.equal(layout.dense_mask(), mask)
    assert torch.equal(layout.base.label_mask, label)
    for b in range(len(ids)):
        assert torch.equal(layout.key_order[b].sort().values, torch.arange(n, dtype=torch.int32))
        for block in range(len(layout.fwd_offsets[b]) - 1):
            start, end = layout.fwd_offsets[b, block:block + 2]
            keys = layout.fwd_keys[b, start:end]
            assert len(keys.unique()) == len(keys)
        for block in range(len(layout.bwd_offsets[b]) - 1):
            start, end = layout.bwd_offsets[b, block:block + 2]
            ranks = layout.bwd_queries[b, start:end]
            assert len(ranks.unique()) == len(ranks)
            for key in layout.key_order[b, block*tile:(block+1)*tile]:
                expected = torch.arange(layout.lo[b, key], layout.hi[b, key])
                assert torch.isin(expected, ranks).all()


def cases(vocab):
    op, cl, pad = vocab.opening_non_terminals[0], vocab.closing_non_terminals[0], vocab.pad
    return {'singleton': [1], 'pads': [pad]*17, 'self': [cl],
            'causal': [1, 2, 3, 4], 'wide': [op]+[1]*70+[cl, cl],
            'mixed': [op, 1, op+1, 2, cl+1, cl+1, cl, cl, pad],
            'internal_pad': [op, 1, pad, 2, cl, cl, 3, pad],
            'repeated': [cl]*7+[1, op, 2, cl, cl]}


def test_random_and_edge_metadata(vocab):
    rng = np.random.default_rng(101)
    examples = list(cases(vocab).values())
    choices = [1, 2, vocab.pad, vocab.opening_non_terminals[0], vocab.closing_non_terminals[0], vocab.eos]
    examples.extend(rng.choice(choices, size=31).tolist() for _ in range(20))
    for row in examples:
        ids = torch.tensor([row])
        layout = build_tg_layout(ids, vocab)
        mask, label = reference(ids)
        assert torch.equal(layout.dense_mask(), mask)
        assert torch.equal(layout.base.label_mask, label)


def test_bulk_base_matches_existing_layout(vocab):
    from dataclasses import fields
    from olmo.attention_kernels.tgnomask import build_tgnomask_layout
    for ids in (tokens(129), torch.tensor([cases(vocab)['internal_pad']])):
        actual = build_tg_layout(ids, vocab).base
        ref = build_tgnomask_layout(ids, vocab)
        for field in fields(ref):
            a, r = getattr(actual, field.name), getattr(ref, field.name)
            assert torch.equal(a, r) if isinstance(a, torch.Tensor) else a == r


@pytest.mark.parametrize('groups', [(('tg', 2), ('tgnomask', 2)), (('tgtree', 2), ('tg', 2)), (('tg', 1), ('tgtree', 2), ('tgnomask', 1))])
def test_mixed_metadata(vocab, groups):
    ids = tokens(129)
    ids[0, -7:] = vocab.pad
    layout = build_mix_tg_layout(ids, vocab, groups)
    mask, label = reference(ids, groups)
    assert torch.equal(layout.dense_mask(), mask)
    assert torch.equal(layout.label_mask, label)


def check(qkv, layout, mask, mixed=False, upstream=None):
    refs = [x.detach().clone().requires_grad_() for x in qkv]
    if upstream is None:
        upstream = torch.randn_like(qkv[0])
    with sdpa_kernel(SDPBackend.MATH):
        expected = F.scaled_dot_product_attention(*refs, attn_mask=mask.cuda())
    actual = (mix_tg_attention if mixed else tg_attention)(*qkv, layout)
    actual.backward(upstream)
    expected.backward(upstream)
    dtype = qkv[0].dtype
    tol = 0.012 if dtype == torch.bfloat16 else 0.002 if dtype == torch.float16 else 0.0002
    for value, ref in [(actual, expected)] + [(x.grad, r.grad) for x, r in zip(qkv, refs)]:
        assert value.isfinite().all()
        error = (value.float()-ref.float()).norm()/ref.float().norm().clamp_min(1e-10)
        assert error < tol, float(error)
        torch.testing.assert_close(value, ref, atol=.035 if dtype==torch.bfloat16 else .006 if dtype==torch.float16 else 3e-5,
                                   rtol=.04 if dtype!=torch.float32 else 5e-4)
    return actual


@GPU
@pytest.mark.parametrize('kind', ['singleton', 'pads', 'self', 'causal', 'wide', 'mixed', 'internal_pad', 'repeated'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_gpu_edges(vocab, kind, dtype):
    torch.manual_seed(92)
    ids = torch.tensor([cases(vocab)[kind]])
    layout = build_tg_layout(ids, vocab, device='cuda')
    qkv = [torch.randn(1, 2, ids.numel(), 32, device='cuda', dtype=dtype, requires_grad=True) for _ in range(3)]
    check(qkv, layout, reference(ids)[0])
    if kind in ('singleton', 'pads', 'self'):
        assert torch.count_nonzero(qkv[0].grad) == torch.count_nonzero(qkv[1].grad) == 0


@GPU
@pytest.mark.parametrize('dtype,d,n', [(torch.float32, 16, 127), (torch.float16, 64, 2048),
    (torch.bfloat16, 64, 2048), (torch.bfloat16, 80, 129), (torch.bfloat16, 128, 128)])
def test_gpu_real_strides(vocab, dtype, d, n):
    torch.manual_seed(419)
    ids = tokens(n)
    ids[0, -7:] = vocab.pad
    layout = build_tg_layout(ids, vocab, device='cuda')
    storage = torch.randn(2, n, 3, 2, d*2, device='cuda', dtype=dtype)
    qkv = [storage[:, :, i, :, ::2].transpose(1, 2).detach().requires_grad_() for i in range(3)]
    upstream = torch.randn(2, 2, 1, d, device='cuda', dtype=dtype).expand(2, 2, n, d)
    check(qkv, layout, reference(ids)[0], upstream=upstream)


@GPU
@pytest.mark.parametrize('groups', [(('tg', 2), ('tgnomask', 2)), (('tgtree', 2), ('tg', 2))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_gpu_mixed(vocab, groups, dtype):
    torch.manual_seed(209)
    ids = tokens(129)
    ids[0, -7:] = vocab.pad
    layout = build_mix_tg_layout(ids, vocab, groups, device='cuda')
    qkv = [torch.randn(2, 129, 4, 32, device='cuda', dtype=dtype).transpose(1, 2).detach().requires_grad_() for _ in range(3)]
    check(qkv, layout, reference(ids, groups)[0], mixed=True)


def model_config(groups=(), **overrides):
    from olmo.config import ModelConfig, TGConfig, BlockType, LayerNormType, ActivationType, InitFnType
    values = dict(d_model=64, n_heads=4, n_layers=2, mlp_hidden_size=128,
        vocab_size=50320, embedding_size=50320, max_sequence_length=2048,
        block_type=BlockType.sequential, layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu, rope=True, flash_attention=False,
        flex_attention=False, attention_dropout=0., residual_dropout=0., embedding_dropout=0.,
        init_device='cpu', init_fn=InitFnType.normal, init_std=.02, weight_tying=True,
        transformer_grammar_type='mixing' if groups else 'tg',
        mix_head_type=[TGConfig(grammar_type=kind, n_heads=h) for kind, h in groups])
    values.update(overrides)
    return ModelConfig(**values)


@pytest.mark.parametrize('groups', [(), (('tg', 2), ('tgnomask', 2)), (('tgtree', 2), ('tg', 2))])
def test_collator_and_microbatch(vocab, groups):
    from olmo.config import TrainConfig
    from olmo.data.collator import DataCollator
    from olmo.torch_util import move_to_device
    from olmo.train import Trainer
    from types import SimpleNamespace
    cfg = TrainConfig(model=model_config(groups, tg_typed_attention=True))
    cfg.tokenizer.vocabulary = VOCAB
    cfg.data.generate_attention_mask = False
    cfg.data.generate_doc_lengths = False
    collator = DataCollator.from_train_config(cfg)
    ids = tokens(129, batch=3)
    items = [{'input_ids': row, 'label_mask': torch.ones_like(row, dtype=torch.bool)} for row in ids]
    items[0]['label_mask'][5] = False
    batch = collator(items)
    assert 'attention_bias' not in batch
    layout = batch['tg_layout']
    mask, label = reference(ids, groups or (('tg', 1),))
    assert torch.equal(layout.dense_mask(), mask)
    assert not batch['label_mask'][0, 5]
    trainer = SimpleNamespace(cfg=SimpleNamespace(device_train_microbatch_size=2))
    micro = Trainer.split_batch(trainer, move_to_device(batch, torch.device('cpu')))
    assert [x['tg_layout'].shape[0] for x in micro] == [2, 1]
    assert torch.equal(torch.cat([x['tg_layout'].dense_mask() for x in micro]), mask)


@pytest.mark.parametrize('extra', ['bias', 'mask', 'cache', 'dropout', 'alibi', 'grammar', 'heads'])
def test_model_rejects_ambiguous_layout(vocab, extra):
    from olmo.model import OLMo
    from olmo.exceptions import OLMoConfigurationError
    ids = torch.tensor([[1, 2, 3]])
    groups = (('tgtree', 2), ('tg', 2))
    layout = build_mix_tg_layout(ids, vocab, groups) if extra == 'heads' else build_tg_layout(ids, vocab)
    kwargs = {'attention_bias': layout.dense_mask()} if extra == 'bias' else {'attention_mask': torch.ones_like(ids)} if extra == 'mask' else {'use_cache': True} if extra == 'cache' else {}
    overrides = {'attention_dropout': .1} if extra == 'dropout' else {'alibi': True, 'rope': False} if extra == 'alibi' else {'transformer_grammar_type': 'tgnomask'} if extra == 'grammar' else {}
    cfg = model_config(tuple(reversed(groups)) if extra == 'heads' else (), **overrides)
    with pytest.raises(OLMoConfigurationError):
        OLMo(cfg, init_params=False)(ids, tg_layout=layout, **kwargs)


@GPU
@pytest.mark.parametrize('groups', [(), (('tg', 2), ('tgnomask', 2)), (('tgtree', 2), ('tg', 2))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('group_size', [1, 2])
def test_full_model(vocab, groups, dtype, group_size):
    from olmo.model import OLMo
    torch.manual_seed(203)
    model = OLMo(model_config(groups, block_group_size=group_size)).cuda().to(dtype).train()
    ids = tokens(127)
    ids[0, -5:] = vocab.pad
    mask, label = reference(ids, groups or (('tg', 1),))
    layout = build_mix_tg_layout(ids, vocab, groups, device='cuda') if groups else build_tg_layout(ids, vocab, device='cuda')
    ids = ids.cuda()
    target = ids[:, 1:].reshape(-1)
    keep = label[:, 1:].reshape(-1).cuda() & (target != vocab.pad)
    def run(typed):
        model.zero_grad(set_to_none=True)
        output = model(ids, **({'tg_layout': layout} if typed else {'attention_bias': mask.cuda()})).logits
        losses = F.cross_entropy(output[:, :-1].float().reshape(-1, output.shape[-1]), target, reduction='none')
        loss = losses[keep].mean()
        loss.backward()
        return output.detach().float(), loss.detach(), {name: p.grad.detach().float().clone() for name, p in model.named_parameters() if p.grad is not None}
    ref, ref_loss, ref_grads = run(False)
    actual, loss, grads = run(True)
    atol, rtol = (.015, .04) if dtype == torch.bfloat16 else (2e-5, 5e-4)
    torch.testing.assert_close(actual, ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(loss, ref_loss, atol=atol, rtol=rtol)
    assert grads.keys() == ref_grads.keys()
    error, norm = 0., 0.
    for name, grad in grads.items():
        assert grad.isfinite().all(), name
        torch.testing.assert_close(grad, ref_grads[name], atol=atol, rtol=rtol, msg=name)
        error += float((grad-ref_grads[name]).square().sum())
        norm += float(ref_grads[name].square().sum())
    assert (error/max(norm, 1e-20))**.5 < (.012 if dtype == torch.bfloat16 else .0002)


@GPU
@pytest.mark.parametrize('groups', [(), (('tg', 2),), (('tg', 1), ('tgnomask', 1)), (('tgtree', 1), ('tg', 1))])
def test_retained_graph_accumulation(vocab, groups):
    ids = tokens(129)
    ids[0, :] = vocab.pad
    layout = build_mix_tg_layout(ids, vocab, groups, device='cuda') if groups else build_tg_layout(ids, vocab, device='cuda')
    torch.manual_seed(182)
    qkv = [torch.randn(2, 129, 2, 32, device='cuda', dtype=torch.bfloat16).transpose(1, 2).detach().requires_grad_() for _ in range(3)]
    fn = mix_tg_attention if groups else tg_attention
    out = fn(*qkv, layout)
    do = torch.randn_like(out)
    out.backward(do, retain_graph=True)
    grads = [x.grad.clone() for x in qkv]
    out.backward(do)
    for x, grad in zip(qkv, grads):
        torch.testing.assert_close(x.grad, 2*grad, atol=0, rtol=0)


@GPU
def test_layout_mutation_version_check(vocab):
    ids = tokens(17)
    layout = build_tg_layout(ids, vocab, device='cuda')
    qkv = [torch.randn(2, 2, 17, 32, device='cuda', requires_grad=True) for _ in range(3)]
    out = tg_attention(*qkv, layout)
    layout.lo.add_(0)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        out.sum().backward()
