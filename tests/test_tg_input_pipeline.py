"""CPU layout equivalence, persistent data order, and asynchronous storage lifetimes."""
import pickle
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from olmo.attention_kernels.tg_attention import TGLayout, build_tg_layout, MixTGLayout
from olmo.data.iterable_dataset import IterableDataset
from olmo.data.prefetch import CUDABatchPrefetcher
from olmo.data.tg_mask import SentencepieceVocab
from olmo.torch_util import move_to_device

ROOT = Path(__file__).resolve().parents[1]


def vocab():
    return SentencepieceVocab.from_vocab_file(str(ROOT / 'dataset/bbc-news/TG_GPT2_tokenizer.json'))


def equal_layout(a, b):
    for x, y in ((a, b), (a.base, b.base)):
        for f in fields(x):
            value = getattr(x, f.name)
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.cpu(), getattr(y, f.name).cpu()), f.name
    assert a.base.sparse_capacity == b.base.sparse_capacity


@pytest.mark.parametrize('tile', [16, 32, 64])
@pytest.mark.parametrize('n', [1, 17, 127, 511, 2048])
def test_native_matches_python(tile, n):
    v = vocab()
    data = np.load(ROOT / 'dataset/bbc-news/tg/test.npy', mmap_mode='r')
    ids = torch.tensor(np.array(data[13:13+4*n], dtype=np.int64)).reshape(4,n)
    ids[0,-min(n,7):] = v.pad
    python = build_tg_layout(ids, v, backend='python', pack=False, q_tile=tile, k_tile=tile)
    native = build_tg_layout(ids, v, backend='native', q_tile=tile, k_tile=tile)
    equal_layout(native, python)
    restored = pickle.loads(pickle.dumps(native))
    equal_layout(restored, python)
    assert len({t.untyped_storage().data_ptr() for t in restored._tensors()}) == 2
    assert len(native._buffers) == 2
    equal_layout(native.slice_batch(1,3), python.slice_batch(1,3))
    assert len({t.untyped_storage().data_ptr() for t in native._tensors()}) == 2


def test_random_truncated_and_degenerate():
    v = vocab()
    rng = np.random.default_rng(382)
    alphabet = [v.pad, v.opening_non_terminals[0], v.opening_non_terminals[0]+1,
                v.closing_non_terminals[0], v.closing_non_terminals[0]+1, 10, 20]
    for n in (2, 31, 128):
        for _ in range(10):
            ids = torch.tensor(rng.choice(alphabet, size=(3,n)))
            equal_layout(build_tg_layout(ids, v, backend='native'),
                         build_tg_layout(ids, v, backend='python', pack=False))


def make_dataset():
    return IterableDataset([torch.tensor([i]) for i in range(48)], global_batch_size=4,
                           seed=71, shuffle=True, world_size=1, rank=0, fs_local_rank=0, num_threads=0)


def collect(loader):
    return torch.cat([b['index'] for b in loader]).tolist()


@pytest.mark.parametrize('workers,context', [(0,None),(1,'fork'),(2,'fork'),(4,'fork'),(2,'spawn')])
def test_persistent_workers_resume_epoch(workers, context):
    dataset = make_dataset()
    loader = DataLoader(dataset, batch_size=4, num_workers=workers,
                        persistent_workers=workers>0, multiprocessing_context=context)
    for epoch, start, limit in ((0,0,None),(1,0,None),(1,12,None),(2,4,32)):
        dataset.reshuffle(epoch)
        dataset.start_index = start
        dataset.max_examples = limit
        expected = dataset._build_global_indices()[:limit][start:].tolist()
        assert collect(loader) == expected


def test_disabled_prefetch():
    loader = [{'input_ids': torch.tensor([i])} for i in range(7)]
    with CUDABatchPrefetcher(loader, 'cpu') as batches:
        assert [int(b['input_ids']) for b in batches] == list(range(7))
    assert batches.closed


class LayoutCollator:
    def __call__(self, items):
        ids = torch.stack(items)
        return {'input_ids': ids, 'tg_layout': build_tg_layout(ids, vocab(), backend='native')}


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Slurm CUDA allocation')
@pytest.mark.parametrize('workers', [0, 2])
def test_pinned_ipc_async_lifetime(workers):
    ids = [torch.arange(127,dtype=torch.long) for _ in range(24)]
    loader = DataLoader(ids,batch_size=2,collate_fn=LayoutCollator(),num_workers=workers,
                        pin_memory=True,persistent_workers=workers>0)
    cpu = next(iter(loader))
    assert all(t.is_pinned() for t in cpu['tg_layout']._tensors())
    assert len({t.untyped_storage().data_ptr() for t in cpu['tg_layout']._tensors()}) == 2
    expected = cpu['tg_layout']
    outputs = []
    with CUDABatchPrefetcher(loader,'cuda') as batches:
        for i,batch in enumerate(batches):
            # Operations execute on the consumer stream after its event wait.
            outputs.append(batch['tg_layout'].lo.clone())
            equal_layout(batch['tg_layout'], expected)
            assert batch['input_ids'].is_cuda
            if i == 4:
                break
    # Early close and resetting a persistent loader must preserve the next full epoch.
    with CUDABatchPrefetcher(loader,'cuda') as batches:
        assert sum(1 for _ in batches) == 12
        with pytest.raises(StopIteration):
            next(batches)
    torch.cuda.synchronize()
    assert all(torch.equal(x.cpu(),expected.lo) for x in outputs)
    mixed = MixTGLayout(expected,(('tg',1),('tgtree',1)))
    copied = move_to_device({'x': [mixed]},torch.device('cuda'),non_blocking=True)['x'][0]
    equal_layout(copied.tg,expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Slurm CUDA allocation')
@pytest.mark.parametrize('groups', [(), (('tg',2),('tgnomask',2)), (('tgtree',2),('tg',2))])
def test_prefetch_attention_without_per_batch_sync(groups):
    from olmo.attention_kernels.tg_attention import tg_attention, mix_tg_attention
    from olmo.config import PaddingDirection
    from olmo.data.collator import DataCollator
    v = vocab()
    data = np.load(ROOT / 'dataset/bbc-news/tg/test.npy',mmap_mode='r')
    ids = torch.tensor(np.array(data[13:13+254],dtype=np.int64)).reshape(2,127)
    collator = DataCollator(PaddingDirection.right,v.pad,False,'mixing' if groups else 'tg',
                            tg_typed_attention=True,mix_head_type=groups,tg_layout_backend='native')
    collator.vocab = v
    loader = DataLoader(list(ids)*10,batch_size=2,num_workers=2,pin_memory=True,collate_fn=collator)
    xs = [torch.randn(2,4,127,32,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    upstream = torch.randn_like(xs[0])
    fn = mix_tg_attention if groups else tg_attention
    base = build_tg_layout(ids,v,backend='python',device='cuda')
    reference = MixTGLayout(base,groups) if groups else base
    fn(*xs,reference).backward(upstream)  # Warm JIT before testing stream concurrency.
    results = []
    with CUDABatchPrefetcher(loader,'cuda') as batches:
        for batch in batches:
            for x in xs: x.grad = None
            out = fn(*xs,batch['tg_layout'])
            out.backward(upstream)
            results.append([out.detach().clone()] + [x.grad.clone() for x in xs])
    for x in xs: x.grad = None
    expected = fn(*xs,reference)
    expected.backward(upstream)
    for result in results:
        for actual, ref in zip(result,[expected]+[x.grad for x in xs]):
            torch.testing.assert_close(actual,ref,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires Slurm CUDA allocation')
def test_prefetch_propagates_errors():
    def broken():
        yield {'input_ids': torch.ones(4,dtype=torch.long).pin_memory()}
        raise ValueError('input producer failed')
    with CUDABatchPrefetcher(broken(),'cuda') as batches:
        assert next(batches)['input_ids'].is_cuda
        with pytest.raises(ValueError,match='input producer failed'):
            next(batches)
    assert not batches.thread.is_alive()
