"""Paired real-mask attention timing, including all operator/autograd overhead."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from olmo.data.tg_mask import SentencepieceVocab, TG_attention_bias, KProximal_TG_attention_bias
from olmo.attention_kernels.tg_attention import build_tg_layout, build_mix_tg_layout, tg_attention, mix_tg_attention
from olmo.attention_kernels import kernels as tg_kernels

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {'tg': (('tg', 12),), 'nomask_mix': (('tg', 6), ('tgnomask', 6)),
          'tree_mix': (('tgtree', 6), ('tg', 6))}


def masks_for(tokens, groups, vocab_path):
    masks = []
    n = tokens.shape[1]
    for row in tokens:
        values = []
        for kind, heads in groups:
            if kind == 'tgtree':
                mask = torch.ones(n, n, dtype=torch.bool).tril()
            else:
                gen = TG_attention_bias(str(vocab_path), n) if kind == 'tg' else KProximal_TG_attention_bias(str(vocab_path), n, n, False)
                mask, _ = gen(row)
            values.append(mask.expand(heads, n, n))
        masks.append(torch.cat(values))
    return torch.stack(masks)


def difference(actual, expected):
    a, e = actual.float(), expected.float()
    result = {'finite': bool(a.isfinite().all()), 'relative_l2': float((a-e).norm()/e.norm().clamp_min(1e-12)),
              'max_abs': float((a-e).abs().max())}
    assert result['finite'] and result['relative_l2'] < .012, result
    return result


def measure(fn, xs, upstream, backward, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        if backward:
            for x in xs:
                x.grad = None
            fn().backward(upstream)
        else:
            with torch.no_grad():
                fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)/iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--timing-seed', type=int, default=6198)
    parser.add_argument('--modes', nargs='+', default=list(GROUPS))
    parser.add_argument('--no-flex', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = Path(__file__).with_name(f'benchmark_{os.environ.get("SLURM_JOB_ID", "manual")}.json')
    result = {'args': vars(args), 'shape': [4, 12, 2048, 64], 'dtype': 'bfloat16',
              'seed': 6198, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'config': tg_kernels._CONFIG, 'status': 'running', 'cases': [],
              'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                ('olmo/attention_kernels/tg_attention.py', 'olmo/attention_kernels/layouts.py', 'olmo/attention_kernels/kernels.py',
                                 'olmo/model.py', 'olmo/data/collator.py', 'olmo/data/tg_mask.cpython-310-x86_64-linux-gnu.so',
                                 'validation/tg_attention_kernel_20260915/benchmark.py')},
              'protocol': 'CUDA events, production BNHD transpose strides, same QKV and dO per method; forward or forward+backward, includes output cat and all autograd/layout tensor handling; excludes CPU metadata/dense mask construction/H2D/additive-mask conversion/BlockMask/compile/model projections/MLP. SDPA uses precomputed same-dtype additive mask. CPU layout and H2D measured separately. causal_flash is speed reference only.'}
    def save():
        path.write_text(json.dumps(result, indent=2)+'\n')
    save()
    vocab_path = ROOT/'dataset/bbc-news/TG_GPT2_tokenizer.json'
    vocab = SentencepieceVocab.from_vocab_file(str(vocab_path))
    data = np.load(ROOT/'dataset/bbc-news/tg/test.npy', mmap_mode='r')
    chunks = np.random.default_rng(6198).choice(len(data)//2048, size=args.groups*4, replace=False).reshape(args.groups, 4)
    compiled_flex = torch.compile(flex_attention, dynamic=False)
    b, h, n, d = result['shape']
    order_rng = random.Random(args.timing_seed)
    for gi, group in enumerate(chunks):
        tokens = torch.stack([torch.from_numpy(np.array(data[int(c)*n:(int(c)+1)*n], dtype=np.int64)) for c in group])
        torch.manual_seed(6198+gi)
        xs = [torch.randn(b, n, h, d, device='cuda', dtype=torch.bfloat16).transpose(1, 2).detach().requires_grad_() for _ in range(3)]
        upstream = torch.randn_like(xs[0])
        for mode in args.modes:
            groups = GROUPS[mode]
            begun = time.perf_counter()
            layout = build_tg_layout(tokens, vocab) if mode == 'tg' else build_mix_tg_layout(tokens, vocab, groups)
            cpu_ms = (time.perf_counter()-begun)*1000
            torch.cuda.synchronize()
            begun = time.perf_counter()
            layout = layout.to('cuda')
            torch.cuda.synchronize()
            h2d_ms = (time.perf_counter()-begun)*1000
            mask = masks_for(tokens, groups, vocab_path).cuda()
            assert torch.equal(layout.dense_mask(), mask[:, :1] if mode == 'tg' else mask)
            if mode == 'tg':
                mask = mask[:, :1].contiguous()
            additive = torch.zeros_like(mask, dtype=xs[0].dtype)
            additive.masked_fill_(~mask, torch.finfo(xs[0].dtype).min)
            typed = lambda: (tg_attention if mode == 'tg' else mix_tg_attention)(*xs, layout)
            def sdpa():
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return F.scaled_dot_product_attention(*xs, attn_mask=additive)
            def flash():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(*xs, is_causal=True)
            methods = {'typed': typed, 'sdpa': sdpa, 'causal_flash': flash}
            if not args.no_flex:
                if mode == 'tg':
                    mask3 = mask.squeeze(1)
                    def mask_mod(batch, head, q, k):
                        return mask3[batch, q, k]
                else:
                    def mask_mod(batch, head, q, k):
                        return mask[batch, head, q, k]
                bm = create_block_mask(mask_mod, B=b, H=None if mode=='tg' else h, Q_LEN=n, KV_LEN=n, device='cuda')
                methods['flex'] = lambda: compiled_flex(*xs, block_mask=bm)
            ref = sdpa()
            ref_grads = torch.autograd.grad(ref, xs, upstream)
            actual = typed()
            grads = torch.autograd.grad(actual, xs, upstream)
            errors = [difference(actual, ref)]+[difference(g, r) for g, r in zip(grads, ref_grads)]
            flex_errors = None
            if 'flex' in methods:
                flex_out = methods['flex']()
                flex_grads = torch.autograd.grad(flex_out, xs, upstream)
                flex_errors = [difference(flex_out, ref)] + [difference(g, r) for g, r in zip(flex_grads, ref_grads)]
            case = {'group': gi, 'mode': mode, 'chunks': group.tolist(),
                    'tokens_sha256': hashlib.sha256(tokens.numpy().tobytes()).hexdigest(), 'cpu_layout_ms': cpu_ms,
                    'h2d_ms': h2d_ms, 'errors': errors, 'flex_errors': flex_errors, 'times': {}}
            result['cases'].append(case)
            for backward in (False, True):
                tag = 'train' if backward else 'forward'
                samples = {name: [] for name in methods}
                for fn in methods.values():
                    measure(fn, xs, upstream, backward, 5)
                for _ in range(args.rounds):
                    names = list(methods)
                    order_rng.shuffle(names)
                    for name in names:
                        samples[name].append(measure(methods[name], xs, upstream, backward,
                                                     max(1, args.iterations//2) if backward else args.iterations))
                case['times'][tag] = {name: {'mean': statistics.mean(vals), 'rounds': vals} for name, vals in samples.items()}
                print(gi, mode, tag, {name: round(v['mean'], 4) for name, v in case['times'][tag].items()}, flush=True)
                save()
    result['status'] = 'completed'
    save()


if __name__ == '__main__':
    main()
