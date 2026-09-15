"""Diagnostic kernel timing and layout statistics; no production changes."""
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch
from olmo.data.tg_mask import SentencepieceVocab
from olmo.attention_kernels.tgnomask import build_tgnomask_layout, tgnomask_attention

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)
vocab = SentencepieceVocab.from_vocab_file(str(ROOT / 'dataset/bbc-news/TG_GPT2_tokenizer.json'))
data = np.load(ROOT / 'dataset/bbc-news/tg/test.npy', mmap_mode='r')
groups = json.loads((HERE / 'benchmark_4451.json').read_text())['groups']
result = {'job': os.environ.get('SLURM_JOB_ID'), 'gpu': torch.cuda.get_device_name(),
          'torch': torch.__version__, 'groups': [],
          'protocol': 'Same 10 groups and BF16 B4 H12 N2048 D64 as 4491. Five warmups and five profiled calls per workload. CUDA event durations from profiler, diagnostic only; not replacement end-to-end timings.',
          'sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                     for f in ('olmo/attention_kernels/tgnomask.py', 'olmo/attention_kernels/kernels.py')}}
for index, old in enumerate(groups):
    tokens = torch.stack([torch.from_numpy(np.array(data[c*2048:(c+1)*2048], dtype=np.int64))
                          for c in old['chunk_indices']])
    cpu = build_tgnomask_layout(tokens, vocab)
    edges = int(cpu.offsets[:, -1].sum())
    prefix_pairs = int(cpu.prefix.sum())
    special = int(cpu.self_mask.sum())
    degrees = (cpu.offsets[:, 1:] - cpu.offsets[:, :-1]).flatten()
    degrees = degrees[degrees > 0]
    row = {'index': index, 'q_counts': cpu.q_count.tolist(), 'k_counts': cpu.k_count.tolist(),
           'sparse_counts': cpu.sparse_count.tolist(), 'sparse_edges': edges,
           'sparse_mean_degree': float(degrees.float().mean()), 'sparse_max_degree': int(degrees.max()),
           'visible_pair_ratio_vs_causal': (prefix_pairs + special + edges)/(4*2048*2049/2),
           'kernels_us': {}}
    layout = cpu.to('cuda')
    torch.manual_seed(6198+index)
    q,k,v = [torch.randn(4,2048,12,64,device='cuda',dtype=torch.bfloat16).transpose(1,2).detach().requires_grad_() for _ in range(3)]
    upstream = torch.randn_like(q)
    for workload in ('eval', 'train'):
        with torch.set_grad_enabled(workload == 'train'):
            def step():
                out = tgnomask_attention(q,k,v,layout)
                if workload == 'train':
                    out.backward(upstream)
                    q.grad = k.grad = v.grad = None
            for _ in range(5): step()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(5): step()
                torch.cuda.synchronize()
            durations = collections.defaultdict(float)
            for event in prof.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    durations[event.name] += event.time_range.elapsed_us()/5
            row['kernels_us'][workload] = dict(durations)
    result['groups'].append(row)
    print(json.dumps(row), flush=True)
result['mean_kernels_us'] = {w: {k: statistics.mean(g['kernels_us'][w].get(k, 0.) for g in result['groups'])
    for k in sorted(set().union(*(g['kernels_us'][w] for g in result['groups'])))} for w in ('eval','train')}
result['status'] = 'passed'
path = HERE / f"headroom_profile_{result['job']}.json"
path.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result['mean_kernels_us'], indent=2), flush=True)
