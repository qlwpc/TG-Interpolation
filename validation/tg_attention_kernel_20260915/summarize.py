"""Summarize paired runs without counting repeated input groups as independent."""
import hashlib
import json
from pathlib import Path
import statistics as s
import sys

root = Path(__file__).resolve().parents[2]
paths = [Path(p) for p in sys.argv[1:]]
runs = [json.loads(p.read_text()) for p in paths]
assert len(runs) == 2 and all(r['status'] == 'completed' and len(r['cases']) == 30 for r in runs)
for name in ('olmo/attention_kernels/tg_attention.py', 'olmo/attention_kernels/layouts.py', 'olmo/attention_kernels/kernels.py', 'olmo/model.py', 'olmo/data/collator.py'):
    assert all(r['source_sha256'][name] == hashlib.sha256((root/name).read_bytes()).hexdigest() for r in runs), name

summary = {'runs': [p.name for p in paths], 'paired_input_groups': 10, 'modes': {}}
for mode in ('tg', 'nomask_mix', 'tree_mix'):
    by_run = [[x for x in r['cases'] if x['mode'] == mode] for r in runs]
    assert all(a['tokens_sha256'] == b['tokens_sha256'] for a, b in zip(*by_run))
    entry = {'times': {}, 'per_run': []}
    for work in ('forward', 'train'):
        entry['times'][work] = {}
        for method in ('typed', 'sdpa', 'flex', 'causal_flash'):
            values = [s.mean(a['times'][work][method]['mean'] for a in group) for group in zip(*by_run)]
            entry['times'][work][method] = {'mean_ms': s.mean(values), 'std_ms': s.stdev(values), 'paired_group_means_ms': values}
        times = entry['times'][work]
        entry['times'][work]['typed_over_flash'] = times['typed']['mean_ms']/times['causal_flash']['mean_ms']
        entry['times'][work]['speedup_vs_sdpa'] = times['sdpa']['mean_ms']/times['typed']['mean_ms']
        entry['times'][work]['speedup_vs_flex'] = times['flex']['mean_ms']/times['typed']['mean_ms']
    for cases in by_run:
        entry['per_run'].append({w: {m: s.mean(x['times'][w][m]['mean'] for x in cases)
                                     for m in ('typed', 'causal_flash')} for w in ('forward', 'train')})
    entry['max_relative_l2'] = max(e['relative_l2'] for cases in by_run for x in cases for e in x['errors'])
    entry['cpu_layout_median_ms'] = s.median(x['cpu_layout_ms'] for cases in by_run for x in cases)
    entry['h2d_median_ms'] = s.median(x['h2d_ms'] for cases in by_run for x in cases)
    summary['modes'][mode] = entry
Path(__file__).with_name('summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
