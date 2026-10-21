"""Render the public dedup results from existing validated campaign summaries.

This does not rerun evaluation or certify original checkpoints. Incomplete
coverage remains incomplete; it never extrapolates missing PPL or seed values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('bbc_dedup_500m_local_20260919', 'bbc_dedup_500m_pause_noont_20260922',
         'bbc_dedup_500m_sist_20260922')
LABELS = {'terminal': 'Terminal', 'tree': 'Tree', 'tgtree': 'TGTree',
          'pause1': 'Pause-1', 'pause2': 'Pause-2', 'pause_1': 'Pause-1', 'pause_2': 'Pause-2',
          'tree_noont': 'Tree-NoONT', 'tg': 'TG', 'tgnomask': 'TGNomask',
          'tgtree_mix_tg': 'TGTree-Mix-TG', 'tgnomask_mix_tg': 'TGNomask-Mix-TG'}


def render_report(summaries: dict, root: Path = ROOT, *, check: bool = False) -> None:
    if set(summaries) != set(NAMES):
        raise ValueError('Expected the three registered campaign summaries')
    evidence = root / 'reproducibility/results/dedup500m'
    lines = ['# Additional BBC dedup 500M experiments', '',
             'These models were trained from scratch on the new deduplicated training set; they do not replace the [frozen paper table](paper.md).',
             'SG/BLiMP/BoolQ and XSum R-AVG are percentages; ± denotes the sample standard deviation across five fine-tuning seeds.',
             'Fine-tuning seeds: 42, 2026, 6198, 13171, 31723. These are not five pretraining seeds.',
             'DocPPL uses the full reserved-clean v1 test: 5,000 documents, 129,085 sentences and 3,068,713 scored tokens.',
             'Terminal/Pause use a single path; structural models sum valid top-300 joint probabilities and use model-best history.',
             'As of 2026-09-29, each of the ten models had completed 22/22 evaluation tasks.', '',
             'The 2026-10-07 review found possible duplicate candidates after NoONT projection. Its historical DocPPL is marked † below',
             'and must not be used for current rankings; reevaluation under the [projection-deduplication protocol](../protocols/clean_document_ppl.md) is required.',
             'Original JSON and acceptance summaries retain their historical bytes and completion status; they do not certify the later deduplication requirement.', '',
             '## Results', '',
             '| Evaluation site | Model | Completed tasks | SG ↑ | BLiMP ↑ | clean DocPPL ↓ | XSum R-AVG ↑ | BoolQ ↑ |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        result = summaries[name]
        for model, entry in result['models'].items():
            def single(key):
                rows = entry[key]
                metric = 'avg' if key == 'SG' else 'overall/overall'
                return f"{rows[0]['metrics'][metric] * 100:.4f}" if len(rows) == 1 else 'pending'

            def multi(key, metric):
                values = entry.get(key + '_mean_sd', {})
                if len(entry[key]) != 5 or metric not in values:
                    return f'{len(entry[key])}/5 seed'
                value = values[metric]
                return f"{value['mean']*100:.4f} ± {value['sample_sd']*100:.4f}"

            if 'DocPPL' in entry and entry['document_coverage'] != 5000:
                raise ValueError(f'{name}/{model}: PPL without complete document coverage')
            doc = f"{entry['DocPPL']:.6f}" if 'DocPPL' in entry else f"{entry['document_coverage']}/5000 documents"
            if model == 'tree_noont' and 'DocPPL' in entry:
                doc += ' †'
            lines.append('| ' + ' | '.join([
                'SIST' if name == NAMES[2] else 'RTX3090B', LABELS[model],
                f"{entry['tasks_complete']}/{entry['tasks_total']}", single('SG'), single('blimp'), doc,
                multi('xsum_finetune', 'R-AVG'), multi('boolq', 'eval/downstream/boolq_acc__')]) + ' |')
    lines += ['', '## Evidence and scope', '',
              'The following JSON files contain per-seed values, SG/BLiMP submetrics, document coverage and original collection times.',
              'They are public copies of the original experiment summaries. Aggregation checked task identity, successful exits,',
              'complete evaluation and DocPPL document/sentence/token coverage.',
              'Post-evaluation input and checkpoint SHA-256 hashes were checked against the original manifests.',
              'SIST checks used a local synchronized mirror; remote storage was not rescanned and evaluation was not rerun.', '']
    for name in NAMES:
        result = summaries[name]
        lines.append(f"- [{name}](../../reproducibility/results/dedup500m/{name}/results.json): "
                     f"`{result['status']}`, collected at `{result.get('collected_at', 'not recorded')}`.")
    lines += ['', 'Full training configurations and identities are in the [additional experiment configs](../../reproducibility/README.md#dedup-configs).',
              'The public dedup CLI covers eight recipes. Frozen configs for this campaign\'s Pause-1/2 and TGTree-Mix-TG are stored separately;',
              'TGNomask-Aug or similarly named recipes cannot substitute for them. Data, weights and full-log availability are listed in the',
              '[public asset inventory](../../reproducibility/README.md#assets).', '',
              'This page is generated from the JSON above by `scripts/render_dedup_results.py`. All three batches, aggregation and local acceptance were completed;',
              'counts, result hashes and scope are recorded in the [acceptance summary](../../reproducibility/results/dedup500m/final_status.json).', '']
    # Validate all rows before replacing any public files.
    outputs = {root / 'docs/results/bbc_dedup_500m.md': '\n'.join(lines)}
    for name in NAMES:
        outputs[evidence / name / 'results.json'] = (
            json.dumps(summaries[name], indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        )
    if check:
        stale = [str(path.relative_to(root)) for path, content in outputs.items()
                 if not path.is_file() or path.read_bytes() != content.encode('utf-8')]
        if stale:
            raise ValueError('Generated results are out of date: ' + ', '.join(stale))
        return
    for dest, content in outputs.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + '.tmp')
        temp.write_text(content, encoding='utf-8')
        temp.replace(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'reproducibility/results/dedup500m')
    parser.add_argument('--check', action='store_true', help='Verify generated files without writing them')
    args = parser.parse_args()
    summaries = {name: json.loads((args.results_root / name / 'results.json').read_text()) for name in NAMES}
    render_report(summaries, check=args.check)


if __name__ == '__main__':
    main()
