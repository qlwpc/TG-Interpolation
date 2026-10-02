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


def render_report(summaries: dict, root: Path = ROOT) -> None:
    if set(summaries) != set(NAMES):
        raise ValueError('Expected the three registered campaign summaries')
    evidence = root / 'reproducibility/results/dedup500m'
    lines = ['# BBC dedup 500M 补充实验', '',
             '这些模型使用新 dedup train 重新训练；不替换[冻结论文表](paper_results.md)。',
             'SG/BLiMP/BoolQ 为百分数，XSum 为 R-AVG 百分数；± 为五个微调 seed 的样本标准差。',
             '微调 seed：42、2026、6198、13171、31723；并非五个预训练 seed。',
             'DocPPL 使用 reserved-clean v1 完整 test：5,000 篇、129,085 句、3,068,713 个计分 token。',
             'Terminal/Pause 是单路径；结构模型是 valid top-300 joint sum、model-best 历史。',
             '截至 2026-09-29，十模型各 22/22 项测试完成；五 seed 指微调重复，并非预训练重复。', '',
             '## 结果', '',
             '| 执行地 | 模型 | 完成任务 | SG ↑ | BLiMP ↑ | clean DocPPL ↓ | XSum R-AVG ↑ | BoolQ ↑ |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        result = summaries[name]
        for model, entry in result['models'].items():
            def single(key):
                rows = entry[key]
                metric = 'avg' if key == 'SG' else 'overall/overall'
                return f"{rows[0]['metrics'][metric] * 100:.4f}" if len(rows) == 1 else '待完成'

            def multi(key, metric):
                values = entry.get(key + '_mean_sd', {})
                if len(entry[key]) != 5 or metric not in values:
                    return f'{len(entry[key])}/5 seed'
                value = values[metric]
                return f"{value['mean']*100:.4f} ± {value['sample_sd']*100:.4f}"

            if 'DocPPL' in entry and entry['document_coverage'] != 5000:
                raise ValueError(f'{name}/{model}: PPL without complete document coverage')
            doc = f"{entry['DocPPL']:.6f}" if 'DocPPL' in entry else f"{entry['document_coverage']}/5000 篇"
            lines.append('| ' + ' | '.join([
                'SIST' if name == NAMES[2] else 'RTX3090B', LABELS[model],
                f"{entry['tasks_complete']}/{entry['tasks_total']}", single('SG'), single('blimp'), doc,
                multi('xsum_finetune', 'R-AVG'), multi('boolq', 'eval/downstream/boolq_acc__')]) + ' |')
    lines += ['', '## 可核对证据与范围', '',
              '以下 JSON 包含逐 seed、SG/BLiMP 子项、文档覆盖和原收集时间。它们是原实验汇总的',
              '公开副本；严格汇总已核对任务身份、成功退出、完整评测和 DocPPL 文档/句/token 覆盖。',
              '评测后的输入和 checkpoint 文件 SHA-256 已按原 manifest 再次核对；',
              'SIST 使用本地同步镜像，本次未重新扫描远端存储或重新评测。', '']
    for name in NAMES:
        result = summaries[name]
        dest = evidence / name / 'results.json'
        lines.append(f"- [{name}](../reproducibility/results/dedup500m/{name}/results.json)："
                     f"`{result['status']}`，收集于 `{result.get('collected_at', '未记录')}`。")
    lines += ['', '完整训练配置及身份见[补充实验配置](../reproducibility/README.md#dedup-configs)。',
              '公共 dedup CLI 覆盖八种配方；本批 Pause-1/2 与 TGTree-Mix-TG 的冻结配置另行保存，',
              '不能用 TGNomask-Aug 或其他近名配方代替。数据、权重及完整日志的获取状态见',
              '[公开资产清单](../reproducibility/README.md#assets)。', '',
              '本页由 `scripts/render_dedup_results.py` 从上述 JSON 生成。三批任务、汇总与本地终验均完成；',
              '核对数量、结果哈希和范围见[终验摘要](../reproducibility/results/dedup500m/final_status.json)。', '']
    # Validate all rows before replacing any public files.
    for name in NAMES:
        dest = evidence / name / 'results.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.json.tmp')
        temp.write_text(json.dumps(summaries[name], indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temp.replace(dest)
    report = root / 'docs/bbc_dedup_500m_evaluation_results_20260925.md'
    temp = report.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=ROOT / 'reproducibility/results/dedup500m')
    args = parser.parse_args()
    summaries = {name: json.loads((args.results_root / name / 'results.json').read_text()) for name in NAMES}
    render_report(summaries)


if __name__ == '__main__':
    main()
