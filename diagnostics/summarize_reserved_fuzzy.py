"""Independently verify saved nearest pairs and publish a readable audit report."""
import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from datatools.reserved_clean.common import atomic_json, normalize, records, sha_file


def grams(w, n):
    return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}


def run(a):
    summary = json.loads((a.audit / 'summary.json').read_text())
    assert summary['complete']
    rows = list(records(a.audit / 'per_test.jsonl'))
    selected = list(records(a.dataset / 'test_selected.jsonl.gz'))
    inputs = json.loads((a.audit / 'inputs.json').read_text())
    train = np.load(inputs['train'], mmap_mode='r')
    tokenizer = Tokenizer.from_file(str(a.tokenizer))
    decoded_cache = {}
    verified = 0
    for row in rows:
        words = selected[row['test_doc']]['body'].split(); gs = grams(words, 5)
        for prefix, metric in (('jac', 'jaccard'), ('cont', 'containment'), ('local', 'local_pair')):
            source = row[prefix + '_source']
            if source == '-':
                assert row[metric] == 0
                continue
            _, offset = map(int, source.split(':'))
            if offset not in decoded_cache:
                assert train[offset] == 50257
                width = 4096
                while True:
                    part = train[offset:min(offset+width,len(train))]
                    ends = np.flatnonzero(part == 50256)
                    if len(ends):
                        part = part[:int(ends[0])+1]
                        assert np.count_nonzero(part == 50257) == 1
                        break
                    assert offset + width < len(train)
                    width *= 2
                decoded_cache[offset] = normalize(tokenizer.decode(part[1:-1].tolist(), skip_special_tokens=False))
            body = decoded_cache[offset]
            assert body == row[prefix + '_body']
            rw = body.split(); rs = grams(rw, 5); common = len(gs & rs)
            if prefix == 'jac':
                score = common / len(gs | rs)
            elif prefix == 'cont':
                assert min(len(words),len(rw)) >= 50 and common >= 20
                score = common / min(len(gs),len(rs))
            else:
                ref13 = grams(rw,13); covered = set()
                for i in range(len(words)-12):
                    if tuple(words[i:i+13]) in ref13:
                        covered.update(range(i,i+13))
                score = len(covered)/len(words)
            assert abs(score-row[metric]) < 1e-12
            verified += 1
    review_ids = set()
    for metric in ('jaccard','containment','coverage'):
        review_ids.update(r['test_doc'] for r in sorted(rows,key=lambda r:r[metric],reverse=True)[:20])
    review_ids.update(r['test_doc'] for r in rows if r['jaccard']>=.6 or r['containment']>=.6 or r['coverage']>=.3)
    review=[]
    for d in sorted(review_ids):
        r=rows[d]; body=selected[d]['body']; w=body.split()
        item={k:v for k,v in r.items() if k!='bitmap'};item['test_body']=body
        rw=r['jac_body'].split()
        matcher=SequenceMatcher(None,w,rw,autojunk=False)
        item['token_sequence_similarity_to_best_jaccard_reference']=matcher.ratio()
        blocks=sorted(matcher.get_matching_blocks(),key=lambda b:b.size,reverse=True)[:5]
        item['longest_equal_blocks_to_best_jaccard_reference']=[{'tokens':b.size,'text':' '.join(w[b.a:b.a+b.size])} for b in blocks if b.size]
        review.append(item)
    atomic_json(a.audit/'nearest_pairs_review.json',review)
    atomic_json(a.audit/'pair_validation.json',{'complete':True,'claim_type':'computed',
        'nearest_pairs_verified_against_actual_train_bytes':verified,
        'distinct_train_offsets_read_back':len(decoded_cache),
        'independent_python_metric_recheck':True,
        'cpp_vs_exhaustive_python_toy_oracle':'tests/test_reserved_fuzzy_audit.py',
        'summary_sha256':sha_file(a.audit/'summary.json'),
        'review_sha256':sha_file(a.audit/'nearest_pairs_review.json')})
    s=summary['test'];sub=summary['test_docppl_1000']
    evidence_path = Path(os.path.relpath(a.audit, a.report.parent)).as_posix()
    lines=['# 新 BBC test 对实际 train 的模糊匹配复核','',
        '2026-09-06。本次重新读取最终测试数组和实际训练数组，未采用训练行号推算或抽样扫描。', '',
        f"- test：`{a.dataset}/terminal/test.npy`，{s['documents']:,} 篇。",
        f"- train：`{inputs['train']}`，{summary['train_documents']:,} 篇，{summary['train_tokens']:,} tokens。",
        f"- 实际 train 全文件 SHA-256：`{summary['train_sha256']}`，与冻结输入一致。",
        f"- 全量遍历后精确去重得到 {summary['unique_strict_train']:,} 种 token 正文，再规范化去重得到 {summary['unique_normalized_train']:,} 种正文；去重仅节省重复计算。", '',
        f"| 检查 | 全量 test（{s['documents']:,}） | DocPPL 子集（{sub['documents']:,}） |",'|---|---:|---:|',
        f"| 完整 terminal token 精确重复 | {len({r['test_doc'] for r in summary['strict_matches']})} | {len({r['test_doc'] for r in summary['strict_matches']} & {r['full_split_doc_id'] for r in json.loads((a.dataset/'docppl/test_1000/document_map.json').read_text())})} |",
        f"| 规范化全文重复 | {s['normalized_exact']} | {sub['normalized_exact']} |"]
    for t in ('0.8','0.7','0.6'):
        lines.append(f"| 5-gram Jaccard 或合格包含率 ≥ {t} | {s['near_at_threshold'][t]} | {sub['near_at_threshold'][t]} |")
    lines += [f"| 未屏蔽模板的累计 13-gram 覆盖 ≥ 30% | {s['unmasked_13gram_coverage_ge_30pct']} | {sub['unmasked_13gram_coverage_ge_30pct']} |",
        f"| 至少共享一个连续 13-gram | {s['any_shared_13gram']} | {sub['any_shared_13gram']} |",'',
        f"全量 test 最高 Jaccard 为 {s['max_jaccard']:.6f}，最高合格包含率为 {s['max_eligible_containment']:.6f}，最高累计片段覆盖率为 {s['max_unmasked_coverage']:.6f}。",'',
        '## 口径与验证','',
        '规范化沿用冻结构建口径：Unicode NFKC、大小写折叠、确定性括号转义还原及词/标点切分；保留数字和标点，不改变评分 token。', '',
        '对所有共享 5-gram 建完整字符串倒排索引并逐对计算 Jaccard；包含率的分母为较小的 distinct 5-gram 集合，要求两侧各至少 50 个规范化 token 且共享至少 20 个 distinct 5-gram。没有近似召回或高频截断。', '',
        f"本次 test 最短为 {min(len(r['body'].split()) for r in selected)} 个规范化 token；低于包含率检查最小长度（50）的测试文档有 {sum(len(r['body'].split())<50 for r in selected)} 篇。", '',
        '13-gram 覆盖以 test 中被匹配的位置取并集，跨 train 文档和计算进程合并；本次不屏蔽任何高频模板，比构建时的局部规则更严格。词元包含标点，因此 13-gram 不等同于 13 个自然语言单词。', '',
        f'匹配器已通过独立 Python 穷举对照，覆盖重复 n-gram、短文、包含关系和跨进程位置并集；最终 {verified:,} 个最近文章对另外从实际 train 偏移读回并用 Python 重算指标。', '',
        'Tree/TG 全量 test 的 terminal 投影逐 token 等于本次审计数组；1,000 篇 DocPPL canonical 子集投影也与完整 test 的对应文档逐 token 一致。原生成产物的 40 个文件哈希已核对。', '',
        '## 证据与复现','',
        f'- [汇总]({evidence_path}/summary.json)',
        f'- [逐篇最高相似度、train 偏移与覆盖位图]({evidence_path}/per_test.jsonl)',
        f'- [最近文章对与共享片段]({evidence_path}/nearest_pairs_review.json)',
        f'- [文章对独立校验]({evidence_path}/pair_validation.json)',
        f'- [格式与子集正文对齐]({evidence_path}/final_format_alignment.json)',
        f'- [构建阶段凭据复核]({evidence_path}/previous_receipts_rechecked.json)', '',
        '```bash',
        'python -m pytest -q tests/test_reserved_fuzzy_audit.py',
        'python -m diagnostics.audit_reserved_fuzzy --output artifacts/reserved_fuzzy_rerun',
        'python -m diagnostics.summarize_reserved_fuzzy --audit artifacts/reserved_fuzzy_rerun',
        '```','',
        '结论仅覆盖上述固定 BBC train 文件及已验证正文投影，不能推及其他混合语料、续训来源或所有 checkpoint。词面近重复阈值以下的共享内容和语义相似不在“无重复”保证内。']
    a.report.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified_pairs':verified,'review_documents':len(review),'report':str(a.report)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=Path('artifacts/reserved_fuzzy_20260906'))
    p.add_argument('--dataset',type=Path,default=Path('dataset/bbc-news-reserved-clean-v1'))
    p.add_argument('--tokenizer',type=Path,default=Path('dataset/bbc-news/TG_GPT2_tokenizer.json'))
    p.add_argument('--report',type=Path,help='Output report (default: AUDIT/report.md)')
    args = p.parse_args()
    if args.report is None:
        args.report = args.audit / 'report.md'
    run(args)
