# Native DocPPL：续跑、缓存和严格合并

当前 Pushdown 历史策略为 [model-best](document_ppl_model_best_history.md)；GPST-model
自身的历史策略不由这项改动定义。数据版本选择见 [Evaluation](../Evaluation.md)。

## 可复用行为

- Pushdown 缓存被选历史的 K/V、final hidden、tokens 和 sentence IDs；完整句滑窗后重建缓存。
  `--no-kv-cache` 保留 full-prefix 正确性参考，v1/v2 normalization 各自不变。
- 候选 batch 同时受 token/action 和二次 attention 预算限制。Pushdown OOM 时从同一候选区间
  减半重试，最低为 1；不跳过候选，不复用失败批次的部分结果。
- 完整文档原子写入独立 JSON；`--resume-document-results` 仅在 run fingerprint 相同才复用。
  fingerprint 绑定 checkpoint、corpus、tokenizer、模型类型及计分设置；改变历史策略或 cache 等
  设置使用新目录。`--max-sentences` 可能截断文档，不能与可恢复文档输出合用。
- launcher 从 finalized manifest 读取文档/候选数，等待所有 worker；任何失败均不发布汇总。
- merge 拒绝非有限 NLL、无效计数、重复 ID、文件名/ID 不符、协议或候选计数不符和覆盖缺口；
  完成全部检查后原子发布 aggregate。相同文件名不是相同运行身份。

## 入口

```bash
PYTHONPATH=. python scripts/evaluate_pushdown_document_ppl.py \
  --checkpoint /path/to/pushdown-checkpoint \
  --native-data dataset/bbc-news-reserved-clean-v1/native_model_topk_300_v2/test \
  --start-document 0 --end-document 5000 \
  --attachment-normalization stack_legal \
  --document-result-dir /path/to/new-run/documents --resume-document-results
```

按设备预算分成完整文档区间。`scripts/run_native_document_ppl_shards.sh` 支持显式语料、
checkpoint、输出目录及 `DOCUMENT_BOUNDS`；旧默认路径须核对后覆盖。
GPST 使用 `scripts/gpst/evaluate_document_ppl.py` 的同名 document-result 参数。
`scripts/merge_native_document_ppl.py --expected-documents N` 才要求精确全量覆盖；旧 shard
总数在省略该选项时可汇总，不代表检查了逐文档完整性。clean campaign 使用其专用
[输入审计/终验合同](bbc_reserved_docppl_sist_20260907.md)。

回归覆盖缓存/完整前缀等价、滑窗、OOM 重试、fingerprint 拒绝和严格合并。
结果解释见[论文结果说明](paper_results.md)。
