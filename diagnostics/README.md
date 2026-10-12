# 数据诊断入口

当前 BBC 溯源与污染审计使用 `audit_bbc_*`、`diagnose_bbc_split_overlap.py`、
`find_bbc_terminal_sources.py`、`read_bbc_structure_sources_remote.py`、`verify_bbc_*`、
`audit_reserved_fuzzy.*` 及对应 `summarize_*`。固定输入、口径和证据见
[BBC 数据溯源](../docs/protocols/bbc_data.md)。这些是审计程序，不是新数据生产入口。

[data/bbc_historical_shard_counts.json](data/bbc_historical_shard_counts.json) 保留旧 parsed 94 分片计数及来源哈希。
raw 普查与 split 原因审计直接读该 JSON；不依赖历史采样脚本的执行或源码解析。

新候选生成/独立检查见 [DocPPL 数据入口](../datatools/parse_test_docppl_data/README.md)。
旧修补脚本和逐次诊断只在本地保留。公开计数证据见[数据版本说明](../docs/protocols/bbc_data.md)。

`audit_dataset_versions.py` 对 BBC 小型 dev/test 流做全量结构检查，对 top-300 索引求和、
候选正文和预计算缓存做确定性抽样；只读数据，JSON 写到 stdout：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python diagnostics/audit_dataset_versions.py \
  --root dataset --host 3090B > /tmp/dataset-legality.json
```

需要 NumPy；不调用 parser/GPU，不扫描整个训练集。抽样通过不证明整个大文件合法，
也不替代完整 SHA 或污染审计。具体版本判断记录在本地
`reports/dataset_migration_20261002.md`（未随公开源码发布）。

## 模型与性能工具

除顶层 BBC 数据审计外，工具按检查对象组织。命令从仓库根目录运行，
使用 README 指定的运行环境及 `PYTHONPATH=.`。

| 目录 | 用途 |
|---|---|
| `models/` | checkpoint 完整性、XSum/Pause 生成、Tree-Shuffle terminality、Pushdown/GPST scorer 和协议等价性 |
| `candidates/` | GPST BPE、n-ary 与 word-atom binary 候选审计 |
| `training/` | 训练数据 inspection 与 dataloader 检查 |
| `performance/` | Pushdown/TreeReg profile 分析、depth 微基准、DocPPL profiler 与 stress runner |
| `data/` | 固定历史分片计数；只放审计输入 |

例如：

```bash
PYTHONPATH=. python diagnostics/models/check_checkpoint_health.py --help
PYTHONPATH=. python diagnostics/candidates/audit_gpst_training_bpe_candidates.py --help
```

性能与 stress 工具可能需要 GPU，部分旧工具包含专用输入或默认路径；先阅读参数和源码。
GPU benchmark 不包含在轻量源码检查中。正式运行前的 clean 输入审计
仍由 `scripts/audit_reserved_docppl_inputs.py` 提供。
