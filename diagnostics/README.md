# 数据诊断入口

当前 BBC 溯源与污染审计使用 `audit_bbc_*`、`diagnose_bbc_split_overlap.py`、
`find_bbc_terminal_sources.py`、`read_bbc_structure_sources_remote.py`、`verify_bbc_*`、
`audit_reserved_fuzzy.*` 及对应 `summarize_*`。固定输入、口径和证据见
[BBC 数据溯源](../docs/bbc_data_provenance.md)。这些是审计程序，不是新数据生产入口。

[data/bbc_historical_shard_counts.json](data/bbc_historical_shard_counts.json) 保留旧 parsed 94 分片计数及来源哈希。
raw 普查与 split 原因审计直接读该 JSON；不依赖历史采样脚本的执行或源码解析。

新候选生成/独立检查见 [DocPPL 数据入口](../datatools/parse_test_docppl_data/README.md)。
旧修补脚本和逐次诊断只在本地保留。公开计数证据见[数据版本说明](../docs/bbc_data_provenance.md)。
