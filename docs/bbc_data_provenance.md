# BBC 数据版本与结果解释

新实验默认 dedup train + reserved-clean dev/test。历史论文使用的训练集与旧 test
存在大量全文重复，旧 DocPPL 不能作为独立 held-out 泛化证据。本文保留影响复现与解释的
结论；逐次排错、源行追溯和旧任务记录仅在本地保存。
新旧训练参数、部署路径、结果差异及后续修订使用规则见
[R-08 核定报告](r08_experiment_protocol_reassessment_20261002.md)。

<a id="versions"></a>
## 数据身份

| 数据版本 | 文档数 | 句数 | 计分 terminal/EOS 数 | 用途 |
|---|---:|---:|---:|---|
| 历史索引 test | 5,025 | — | — | 原 split 索引，不等于旧 DocPPL 对齐数组 |
| 历史 DocPPL 对齐 test | 4,966 | 148,836 | 3,284,061 | 冻结稿的旧 BBC DocPPL |
| reserved-clean-v1 dev | 1,000 | — | — | 新训练验证 |
| reserved-clean-v1 test | 5,000 | 129,085 | 3,068,713 | 旧模型 clean 重测及新增实验 |
| dedup train | 1,466,132 | — | 按表示分别统计 | 新 500M 训练 |

clean test 基础数组包含 3,073,713 tokens；每篇 BOS 仅作上下文，EOS 计分。
早期 test-1000 子集不能替代完整 5,000 篇。不同数据版本即使路径末尾同为 `test.npy`，
也不能混合结果；绑定 manifest、tokenizer、候选与 checkpoint 身份。

<a id="overlap"></a>
## 历史训练重叠

对实际旧 `terminal/train.npy` 全量扫描：15,187,326 篇、10,061,025,584 tokens。
旧 DocPPL test 中 4,752/4,966 篇存在完整 token 相同的训练副本（95.6907%）；
涉及 91,394 个训练文档实例和 91,631 个匹配对。按行分开 split 未消除跨分片同文。
[公开摘要](../reproducibility/evidence/historical_overlap.json)保留定义、训练哈希、计数与校验状态。
这不是对全部下游任务的污染结论，也不单独解释所有模型间差异。

<a id="raw-census"></a>
## 训练组装和预留范围

历史下载/解析池为 94 分片，训练组装入口限定
[89 个分片](../datatools/parse_pretrain_data/bbc_train_shards.txt)，拒绝五个
`CC-MAIN-2023-{06,14,23,40,50}` reserved 分片。整分片预留仍可能有同文副本，
不能直接视为 clean。新训练还要使用去重数组；普通历史 `assemble` 不自动生成 dedup train。
组装保护不会改变历史 checkpoint 的训练经历。

<a id="clean-audit"></a>
## clean 的筛选与审计范围

[构建程序与冻结规则](../datatools/reserved_clean/README.md)使用 exact、5-gram near
和 13-gram local reuse 筛选；计算表示做 NFKC/casefold/PTB/空白规范化，评分正文不改写。
近重复采用 5-gram Jaccard 或合格包含率 ≥0.8；后者要求双方至少 50 tokens、20 个共享
5-grams。共享 13-gram 位置并集覆盖 ≥30% 时隔离；此局部规则屏蔽冻结的高频模板。

最终 5,000 篇对指定历史 train 的独立复核未发现完整文本重复，也未发现所检查的
近重复（Jaccard/合格包含率 ≥0.6）。但存在共享片段：未屏蔽模板时 41 篇局部并集覆盖
达到 30%；按冻结规则屏蔽模板并合并旧 dev/test 参考后该阈值命中为 0。
不能据此声称“零文本重合”。[公开量化摘要](../reproducibility/evidence/clean_overlap.json)
保留完整/子集口径、训练 hash 和限制。

这些审计相对于固定历史 train 及其阈值。新 dedup train 与 clean dev/test 的独立完整
重叠审计未在本次整理中重做；文件传输 hash、配置生成和输入格式通过不替代该项检查。

<a id="provenance"></a>
## 历史重建的限制

旧索引 5,025 篇与 DocPPL 4,966 篇的版本差异包括组装范围、边界及正文修订。
历史字节重建需要固定配方，不能从今天重新下载/解析直接推断。91 篇的 403 个变化句
包括 345 句成分区间变化、32 句仅标签变化、26 句 unary 层数变化；差异已存在于原始
top300 文本及其分片数组。167 个旧树在候选 1–299 中存在（82 个在 candidate 1），
236 个不在导出候选中，不能归因于统一的 top-1/top-2 交换。

2026-10-02 本地复核重现上述结构计数、历史阶段对照及全部 403 句的候选命中/最低排名。
本地后处理数组有 4 句增加了重复命中槽位，不改变上述计数，不能与原始导出逐槽混同。
当次 parser 权重身份、完整配置和候选分数仍无法确认，结构变化的解析原因保留为未知。
按用户决定关闭 R-03，不再作为待修复项追查；关闭不表示原因已解决，也不改变历史
数据或结果身份。后续新实验继续使用 dedup train + reserved-clean dev/test。
历史 train 的一次重编码即使数量一致也未匹配字节 hash；不声称已精确重建旧训练数组。

<a id="paper-impact"></a>
## 对论文结论的影响

[论文结果说明](paper_results.md)并列冻结稿与旧模型 clean 重测，包含三项正确 grammar
重测值。clean 下 Tree 优于 Terminal、混合注意力在 100M PPL 中领先等观察仅适用于
对应 checkpoint、数据及评分协议；没有显著性检验，也不能把新旧差值全部归因于泄漏。
新增 dedup 500M 是另一批训练实验，不能作为旧模型重测的同一行。

<a id="reproduction"></a>
## 复现入口

- [历史数据构建](pretraining_reproduction.md)：下载、解析、89 分片组装与数据版本校验。
- [clean 筛选](../datatools/reserved_clean/README.md)、[候选生成](../datatools/parse_test_docppl_data/README.md)。
- [审计程序](../diagnostics/README.md)：实际 train exact 扫描、raw census、fuzzy 扫描及回归。
- [公开资产清单](../reproducibility/README.md#assets)：数据数组、权重与完整日志的获取状态。

公开摘要足以复核本页列出的计数和口径；重跑全量数据审计仍需要原数据资产。
