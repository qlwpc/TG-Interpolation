# 文档索引

| 阅读目的 | 入口 |
|---|---|
| 运行一个复现实验 | [预训练工作流](pretraining_workflow.md) |
| 理解论文结果、旧模型重测与适用边界 | [论文结果说明](paper_results.md) |
| 区分新旧实验、准备后续论文数据修订 | [R-08 标准、路径与结果差异核定（2026-10-02）](r08_experiment_protocol_reassessment_20261002.md) |
| 新增 dedup 500M 实验 | [结果及完成范围](bbc_dedup_500m_evaluation_results_20260925.md)、[配置生成器](bbc_dedup_500m_pretraining.md) |
| 数据、权重与证据获取 | [公开材料与资产状态](../reproducibility/README.md)、[数据版本与污染说明](bbc_data_provenance.md) |
| 按模型和任务评测 | [Evaluation](../Evaluation.md) |

## 复现时按需阅读

- 数据构建与历史配置：[预训练细则](pretraining_reproduction.md)、[clean 数据筛选](../datatools/reserved_clean/README.md)、[候选生成](../datatools/parse_test_docppl_data/README.md)。
- 模型实现：[Pause](pause_protocol.md)、[TG 输入与内核](tg_input_pipeline.md)、[GPST](gpst_implementation.md)。
- 文档 PPL：[clean 评测](bbc_reserved_docppl_sist_20260907.md)、[model-best 历史](document_ppl_model_best_history.md)、[续跑与严格合并](native_document_ppl_recovery.md)。
- 候选与概率：[native top-K](native_model_topk_300_v2_format.md)、[binary 存储](native_binary_storage.md)、[n-ary 兼容格式](native_nary_300_format.md)、[Pushdown 概率](pushdown_word_atom_strict_binary_document_ppl_protocol.md)、[GPST binary 对照](gpst_binary_pushdown_document_ppl_protocol.md)、[Pushdown 实现比较](pushdown_vs_original_repo.md)。
- 分布式限制：[FSDP 下游评测](FSDP_DOWNSTREAM_EVAL_RISKS.md)。

结果表从冻结稿或原实验凭据提取，新增结果页由收集器生成。修改结果需同时更新对应证据，
不能把历史结果改贴为当前协议。日常协作登记、排队状态与修复过程仅在本地保存。
