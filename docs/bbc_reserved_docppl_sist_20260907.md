# reserved-clean Document-PPL 复现协议

完整 test：5,000 篇、129,085 句、3,068,713 个计分 terminal/EOS token。
旧 checkpoint 的有效结果见[论文结果说明](paper_results.md)，新 dedup 模型见
[补充实验](bbc_dedup_500m_evaluation_results_20260925.md)。本页保存可复用协议。

## 输入与模型

根目录为 `dataset/bbc-news-reserved-clean-v1/`。校验基础 manifest、tokenizer、
canonical 边界及 terminal 投影。Tree/TG 读取 `testppl/{tree300,tg300}`，仅评分
`valid_counts.npy` 指定的唯一候选；填充槽不进入求和。TG 应逐 token 等于 Tree closing
符号原位重复后的转换。native 候选另行检查词边界、有效数量、唯一性与 padding。
terminal 输入为 `evaluation_terminal/`；句长/文档句数索引从 canonical 派生。

NoONT、Compress、TripleCNT 的历史配置有残留 `tgtree` grammar。构造模型及转换数据
**之前**分别绑定 `tree_noont/tree_compress/tree_triplecnt`。共享底层 tree300 并不代表
可共享 grammar。旧错误三项永久排除；公开汇总采用三项修正重测。
Pause 使用 dedicated SEP；Tree-Shuffle 使用 masked checkpoint，基础评测走 terminal。

## 概率与数值

Terminal/Pause 单路径；结构模型逐句有效候选的 joint probability 作 truncated sum，
不除以 K。Tree/TG 及 standalone Pushdown 选择本句 joint NLL 最小候选提交后续历史，
详见 [model-best](document_ppl_model_best_history.md)。Pushdown 使用 native n-ary、
`stack_legal` v1；binary/v2 等不同协议不可混入。

共享 Trainer clean 重测采用 FP32/SDPA、关闭 TF32；Pushdown standalone 也是 FP32。
同一句各微批使用共同最大 padding 长度；历史只提交获选候选真实长度，超长单句报错。
BOS 仅作上下文；每篇边界重置 cache，句间继续历史。计分分母不含结构/Pause 位置。

## 运行与合并

从 `--help` 检查参数后，以明确 checkpoint、grammar、输入和新输出目录运行：

```bash
python scripts/audit_reserved_docppl_inputs.py --help
python scripts/evaluate_reserved_document_ppl.py --help
python scripts/merge_reserved_docppl.py --help
python scripts/finalize_reserved_docppl.py --help
```

Pushdown 的完整调用示例见[续跑协议](native_document_ppl_recovery.md)。
按完整文档划分 `[start,end)` 区间；文档结果原子保存。fingerprint 绑定权重、配置、
数据、tokenizer、源码、扩展及概率协议，身份不同不能续接。仅重做中断的当前文档。

严格合并检查：恰好覆盖文档 0–4999、句子连续、逐文档 token 分母、有限 NLL、
候选范围和历史计数、所有评分 fingerprint 一致。终验再核对输入/权重/源码身份。
缺片、错误 grammar 或身份冲突不能通过；Slurm 0:0、文件存在或覆盖齐全单独都不充分。

公开 [clean JSON](../reproducibility/evidence/old_checkpoint_clean.json)是原结果与三项修正
结果的有效合并，保留 NLL 和权重 hash。原始逐文档文件及完整运行凭据尚未随源码发布，
见[资产状态](../reproducibility/README.md#assets)。本次整理没有重跑评测。
