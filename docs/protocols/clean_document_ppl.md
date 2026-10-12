# reserved-clean Document-PPL 复现协议

完整 test：5,000 篇、129,085 句、3,068,713 个计分 terminal/EOS token。
旧 checkpoint 的有效结果见[论文结果说明](../results/paper.md)，新 dedup 模型见
[补充实验](../results/bbc_dedup_500m.md)。本页保存可复用协议。

## 输入与模型

根目录为 `dataset/bbc-news-reserved-clean-v1/`。校验基础 manifest、tokenizer、
canonical 边界及 terminal 投影。Tree/TG 读取 `testppl/{tree300,tg300}`，仅评分
`valid_counts.npy` 指定的原始有效候选；填充槽不进入求和。随后按模型 grammar 转换，
在每一句内对完整、未 padding 的 token 序列再次去重，保留首次出现的候选。
**原始树唯一不保证 NoONT/Compress 转换后的序列唯一。** 同一序列只贡献一次 joint
probability，不按原始树重数加权；去重后的 K 决定句末历史提交时机。
TG 应逐 token 等于 Tree closing
符号原位重复后的转换。native 候选另行检查词边界、有效数量、唯一性与 padding。
terminal 输入为 `evaluation_terminal/`；句长/文档句数索引从 canonical 派生。

NoONT、Compress、TripleCNT 的历史配置有残留 `tgtree` grammar。构造模型及转换数据
**之前**分别绑定 `tree_noont/tree_compress/tree_triplecnt`。共享底层 tree300 并不代表
可共享 grammar。旧 grammar 错误三项永久排除。2026-10-07 进一步确认此前 grammar
修正重测仍未处理 NoONT/Compress 投影重复，其旧 PPL 已从修订稿撤下，须采用新输出
目录重新评测；不得仅改汇总文件或给旧结果重新标注去重协议。
Pause 使用 dedicated SEP；Tree-Shuffle 使用 masked checkpoint，基础评测走 terminal。

## 概率与数值

Terminal/Pause 单路径；结构模型逐句有效候选的 joint probability 作 truncated sum，
不除以 K。Tree/TG 及 standalone Pushdown 选择本句 joint NLL 最小候选提交后续历史，
详见 [model-best](document_history.md)。Pushdown 使用 native n-ary、
`stack_legal` v1；binary/v2 等不同协议不可混入。

共享 Trainer clean 重测采用 FP32/SDPA、关闭 TF32；Pushdown standalone 也是 FP32。
同一句各微批使用共同最大 padding 长度；历史只提交获选候选真实长度，超长单句报错。
BOS 仅作上下文；每篇边界重置 cache，句间继续历史。计分分母不含结构/Pause 位置。

## 运行与合并

从 `--help` 检查参数后，以明确 checkpoint、grammar、输入和新输出目录运行：

```bash
python scripts/audit_reserved_docppl_inputs.py --help
python scripts/audit_projected_docppl.py --help
python scripts/evaluate_reserved_document_ppl.py --help
python scripts/merge_reserved_docppl.py --help
python scripts/finalize_reserved_docppl.py --help
```

Pushdown 的完整调用示例见[续跑协议](../guides/document_ppl_recovery.md)。
NoONT/Compress 的 runner 必须传入 `--projection-audit`，指向完整投影审计目录；
逐句核验唯一候选数量、首次出现索引、原候选到唯一候选的映射及 token 分母。
2026-10-07 的三个模型使用 `scripts/campaigns/projected_docppl_20261007.py` 合并，
以投影后的计数校验结果，不能用只检查原始 `valid_counts.npy` 的旧合并入口。
按完整文档划分 `[start,end)` 区间；文档结果原子保存。fingerprint 绑定权重、配置、
数据、tokenizer、源码、扩展及概率协议，身份不同不能续接。仅重做中断的当前文档。

严格合并检查：恰好覆盖文档 0–4999、句子连续、逐文档 token 分母、有限 NLL、
候选范围和历史计数、所有评分 fingerprint 一致。终验再核对输入/权重/源码身份。
缺片、错误 grammar 或身份冲突不能通过；Slurm 0:0、文件存在或覆盖齐全单独都不充分。

公开 [clean JSON](../../reproducibility/evidence/old_checkpoint_clean.json)是原结果与三项 grammar 修正
结果的历史合并，保留 NLL 和权重 hash；其中 NoONT/Compress 尚不满足上述投影去重协议。
原始逐文档文件及完整运行凭据尚未随源码发布，
见[资产状态](../../reproducibility/README.md#assets)。本次整理没有重跑评测。
原 22 模型总终验回执为 `failed`，后续三个 grammar 修正模型的 scoped final receipt
为 `complete`；不能将三模型终验扩大为全部旧模型总终验成功。有效结果与验收边界见
[R-08 §7](../results/protocol_comparison.md#7-完成状态证据可信度与未解决项)。
