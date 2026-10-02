# GPST strict-binary 候选的 Pushdown DocPPL 合同

这里的输入 v2 指 `native_model_topk_300_v2` 存储格式，与 attachment normalization 的
v1/v2 是两个版本轴。当前历史选择为 **model-best**，见 [历史策略](document_ppl_model_best_history.md)。
旧 candidate-0 测量集中在 [word-atom 协议](pushdown_word_atom_strict_binary_document_ppl_protocol.md)。

## 输入和结构

实现：`olmo/eval/gpst_binary_pushdown_document_ppl.py`；运行/合并：
`scripts/evaluate_gpst_binary_pushdown_document_ppl.py`、`scripts/merge_gpst_binary_pushdown_document_ppl.py`。

- GPST 轴读取 `terminal_tokens`、`content_bounds`、`document_ids`、`gpst_valid_counts`、
  `gpst_merge_orders`；不读取 Pushdown 轴，不把 proposal score 当模型概率权重。
- 只消费前 K 个有效候选；300 是物理容量，padding 不进入 forward、求和或计数。
- T 个内容 token 的 merge order 必须是 gap `0..T-2` 的排列。维护活动区间左右边界，
  每次合并 gap 两侧 constituent，输出原 terminal 坐标的 `(left, split, right)`。
  每棵树恰有 T−1 个 spans，最后覆盖全部内容；转换复杂度 O(KT)。
- 用 closing counts 恢复 literal stack actions。合法位置依次是当前 token、stack top 至 bottom
  的 right endpoint；gold target 由本 token 的 reduce 次数确定。句末只剩一个 root。
  `sentence_ids=-1` 的控制位置没有 attachment target，但仍可能需要 LM 计分。
- multi-BPE word 保持固定右递归词内子树，与该 checkpoint 训练表示一致；不得改为 BPE-spliced。

## 计分、历史与计数

对同一句所有有效候选使用同一历史，计算 `joint_nll = token_nll + attachment_nll`，
`sentence_ll = logsumexp(-joint_nll)`，不除以 K、不乘 proposal 权重。

| 概率轴 | 定义 |
|---|---|
| v1 `stack_legal` | attachment softmax 只在当前 stack 合法位置归一化 |
| v2 `sentence_causal` | 在完整句内因果位置归一化；与 attachment 训练 CE 的 support 一致 |

固定路径下 v2 NLL ≥ v1，但不能据此比较不同搜索 support。选本句 joint NLL 最小的候选
提交历史，平局取最早者；文档边界清空状态。上下文仅丢弃完整历史句，滑窗后重建 KV/hidden
与坐标；当前单句超过模型长度时报错。非首句记录级 BOS 去除时同步平移结构坐标。

`joint_document_perplexity_v1` 等历史命名字段必须连同 normalization 元数据解释。
`candidate0_structured_terminal_perplexity` 只累计**当前句 candidate 0** 的 token NLL，
其条件历史仍为 model-best；它受树结构影响，不是 flat terminal baseline。
两指标分母相同：BOS 不计，ordinary terminals/EOS 计分。

合并必须验证：协议/历史策略/数据/tokenizer/checkpoint 哈希相同；整文档范围连续无重复无缺口；
NLL 有限；`model_candidate_forwards == valid_candidate_count`；
`candidate_slots == sentence_count × max_candidates_per_sentence`。`--max-sentences` 只用于 smoke，
不将部分文档结果合并成全量。候选 batch/OOM 重试不得跳过或重复计数。

## 运行边界

从仓库根目录，显式设置 checkpoint、语料和新输出路径。例如 clean 语料：

```bash
python scripts/evaluate_gpst_binary_pushdown_document_ppl.py \
  --checkpoint /path/to/pushdown-checkpoint \
  --native-data dataset/bbc-news-reserved-clean-v1/native_model_topk_300_v2/test \
  --candidate-source gpst-strict-binary \
  --attachment-normalization stack_legal \
  --max-sequence-length 2048 --output /path/to/new-result.json
```

脚本的旧 BBC 默认路径不代表当前 clean 入口；分片按完整文档的 `[start,end)` 划分。
`--disable-kv-cache` 是 full-prefix 参考路径。合并器的 `--require-full-bbc` **专指旧 4,966 篇**，
不能拿它验证 clean 5,000 篇。clean 的数据、汇总与完成门禁见
[运行记录](bbc_reserved_docppl_sist_20260907.md)。

旧 GPST 轴完整不变量为 4,966 文档、148,836 句、3,284,061 计分 tokens、37,227,054 个有效候选、
44,650,800 物理槽位。K=1/2/5/14/42/132/300 的句数分别为
9,517 / 2,740 / 3,575 / 3,916 / 3,673 / 3,806 / 121,609；不套用到其他语料。

回归入口 `tests/test_gpst_binary_pushdown_document_ppl.py` 覆盖转换/literal-stack 对照、
v1 稀疏/dense 等价、poison padding、双指标、KV/full-prefix、滑窗、microbatch、预取顺序及严格合并。
这些门禁不证明已完成新的全量运行；所有结果仍按实际 run 身份登记。
