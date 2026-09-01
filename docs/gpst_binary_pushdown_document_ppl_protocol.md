# v2 GPST Strict-Binary → Pushdown Document-PPL 评测协议

> 范围说明：标题中的 `v2` 指输入数据格式
> `native_model_topk_300_v2`。本文第 6–12 节锁定的是论文兼容的 evaluator-v1
> `stack_legal` 协议。2026-08-31 新增、与 checkpoint attachment 训练 CE 更一致的
> evaluator-v2 `sentence_causal` 协议及两种训练树表示的交叉验证，见
> `docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`；两种 normalization
> 的结果必须分开报告。

## 1. 状态与目的

本文档固定 `native_model_topk_300_v2` 中独立 GPST strict-binary 候选在
Pushdown 模型上的 document-level perplexity 评测协议。它同时规定数据转换、
历史结构、上下文截断、attachment 概率、候选边缘化、terminal 计数和运行时不变量。

当前实现入口：

- 数据转换和 evaluator：`olmo/eval/gpst_binary_pushdown_document_ppl.py`
- 运行脚本：`scripts/evaluate_gpst_binary_pushdown_document_ppl.py`
- 分片合并：`scripts/merge_gpst_binary_pushdown_document_ppl.py`
- 回归测试：`tests/test_gpst_binary_pushdown_document_ppl.py`

该协议与现有 `NativePushdownTopKCorpus` 的 nary 候选评测相互独立，不复用 v2
数据中的 Pushdown 候选轴。

经 2026-08-31 对 checkpoint 实际预计算数据反向核验，GPST 轴记录的
`fixed-per-word-right-recursive-v1` 正是预训练树表示：每个 parser preterminal 被
保留，多 BPE 单词先形成固定右递归词内子树，再接入词级 right-CNF 树。因此本协议
不是 BPE adapter mismatch；另见
`docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`。

## 2. 固定数据源与结构语义

默认数据目录：

```text
dataset/bbc-news/testppl/native_model_topk_300_v2
```

每个句子只读取以下 GPST 字段：

- `terminal_tokens`
- `content_bounds`
- `document_ids`
- `gpst_valid_counts`
- `gpst_merge_orders`

候选集合是独立 binary CKY 得到的、去重后的 unlabeled strict-binary topology。
`gpst_proposal_scores` 不参与 Pushdown 概率计算，也不作为候选权重。

对于句子 $i$，有效候选数记为 $K_i$。评测只能读取
`gpst_merge_orders[:K_i]`；物理上补齐到 300 的行不属于候选集合，不得进入模型、
求和、平均或计数。

## 3. Strict-binary merge order 转 Pushdown spans

设句子结构内容区间为 `[content_start, content_end)`，结构 token 数为
$T=content\_end-content\_start$。每个候选的 merge order 是原始叶间 gap
`0..T-2` 的一个排列。

转换维护两个活动边界表：

```text
left_by_right[r] = 当前以 r 结尾的 constituent 的左边界
right_by_left[l] = 当前以 l 开始的 constituent 的右边界
```

依次处理 merge gap `s`：

```text
l = left_by_right[s]
r = right_by_left[s + 1]
emit (l + content_start, s + content_start, r + content_start)
left_by_right[r] = l
right_by_left[l] = r
```

输出顺序就是 binary tree 的 postorder spans。对 $K_i$ 个候选进行批量转换，
时间复杂度为 $O(K_iT)$，不构造括号树，也不经过字符串解析。

必须验证：

1. 每行 merge order 是 `0..T-2` 的排列；
2. 每个候选恰有 `T-1` 个 binary spans；
3. 最后一个 span 覆盖完整内容区间；
4. span 坐标使用原始 terminal-token 坐标，而不是裁剪后的局部坐标。

## 4. Pushdown attachment gold actions

对每个 binary 候选，从 spans 的 closing counts 恢复 literal Pushdown stack
transition。对内容 token $k$，令 $c_k$ 为在该 token 结束的 binary
constituent 数量。attachment 决策前的合法集合为：

```text
[当前 token k, stack top 的 right endpoint, ..., stack bottom 的 right endpoint]
```

gold target 是上述合法集合中的第 $c_k$ 项。完成 $c_k$ 次 reduce 后，将当前
constituent 压回 stack。完整句处理结束后 stack 必须只剩一个 root。

`sentence_ids == -1` 的控制位置没有 attachment target，但仍可能是语言模型需要
预测的 terminal token。该 binary 专用推导必须与
`derive_gold_attachment_actions` 的 literal-stack 参考实现数值一致。

## 5. Document 历史与上下文截断

### 5.1 Candidate-0 历史

句子 $i$ 的所有候选共享同一个文档历史。完成该句评测后，只将候选 0 的
terminal stream 和 binary Pushdown spans 提交到历史中：

```text
history <- history + candidate[i, 0]
```

不得根据当前 checkpoint 的概率重新选择历史树，也不得为不同当前候选使用不同
的历史 prefix。文档边界处必须清空 tokens、spans、KV cache 和 final-hidden cache。

### 5.2 有界上下文

设模型上下文长度为 $C$，当前句 token 数为 $Q$。从 candidate-0 历史末尾向前
保留尽可能多的完整句子，使得：

\[
\sum_{s\in\text{retained history}} |s| + Q \le C.
\]

截断只能丢弃完整的历史句，不得从一棵树或一句 terminal stream 的中间切开。
如果单个当前句本身满足 $Q>C$，评测必须报错，不能静默裁剪当前句。

发生左截断后，保留下来的 prefix 坐标从零重新组织，并重建一次 candidate-0
KV/final-hidden cache；不能直接切片旧 KV cache，因为位置和 span 坐标已经变化。

非文档首句如果仍带有记录级 BOS，追加前移除该 BOS，并同步平移 spans、targets
和合法 attachment indices。文档首句必须以 tokenizer BOS 开始；BOS 只提供 LM
上下文，不进入 PPL 分母。

## 6. Evaluator-v1 attachment 概率

joint 指标固定使用 evaluator v1，即 `stack_legal` normalization。对 query $q$，
模型 attachment head 的未归一化分数记为 $a_{q,r}$，literal stack 给出的合法位置
集合记为 $\mathcal A_q$。gold attachment 概率为：

\[
\log p_{\mathrm{v1}}(r_q^*)
=a_{q,r_q^*}-\log\sum_{r\in\mathcal A_q}\exp a_{q,r}.
\]

这等价于先将所有 $r\notin\mathcal A_q$ 的 logits 设为 $-\infty$，再进行
softmax。实现可以只 gather 合法位置后计算 `logsumexp`，但必须与 dense
`illegal mask → softmax` 结果一致。

不得使用 `sentence_causal`/v2 normalization。v2 会在完整 sentence-causal row
上归一化后再丢弃非法 action，与本协议的条件概率不同。

## 7. 两个正式指标

### 7.1 Joint document-level perplexity

对文档 $d$ 的句子 $i$ 和有效候选 $k<K_{d,i}$，定义：

\[
N^{\mathrm{joint}}_{d,i,k}
=N^{\mathrm{tok}}_{d,i,k}+N^{\mathrm{att,v1}}_{d,i,k}.
\]

固定 candidate-0 历史条件下，句子截断候选质量为：

\[
\ell^{\mathrm{joint}}_{d,i}
=\log\sum_{k=0}^{K_{d,i}-1}\exp(-N^{\mathrm{joint}}_{d,i,k}).
\]

语料 log likelihood 与 perplexity 为：

\[
\mathcal L_{\mathrm{joint}}
=\sum_d\sum_i\ell^{\mathrm{joint}}_{d,i},
\qquad
\mathrm{PPL}_{\mathrm{joint,v1}}
=\exp\left(-\frac{\mathcal L_{\mathrm{joint}}}{M}\right).
\]

这里是候选 joint probability 的截断 **sum**。禁止减去 `log(K_i)`，禁止除以
$K_i$，禁止加入 proposal-score 权重，禁止让 padding rows 贡献概率质量。

正式输出字段：

```text
joint_log_likelihood_v1
joint_document_perplexity_v1
```

### 7.2 Candidate-0 structured terminal perplexity

第二个指标只累计每组树 candidate 0 的 terminal-token 概率，完全排除 attachment
概率：

\[
\mathcal L_{\mathrm{tok0}}
=-\sum_d\sum_i N^{\mathrm{tok}}_{d,i,0},
\qquad
\mathrm{PPL}_{\mathrm{tok0}}
=\exp\left(-\frac{\mathcal L_{\mathrm{tok0}}}{M}\right).
\]

不得对所有候选的 token NLL 做 `logsumexp`。该指标仍然经过 candidate-0 binary
Pushdown depth bias，因此应称为 **candidate-0 structured terminal PPL**，不能解释成
flat/terminal-only baseline。

正式输出字段：

```text
candidate0_terminal_log_likelihood
candidate0_structured_terminal_perplexity
```

### 7.3 统一分母

$M$ 是实际被语言模型预测的 terminal token 总数：

- 文档 BOS 不计入；
- 普通 terminal、EOS 和其他有效 LM terminal 计入；
- attachment 是否存在不改变 terminal 分母；
- 两个指标使用完全相同的 $M$。

## 8. Ragged candidate count 不变量

每个句子的计算量和概率求和范围必须由 `gpst_valid_count` 决定。即使物理文件为
每句保留 300 个 slots，也只能执行：

```text
orders = gpst_merge_orders[:gpst_valid_count]
sum/logsumexp over exactly gpst_valid_count candidates
```

输出同时报告：

- `valid_candidate_count`：真正求和和 forward 的候选总数；
- `candidate_slots`：物理候选容量，即 `sentence_count × 300`；
- `model_candidate_forwards`：实际模型候选 forward 数，应等于
  `valid_candidate_count`。

BBC test-PPL 完整 v2 数据当前已核实的语料不变量为：

```text
document_count          = 4,966
sentence_count          = 148,836
terminal_count          = 3,284,061
valid_candidate_count   = 37,227,054
candidate_slots         = 44,650,800
invalid_padding_slots   = 7,423,746
```

有效候选数分布：

```text
K=1:     9,517 sentences
K=2:     2,740
K=5:     3,575
K=14:    3,916
K=42:    3,673
K=132:   3,806
K=300: 121,609
```

正式全量结果必须与上述计数一致；分片结果合并后也必须恢复相同计数。

## 9. 等价的效率优化

所有优化必须保持第 3–8 节的概率语义不变。

1. **批量 merge-order 转换**：按句子同时转换 $K_i$ 行，复杂度 $O(K_iT)$。
2. **binary closing-count attachment 推导**：直接从 strict-binary spans 的 closing
   counts 得到 literal stack actions，避免逐候选构造 Torch closure 字典。
3. **Candidate-0 KV cache**：prefix Transformer K/V 和 pre-`ln_f` final hidden 只存
   candidate 0；当前候选共享并扩展该 cache。
4. **截断后单次重建**：上下文滑动时以 batch size 1 重建保留的 candidate-0
   suffix，随后所有当前候选共享它。
5. **稀疏 v1 attachment**：只计算 stack-legal target logits，不构造 dense legal
   boolean mask；结果必须与 dense mask→softmax 等价。
6. **`Q×N` depth tape**：KV-cache forward 只为当前句的 $Q$ 个 query 构造对完整
   $N$-token key prefix 的 depth rows，避免 `N×N` depth matrix。
7. **动态 microbatch**：batch size 同时受候选上限、model-input token budget 和
   `Q×N` attention-cell budget 限制。
8. **OOM 精确重试**：OOM 时将同一候选区间的 batch size 减半；不能跳过候选或
   继续使用部分结果。
9. **CPU 预取**：后台只预转换后续句子；必须保持文档顺序、candidate 顺序和
   ragged $K_i$ 不变。
10. **单次双指标 forward**：candidate 0 必须位于首个 microbatch；joint 和
    terminal 指标从同一组 token NLL 中取得，不额外执行 terminal-only forward。
11. **多 GPU 文档分片**：只允许按完整文档划分 `[start_document,end_document)`；
    不能从文档中间开始，否则 candidate-0 prefix 不完整。

## 10. 输出协议

正式 aggregate JSON 至少包含：

```text
protocol_version
structure_source = v2_gpst_strict_binary_to_pushdown
prefix_policy = candidate0
context_truncation = left_drop_complete_sentences
attachment_normalization = stack_legal
candidate_aggregation = valid_unique_truncated_joint_sum
divide_by_candidate_count = false
ppl_denominator = terminal_count

joint_log_likelihood_v1
joint_document_perplexity_v1
candidate0_terminal_log_likelihood
candidate0_structured_terminal_perplexity

terminal_count
sentence_count
document_count
valid_candidate_count
candidate_slots
model_candidate_forwards
kv_cache_hits
kv_cache_rebuilds
oom_retries
max_sequence_length
max_candidates_per_sentence

checkpoint_model_sha256
native_manifest_sha256
tokenizer_sha256
```

不同分片只有在协议字段完全一致、文档区间不重叠且连续时才能合并。
合并还必须拒绝 checkpoint、native manifest 或 tokenizer 哈希不同的分片，并验证
`model_candidate_forwards == valid_candidate_count` 以及
`candidate_slots == sentence_count × max_candidates_per_sentence`。
每个 microbatch 的 token、attachment、joint NLL 和每个分片的两个 log likelihood
都必须有限；发现非有限值时评测必须携带文档、句子和候选区间立即失败，merge 也必须
拒绝该分片。JSON 使用严格模式，禁止把 `NaN`/`Infinity` 当作成功结果序列化。
`--max-sentences` 只用于 smoke test；它可能产生部分文档结果，禁止作为正式分片
进入 aggregate merge。

## 11. 正确性验收

实现至少需要通过以下检查：

1. merge order 转 spans 与已物化 binary tree 的 postorder spans 一致；
2. binary closing-count targets/legal sets 与 literal-stack 参考实现一致；
3. 稀疏合法位置 v1 NLL 与 dense illegal-mask→softmax NLL 一致；
4. `K_i<300` poison-padding 测试证明 padding rows 不影响结果；
5. candidate-0 token 指标严格等于 `sum(token_nll[0])`；
6. KV-cache scoring 与 full-prefix teacher forcing 的 token、attachment 和 joint NLL
   一致；
7. 截断只保留完整历史句，单句超过上下文时明确报错；
8. `Q×N` cached depth rows 与完整 `N×N` depth matrix 的对应后缀行一致；
9. microbatch 大小变化不改变 log likelihood；
10. 同步读取与 CPU 预取的 document/candidate 顺序完全一致；
11. 分片合并拒绝重叠、缺口、协议不一致和部分文档 smoke 结果；
12. 两个 PPL 有限，joint NLL 恒等于 token NLL 加 v1 attachment NLL；
13. 完整 BBC 运行恢复第 8 节的全部语料计数。

## 12. 运行方法

单分片或完整运行：

```bash
PYTHONPATH=. python scripts/evaluate_gpst_binary_pushdown_document_ppl.py \
  --checkpoint <pushdown-checkpoint> \
  --native-data dataset/bbc-news/testppl/native_model_topk_300_v2 \
  --max-sequence-length 2048 \
  --output results/gpst_binary_pushdown_docppl.json
```

完整文档分片示例：

```bash
PYTHONPATH=. python scripts/evaluate_gpst_binary_pushdown_document_ppl.py \
  --checkpoint <pushdown-checkpoint> \
  --start-document 0 \
  --end-document 1242 \
  --output results/shard_00.json
```

合并连续、互不重叠的分片：

```bash
PYTHONPATH=. python scripts/merge_gpst_binary_pushdown_document_ppl.py \
  results/shard_00.json results/shard_01.json \
  results/shard_02.json results/shard_03.json \
  --require-full-bbc --output results/aggregate.json
```

使用 `--disable-kv-cache` 可运行 full-prefix correctness reference，但不适合作为
全量默认配置。使用 `--skip-merge-order-validation` 只能省略已完成数据的重复排列
检查，不能改变 spans、候选集合或概率计算。

## 13. 解释边界

- joint 指标是固定 candidate-0 历史条件下，对当前句外部 top-K binary support 的
  截断概率和；它不是所有可能 document parse 的精确边缘概率。
- candidate-0 历史是一项固定评测近似，不等同于每一步对历史树进行模型后验
  边缘化。
- candidate-0 structured terminal PPL 排除了 attachment 概率，但仍由指定的
  binary Pushdown 结构影响 hidden states，因此不能与 flat LM PPL 混为一谈。
- 由于候选 support 随句长和 CKY 可用 topology 数变化，`K_i<300` 是协议允许的
  ragged support，不是需要补齐或归一化修正的缺失数据。

## 14. 完整结果与后续 v2 扩展

同一 checkpoint、数据、candidate-0 history、2,048-token 完整句截断和 terminal
分母下，direct GPST strict-binary 完整结果为：

```text
evaluator-v1 stack_legal:
  joint LL  = -9,124,537.0987
  joint PPL = 16.09375107

evaluator-v2 sentence_causal:
  joint LL  = -9,157,799.3091
  joint PPL = 16.25758293
```

两者均覆盖 4,966 documents、148,836 sentences、3,284,061 terminals 和
37,227,054 candidates。v1 是本文正文及论文口径比较使用的条件概率；v2 是训练
attachment head 完整句内因果 softmax 下的 top-K 有效树质量。v2 不覆盖 v1，也不
替代论文 Table-4 的 historical native-nary-v1 `13.293598`。
