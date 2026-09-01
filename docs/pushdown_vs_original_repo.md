# Pushdown document PPL：本仓库协议与原仓库的稳定对照

> [!NOTE]
> 本文只说明原仓库与本仓库评测语义的差异，不作为结果总表。2026-08-31 的 checkpoint
> 预处理审计确认训练表示保留 fixed right-recursive multi-BPE word atom；当前完整
> v1/v2 数值和解释以
> [`pushdown_word_atom_strict_binary_document_ppl_protocol.md`](pushdown_word_atom_strict_binary_document_ppl_protocol.md)
> 为准。v1 与 v2 是两种定义明确但不可混合的概率协议。

## 1. 结论与适用范围

本仓库的 Pushdown document PPL 是在本地 OLMo Pushdown checkpoint 上定义的
**文档级、外部 top-K 结构截断边缘化指标**，不是原作者
`MurtyShikhar/Pushdown-Layers` 的评测脚本复现。

本文对照的原仓库版本固定为
[`MurtyShikhar/Pushdown-Layers@6a4c543`](https://github.com/MurtyShikhar/Pushdown-Layers/tree/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30)，
本仓库实现过 v1 `stack_legal` 与 v2 `sentence_causal` 两种概率协议；对应实现提交为
[`47564d9`](https://github.com/qlwpc/TG-Interpolation/commit/47564d9bb91dffab3f45b8c3b7012ce1e77f6c4b)。
训练表示对齐的当前评测入口是：

```text
checkpoint: saved_models/pushdown_terminalonly/step34354-unsharded
candidate data: direct strict-binary GPST, or native n-ary converted to word-atom right-CNF
entry point: scripts/evaluate_gpst_binary_pushdown_document_ppl.py
structure_source: fixed-word-atom strict-binary external top-K
attachment_normalization: stack_legal (v1) or sentence_causal (v2)
prefix_policy: candidate0
```

因此，结果应命名为“本地 Pushdown fixed-word-atom strict-binary top-K
truncated-marginal document PPL（并标明 v1/v2）”，不能直接称为“原仓库 Pushdown
PPL”或“原仓库 gold-300 复现”。

## 2. 原仓库实际存在的三条概率路径

讨论 attachment 概率时，必须把原仓库的训练、给定树评测和 beam 解码分开。

### 2.1 训练

原仓库直接将 attachment head 输出交给交叉熵，只屏蔽 padding／无目标 query；
没有在 CE 前按当前 stack 的合法 action 集合重新归一化。也就是说，训练目标是
当前 causal support 上的 gold attachment 概率。

证据：原仓库
[`interfaces/lm_interface.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/interfaces/lm_interface.py#L120-L162)。

### 2.2 给定 gold tree 的 teacher-forced 评测

原仓库的 `gold` 模式对每个 query 先在 causal attachment logits 上执行
`log_softmax`，再读取 gold target。它同样没有先移除已经归约的位置，也没有进行
stack-legal 条件化。

证据：原仓库
[`pushdown_util.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/pushdown_util.py#L16-L34)
和
[`eval_utils/eval_pushdown_model.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/eval_utils/eval_pushdown_model.py#L176-L197)。

### 2.3 Beam 解码

beam 路径先对全部当前 causal attachment logits 做 `log_softmax`，之后才把已经归约
的位置改成 `-inf`。这一步用于禁止非法 beam 扩展，但不会重新归一化剩余 action。
所以它保留的是训练分布中的原始 log probability，而不是
`p(action | action is stack-legal)`。

证据：原仓库
[`models/pushdown_transformer_lm.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/models/pushdown_transformer_lm.py#L326-L336)。

结论：原仓库确实是“训练用 teacher forcing、部分测试用 beam”，但两者都没有把
attachment 概率重新定义为 stack-legal 集合内的条件概率。训练与 gold-tree 打分用
完整 causal support；beam 只在 softmax 之后禁止非法扩展。

## 3. 本地 v1/v2 概率协议

### 3.1 protocol v1：历史 Table-4 evaluator 口径

本地旧 evaluator 曾在 CE 前屏蔽非 stack-legal action，再对剩余 action 归一化：

```text
p_v1(r_k) = p(r_k | r_k is stack-legal)
```

这与本地 checkpoint 的训练目标和原仓库给定树打分不是同一个概率，但它是论文
Table-4 的历史 evaluator 口径。历史 native n-ary v1 只按这一旧结构协议保留；当前
fixed-word-atom v1 已独立完整运行，可用于同归一化口径比较。两种结构表示都不能与
v2 数值混写或合并。

### 3.2 protocol v2：训练 CE 口径

当前 evaluator 直接在 attachment head 已有的 causal、padding、sentence-local mask
之后做 CE：

```text
p_v2(r_k) = softmax(sentence-causal attachment logits)[r_k]
```

stack legality 仅用于：

- 验证给定 gold／候选 target 是否表示合法的 stack transition；
- beam search 时阻止非法扩展。

它不参与 teacher-forced likelihood 的再次归一化。实现见
[`olmo/eval/pushdown_document_ppl.py`](../olmo/eval/pushdown_document_ppl.py)；
协议元数据固定为：

```text
protocol_version=2
attachment_normalization=sentence_causal
```

断点续跑和结果合并会检查这些字段，从而拒绝缺少协议元数据的旧结果。v1/v2 的完整
数值见当前协议文档，本对照不复制第二套结果表。

## 4. 评测协议逐项对照

| 维度 | 原仓库 | 本仓库 fixed-word-atom document PPL |
|---|---|---|
| 主要目的 | 句子 PPL、解析、BLiMP、SyntaxGym surprisal | BBC-News 的 document-level PPL |
| 模型与 checkpoint | 原作者 16-layer Pushdown LM／BLLIP-LG checkpoint | 本地 OLMo Pushdown／`saved_models/pushdown_terminalonly/step34354-unsharded` |
| attachment head | `StackPredictor`：基于 `PushdownAttention`，含 depth embedding/composer、q/k MLP 和相对位置项 | 论文公式启发的两层 MLP＋单个 bilinear `W`；不是原仓库 `StackPredictor` 模块，见 [`olmo/attachment.py`](../olmo/attachment.py) |
| 基本评测单位 | 独立句子，显式 SOS/EOS | 按文档顺序的句子，跨句保留 token 和结构上下文 |
| 结构来源 | 给定单棵 gold tree，或模型自身 beam | 外部 parser 产生并排序的最多 300 个候选；不运行 Pushdown beam |
| 结构表示 | GPT-2 tokenization 后的无标签二叉树 | unary-collapse、word-level right-CNF、multi-BPE word 使用 fixed right-recursive atom |
| attachment 归一化 | causal support；beam 在 softmax 后禁非法 action | v1=stack-legal 条件化；v2=sentence-local causal support |
| 候选聚合 | gold 模式不聚合；beam 可对保留 beam 做 `logsumexp` | 对外部 top-K 唯一候选的模型联合概率做 `logsumexp` |
| 候选权重 | beam 分数来自模型自身联合概率 | parser proposal 分数只负责选择和排序，不乘入模型概率 |
| 跨句前缀 | 无 | 所有候选共享历史；每句 candidate 0 更新后续前缀 |
| 长上下文 | 原仓库句子级路径不需要文档裁剪 | 超长时从左侧按完整历史句裁剪，最大长度 2048 |
| 加速 | 原仓库增量 beam state | candidate 0 K/V cache 和 final-hidden cache |
| 分母 | 原仓库按其句子 token／attachment 步数定义 | terminal token 数；文档 BOS 不计，普通空白／换行 token 计入 |
| 输出含义 | gold-tree joint PPL 或模型 beam 近似 | 给定外部 top-K support 的截断结构边缘化 document PPL |

原仓库入口及公开评测说明见
[`eval_utils/eval_pushdown_model.py`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/eval_utils/eval_pushdown_model.py)
和原仓库
[`README.md`](https://github.com/MurtyShikhar/Pushdown-Layers/blob/6a4c543d105ab6f5df77f9e86bfd5650d1ed3f30/README.md)。

## 5. 当前仍然存在的关键可比性限制

### 5.1 本地模型不是原作者 checkpoint 的直接移植

本仓库使用 OLMo 主干和本地 attachment head；网络结构、训练实现及 checkpoint 均与
原仓库不同。因此，protocol v2 只能说明 attachment 的概率归一化语义与训练目标／
原作者 likelihood 路径一致，不能说明整个模型实现复现了原仓库。

### 5.2 已关闭：候选结构与 checkpoint 训练表示不一致

旧 native n-ary 路径没有补回 checkpoint 训练时的人工二叉节点和 multi-BPE word atom，
因此只保留为不同结构协议的历史/消融。当前生产协议将 direct strict-binary proposal，
或 native n-ary proposal 的 word-level right-CNF 转换，统一映射到训练使用的
fixed-word-atom 表示。转换与支持集审计见当前协议文档。

### 5.3 top-K `logsumexp` 是截断和，不是完整句子概率

当前主值为：

```text
log p_topK(x) = logsumexp_y_in_external_topK log p_model(x, y)
```

它只覆盖外部 proposal 给出的有效唯一候选，不保证穷尽模型结构空间。不同句子的
有效 `K` 可以小于 300。当前同时报告减去 `log(K)` 的 uniform-mixture 结果，但该值
只是诊断口径，不是原仓库定义。

### 5.4 candidate 0 文档前缀是本地扩展

当前每句的所有候选共享相同历史，并固定用 candidate 0 的结构更新下一句上下文。
这是为了与本仓库其他 document-PPL evaluator 对齐而作的确定性定义，不来自原仓库
的句子级测试标准。

## 6. 可以和不可以据此声称的结论

可以声称：

- protocol v2 的 attachment CE 不再做 stack-legal 条件化；
- 该归一化与本地训练 objective 以及原仓库给定树 likelihood 的语义一致；
- 当前结果是在外部 fixed-word-atom strict-binary top-K support 上得到的本地文档级截断边缘化指标。

不可以声称：

- 当前 native document PPL 是原仓库评测的逐项复现；
- 外部 top-K 候选等价于原仓库模型 beam；
- 外部 top-K support 穷尽了模型的全部合法结构；
- protocol v1 的历史结果可与 protocol v2 直接比较或合并。

若目标是复现原仓库，应使用原仓库代码、checkpoint、tokenizer／二叉树预处理及其
`gold` 或 `beam` 入口。若目标是本仓库的训练表示对齐 top-K 指标，应使用
[`pushdown_word_atom_strict_binary_document_ppl_protocol.md`](pushdown_word_atom_strict_binary_document_ppl_protocol.md)；
它仍然是本地 OLMo Pushdown 指标，而不是原作者 checkpoint 复现。

## 7. 本仓库实现入口

- evaluator：[`olmo/eval/pushdown_document_ppl.py`](../olmo/eval/pushdown_document_ppl.py)
- attachment head：[`olmo/attachment.py`](../olmo/attachment.py)
- CLI：[`scripts/evaluate_gpst_binary_pushdown_document_ppl.py`](../scripts/evaluate_gpst_binary_pushdown_document_ppl.py)
- merge/protocol 检查：[`scripts/merge_gpst_binary_pushdown_document_ppl.py`](../scripts/merge_gpst_binary_pushdown_document_ppl.py)
