# Pushdown Gold-300 Document-Level Perplexity 设计方案

> [!WARNING]
> **设计阶段历史文档，不是当前正式运行协议。** 其中接口和假设可用于追溯设计，但
> checkpoint-training representation、v1/v2 attachment normalization 与完整结果以
> [`pushdown_word_atom_strict_binary_document_ppl_protocol.md`](pushdown_word_atom_strict_binary_document_ppl_protocol.md)
> 为准；已废弃的 BPE-spliced 方案见本仓库纠错索引
> [`../REPOSITORY_CLEANUP_MEMORY.md`](../REPOSITORY_CLEANUP_MEMORY.md) M-06/M-07。

## 1. 目标

为本仓库的 Pushdown OLMo 模型实现一个 document-level perplexity evaluator。评估时不运行 beam search，而是直接使用测试集为每个句子提供的 300 棵 gold parse tree，对固定结构下的 token 概率和 attachment 概率进行 teacher-forced 评分。

该指标需要与现有 OLMo `TG_doc` 和 GPST gold-300 document PPL 保持一致的文档组织方式：

- 一个句子固定读取 300 个 parse candidates；
- 300 个候选共享同一个历史文档上下文；
- 完成一句后，使用 candidate 0 更新下一句的共享上下文；
- 文档切换时清空上下文；
- 句子概率通过 300 个候选的 `logsumexp` 聚合；
- perplexity denominator 只统计 terminal tokens；
- 全流程不调用 `pushdown_beam_search()`。

第一版以概率定义和结构约束正确为优先，使用完整前缀重算。KV-cache 优化在获得正确性基准后单独实现。

## 2. 与 GPST gold-300 指标的对应关系

| GPST | Pushdown |
|---|---|
| terminal token target | terminal token target |
| SHIFT/REDUCE action target | attachment/reduce target `r_k` |
| gold merge trajectory | gold constituent spans + stack state |
| action transformer probability | attachment-head probability |
| candidate 0 更新文档前缀 | candidate 0 更新文档前缀 |
| 300-tree `logsumexp` | 300-tree `logsumexp` |

两种模型的结构因子不同，因此结构 NLL 的具体项不同，但都计算模型自身定义的联合概率 `p(x, y)`。

## 3. 概率定义

设第 `s` 个句子的第 `k` 个 gold parse 为 `y_{s,k}`，terminal sequence 为 `x_s`。固定该 parse 的联合负对数似然为：

```text
tree_nll[s, k]
  = token_nll[s, k]
  + attachment_nll[s, k]
```

其中：

```text
token_nll
  = -sum_t log p(x_t | document_prefix, x_<t, y_<=t-1)

attachment_nll
  = -sum_t log p(r_t_gold | document_prefix, x_<=t, y_<t)
```

约束如下：

- BOS 仅作为上下文，不计入 token NLL 和 denominator；
- ordinary whitespace/newline tokens 计入 token NLL；
- tree 外部的 whitespace/newline 不产生 attachment target；
- attachment target 只在 `pushdown_sentence_ids >= 0` 的位置计算；
- attachment softmax 只在当前 stack 状态允许的合法 actions 中归一化；
- 不加入训练时的 `pushdown_attachment_weight`；该参数是优化目标权重，不是概率指数；
- 不加入其他训练辅助 loss；
- 所有候选 NLL 使用 float64 聚合。

### 3.1 句子级聚合

为了复现当前 OLMo `TGPerplexityDocumentLevelMetric`：

```python
legacy_sentence_ll = torch.logsumexp(-tree_nll, dim=0)
```

当前 OLMo legacy metric 没有减去 `log(300)`。为避免统计口径含糊，Pushdown evaluator 应同时报告标准均匀混合版本：

```python
uniform_mixture_sentence_ll = (
    torch.logsumexp(-tree_nll, dim=0)
    - math.log(num_candidates)
)
```

最终输出至少包括：

- `legacy_perplexity`：与现有 OLMo 指标直接对齐；
- `uniform_mixture_perplexity`：标准的 300-tree 均匀混合概率；
- `token_only_perplexity`：诊断项，不计 attachment probability；
- `terminal_count`；
- `sentence_count`；
- `document_count`；
- `samples_per_sentence`；
- `beam_search: false`。

### 3.2 文档级聚合

```python
total_ll = sum(sentence_ll for sentence in corpus)
ppl = exp(-total_ll / total_terminal_count)
```

这里的“document-level”表示句子概率在前文上下文下计算，而不是先分别计算每篇文档的 PPL 再取算术平均。

## 4. 数据输入与 gold tree 转换

默认数据：

```text
dataset/testppl_tree/tree_300.npy
dataset/testppl_tree/tree_sent_index.npy
dataset/testppl_tree/tree_doc_index.npy
dataset/bbc-news/TG_GPT2_tokenizer.json
```

使用现有 `parse_chunk_slice()`：

```python
parsed = parse_chunk_slice(
    tree_record,
    tree_vocab,
    direction="right",
    binarize=True,
    collapse_unary=True,
    drop_singleton_spans=True,
)
```

输出：

- `input_ids`：仅含 terminal/control tokens，不含 NT bracket tokens；
- `spans`：terminal 坐标下的 `(left, split, right)`；
- `sentence_ids`：top-level tree 内 token 为非负 ID，tree 外 token为 `-1`；
- `word_boundaries`：该指标暂不直接使用，但保留以支持诊断。

不得使用原始 unary tree 直接生成 Pushdown spans。预处理必须与 terminal-only Pushdown checkpoint 保持一致：

```text
collapse unary → right CNF → drop singleton spans
```

### 4.1 已验证的真实数据不变量

本地 `tree_300` 第一条句子具有：

- 300 个候选；
- 300 个候选的 terminal sequence 完全相同；
- 每个候选转换后有 12 个输入 token；
- 第一个 token 是 BOS，因此有 11 个被评分 token；
- 300 个 serialized parses 在 Pushdown 的 unary-collapsed、无标签 span 表示下映射为 5 个唯一结构。

主指标仍保留全部 300 个 slots，以复现 OLMo legacy protocol。可额外提供 `deduplicated_tree_perplexity` 作为诊断，但不能替代主指标。

## 5. 数据结构

建议新增：

```python
@dataclass(frozen=True)
class PushdownGoldCandidate:
    tokens: tuple[int, ...]
    spans: tuple[tuple[int, int, int], ...]
    sentence_ids: tuple[int, ...]
    attachment_targets: tuple[int, ...]
    legal_attachment_targets: tuple[tuple[int, ...], ...]
```

其中 `legal_attachment_targets[q]` 是 query `q` 对应的当前 stack 合法 attachment key 集合。tree 外位置使用空 tuple。

```python
class PushdownGold300Corpus:
    def document_id(self, sentence_index: int) -> int: ...

    def sentence_candidates(
        self, sentence_index: int
    ) -> tuple[PushdownGoldCandidate, ...]: ...
```

`tree_sent_index.npy` 保存每条 record 的长度。初始化时构造前缀和 offsets；`tree_doc_index.npy` 保存每篇文档的句子数，使用累积和定位 `document_id`。

为了避免把完整 4.9 GB tree array 读入内存，`tree_300.npy` 必须使用 mmap。

## 6. Gold attachment action 构造

不能直接把 attachment logits 在所有 `j <= k` 上做普通 CE。当前 `pushdown_beam_search()` 在当前 stack 允许的合法 attachment actions 内归一化，因此固定 gold tree 也必须使用相同 action space。

对每棵 gold tree 按 token 从左到右模拟 stack：

1. 在 token `k` 到来前，stack 保存由 gold prefix 形成的 constituent frontier；
2. SHIFT 当前 singleton `(k, k)`；
3. 合法 actions 包含：
   - shift-only：target `k`；
   - 从 stack top 开始依次 reduce 1、2、... 次得到的 target；
4. 根据所有 `right == k` 的 gold spans 确定本步 gold reduction；
5. 记录 `gold_target`；
6. 应用 gold reduction，更新 stack；
7. top-level sentence 边界处重置 sentence-local stack。

建议在 `olmo/attachment.py` 增加：

```python
def derive_gold_attachment_actions(
    spans: torch.Tensor,
    sentence_ids: torch.Tensor,
    span_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[list[list[int]]]]:
    """Return gold targets and ragged legal targets for every query."""
```

collator 再将当前 batch 的 ragged legal targets 转换成仅覆盖当前句 query range 的 bool mask：

```text
(batch, current_query_length, full_context_length)
```

结构 NLL：

```python
legal_logits = attachment_logits.masked_fill(~legal_action_mask, -torch.inf)
attachment_nll = F.cross_entropy(
    legal_logits.transpose(1, 2),
    gold_attachment_targets,
    reduction="none",
)
attachment_nll = (attachment_nll * current_attachment_query_mask).sum(dim=1)
```

需要验证每个非忽略 gold target 都位于 legal mask 中。

## 7. 文档上下文

对一个文档按句子顺序处理：

```python
prefix = ()

for sentence in document:
    candidates = sentence.candidates  # exactly 300
    score(prefix, candidates)
    prefix = prefix + candidates[0]
```

对于当前句的 300 个候选：

- prefix tokens 相同；
- prefix spans 相同；
- prefix 来自之前每句的 candidate 0；
- 当前 tokens 相同；
- 只有当前 spans、attachment targets 和 legal actions 随 candidate 改变。

拼接时必须：

- 将当前 spans 整体平移 `prefix_token_count`；
- 将每个 top-level tree 的 `sentence_ids` 重映射成文档内唯一 ID；
- 保持 BOS 只在原始文档开头出现；
- 不为后续句人工插入 ROOT/BOS；
- label/token loss slice 必须覆盖当前句的第一个 token。

如果 prefix 长度为 `P`，当前句长度为 `C`：

```text
target_start = max(P, 1)
target_end   = P + C
logit slice = [target_start - 1 : target_end - 1]
label slice = [target_start     : target_end]
```

当 `P > 0` 时，前缀最后一个位置的 logits 负责预测当前句第一个 token。

### 7.1 上下文截断

第一版使用完整前缀重算。达到 `max_sequence_length` 时，从左侧删除完整历史句子，直到：

```text
len(retained_prefix) + len(current_sentence) <= max_sequence_length
```

不能从一棵树中间截断，因为这会产生不完整 spans 和错误 stack tape。如果当前单句自身超过上限，应明确报错或单独记录为 skipped/error，不能静默截断 gold tree。

## 8. 模型接口优化

完整 transformer 前向仍然需要 prefix hidden states，但不应对整个 prefix 生成 vocabulary logits 或完整 attachment matrix。

### 8.1 Token logits range

在 `OLMo.forward()` 增加：

```python
logits_range: Optional[tuple[int, int]] = None
```

在 LM head 前切片：

```python
if logits_range is not None:
    start, end = logits_range
    x_for_logits = x[:, start:end]
else:
    x_for_logits = x

logits = self.transformer.ff_out(x_for_logits)
```

这样只物化当前句所需的 `(B, current_len, vocab_size)` logits。

### 8.2 Attachment query range

在 `OLMo.forward()` 和 `PushdownAttachmentHead.forward()` 增加：

```python
attachment_query_range: Optional[tuple[int, int]] = None
```

attachment head 只为当前 query range 生成：

```text
(B, current_query_length, full_context_length)
```

实现时仍需要：

- full-context key hidden states `h_j`；
- 当前 query 的 `emb(x_k)`；
- 当前 query 对应的 `h_{k-1}`；
- 矩形 causal mask；
- 矩形 same-sentence mask；
- self-action diagonal位于 global key coordinate `j == k`。

该优化可显著降低 `B × context_len × context_len` attachment logits 的内存。

## 9. Scorer 接口

建议新增：

```python
@torch.no_grad()
def score_pushdown_gold_candidates(
    model: OLMo,
    prefix: tuple[PushdownGoldCandidate, ...],
    candidates: tuple[PushdownGoldCandidate, ...],
    device: torch.device,
    eval_batch_size: int,
    include_attachment_probability: bool = True,
) -> PushdownCandidateScores:
    ...
```

返回：

```python
@dataclass
class PushdownCandidateScores:
    joint_nll: torch.Tensor       # (K,), float64 CPU
    token_nll: torch.Tensor       # (K,), float64 CPU
    attachment_nll: torch.Tensor  # (K,), float64 CPU
```

核心模型调用：

```python
output = model(
    input_ids=batch.input_ids,
    attention_mask=batch.attention_mask,
    tree_spans=batch.tree_spans,
    pushdown_sentence_ids=batch.sentence_ids,
    compute_attachment_logits=True,
    logits_range=batch.current_logit_range,
    attachment_query_range=batch.current_attachment_range,
)
```

token softmax 和 attachment softmax 应在 float32 中计算；句子/文档聚合使用 float64。

## 10. Evaluator 接口

建议新增：

```python
@dataclass(frozen=True)
class PushdownDocumentPPLResult:
    legacy_perplexity: float
    uniform_mixture_perplexity: float
    token_only_perplexity: float
    legacy_log_likelihood: float
    uniform_mixture_log_likelihood: float
    terminal_count: int
    sentence_count: int
    document_count: int
    samples_per_sentence: int
    deduplicated_trees: bool
    beam_search: bool = False
```

```python
def evaluate_pushdown_document_ppl(
    model: OLMo,
    corpus: PushdownGold300Corpus,
    device: torch.device | str,
    eval_batch_size: int = 4,
    max_sequence_length: int = 2048,
    deduplicate_trees: bool = False,
    include_attachment_probability: bool = True,
    progress: Optional[Callable] = None,
) -> PushdownDocumentPPLResult:
    ...
```

主循环：

```python
for document in corpus:
    prefix = ()

    for sentence in document:
        scores = score_pushdown_gold_candidates(
            model,
            prefix,
            sentence.candidates,
            device,
            eval_batch_size,
        )

        legacy_ll += torch.logsumexp(-scores.joint_nll, dim=0)
        mixture_ll += (
            torch.logsumexp(-scores.joint_nll, dim=0)
            - math.log(len(scores.joint_nll))
        )
        token_only_ll += torch.logsumexp(-scores.token_nll, dim=0)
        terminal_count += sentence.terminal_count

        prefix = prefix + (sentence.candidates[0],)
```

## 11. CLI

新增：

```text
scripts/evaluate_pushdown_document_ppl.py
```

命令草案：

```bash
python scripts/evaluate_pushdown_document_ppl.py \
  --checkpoint saved_models/pushdown_terminalonly/step34354-unsharded \
  --tree-data dataset/testppl_tree/tree_300.npy \
  --sentence-index dataset/testppl_tree/tree_sent_index.npy \
  --document-index dataset/testppl_tree/tree_doc_index.npy \
  --tokenizer-path dataset/bbc-news/TG_GPT2_tokenizer.json \
  --eval-batch-size 4 \
  --device cuda
```

建议参数：

```text
--samples-per-sentence 300
--max-sentences N
--max-documents N
--max-sequence-length 2048
--deduplicate-trees
--token-only
--device cuda:0
--log-every 10
```

CLI 启动时必须打印：

```text
structure_source=gold300
beam_search=false
context_update=candidate0
attachment_probability=true|false
mixture_reporting=legacy_and_normalized
```

## 12. Checkpoint 约束

主指标默认面向：

```text
saved_models/pushdown_terminalonly/step34354-unsharded
```

该 checkpoint config 中：

```text
transformer_grammar_type: pushdown
pushdown_attachment_weight: 1.0
pushdown_use_attachment_head_inference: true
```

加载时必须检查 attachment-head weights 是否存在。若用户要求联合 PPL，但 checkpoint 缺少 attachment head：

- 必须报错；
- 不能用随机初始化 attachment head 继续评分；
- 可显式选择 `--token-only`；
- 或实现 normalized-uniform structural prior 并在结果中明确标记。

旧 checkpoint 的 `strict=False` 加载方式不能直接用于正式联合 PPL。

## 13. 为什么第一版不使用 KV cache

当前普通 OLMo `TG_doc_eval_step` 不能直接用于 Pushdown gold-tree 评分：

- 它没有传入每个候选的 `tree_spans`；
- Pushdown depth bias 依赖当前 query 与全部历史 key 之间的 stale depth row；
- 缓存场景满足 `query_len != key_len`；
- 当前 `_pushdown_attention()` 的 depth matrix 路径仍按 query length 构造方阵，不能正确表达 rectangular cached attention；
- tree spans/stack 状态会随 gold candidate 改变；
- 直接套普通 KV cache 可能输出有限数值，但概率语义错误。

第一版使用完整前缀重算，作为 correctness oracle。

后续优化需要单独实现：

```python
compute_depth_rows_gpu(
    tree_spans,
    query_start,
    query_length,
    key_length,
) -> Tensor  # (B, query_length, key_length)
```

并验证 cached forward 与完整前缀 forward 在 token logits、attachment logits 和最终 NLL 上一致后，才能启用 Pushdown document KV cache。

## 14. 测试计划

建议新增：

```text
tests/test_pushdown_document_ppl.py
tests/test_pushdown_gold_actions.py
tests/test_pushdown_document_cache_parity.py  # 第二阶段
```

### 14.1 数据转换测试

1. 真实第一句 300/300 candidates 均成功转换；
2. 300 个 candidates 的 terminal sequence 完全相同；
3. spans 满足 `0 <= left <= split <= right < len(tokens)`；
4. unary/preterminal singleton spans 已删除；
5. BOS 只在文档第一句出现；
6. tree 外 suffix 的 `sentence_ids == -1`；
7. 主指标保留全部 300 slots；
8. dedupe 模式只作为额外诊断。

### 14.2 Gold action 测试

使用手写左分支树和右分支树验证：

1. gold attachment targets 不同；
2. 每个 gold target 位于对应 legal action set；
3. shift-only target 为当前 token；
4. reduce target 为当前 stack 中合法 constituent 的 right endpoint；
5. sentence boundary 会重置 stack；
6. tree 外 token 不产生结构 target。

### 14.3 概率测试

1. legal structural softmax 概率和为 1；
2. 非法 actions 被置为 `-inf`；
3. joint NLL 等于 token NLL 加 attachment NLL；
4. 不受 `pushdown_attachment_weight` 数值影响；
5. synthetic 300 个零 NLL 时：
   - legacy LL 为 `log(300)`；
   - normalized LL 为 0；
6. BOS 不进入 token NLL；
7. denominator 与真实 terminal count 一致。

### 14.4 文档上下文测试

1. 第二句能读取第一句 candidate 0 上下文；
2. 第二句不能读取第一句其他 candidate；
3. 新文档正确清空 prefix；
4. 当前句第一个 token 由 prefix 最后一个位置预测；
5. 删除完整历史句子后的 spans 坐标正确重映射；
6. 不允许截断半棵当前树。

### 14.5 模型与真实 checkpoint 测试

1. `logits_range` 与完整 logits 对应 slice 数值一致；
2. `attachment_query_range` 与完整 attachment logits 对应 slice 数值一致；
3. FlexAttention 与 SDPA fallback 在小样本上数值接近；
4. 真实 checkpoint、第一句、完整 300 candidates 输出有限 NLL/PPL；
5. 同一文档两句 smoke test 输出有限 PPL；
6. evaluator 运行期间不调用 `pushdown_beam_search()`；
7. 缺失 attachment-head weights 时联合模式明确失败。

## 15. 实施顺序

### Phase 1：数据与 gold actions

- 实现 `PushdownGold300Corpus`；
- 复用 `parse_chunk_slice()`；
- 实现 gold stack simulator；
- 完成真实第一句 300-tree 数据测试。

### Phase 2：范围化模型输出

- 为 `OLMo.forward()` 增加 `logits_range`；
- 为 attachment head 增加 `attachment_query_range`；
- 验证 range/full parity。

### Phase 3：正确性优先 evaluator

- 实现完整前缀重算；
- 实现 joint/token-only NLL；
- 同时输出 legacy 和 normalized PPL；
- 完成真实 checkpoint 一句和两句 smoke tests。

### Phase 4：Trainer 集成

- 新增 evaluator type，例如 `pushdown_gold_doc`；
- 增加 `Trainer.pushdown_gold_doc_eval_step()`；
- 增加 YAML evaluator 配置；
- 支持 distributed document-level sampling，保证同一文档不会跨 rank。

### Phase 5：KV-cache 优化

- 实现 rectangular depth rows；
- 实现 tree-aware cached forward；
- 完整前缀与 cache 逐 token parity；
- parity 通过后才允许全量评估默认启用 cache。

## 16. 完成标准

实现可以被认为正确完成，必须同时满足：

- 固定读取每句 300 个 gold parses；
- 没有 beam search 调用；
- token 与 attachment 概率均按 Pushdown 联合模型计算；
- attachment 只在合法 stack actions 内归一化；
- candidate 0 正确维护文档上下文；
- legacy 公式与现有 OLMo metric 一致；
- 同时报告 normalized mixture，避免指标解释歧义；
- terminal denominator 与 GPST/OLMo 数据定义一致；
- 缺少 attachment head 时不会静默使用随机参数；
- 真实 300-tree checkpoint smoke test 通过；
- 所有新增回归测试通过。
