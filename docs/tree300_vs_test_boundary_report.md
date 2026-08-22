# BBC News `tree_300` 与 `tree/test.npy` 文档、句子及 terminal 对齐报告

核验日期：2026-08-22

## 结论

`dataset/bbc-news/testppl_tree` 不是 `dataset/bbc-news/tree/test.npy` 的逐句、逐下标复制。两者的正确关系是：

1. `testppl_tree` 的 4,966 个文档对应 `tree/test.npy` 从文档 59 开始的后缀，即候选映射为
   `testppl_doc_id = d` 对应 `test_doc_id = d + 59`（均为 0-based）。
2. 两边以“顶层句法树内部的 BPE terminal ID 序列”为内容判据。标签、unary chain、括号 token、BOS/EOS 及树外空白不参与 terminal 相等判断。
3. 4,966 个对应文档中，4,960 个文档的拼接 terminal 串完全相同。
4. 仍有 6 个异常文档。其中 4 个文档的句子数量不同，另外 2 个句子数量相同但 terminal 串不完全相同。

因此，后续生成 GPST/Pushdown native top-k 树时，不能简单断言两个数据源的句子边界逐项相等。应采用 `testppl_tree/tree_doc_index.npy` 作为 document-level PPL 的规范评测边界，同时以 terminal 序列建立并验证到 `tree/test.npy` 的对应关系。

## 数据文件与物理规模

| 文件 | shape | dtype | 文件大小 | 含义 |
|---|---:|---:|---:|---|
| `tree/test.npy` | `(8178128,)` | `uint16` | 16,356,384 B | 原始 test tree stream；每个文档含 BOS/EOS，文档内可有多个顶层句树 |
| `terminal/test.npy` | `(3328360,)` | `uint16` | 6,656,848 B | 与原始 test tree stream 对应的 terminal-only stream |
| `testppl_tree/tree_300.npy` | `(2594352932,)` | `uint16` | 5,188,705,992 B | 每句 300 个带标签候选树顺序拼接后的 token stream |
| `testppl_tree/tree_sent_index.npy` | `(44650800,)` | `uint16` | 89,301,728 B | 每个候选树记录的 token 长度；`148,836 × 300` 条记录 |
| `testppl_tree/tree_doc_index.npy` | `(4966,)` | `uint32` | 19,992 B | 每个评测文档包含的句子数 |

注：表中的 shape 按当前 `.npy` 文件头记录，均为一维数组；数字中的千位分隔符只在正文统计中使用。

## 总体边界差异

| 统计量 | `tree/test.npy` 全集 | `tree/test.npy[doc 59:]` | `testppl_tree` |
|---|---:|---:|---:|
| 文档数 | 5,025 | 4,966 | 4,966 |
| 顶层句树/评测句数 | 150,738 | 148,884 | 148,836 |

由此可见：

- `test.npy` 的前 59 个文档不在 `testppl_tree` 中；这 59 个文档共含 1,854 个顶层句树。
- 即便去掉这 59 个文档，`test.npy` 后缀仍比 `testppl_tree` 多 48 个句子。
- 文档数量及顺序可以用固定偏移 59 对齐，但句子全局下标不能只用一个固定偏移对齐。

## 句子数量不一致的 4 个文档

下表中 `ppl doc` 是 `testppl_tree` 的 0-based 文档号，`test doc` 是 `tree/test.npy` 的 0-based 文档号。

| ppl doc | test doc | `test.npy` 句数 | `testppl_tree` 句数 | 差值 test − ppl |
|---:|---:|---:|---:|---:|
| 1,063 | 1,122 | 26 | 29 | -3 |
| 1,090 | 1,149 | 70 | 15 | +55 |
| 1,121 | 1,180 | 37 | 42 | -5 |
| 1,122 | 1,181 | 10 | 9 | +1 |
| **合计** |  | **143** | **95** | **+48** |

其余 4,962 个对应文档的句子数量相同。总句数差 48 完全由以上 4 个文档产生，说明主要差异来自这几个文档采用了不同的长句切分或预处理边界，而不是整个语料发生了持续性错位。

## terminal 内容不一致的 6 个文档

以下统计把一个文档内所有顶层句树的 terminal 依次拼接，忽略句子切分点后比较 BPE token ID。

| ppl doc | test doc | test 句数 | ppl 句数 | test terminal 数 | ppl terminal 数 | 类型 |
|---:|---:|---:|---:|---:|---:|---|
| 1,063 | 1,122 | 26 | 29 | 699 | 673 | 边界及内容差异 |
| 1,090 | 1,149 | 70 | 15 | 1,250 | 319 | 边界及内容差异 |
| 1,121 | 1,180 | 37 | 42 | 871 | 835 | 边界及内容差异 |
| 1,122 | 1,181 | 10 | 9 | 204 | 221 | 边界及内容差异 |
| 1,550 | 1,609 | 句数相同 | 句数相同 | 3,029 | 3,023 | 少量 terminal 差异 |
| 2,626 | 2,685 | 句数相同 | 句数相同 | 759 | 753 | 少量 terminal 差异 |

除这 6 个文档外，其余 4,960 个文档在固定 `+59` 映射下具有完全相同的文档级 terminal 串，占 99.879%（`4960 / 4966`）。

另外，以单句 terminal 序列在整个 `test.npy` 中查找时，`testppl_tree` 的 148,836 个句子中有 148,740 个能找到相同 terminal 内容。这个数字说明内容高度重合，但它不能替代逐文档边界映射：重复句子会造成多义匹配，异常文档中的重新切句也会破坏逐句一一对应。

## 比较口径

本报告使用如下口径，避免标签和序列化格式干扰内容判断：

1. 使用 `dataset/bbc-news/TG_GPT2_tokenizer.json` 识别 opening/closing non-terminal token。
2. `tree/test.npy` 以 BOS/EOS 划分文档，以 bracket depth 回到 0 划分顶层句树。
3. `tree_300.npy` 使用 `tree_sent_index.npy` 定位记录，每句只读取 candidate 0 来核对语料内容与边界。
4. 对每棵树只保留根括号内部的普通 token ID，即 parser tree 的 BPE terminals。
5. 不比较 non-terminal 标签、unary chain、括号数量、候选树长度、BOS/EOS 或句树外的空格/换行 token。

这里只用 candidate 0 做跨数据集核验，是因为 300 个候选的区别应当只在树结构；native top-k 生成器仍需另外执行“同一句所有有效候选 terminals 一致”的输出断言，不能把 candidate-0 核验当作该断言的替代品。

## 对 native n-ary top-k=300 生成的约束

推荐将两类信息分开保存：

- **评测边界真值**：以 `testppl_tree/tree_doc_index.npy` 的 4,966 个文档、148,836 个句子为准，保证新结果可直接接入现有 document-level PPL 协议。
- **语料来源关系**：保存 `ppl_doc_id -> test_doc_id = ppl_doc_id + 59`，并对每句/每文档保存 terminal hash 或显式对齐状态。

生成阶段必须满足：

1. 不按 `test.npy` 后缀的 148,884 个句子直接输出，否则会比评测边界多 48 句。
2. 对 4,960 个完全一致文档执行严格 terminal equality 断言。
3. 对上述 6 个异常文档建立显式异常清单；不得用“下一个句子”自动补位，也不得静默接受 terminal 不一致。
4. 最终 shard 合并后必须断言：文档数为 4,966、句子数为 148,836、每句物理候选槽为 300。
5. GPST 与 Pushdown 的候选树可以采用不同的模型适配表示，但同一句的 terminal 序列和 document/sentence 边界必须共享同一份规范索引。

## 最终判断

两套数据的关系应表述为：**文档顺序以固定 `+59` 偏移对应，绝大多数内容在 document terminal 层完全一致，但 sentence segmentation 并非处处相同。**

因此，“他们是 terminal 一致的关系”在语料对应原则上是正确的，但必须附加 6 个已知异常文档的限定。对于 document-level perplexity，`testppl_tree` 的边界是规范边界，`test.npy` 是需要通过 terminal 对齐接入的来源，不能反过来直接用 `test.npy` 的全部句子边界覆盖评测边界。
