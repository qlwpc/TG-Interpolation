# BBC News `tree_300` 与 `tree/test.npy` 文档、句子及 terminal 对齐报告

核验日期：2026-08-22

## 结论

`dataset/bbc-news/testppl_tree` 不是 `dataset/bbc-news/tree/test.npy` 的逐句、逐下标复制。两者的正确关系是：

1. `testppl_tree` 的 4,966 个文档对应 `tree/test.npy` 从文档 59 开始的后缀，即候选映射为
   `testppl_doc_id = d` 对应 `test_doc_id = d + 59`（均为 0-based）。
2. 两边以“顶层句法树内部的 BPE terminal ID 序列”为内容判据。标签、unary chain、括号 token、BOS/EOS 及树外空白不参与 terminal 相等判断。
3. 4,966 个对应文档中，4,960 个文档的拼接 terminal 串完全相同。
4. 仍有 6 个异常文档。其中 4 个不是同一文章的重新切句，而是两套数据放了不同的新闻全文；另外 2 个是同一文章中各有一句多出字面量 `(ADJ ... ADJ)`。
5. RTX3090 上的未清理版 `dataset/testppl_tree_deprecated` 与 `tree/test.npy` 有 4,962 个文档完全一致；其 4 个异常正是前述“整篇新闻不同”的文档。因此 ADJ 清理只造成当前版本额外的 2 个差异，不能解释前 4 个差异。

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

其余 4,962 个对应文档的句子数量相同。总句数差 48 完全由以上 4 个文档产生，但逐文本检查表明这不是长句重新切分：四组标题和正文均不同，是局部的整篇文档替换。前后相邻文档仍然精确对齐，所以它们没有引起持续性的 document offset 错位。

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

另外，以单句 terminal 序列在整个 `test.npy` 中查找时，`testppl_tree` 的 148,836 个句子中有 148,740 个能找到相同 terminal 内容。这个数字说明内容高度重合，但它不能替代逐文档边界映射：重复句子会造成多义匹配，而整篇文章替换和局部字面量差异会破坏逐句一一对应。

## 6 个异常文档的具体差异

| ppl doc | test doc | `tree/test.npy` 文章 | `testppl_tree` 文章 | 判定 |
|---:|---:|---|---|---|
| 1,063 | 1,122 | *Avigdor Lieberman wants to put off talks...* | *Palestinian militant group Hamas says it has fired rockets...* | 整篇新闻不同；26 句/699 terminals 对 29 句/673 terminals |
| 1,090 | 1,149 | *Stores in the US, UK, Canada, Ireland and Puerto Rico are affected* | *Concern at plan for third bookmakers in Merkinch area of Inverness* | 整篇新闻不同；70 句/1,250 terminals 对 15 句/319 terminals |
| 1,121 | 1,180 | *Gove rebuts claims of American author ban* | *The Lebanon donor conference in Stockholm has raised...* | 整篇新闻不同；37 句/871 terminals 对 42 句/835 terminals |
| 1,122 | 1,181 | *The training pool at DG One will be closed...* | *Work has started on exhuming 50 Victorian corpses...* | 整篇新闻不同；10 句/204 terminals 对 9 句/221 terminals |
| 1,550 | 1,609 | `... it's been (ADJ worth the wait ADJ) .` | `... it's been worth the wait .` | 同文章、同句界；当前 `testppl_tree` 删除了 `(ADJ` 和 `ADJ)`，合计 6 个 BPE terminals |
| 2,626 | 2,685 | `Phase one is (ADJ worth a few million pounds ADJ) .` | `Phase one is worth a few million pounds .` | 同文章、同句界；当前 `testppl_tree` 删除了同样的 6 个 BPE terminals |

前 4 组文章在另一套语料的其他文档中也没有找到 document-terminal 全串精确匹配，因此不应通过局部搜索后重排文档来“修复”。对它们应直接选定一个权威来源；若目标是复现现有 document-PPL 协议，该来源必须是 `testppl_tree` candidate 0。

## deprecated 版本核验

2026-08-22 在 SSH 主机 RTX3090 上原位读取了：

- `/home/wangpch/TG-Interpolation/dataset/testppl_tree_deprecated`（未清理 ADJ）；
- `/home/wangpch/TG-Interpolation/dataset/bbc-news/testppl_tree`（当前清理版）；
- `/home/wangpch/TG-Interpolation/dataset/bbc-news/tree/test.npy`。

| 版本 | 与 `tree/test.npy[doc 59:]` 完全一致的文档 | 异常 ppl docs | candidate-0 内部 terminals |
|---|---:|---|---:|
| deprecated | 4,962 / 4,966 | 1,063、1,090、1,121、1,122 | 3,068,837 |
| current | 4,960 / 4,966 | 上述 4 篇 + 1,550、2,626 | 3,068,825 |

两版的 `tree_doc_index.npy` 内容相同，文档数和句数也相同。当前版相对 deprecated 只在 ppl doc 1,550 和 2,626 的 candidate-0 terminal 内容上变化，共删除 12 个 BPE terminals。`tree_sent_index.npy` 则不能逐项相等，因为清理/修复还改变了其他 candidate 记录的物理长度；这不影响上述 candidate-0 语料结论。

## 当前 `testppl_tree` 文本质量核验

在 BOS/EOS 规范化前，对 `tree_300.npy` 全部 2,594,352,932 个 token 扫描了 ADJ 及另一个 tokenizer 缺失标签 NX 的有/无前导空格开闭字面量模式，所有计数均为 0。因此不仅 candidate 0，整个 300-candidate 物理文件中都没有已知的 `(ADJ ... ADJ)` 或 `(NX ... NX)` 标签泄漏。后续边界规范化只插入 BOS/EOS，不改变这项结论。

candidate-0 文档层检查结果为：

- 4,966 个文档、148,836 个句记录，没有空文档或空句；
- 148,836 个 candidate-0 记录均只含一个完整、括号平衡的顶层树，解析失败数为 0；
- 单句内部 terminal 长度为 1--304，中位数 20；文档内部 terminal 长度为 60--17,397，中位数 466；
- 有 8,512 个长度不超过 2 terminals 的微型句树，多为独立引号、标题片段、`PHOTO :` 等 BBC 页面/句法切分产物。它们不造成文档 terminal 丢失，但说明 sentence boundary 不应被当作经过人工清洗的自然句边界。

因此，当前版可判定为**无已知 ADJ/NX 字面量泄漏，且在结构、非空性和文本连贯性层面质量良好**。但它不是 `tree/test.npy` 的完全同源副本：4 个文档位置是另一篇完整、可读的 BBC 新闻。这是 provenance/可比性问题，不是文章本身损坏。

## 构建与 `testppl_tree` 对齐的 terminal `test.npy`

必须先区分两种“一模一样”：

1. **历史字面投影版**：对规范化前的每句 candidate 0 删掉 non-terminal token，保留记录中其他所有 token，然后按 `tree_doc_index.npy` 顺序拼接。历史结果 shape 为 `(3284061,)`、dtype 为 `uint16`；其中每篇文档只含 BOS 或 EOS 之一。完成本文后述的物理规范化后，同样的字面投影 shape 变为 `(3289027,)`。
2. **标准文档边界版（推荐用于 terminal LM）**：仍以 candidate 0 投影为内容，但每个文档先移除已有 BOS/EOS，再规范化为 `[BOS, document terminals..., EOS]`。结果 shape 应为 `(3289027,)`、dtype 应为 `uint16`。

需要边界规范化是因为 `testppl_tree` candidate-0 记录每文档只带一个边界 token：88 个文档只带 BOS，其余 4,878 个只带 EOS，总数恰为 4,966。这不是 `tree_doc_index.npy` 的文档边界错误，而是 `datatools/tokenize_testppl.py` 分别处理 88 个 FineWeb BBC `CC-MAIN-*` source split 时的组装 off-by-one：`doc_id` 从 1 开始，又对累计句数在更新前后判断 BOS/EOS，因此每个 source split 的第一篇文档保留 BOS 而没有 EOS，其后文档保留 EOS 而没有 BOS。88 个 BOS 文档正好对应 88 个 source split 的起点。直接字面投影虽然最严格，但会把这个历史组装细节带入普通 terminal LM 评测。

实际采用的安全流程是先生成独立文件，核验后再安装到 `dataset/bbc-news/{terminal,tree,tg}/test.npy`。生成器执行以下硬断言：

1. `tree_sent_index.npy.reshape(-1, 300)` 的行数为 148,836，`tree_doc_index.npy.sum()` 也为 148,836；
2. 句记录起点是前面所有句的 300 个 candidate 长度之和，不能用 candidate-0 长度单独做全局 cumsum；
3. 每句只投影 candidate 0，且根括号必须平衡、只有一个顶层树；
4. 从规范化前备份重建历史字面投影时 shape 必须为 3,284,061；当前物理数据的字面投影及标准边界版均必须为 3,289,027，且每文档恰有一个首 BOS 和一个尾 EOS；
5. 另存 manifest，记录输入三个 `.npy` 的 SHA-256、输出 SHA-256、构建参数和 6 个已知异常文档；
6. 用原 `terminal/test.npy` 的 `+59` 后缀做回归比较：规范化去除 BOS/EOS 后应有 4,960 个文档逐 token 完全相同，且异常 ID 必须精确等于 `{1063, 1090, 1121, 1122, 1550, 2626}`。

### 已生成产物

`scripts/build_testppl_aligned_test.py` 已按上述“标准文档边界版”实现并生成：

| 格式 | 正式路径 | shape | SHA-256 |
|---|---|---:|---|
| terminal | `dataset/bbc-news/terminal/test.npy` | `(3289027,)` | `c9ce654523b36bb1938fe78fd9b034bed2e5f2e48e137c81b24b4178c53df1ae` |
| Tree | `dataset/bbc-news/tree/test.npy` | `(8082269,)` | `fcf10cd1be5dc11ba4e00afc9477a27e73573792456599562d07cc62e1679339` |
| TG | `dataset/bbc-news/tg/test.npy` | `(10478890,)` | `1a40043b642876606de9e6fb2f8a392ea05a1c5a51cbc59b4f321a8897032d2f` |

每个格式同时保存 `test_sent_index.npy` 和 `test_doc_index.npy`。独立构建产物及 manifest 位于 `dataset/bbc-news/testppl_aligned/`。原 terminal/Tree 测试集分别备份为 `test.pre_testppl_alignment.npy`；TG 目录原先没有 `test.npy`。

已逐句验证 Tree 删除 non-terminal 后与 terminal 完全一致，Tree 复制 closing non-terminal 后与 TG 完全一致；三种格式的 sentence index 均完整覆盖 token stream，且 4,966 个文档均恰有一个首 BOS 和一个尾 EOS。

## 300-candidate 数据与评测代码的 BOS/EOS 规范化

2026-08-22 已进一步同时修改数据和代码，使物理数据与运行时都执行同一约束：**每个 candidate 的文档首句以 BOS 开始，文档末句以 EOS 结束**。中间句不得出现 BOS/EOS，任何记录不得出现 PAD。

### 物理数据

`scripts/normalize_testppl_document_boundaries.py --install` 对 Tree/TG 的全部 44,650,800 个候选记录进行流式改写。每种格式均保留所有原 token，仅补入缺失边界：1,463,400 个 BOS 和 26,400 个 EOS。`tree_doc_index.npy` 与 `tg_doc_index.npy` 不变，句子记录长度索引同步更新。

| 格式 | 规范化前 shape | 规范化后 shape | 规范化后 SHA-256 |
|---|---:|---:|---|
| Tree | 2,594,352,932 | 2,595,842,732 | `0eac90749f854df5e7fad016ffe4aae761f686392bedfa8a5bd3283df14489b2` |
| TG | 3,398,920,248 | 3,400,410,048 | `2011134743d36d9d92a41759ed41ddc9e534a03435ee80822b42ab746887939a` |

规范化后 Tree/TG 各有 1,489,800 个 BOS 和 1,489,800 个 EOS，即 `4,966 documents × 300 candidates`；PAD 为 0。Tree 与 TG 的逐记录长度差在规范化前后完全相同，说明改写没有碰触 TG 的 closing-nonterminal 复制结构。完整机器可读记录位于 `dataset/bbc-news/testppl_boundary_normalization.json`。

原始数组与索引保存在：

- `dataset/bbc-news/testppl_tree/tree_300.pre_bos_eos_normalization.npy`；
- `dataset/bbc-news/testppl_tree/tree_sent_index.pre_bos_eos_normalization.npy`；
- `dataset/bbc-news/testppl_tg/tg_300.pre_bos_eos_normalization.npy`；
- `dataset/bbc-news/testppl_tg/tg_sent_index.pre_bos_eos_normalization.npy`。

当前文件系统不支持 reflink，因此备份采用硬链接保存旧 inode；正式文件通过原子替换指向新 inode。不要对这些备份文件做原地写入。

### 运行时代码

`olmo/eval/downstream.py` 的 document-PPL Dataset 默认开启幂等规范化：旧数据缺边界时补齐，新数据已规范化时不重复插入；重复、错位的 BOS/EOS 或 PAD 会直接报错。sentence-PPL 默认不启用该文档边界逻辑，避免改变其既有口径。可通过 `normalize_document_boundaries` 显式覆盖默认值。

真实 Tree/TG Dataset 加载验证均通过：各 44,650,800 个 candidate 记录，运行时规范化在新数据上保持幂等。用于 perplexity 指数分母的是 **3,284,061 = 3,279,095 普通 terminal + 4,966 EOS**；BOS 只作为条件上下文，不计入预测 token 数。15 个边界单元测试覆盖补齐、幂等、一词文档、重复/错位特殊 token 和 PAD 拒绝。

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

两套数据的关系应表述为：**文档顺序以固定 `+59` 偏移对应，绝大多数内容在 document terminal 层完全一致；但 4 个位置是整篇不同的新闻，另有 2 个位置受 ADJ 字面量清理影响。**

因此，“他们是 terminal 一致的关系”在语料对应原则上是正确的，但必须附加 6 个已知异常文档的限定。对于 document-level perplexity，`testppl_tree` 的边界是规范边界，`test.npy` 是需要通过 terminal 对齐接入的来源，不能反过来直接用 `test.npy` 的全部句子边界覆盖评测边界。
