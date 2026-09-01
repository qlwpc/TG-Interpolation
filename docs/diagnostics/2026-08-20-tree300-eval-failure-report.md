# 诊断报告：tree_300 评测任务失败 (Jobs 45196 / 45195)

> [!CAUTION]
> **2026-08-31 纠错：本报告原 §1.6.2、§1.6.3 和 §1.10 夸大了 `tree_spans`
> 缺陷对 Pushdown 的影响。** 后续审计证明变化只发生在 `split`，所有已审计
> `(left,right)` 均不变。GPST job 45196 的 merge-order 崩溃与 TreeReg split 监督污染
> 成立；Pushdown 深度矩阵和 attachment gold action 不受这一代码缺陷影响。注意这不
> 否定 job 45195 的**另一项独立数据错误**（候选 terminal 分组错乱）。量化证据见
> [`../../diagnostics/results/tree_spans_contamination_summary.md`](../../diagnostics/results/tree_spans_contamination_summary.md)，
> 纠错索引见 [`../../REPOSITORY_CLEANUP_MEMORY.md`](../../REPOSITORY_CLEANUP_MEMORY.md) M-05。

**生成日期**：2026-08-20
**代码版本**：`main` 分支 (commit f076810)
**数据产物**：`dataset/testppl_tree/tree_300.npy` (2025-08-12 生成)
**诊断脚本**：`diagnostics/diag_tree300_consistency.py`、`diagnostics/diag_trace_sentence2_cand1.py`

---

## 0. 任务背景

两个 SLURM 评测任务在 2026-08-19 失败，均针对 `dataset/testppl_tree/tree_300.npy` 做 gold-tree 文档级 perplexity 评测：

| Job ID | 作业名 | 评测模型 | 状态 | 退出码 |
|--------|--------|----------|------|--------|
| 45196 | gpst-tree300-docppl | GPST (gpst-bbc-unsup) | FAILED | 1:0 |
| 45195 | pushdown-terminalonly-tree300-docppl | Pushdown (pushdown_terminalonly) | FAILED | 1:0 |

诊断结论：两个 job 是**两个不同的根因**——Job 45196 是**代码 bug**（`tree_spans` 的 `id()` 键冲突），Job 45195 是**数据 bug**（`tree_300.npy` 候选分组错乱）。

> 备注：用户最初提到的 "49251" 在集群上不存在（今日 job ID 上限 45196，SLURM ID 单调递增，`squeue` 为空），系笔误，实际为 45196 与 45195。

---

## 1. Job 45196 (GPST) — 代码 bug：`tree_spans` 的 `id()` 冲突

### 1.1 报错信息

```
Traceback (most recent call last):
  File ".../scripts/gpst/evaluate_document_ppl.py", line 113, in <module>
    main()
  File ".../scripts/gpst/evaluate_document_ppl.py", line 98, in main
    result = evaluate_gold_tree_document_ppl(
  File ".../olmo/gpst/eval/document_ppl.py", line 324, in evaluate_gold_tree_document_ppl
    nll_parts.append(_score_items(
  File ".../olmo/gpst/eval/document_ppl.py", line 232, in _score_items
    batch = _move_batch(collator(items), device)
  File ".../olmo/gpst/reader/dataset_gold.py", line 217, in __call__
    raise ValueError("grouped merge orders are not a global gap permutation")
ValueError: grouped merge orders are not a global gap permutation
```

### 1.2 缺陷位置

**文件**：`olmo/data/parse_align.py`
**函数**：`tree_spans`（第 416–484 行）
**缺陷行**：第 447、456、468、483 行

```python
447    ranges: Dict[int, Tuple[int, int]] = {}        # ← key = id(node)
...
456                ranges[id(node)] = (idx, idx)      # ← 叶子写入：被同 id 的后继叶子覆盖
...
468        child_ranges = [ranges[id(c)] for c in children]   # ← 父节点读取：可能读到错误位置
...
483            ranges[id(node)] = (left, right)       # ← 内部节点也存在同类风险
```

### 1.3 缺陷机制（逐步）

1. `tree_spans` 用迭代后序遍历，把每个节点的叶子区间存入 `ranges`，**键是 `id(node)`**（第 445–447 行注释明确说明 "keyed by id(node)"）。
2. 叶子是裸 `int`（token id）。CPython **interns `[-5, 256]` 区间的整数**，即对这些值，`id(x)` 在整个进程内全局唯一，与出现位置无关。
3. 当同一棵树里 token id ≤ 256 的叶子重复出现时（例如 token 82 出现在位置 5 和位置 9）：
   - 位置 5 处理时：`ranges[id(82)] = (5, 5)`
   - 位置 9 处理时：`ranges[id(82)] = (9, 9)` **← 覆盖了位置 5 的记录**
4. 某个父节点的左子树本应读到位置 5 的范围，却从 `ranges[id(82)]` 读到 `(9, 9)`，于是该父节点的 `split` 被算成 9（而非正确的 5），right 也可能错算。
5. 结果：一个 span 出现重复 gap（9 出现两次），另一个本应有的 gap（5）丢失。

### 1.4 复现证据（真实数据）

数据：`dataset/testppl_tree/tree_300.npy`，sentence 2 candidate 1（offsets 索引 `2*300+1`）。

#### 叶子序列与 id 冲突

```
leaves = [4492, 1839, 262, 1450, 705, 82, 290, 1466, 705, 82, 1074, ...]
                       [4]  [5]              [8]  [9]
```

| 位置 | token | id(leaves[i]) | interned? | 冲突 |
|------|-------|---------------|-----------|------|
| 5 | 82 | `140256761383696` | 是（≤256） | 与位置 9 **相同** |
| 9 | 82 | `140256761383696` | 是 | 与位置 5 **相同** |
| 4 | 705 | `140242133756816` | 否（>256） | — |
| 8 | 705 | `140242133756912` | 否 | 与位置 4 **不同**（无冲突） |

直接断言验证：
- `id(leaves[5]) == id(leaves[9])` → **True**
- `id(leaves[4]) == id(leaves[8])` → **False**

#### Binarized 树的 span 输出（buggy 版本）

关键片段（叶子 [5..9] 区域）：
```
(NP|<) l=8 split=8 r=9   # gap 8
(NP|<) l=7 split=7 r=9   # gap 7
(NP|<) l=6 split=6 r=9   # gap 6
(NP|<) l=5 split=9 r=9   ← 非法：二节点但 split==r==9（叶子 9 被重复计数）
(NP|<) l=4 split=4 r=9   # gap 4
```
`(NP|<) l=5 split=9 r=9` 是一个**不可能存在于真实二叉树中的节点**：`split=9, r=9` 意味着左子树结束于叶子 9，右子树也结束于叶子 9——叶子 9 被重复计数。这正是 id 冲突导致父节点误读左子树范围的直接后果。

#### merge_orders 对比

```
BUGGY  : [8, 7, 6, 9, 4, 3, 12, 11, 14, 13, 10, 9, 2, 18, 17, 20, 19, 16, 15, 1, 21, 0]
                                              ↑ 重复 gap 9，缺失 gap 5
sorted  : [0,1,2,3,4,6,7,8,9,9,10,11,12,13,14,15,16,17,18,19,20,21]   ← 9 重复，5 缺失
want    : [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
valid permutation? False ❌
```

#### 位置-keyed 重写版（对照组）

用每个栈帧的独立 sentinel 对象作 key（而非 `id(node)`），同一棵树：
```
STABLE : [8, 7, 6, 5, 4, 3, 12, 11, 14, 13, 10, 9, 2, 18, 17, 20, 19, 16, 15, 1, 21, 0]
                                              ↑ gap 5 恢复，无重复
valid permutation? True ✅
```

两者唯一的差异正是受 id 冲突影响的位置 [5..9] 子树，**确证根因**。

### 1.5 触发条件

满足**全部**条件即触发：

1. 叶子是 Python `int`（本仓库叶子为 token id，满足）。
2. 同一棵树内**重复出现某个 token id ≤ 256**。GPT-2 词表中 ≤256 的 token 包括高频词（"the"/"a"/"and"）、标点、空格前缀 token——在真实英文句子里**几乎必然出现**。
3. 该重复叶子的两个出现位置**位于不同子树**，且其中一个是某个二叉节点的左子树右端点。

因此该 bug **不是偶发**：对自然语言 parse 树，受影响句子的比例很高。

### 1.6 影响分析

`tree_spans` 有两个调用点：

#### 1.6.1 GPST 评测路径（显式崩溃）— Job 45196

调用链：
```
scripts/gpst/evaluate_document_ppl.py
  → olmo/gpst/eval/document_ppl.py:evaluate_gold_tree_document_ppl
  → olmo/gpst/reader/dataset_gold.py:tree_to_merge_orders  (line 56)
  → olmo/data/parse_align.py:tree_spans                      (line 416)
```
`tree_to_merge_orders`（`dataset_gold.py:56`）调用 `tree_spans(bin_tree)` 取 spans，再 `merge_orders = [split for (l,split,r) in spans if l != r]`。由于 buggy spans 含重复 gap，得到的 `merge_orders` 不是 `range(L-1)` 的置换。

下游 `GoldTreeCollator.__call__`（`dataset_gold.py:216`）校验 `sorted(global_orders) != list(range(width))` 失败，抛出：
```
ValueError: grouped merge orders are not a global gap permutation
```
**job 45196 实际崩溃点**：sentence 2, candidate 1。

#### 1.6.2 Pushdown / TreeReg 训练 + 评测路径（原结论部分错误，已纠正）

调用链：
```
olmo/data/parse_align.py:_binarize_segments  (line 557)
  → tree_spans(btree)
  → 产出的 (leaves, spans) 进入 ParseAlignedDataset / 下游
```
`_binarize_segments`（第 549–561 行）对每个 tree segment 调 `tree_spans`，结果 `spans` 被打包进数据项的 `tree_spans` 字段，一路流向：

| 消费者 | 用法 | 受影响？ |
|--------|------|----------|
| `compute_depth_matrix` / `compute_depth_matrix_gpu` | pushdown 深度偏置，仅用 `(left, right)` | **否**：后续审计确认 `(left, right)` 不变 |
| TreeReg CE loss | 用 `split` 作二分类目标 | **是**：错误 `split` → 错误监督信号 |
| `derive_gold_attachment_actions` | attachment head 的 gold 动作，只用 `(left, right)` | **否**：`split` 被忽略 |

**纠正后的关键结论**：terminals（`input_ids`）正确，span 的 `left/right` 也保持不变；
只有部分 `split` 错误。因此该缺陷是 **TreeReg 监督中的隐性污染**，不是 Pushdown
结构输入污染。完整 dev/test 与两份独立 train 抽样中，所有差异均只出现在 `split`；
train 抽样估计受污染 TreeReg decision 为 `0.027710%`。

#### 1.6.3 影响范围总结

- GPST gold-tree PPL 评测：**崩溃**（job 45196）。
- TreeReg：消费 `split` 的监督信号受到稀疏污染；影响比例见污染审计。
- Pushdown：深度偏置与 attachment action 只消费 `(left,right)`，**不受本代码缺陷影响**。
- Pushdown job 45195 仍受 §2 的候选 terminal 分组数据错误影响；它与本节代码缺陷不是
  同一根因。
- 该 bug 在 `tree_300.npy`（2025-08-12 生成）的数据上是可复现的，说明**自该函数引入以来一直存在**。

### 1.7 为什么 `collapse_unary_tree` / `binarize_tree` 不受影响

这两个函数（第 288、327 行）同样用 `id()` 作 `result` 字典的键，但它们对叶子存的是**叶子自身**（identity 值）：

```python
# collapse_unary_tree, line 310
result[id(current)] = current          # 叶子: result[id(82)] = 82 (值相同, 覆盖无害)
# binarize_tree, line 376
result[id(node_i)] = node_i            # 同理
```

对叶子，覆盖前后**值完全相同**（82 还是 82），所以 id 冲突无害。只有 `tree_spans` 存的是**位置依赖值** `(idx, idx)`，位置随出现而变，覆盖才会引入错误。因此**唯一需要修复的是 `tree_spans`**。

### 1.8 修复方案

**核心**：把 `tree_spans` 的遍历 key 从 `id(node)`（对象标识，对 interned 小整数会碰撞）改为 **occurrence-unique sentinel**（每次入栈用一个独立对象，保证全局唯一）。

#### 方案 A（推荐）：栈帧 sentinel

每个栈帧自带一个唯一的 frame 对象，`ranges` 以 frame 的 `id()` 为键。已用此法在真实数据上验证为合法置换（§1.4）。

伪代码要点：
```python
ranges: Dict[int, Tuple[int, int]] = {}
stack: List[Tuple] = [("node", tree, None)]   # 第三项 = frame
while stack:
    phase, node, frame = stack.pop()
    if phase == "node":
        my_frame = object()                   # ← 每个节点一个独立 sentinel
        if isinstance(node, int):
            idx = len(leaves); leaves.append(node)
            ranges[id(my_frame)] = (idx, idx) # ← key = frame, 不再是 id(node)
            continue
        label, children = node
        left = len(leaves)
        stack.append(("ready", node, my_frame, left))
        child_frames = []
        for c in reversed(children):
            cf = object()
            child_frames.append(cf)
            stack.append(("node", c, cf))
        # 记住 child_frames 供 ready 阶段读取（挂到 ready entry）
    else:  # 'ready'
        _, node, my_frame, left, child_frames = ...
        child_ranges = [ranges[id(cf)] for cf in child_frames]  # ← 读 frame key
        ...
        ranges[id(my_frame)] = (left, right)
```

#### 方案 B（更简单）：递归 + 位置传递

`tree_spans` 当初改迭代是为了避免深树 `RecursionError`（第 440–446 行注释）。若保留迭代，方案 A 最稳；若能确认评测期树深可控，可用纯递归版本（位置天然由调用栈传递，无需 dict）。但训练 split 树很深，**不建议**放弃迭代。

### 1.9 验证要求

修复后必须通过：
1. `tests/test_parse_align.py`（含 `test_tree_spans_*` 系列，第 163–276 行）。
2. 新增回归测试：构造一棵含重复 token id ≤ 256 叶子的树，断言 `merge_orders` 是 `range(L-1)` 的置换。
3. 在 `tree_300.npy` sentence 2 candidate 1 上重跑诊断脚本，确认输出与 §1.4 的 STABLE 版一致。

### 1.10 风险与注意事项

- **修复仅限 `tree_spans`**，不要动 `collapse_unary_tree` / `binarize_tree`（它们无害，改了反而可能引入回归）。
- 修复会改变 TreeReg 使用的部分 `split` 监督；已训练 TreeReg checkpoint 的污染程度应按
  `tree_spans_contamination_summary.md` 解释。它不会改变已审计的 `(left,right)`，因此
  不改变 Pushdown 深度偏置或 attachment action 语义。

## 附录：关键文件索引

| 文件 | 相关内容 |
|------|----------|
| `olmo/data/parse_align.py:416-484` | `tree_spans`（缺陷函数） |
| `olmo/data/parse_align.py:540-561` | `_binarize_segments`（pushdown/TreeReg 调用点） |
| `olmo/data/parse_align.py:288-324` | `collapse_unary_tree`（不受影响） |
| `olmo/data/parse_align.py:327-410` | `binarize_tree`（不受影响） |
| `olmo/gpst/reader/dataset_gold.py:37-61` | `tree_to_merge_orders`（GPST 调用点） |
| `olmo/gpst/reader/dataset_gold.py:148-226` | `GoldTreeCollator`（Job 45196 崩溃点） |
| `olmo/gpst/eval/document_ppl.py` | GPST 文档 PPL 评测 |
| `olmo/eval/pushdown_document_ppl.py:62-106` | `PushdownGold300Corpus`（Job 45195 崩溃点） |
| `scripts/gpst/evaluate_document_ppl.py` | GPST 评测入口 |
| `scripts/evaluate_pushdown_document_ppl.py` | pushdown 评测入口 |
| `datatools/gen_tgppl_fromtree.py` | `tree_300.npy` 生成脚本 |
| `tests/test_parse_align.py:161-276` | `tree_spans` 现有测试 |
| `diagnostics/diag_tree300_consistency.py` | 复现诊断脚本 |
| `diagnostics/diag_trace_sentence2_cand1.py` | 追踪诊断脚本 |
