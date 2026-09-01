# Evaluation 工作流程文档

> [!WARNING]
> 本文是 2026-08-23 的个人实现快照，`file:line` 随代码演进已经漂移，也不包含之后
> Tree-Shuffle checkpoint、Pause XSum v2、Pushdown fixed-word-atom v1/v2 等完整纠错。
> 当前实验状态先查 [`REPOSITORY_CLEANUP_MEMORY.md`](REPOSITORY_CLEANUP_MEMORY.md) 和
> [`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md)；本文主要用于
> 理解历史 evaluator 分发，不应单独作为数值或协议依据。

**代码库**: /home/wangpch/TG-Interpolation (OLMo-based)
**更新日期**: 2026-08-23
**用途**: 个人参考文档，记录 evaluation 系统如何运作，含 file:line 引用和关键数值。

---

## 1. Evaluator 类型与分发

### EvaluatorType 枚举 (`olmo/config.py:746-752`)
```python
class EvaluatorType(StrEnum):
    downstream = "downstream"      # ICL 多选 (boolq/cb/copa 等)
    lm = "lm"                      # legacy token-stream PPL
    tg_doc = "tg_doc"              # TG 格式文档级困惑度
    tg_sent = "tg_sent"            # TG 格式句子级困惑度
    terminal_doc = "terminal_doc"  # terminal/pause 文档级 PPL
    rouge = "rouge"                # 摘要 ROUGE (xsum)
    beam_search_icl = "beam_search_icl"
```

### 三处分发

**1. 构造分发** (`olmo/eval/__init__.py:166-196`): `build_evaluator` 按 `eval_config.type` 分发
- `{tg_doc, terminal_doc, tg_sent, downstream, rouge, beam_search_icl}` → `build_downstream_evaluator`
- `lm` → `build_eval_dataloader` + `MeanMetric`
- 其他 → `ValueError`

**2. Metric 类分发** (`olmo/eval/__init__.py:90-143`): `build_downstream_evaluator` 内按 type/label 选 metric
- `tg_sent` → `TGPerplexitySentenceLevelMetric`
- `tg_doc` → `TGPerplexityDocumentLevelMetric`
- `terminal_doc` → `TerminalDocumentPerplexityMetric`（FP64 NLL，严格检查每句恰好一次）
- label `"syntactic_generalization"` → `SyntacticGeneralizationMetric`
- label `"BLiMP"` → `BLiMPMetric`
- type `rouge` → `RogueMetric`
- type `beam_search_icl` → `BeamSearchICLMetric`
- label 以 `"_decomp"` 结尾 → `DecomposedICLMetric`
- else → `ICLMetric`

**3. 运行时分发** (`olmo/train.py:1301-1311`): `eval()` 方法按 batch 分发到不同 eval_step
```python
if evaluator.type in (EvaluatorType.tg_doc, EvaluatorType.terminal_doc):
    self.TG_doc_eval_step(eval_batch, evaluator)
elif evaluator.label == "syntactic_generalization":   # SG 按 label 分发 (type 是 downstream)
    self.SG_eval_step(eval_batch, evaluator)
elif evaluator.type == EvaluatorType.rouge:
    self.summarization_eval_step(eval_batch, evaluator)
elif evaluator.type == EvaluatorType.beam_search_icl:
    self.beam_search_icl_eval_step(eval_batch, evaluator)
else:   # lm, tg_sent, downstream (ICL 多选)
    self.eval_step(eval_batch, evaluator)
```
注意: SG 按 **label** 分发 (其 type 是 `downstream`)。catch-all `eval_step` 处理 `lm`、`tg_sent`、`downstream`。

### 任务 → Evaluator 映射总表

| Task | EvaluatorConfig label | Type | Metric 类 | Dataset 类 |
|------|----------------------|------|-----------|------------|
| pretrain | TG-ppl-validation | lm | MeanMetric | MemMapDataset |
| docppl (terminal/pause，当前标准) | terminal_doc_ppl | terminal_doc | TerminalDocumentPerplexityMetric | TerminalDocumentPerplexityDataset |
| docppl (terminal/pause，旧口径) | TG-ppl-validation + -test | lm | MeanMetric | MemMapDataset |
| docppl (tg/tree) | tg_approx_doc / txl_approx_doc | tg_doc | TGPerplexityDocumentLevelMetric | TGPerplexityApproximationDataset |
| boolq/cb/copa/... | {task} | downstream | ICLMetric | ICLMultiChoiceTaskDataset 子类 |
| SG | syntactic_generalization | downstream | SyntacticGeneralizationMetric | SGDataset |
| blimp | BLiMP | downstream | BLiMPMetric | BLiMPApproximationDataset |
| xsum | xsum | rouge | RougeMetric | XsumDataset |

---

## 2. Eval 触发机制

### eval_on_load (`olmo/train.py:1452-1455`)
`fit()` 开始时，训练前:
```python
if self.cfg.eval_on_load:
    eval_metrics = self.eval()
    wandb.log(eval_metrics, step=self.global_step)
```
在 `global_step=0` (或恢复的 step) 运行 eval。之后 line 1458 设回 `train()` 模式。

### stop_at 钳制 (`olmo/train.py:1388-1395, 1476`)
```python
# stop_after 可设定相对步数
if self.cfg.stop_after is not None:
    self.cfg.stop_at = min(self.cfg.stop_at, self.global_step + self.cfg.stop_after) if self.cfg.stop_at else self.global_step + self.cfg.stop_after
if self.cfg.stop_at is None:
    self.cfg.stop_at = self.max_steps + 10
# line 1476: 钳制到 max_steps
stop_at = self.cfg.stop_at if self.cfg.stop_at <= self.max_steps else self.max_steps
```

### 训练中 eval 触发 (`olmo/train.py:1603-1616`)
```python
if not cancel_initiated and (
    self.global_step % self.cfg.eval_interval == 0 or self.global_step >= stop_at
):
    eval_metrics = self.eval()
    ...
    self.dist_model.train()   # eval 后切回训练模式
```
触发条件: `global_step` 是 `eval_interval` 的倍数 **或** `global_step >= stop_at` (训练结束)。

### 训练循环退出 (`olmo/train.py:1623-1624`)
`if self.global_step >= stop_at: break`

### 最终 checkpoint (`olmo/train.py:1652`)
`if save_checkpoints and not self.cfg.eval_no_save:` — `eval_no_save` 抑制保存 (test_only 运行用)。

---

## 3. test_only vs finetune 参数

定义于 `scripts/init_cfg_and_sbatch.py:122-123`:
```python
test_only_params = {
    "eval_on_load": True, "eval_no_save": True, "max_duration": 0,
    "reset_optimizer_state": True, "reset_trainer_state": True,
}
finetune_params = {"reset_optimizer_state": True, "reset_trainer_state": True, "eval_interval": 1000000}
```

### test_only_params (用于 docppl, xsum_test, blimp, SG, hellaswag, winogrande)
- `eval_on_load=True`: 训练前在 step 0 eval
- `eval_no_save=True`: 不保存 checkpoint
- `max_duration=0`: 0 训练步，eval 通过 `eval_on_load` 触发
- `reset_optimizer_state=True` / `reset_trainer_state=True`: 恢复时只加载
  `model.pt`，不读取 `optim.pt` 或 `train.pt`
- **行为**: 加载 model-only checkpoint → step 0 eval → 无训练 → 无保存

### finetune_params (用于 boolq, cb, copa, multirc, record, rte, wic, wsc, xsum_finetune)
- `reset_optimizer_state=True`: 清空 Adam 动量，finetune 从零开始优化器状态
- `reset_trainer_state=True`: 清空 trainer 状态 (步数计数器等)
- `eval_interval=1000000`: 训练中禁用周期性 eval (1M 步远超任何运行)
- **行为**: 加载 checkpoint → 重置优化器/trainer → 训练 N epochs → 通过 `global_step >= stop_at` 触发 eval → 保存

### xsum_finetune 特殊 (`scripts/init_cfg_and_sbatch.py:130-131`)
额外设 `eval_interval: 1000000` + `device_eval_batch_size: 1` + finetune_params。

---

## 4. document-level PPL（当前可比口径）

`task="docppl"` 对 terminal 和所有 pause grammar 现在统一产生：

```python
EvaluatorConfig(
    label="terminal_doc_ppl", type=EvaluatorType.terminal_doc,
    device_eval_batch_size=1,
)
```

它读取 `dataset/bbc-news/terminal/{test.npy,test_sent_index.npy,test_doc_index.npy}`，一条路径一条句子地延续 KV cache；每篇文章从 BOS 开始、以 EOS 结束。BOS 只提供上下文，所有普通 terminal 与 EOS 都计分。当前数据集固定为 **148,836 句、4,966 篇文档、3,284,061 个计分 token**。

pause 模型按其 grammar 在同一文档内连续插入 pause（下一篇文档相位重置），并以 `label_mask` 排除插入位置；因此分子和分母与普通 terminal 完全对齐，能直接比较 PPL。`terminal_doc` 强制 batch=1、num_workers=0，防止乱序破坏 KV cache。

句子间首 token 使用上一句冻结的末位分布计分；该修正不能用当前句缓存提交后的分布覆盖。

### 本机完整运行（2026-08-23）

| 模型 | checkpoint | Slurm job | 状态 |
|---|---|---:|---|
| terminal | `saved_models/Terminal-lr005-bs144/step34115-unsharded` | 3444 | 完成：PPL 9.88981（148,836 句） |
| pause1 | `saved_models/pretrain_pause1_100M/step45487-unsharded` | 3445 | 完成：PPL 9.83125（148,836 句） |

评测脚本必须显式进入固定的 `LLM` conda 环境，避免非交互 Slurm shell 缺少
`torchrun`。失败任务及其编号不作为实验记录保留。

生成的可复现配置、脚本和日志在 `artifacts/evaluation/terminal_doc_ppl_20260823/{terminal,pause1}/`。结果从各自 `logs/slurm-*.out` 的 `eval/downstream/terminal_doc_ppl_doc_ppl` 读取；两次运行均完成于 2026-08-23。

### 旧的 lm 路径（仅保留作历史对照）

两条路径取决于 `input_format`:

### 路径 A: terminal-input docppl (`scripts/init_cfg_and_sbatch.py:284-291`)
当 `task=="docppl" and input_format=="terminal"` (或 `task[:8]=="pretrain"`):
```python
Evallist = [EvaluatorConfig(label="TG-ppl-validation"), EvaluatorConfig(label="TG-ppl-validation-test")]
Evallist[0].data.paths = [dev.npy]
Evallist[1].data.paths = [test.npy]
Evallist[0].data.generate_doc_lengths = Evallist[1].data.generate_doc_lengths = (input_format!="tg")
grammar_type = cfg.model.transformer_grammar_type
cfg.model.flex_attention = (
    grammar_type == "tg"
    or grammar_type.startswith("tgproximal")
    or grammar_type.startswith("tgnomask")
)
```
- type 默认 `lm` → `build_eval_dataloader` → `MemMapDataset`
- `generate_doc_lengths=True` (非 tg 格式): `MemMapDataset` 预计算文档边界长度
- TG/tgproximal/tgnomask 才允许进入长度感知路由：短/动态结构化输入走
  SDPA，达到评测阈值的重复长序列才走 Flex；非 128 整数倍的 Flex 形状
  自动补齐。mixing 在真实逐头基准完成前保持关闭
- terminal/tree 格式不生成 TG bias，保持 `flex_attention=False`，文档边界
  路径使用 `flash_attn_varlen_func`
- `MeanMetric` (`olmo/eval/__init__.py:177-178`) 累积每实例 CE loss
- `compute_metrics` (`olmo/eval/evaluator.py:51-77`) 输出 `eval/{label}/CrossEntropyLoss` 和 `Perplexity = exp(CE)`

### 路径 B: TG-format docppl (`scripts/init_cfg_and_sbatch.py:292-293`)
当 `task=="docppl"` 但 `input_format != "terminal"`:
- label 改为 `"tg_approx_doc"` (tg) 或 `"txl_approx_doc"` (tree)
- type `tg_doc`, `device_eval_batch_size=60`
- `TGPerplexityDocumentLevelMetric` (`downstream.py:1090-1129`): 单设备 (`sync_on_compute=False`)，`loglikelihoods` buffer 形状 `(dataset_length // samples_per_sent, samples_per_sent=300)`
- `compute`: `ppl = exp(-logsumexp(-loglikelihoods, dim=1).sum() / data_numwords)`，`data_numwords = sum(term_length)` (term_length 只数 terminal token, `downstream.py:1320`)

### TG_doc_eval_step (`train.py:979-1054`)
顺序处理文档 + KV cache，每 300 token 快照 KV cache (`doc_kv_cache`)。CE 在所有位置计算 (line 1045-1048)，减去前一块边界的首 token log-prob 修正 (line 1049-1050)。

### 旧 lm 路径的 pause 稀释
- `TG-ppl-validation[-test]` 的 `lm` 路径仍会把 pause 展开位置纳入平均 CE，不能与 terminal 或 `terminal_doc` PPL 比较。
- 不要用简单的展开倍数换算其 PPL；新的 `terminal_doc_ppl` 是唯一报告用的 terminal/pause 文档 PPL。

---

## 5. 下游 ICL 多选 (boolq/cb/copa 等)

### ICLMetric.update (`downstream.py:60-167`)
对 batch 中每个样本，在 `[ctx_len-1 : ctx_len+cont_len-1]` 位置收集 continuation log-likelihood (line 71-73):
```python
lm_cont_logits = lm_logits[idx][ctx_len-1 : ctx_len+cont_len-1]
lm_log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens).squeeze()
```
对 `metric_type=="acc"` (line 111-113): `log_likelihood = lm_log_likelihood.sum()`

### ICLMetric.compute (`downstream.py:169-248`)
按 `doc_id` 分组 log-likelihood，填 `[num_continuations]` 张量；对 acc: `correct = argmax(loglikelihoods) in label_dict[doc_id]` (line 221)。预测 = 总 log-likelihood 最高的 continuation。返回 per-group 准确率 + `"_"` 总平均。

### prep_examples 流程 (`downstream.py:554-709`)
1. `continuations = self.doc_to_continuations(doc)` — continuation 字符串列表 (boolq: `[" yes", " no"]`)
2. `ctx = self.token_encode(doc_text)` — `encode_TG_string` + `convert_grammar_input` (line 846-849)
3. `continuation = self.token_encode(continuation_str)` (line 577)
4. `query = ctx + continuation` (line 582)
5. `query = query[-model_ctx_len:]` — 左截断 (line 589)
6. `query = self.convert_grammar_input(query)` (line 590)
7. `full_query = query` — 保存未截断副本 (line 599)
8. **split != "train" 截断**: `query = query[:-1]` (line 602) — 丢最后一个 token 让模型预测它
9. `actual_ctx_len = len(query) - len(continuation) + 1` (line 606) — `+1` 补偿 `[:-1]` 截断

### PAUSE 路径 (line 620-673，当 `self.ispause` 为真)
- `ctx_real = len(full_query) - len(continuation)` (line 654) — **不用** `actual_ctx_len` (它带 `+1` 偏移)
- `query = pause_input_ids(full_query, self.pause_token_id, pause_num=gtype)` (line 656) — 展开**完整** ctx+cont
- `split = pause_expanded_len(ctx_real, p, q)` (line 658) — 第一个 continuation real token 的展开位置
- `trim = pause_trailing_trim(ctx_real + cont_real, p, q)` (line 659) — 末尾 pause
- `continuation = query[split : len(query) - trim]` (line 660) — 评分的 continuation 是展开 query 的连续切片
- `actual_ctx_len = split` (line 661) — 覆盖，使 ICLMetric.update 在正确展开位置收集

### shots_num
- BoolQ 默认 `shots_num=3` (`downstream.py:2018`)
- 当 `train_config.finetune_task is not None`，`build_downstream_evaluator` 强制 `shots_num=0` (`olmo/eval/__init__.py:54-55`)
- shots 仅在 `shots_num != 0 and split != "train"` 时准备 (line 488)。finetune 时 split="train" 且 shots_num=0，无 shots。

---

## 6. BLiMP

### BLiMPApproximationDataset (`downstream.py:1800-1991`)
加载 `blimp_{dataset_name}.npy`，`dataset_name`:
- `"terminal"` — terminal/pause 类型
- `"tree_300"` — tree 类型
- `"tg_300"` — TG 类型 (lines 1852-1857)

### SENT_SIZE (`downstream.py:1835-1847`)
- terminal/pause: `SENT_SIZE = 1` (300 棵树转换成相同 terminal 序列，每句 1 样本)
- tree/tg: `SENT_SIZE = samples_per_sent = 300`

### TASK_SIZE (line 1848)
`TASK_SIZE = 2 * pair_per_task(1000) * SENT_SIZE`
- terminal/pause: 2000
- tree/tg: 600000

### device_eval_batch_size (`scripts/init_cfg_and_sbatch.py:294-298`)
- terminal/pause (`[:8]=="terminal"` 或 `[:5]=="pause"`): 100
- 其他 (tree/tg): 150

### Pause 展开 (`downstream.py:1927-1930`)
`__getitem__` 中 `self.ispause` 时:
```python
input_ids = pause_input_ids(input_ids, self.pause_token_id, pause_num=self.transformer_grammar_type)
```
BLiMP 句子在数据加载时 pause 展开，CE 在所有展开位置计算。

### BLiMPMetric (`downstream.py:1669-1797`)
- `loglikelihoods` buffer: `SENT_SIZE==1` → 1D `(dataset_length,)`；`SENT_SIZE>1` → 2D `(dataset_length//SENT_SIZE, SENT_SIZE)`
- `update` (line 1712): 所有位置 CE (左移 labels)，可选 mask 非 terminal (`tree_eval_type=="terminal"`)，存每句 CE
- `compute` (line 1759): 
  - `SENT_SIZE>1`: `logsumexp(-loglikelihoods, dim=1)` (tree-marginalized) 或 `-mean(...)` (terminal eval)
  - `SENT_SIZE==1`: `-loglikelihoods` (负 CE = log-prob)
  - 对 67 个 task pair (每 task 1000 pair) 比较 `p_good > p_bad`，返回 per-task/category 准确率 + `overall/overall`

### 整除断言 (`olmo/eval/__init__.py:145-149`)
`SENT_SIZE % eval_batch_size == 0 or TASK_SIZE % eval_batch_size == 0`

---

## 7. SG (Syntactic Generalization)

### SG_eval_step (`train.py:1056-1096`)
- terminal/pause 模型 (`grammar_type[:8]=="terminal" or ispause`, line 1067): 前向 `model_forward`，`score = sum(ce_loss[0] * tag[0])` (line 1071) — tag mask 选择贡献 surprisal 的位置
- tree/tg 模型: `word_sync_beam_search` (line 1077-1090)，`beam_size=samples_per_sent=300`，`nc=max(int(sg_nc_ratio*term_len), 5)`，`max_len=max(3*term_len, 10)`，`pc=sg_pc=3`

### prep_examples pause 处理 (`downstream.py:1575-1609`)
```python
if self.ispause:
    if is_gpt2:
        sent["tag"][0] = [0] + sent["tag"][0]   # BOS 前置
    sent["tag"][0] = pause_input_ids(sent["tag"][0], pause_token_id=None, pause_num=gtype)[1:]
    sent_paused = pause_input_ids(sent["input_ids"][0], self.pause_token_id, pause_num=gtype)
    sent["input_ids"] = sent_paused.unsqueeze(0)
```
- tag (0/1 mask) 用 `pause_token_id=None` (广播模式) 展开，每个 real token 的 tag 值重复到其 pause 槽
- 前导 BOS tag 项 `[1:]` 去掉
- input_ids 用实际 pause token 展开
- `score = sum(ce_loss * tag)` 只在 real-token 位置求和 (pause 位置 tag 值 = 前一 real 的值，但模型预测 pause 容易，贡献可忽略；tag 展开保证对齐)

### SyntacticGeneralizationMetric (`downstream.py:1466-1519`)
6 类 (`test_suite_dict`, lines 1455-1464):
1. **Agreement**: number_orc, number_prep, number_src
2. **Center_Embedding**: center_embed, center_embed_mod
3. **Garden_Path_Effects**: mvrr, mvrr_mod, npz_ambig, npz_ambig_mod, npz_obj, npz_obj_mod
4. **Gross_Syntactic_Expectation**: subordination, subordination_orc-orc, subordination_pp-pp, subordination_src-src
5. **Licensing**: npi_orc_any/ever, npi_src_any/ever, reflexive_orc/prep/src_fem/masc
6. **Long_Distance_Dependencies**: fgd_subject/object/pp, fgd-embed3/4, fgd_hierarchy, cleft, cleft_modifier

- `update` (line 1486): 从 `formula_dict` 查公式，代入 condition surprisal，求值 (如 `[ (%plaus%) ] < [ (%implaus%) ]`)，布尔结果加入类别列表
- `compute` (line 1506): 返回 per-category 准确率 + `avg` (排除 `nn-nv-rpl`)

### 关键数值
- 32 tasks in `task_list`，6 categories
- `samples_per_sent=300` (beam_size 默认)，`sg_nc_ratio=1.0`，`sg_pc=3`

---

## 8. xsum (rouge)

### summarization_eval_step (`train.py:1157-1191`)
- **terminal 模型** (line 1160-1168): `self.dist_model.module.generate(input_ids, max_steps=MAX_SUMMARY_LENGTH=150, beam_size=6)`，标准自回归生成，`predictions[:, 0, :]` 取最佳 beam
- **非 terminal 模型** (line 1169-1186): `word_sync_beam_search`，`beam_size=6`，`max_length=150`，`max_word_steps=75`，预测 = `predictions[0]["input_ids"]`
  - **Pause 修复** (line 1182-1184): 若 `grammar_type[:5]=="pause"`:
    ```python
    p, q = self.cfg.model.pause_spec
    predictions = extract_real_tokens(predictions, p, q, skip_first=True)
    ```
    去除生成的 pause token，`skip_first=True` 丢前导 BOS
  - 然后 `predictions = vocab.convert_treenpy_to_terminal(predictions)` (line 1185) 转 terminal token

### RougeMetric
- `update` (`downstream.py:1057-1066`): 打印 `<New Passage>: {passage} {prediction}` (line 1062)，存 prediction 和 reference token id
- `compute` (`downstream.py:1072-1088`): HuggingFace `evaluate.load('rouge')`，`rouge_types=['rouge1','rouge2','rougeL']`，`use_stemmer=True`，`use_aggregator=True`，返回 rouge1/rouge2/rougeL + `R-AVG` (三者均值)

### extract_real_tokens (`olmo/data/util.py:456-487`)
反转 `pause_input_ids` (pause_token_id=None 版)。遍历 real-token 索引 `j=0,1,2,...`，emit `paused[j + (j//q)*p]`。`skip_first=True` 从 `j=1` 开始 (丢 BOS real token)。

### 关键数值
- `MAX_SUMMARY_LENGTH=150`，`beam_size=6`
- xsum test set: 11,333 句 (慢的 beam-search eval ~6h on 4 GPU)
- xsum_finetune: 3 epoch 训练 + eval

---

## 9. BoolQ Eval Bug (已修复)

### 根因
bug 在 `ICLMultiChoiceTaskDataset.prep_examples` 的 pause 路径。BoolQ finetune eval 用 `split="train"` (finetune task 构造设定)，所以 `query[:-1]` 截断 (line 602) **被跳过** (条件是 `if self.split != "train"`)。但 `actual_ctx_len` (line 606) 仍按 `len(query) - len(continuation) + 1` 计算，带 `+1` 偏移 (本为补偿 `[:-1]` 截断，但该截断未运行)。

pause 路径旧代码用 `actual_ctx_len` (对 train split = `len(ctx)+1`) 作为 `ctx_real`。BoolQ continuation 是单 token (`" yes"`/`" no"`)，`full_query = ctx + continuation` 的 `len = len(ctx)+1`。buggy `ctx_real = actual_ctx_len = len(ctx)+1 = len(full_query)`，使展开 split 点 `pause_expanded_len(ctx_real, p, q)` 落在展开 query 末尾或之后 → continuation 切片 `query[split:end]` 为空 `[]`。

空 continuation → 两选项 ("yes"/"no") 都得 `log_likelihood = 0.0` (零元素求和) → `torch.argmax` 平局返回 0 ("yes") → BoolQ 多数类 ~62% "yes" → 始终预测 "yes" → 准确率 **0.6217**。

### 修复 (`downstream.py:654`)
```python
ctx_real = len(full_query) - len(continuation)
```
正确计算 `full_query = ctx + continuation` 中真实 ctx token 数，无 `+1` 偏移。注释 (lines 646-653) 说明为何不能用 `actual_ctx_len`。同时展开 `full_query` (未截断 ctx+cont) 而非 trimmed `query`，使 continuation token 保留在展开序列中。`actual_ctx_len = split` (line 661) 覆盖为正确展开 split 点。

### 验证
- pause1 boolq: 0.6217 → 0.6780
- 未微调 pause1 boolq: 0.6217 → 0.4413 (eval 不再退化)
- terminal boolq: 0.6835 (从未受影响)
- 所有 pause 模型 boolq 重跑后均在 0.67-0.69

### Terminal (非 pause) 不受影响
pause 路径 (line 620) 仅当 `self.ispause` 为真时执行。terminal `pause_spec=(0,1)`，`ispause=0`，走标准路径 `actual_ctx_len = len(query) - len(continuation) + 1` (正确)。

---

## 10. 配置生成 (init_cfg_and_sbatch.py)

### Evaltasks dict (`scripts/init_cfg_and_sbatch.py:182-199`)
```python
"pretrain": [EvaluatorConfig(label="TG-ppl-validation")]                    # lm
"docppl": [EvaluatorConfig(label="tg_approx_doc", type=tg_doc, device_eval_batch_size=60)]
"xsum_test"/"xsum_finetune": [EvaluatorConfig(label="xsum", type=rouge)]
"SG": [EvaluatorConfig(label="syntactic_generalization", type=downstream)]
"blimp": [EvaluatorConfig(label="BLiMP", type=downstream, device_eval_batch_size=100)]
"boolq"/"cb"/...: [EvaluatorConfig(label="{task}", type=downstream)]
"hellaswag"/"winogrande": [EvaluatorConfig(label="{task}", type=downstream, device_eval_batch_size=5)]
```

### generate_config (`scripts/init_cfg_and_sbatch.py:258-312`)
加载目标 checkpoint 的 `config.yaml` 作为**模型结构基线** → 应用
`Models[modelname]` + `train_params[task]` 覆盖 → 从 `INPUTFORMAT` 解析
`input_format`。随后重建 workspace、tokenizer、训练占位数据、load path、
evaluator 与状态控制；不继承 checkpoint 的数据 shards、`try_load_latest_save`、
`stop_at` 或 optimizer/trainer state。

这避免了 1B checkpoint 的 FineWeb-Edu shard 列表污染 BBC News 测评，也使仅有
`model.pt` 的 checkpoint（例如已知的 Tree-Shuffle）能够进行 test-only 评测。

### INPUTFORMAT 解析 (lines 70-77, 276-280)
```python
"terminal": ["terminal", "pause1", "pause1_label", "pause1/2", "pause1/2_label", "pause2", "pause3", "pause1/3", "pause1/4"]
"tree": ["tree", "tree_shuffle", "tree_shuffle_mask"]
"tg": ["tg", "mixing", "tgnomask", "tgnomask_aug", "tgtree"]
"tree_compress", "tree_noont", "tree_triplecnt": 各自独立
```
决定数据路径: `train_path = {workspace}/dataset/bbc-news/{input_format}/train.npy` (line 281)。
**注意**: 新 grammar type 必须加入对应 INPUTFORMAT 列表，否则 `input_format=None` → 数据路径 `bbc-news/None` 失败。

### docppl/terminal-input 分支
当 `task=="docppl" and input_format=="terminal"`（含 pause）时，创建唯一的 `terminal_doc_ppl` / `terminal_doc` evaluator，batch=1。`pretrain*` 仍保留旧的 `TG-ppl-validation` lm 监控，不应当作上述正式比较指标。

### docppl 非 terminal 分支 (lines 292-293)
`task=="docppl"` 但 `input_format != "terminal"`: label 改 `tg_approx_doc` (tg) 或 `txl_approx_doc` (tree)，type `tg_doc`，batch 60。

### blimp batch size 分支 (lines 294-298)
```python
if grammar_type[:8]=="terminal" or grammar_type[:5]=="pause":
    device_eval_batch_size = 100
else:
    device_eval_batch_size = 150
```

### Tree-Shuffle terminal syntax protocol

Tree-Shuffle 的 SG/BLiMP 若要与普通 terminal 模型比较，必须显式设置：

```yaml
structure_mode: terminal
tree_eval_type: terminal
beam_search: false
```

代码分支也已硬编码支持该协议：SG 在 `structure_mode=terminal` 时走
teacher-forced causal logits，不调用 `word_sync_beam_search`；BLiMP 通过
`force_terminal` 使用 terminal 数据且 `SENT_SIZE=1`，不加载 gold300。

最后 `cfg.evaluators = Evaltasks[task]` (line 306)。

---

## 关键数值速查

| 项目 | 值 |
|------|-----|
| BLiMP pair_per_task | 1000 |
| BLiMP SENT_SIZE | 1 (terminal/pause) / 300 (tree/tg) |
| BLiMP TASK_SIZE | 2000 / 600000 |
| BLiMP task 数 | 67 |
| BLiMP category 数 | 12 |
| SG task 数 | 32 |
| SG category 数 | 6 |
| SG samples_per_sent (beam_size) | 300 |
| SG sg_nc_ratio | 1.0 |
| SG sg_pc | 3 |
| docppl tg_doc SENT_SIZE | 300 |
| docppl tg_doc batch | 60 |
| terminal_doc batch / workers | 1 / 0 |
| terminal_doc 测试集 | 148,836 句 / 4,966 文档 / 3,284,061 tokens |
| xsum MAX_SUMMARY_LENGTH | 150 |
| xsum beam_size | 6 |
| xsum test set | 11,332 句 |
| BoolQ shots_num 默认 | 3 |
| BoolQ shots_list | [0,1,7,11,3,4,5] |
| BoolQ metric_type | acc |
| BoolQ split 默认 | val |
| BoolQ 多数类准确率 | ~0.6217 (bug 退化值) |
| finetune shots_num | 0 (强制) |
| test_only_params | eval_on_load=True, eval_no_save=True, max_duration=0 |
| finetune_params | reset_optimizer_state=True, reset_trainer_state=True, eval_interval=1000000 |
| stop_at 钳制 | min(cfg.stop_at, max_steps) (line 1476) |

## 运行方式

1. 用 `scripts/init_cfg_and_sbatch.py` 生成 config + sbatch 脚本:
   ```python
   Device = "RTX3090"
   modelnames = ["pause1"]  # 或其他
   tasks = ["docppl", "SG", "blimp", "boolq", "xsum_finetune"]  # 
   ```
2. sbatch 脚本需注入 `conda activate LLM` (在 `cd ${workspace}` 后)
3. `sbatch run_folder/{model}/{model}_{task}_test.sh`
4. 结果在 `slurm-{jobid}.out`:
   - terminal/pause docppl: `eval/terminal_doc_ppl/Perplexity=`
   - SG: `avg=`
   - BLiMP: `overall/overall=`
   - boolq: `eval/downstream/boolq_acc__=`
   - xsum: `rouge1=` / `rouge2=` / `rougeL=`

## 已知坑

1. **新 grammar type 必须加入 INPUTFORMAT** — 否则数据路径解析为 None
2. **不要混用旧 lm docppl** — 它的 pause CE/PPL 被插入位置稀释；报告 terminal/pause 对比时只使用 `terminal_doc_ppl`
