# TG-Interpolation

本仓库是论文 *A Scaled-Up Empirical Study of Syntactic Language Models* 从语料构建、
预训练到评测的复现入口，包含 BBC News 实验、FineWeb-Edu-100BT 扩展实验，以及
Terminal、Pause、Tree/TG、TreeReg、Tree-Shuffle 和 Pushdown 等模型分支。

> [!IMPORTANT]
> README 只负责导航和最短执行路径，不保存第二份结果表。checkpoint 身份、实际运行协议、
> 任务状态和可引用结果统一登记在
> [`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md)。引用任何旧结果前，
> 先检查 [`REPOSITORY_CLEANUP_MEMORY.md`](REPOSITORY_CLEANUP_MEMORY.md)。

## 0. 开始前先确定口径

| 要确认的内容 | 首选入口 | 使用边界 |
|---|---|---|
| 论文声称的方法与表格 | [`camera_ready/paper.tex`](camera_ready/paper.tex) | 表示论文口径；若与实际运行记录冲突，不做静默合并 |
| checkpoint、实际 config、运行状态与结果 | [`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md) | 当前唯一人工总登记 |
| 数据构建与预训练 campaign | [`docs/pretraining_reproduction.md`](docs/pretraining_reproduction.md) | 当前可执行操作说明 |
| 已确认错误、协议差异和未关闭风险 | [`REPOSITORY_CLEANUP_MEMORY.md`](REPOSITORY_CLEANUP_MEMORY.md) | 错误数值不保留；有意义的替代协议带限定条件保留 |
| evaluator、数据入口与配置路由 | [`Evaluation.md`](Evaluation.md) | 当前评测协议导航；结果与 checkpoint 身份仍以总登记表为真源 |

发生冲突时，先区分三件事：论文原本声称什么、代码实际执行什么、某个 checkpoint
当时使用了什么。修正应回填总登记表和纠错记忆，不能直接用一种口径覆盖另一种口径。

### 实验范围

| 语料 / 规模 | 主要模型 | 主要评测 |
|---|---|---|
| BBC News 100M | Terminal、Tree/TG、Pause、结构消融、TreeReg、Tree-Shuffle、Pushdown | Doc-PPL、SG、BLiMP、XSum、BoolQ |
| BBC News 500M / 补充 1B | Terminal、Tree、TGTree、TGNomask-Aug | 以总登记表中的已完成项为准 |
| FineWeb-Edu-100BT 1B | Terminal、Tree、TGTree、Pause-1、Pause-2 | OLMES 11-task terminal-format 评测及待补充下游项 |

## 1. 环境

```bash
conda env create -f environment.yml
conda activate LLMH100
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

解析语料前检查 Benepar、spaCy 和 tokenizer 依赖；只有缺失时才执行第二条下载命令：

```bash
python -m datatools.parse_pretrain_data.setup_parse_deps --check
python -m datatools.parse_pretrain_data.setup_parse_deps
```

## 2. 构建预训练数据

统一入口按阶段运行；下载/解析/tokenization 支持 Slurm 数组与断点续跑：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data plan --corpus bbc
python -m datatools.parse_pretrain_data.build_pretrain_data plan --corpus fineweb-edu
```

### BBC News

```text
94 个 FineWeb BBC configs
  -> benepar_en3_large 解析
  -> GPT-2 扩展 tokenizer
  -> tree / terminal / tg uint16 shards
  -> 固定 train / dev / test streams
  -> 线性化消融与 TreeReg/Pushdown 对齐数据
```

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data tokenizer --corpus bbc

# 调度器数组 0-93；TASK_ID 也可由 SLURM_ARRAY_TASK_ID 自动读取。
python -m datatools.parse_pretrain_data.build_pretrain_data download --corpus bbc --task-index "$TASK_ID"
python -m datatools.parse_pretrain_data.build_pretrain_data parse    --corpus bbc --task-index "$TASK_ID"

python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus bbc --jobs 1
python -m datatools.parse_pretrain_data.build_pretrain_data validate-splits --corpus bbc
python -m datatools.parse_pretrain_data.build_pretrain_data assemble  --corpus bbc
python -m datatools.parse_pretrain_data.build_pretrain_data variants  --corpus bbc --jobs 4
python -m datatools.parse_pretrain_data.build_pretrain_data baselines --corpus bbc --workers 8
python -m datatools.parse_pretrain_data.build_pretrain_data validate  --corpus bbc --scope all
```

仓库包含 `dataset/bbc-news/{dev,test}_index.json`，分别选择 **4,980 / 5,025** 个文档；
默认流程校验固定 SHA-256 并保留列表顺序，不重新抽样。未列入索引的五个 2023 config
全部进入 train。索引一致不等于历史语料逐字节一致，也不能替换独立的 DocPPL 评测语料。

`tokenize` 要求全部解析任务完成，逐文档写盘，坏行直接失败；续跑核对源文件、tokenizer 和
三路输出指纹。`assemble` 先检查每个 shard 的文档对齐与索引范围，再用 mmap 写标准 `.npy`。
`validate --scope all` 校验 shard 完成记录与最终三路 train/dev/test 的哈希；消融和架构基线
仍需各自预计算及下文 `--validate-data` 路径门禁。已有最终数据不会默认覆盖，重建建议指定
新的 `--final-dir`。自定义 split、半成品恢复及历史协议边界见
[`docs/pretraining_reproduction.md`](docs/pretraining_reproduction.md)。

论文 BBC terminal `train.npy` 为 `10,061,025,584` 个 `uint16` token；LIN1/LIN2 约为
24.7B/32B。GPT-2 扩展 tokenizer 应为 vocab 50320、`<|SEP|>=50261`。

### FineWeb-Edu-100BT

```text
sample-100BT -> 984 Arrow shards -> 246 个四分片解析任务
             -> tree / terminal / tg uint32 shards
```

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data tokenizer --corpus fineweb-edu
python -m datatools.parse_pretrain_data.build_pretrain_data download  --corpus fineweb-edu

# 调度器数组 0-245。
python -m datatools.parse_pretrain_data.build_pretrain_data parse --corpus fineweb-edu --task-index "$TASK_ID"

python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus fineweb-edu --jobs 1
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus fineweb-edu
```

FineWeb-Edu 使用历史 `benepar_en3` 与 Qwen3 扩展 tokenizer；后者应为 vocab 151732、
`<|SEP|>=151673`，因此必须保存为 `uint32`。预训练直接消费 246 个 shard pattern；
terminal/LIN1/LIN2 约为 98B/233B/301B token。

## 3. 生成全部论文预训练配置

机器可读清单
[`train_configs/paper_pretraining_manifest.json`](train_configs/paper_pretraining_manifest.json)
由复现登记表 §2A 固化，并逐项绑定已核对 checkpoint config 或固定哈希的原始提交配置。
默认生成 27 个已登记 run，其中 BBC Pause-1/2 使用最终论文的 dedicated-SEP 协议；
另外两个历史 repeat-token 对照仅在显式选择时生成：

```bash
python scripts/prepare_paper_pretraining.py --list
python scripts/prepare_paper_pretraining.py \
  --campaign-dir artifacts/experiment/paper_pretraining_reproduction
```

| 实验组 | 生成的模型 |
|---|---|
| BBC 100M | Terminal；Tree、TGTree、TG、TGNomask、TGNomask-Aug、Tree-NoONT、Tree-Compress、Tree-TripleCNT、Tree-Shuffle、TGNomask-Mix-TG、TGTree-Mix-TG；dedicated-SEP Pause-1/2 |
| BBC 100M 架构基线 | Pushdown |
| BBC 500M | Terminal、Tree、TGTree、TGNomask-Aug |
| BBC 1B 补充 | Terminal、Tree |
| FineWeb-Edu 1B 主实验 | Terminal、Tree、TGTree、Pause-1、Pause-2 |

也可只生成一组：

```bash
python scripts/prepare_paper_pretraining.py --groups bbc-100m
python scripts/prepare_paper_pretraining.py --groups bbc-100m-baselines bbc-500m
python scripts/prepare_paper_pretraining.py --groups fineweb-edu-1b
```

输出的每个 run 都包含 `config.yaml`、`launch.sh` 和 `protocol.json`；生成器会核对规模、
grammar、LR、global batch、dtype、pause id 与 mixing heads，并清除旧绝对路径、评测污染路径
和 checkpoint restore state。正式运行前加 `--validate-data` 重新生成，以验证全部输入路径。

BBC 100M 论文 Pause-1/2 使用 SEP 50261，FineWeb-Edu 1B 使用 SEP 151673。
BBC SEP 两个 run 保留实际重训的 8-GPU、global batch 216/272 与 sequence length 2048/2049；
完整参数、checkpoint 和评测命令见 [`docs/pause_protocol.md`](docs/pause_protocol.md)。
历史 `pause_token_id=null` 权重仅通过 `--groups bbc-100m-historical` 或显式
`--models bbc_100m_pause1_repeat bbc_100m_pause2_repeat` 选择，不能通过修改 YAML 冒充 SEP 模型。

## 4. 验证

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_pretraining_reproduction_pipeline.py \
  tests/test_paper_pause_protocol.py \
  tests/test_parse_pipeline_smoke.py \
  tests/test_make_tree_variant.py \
  tests/test_submit_pause_sep_pretrain.py \
  tests/test_step_law.py
```

## 5. 评测与结果登记

`Evaluation.md` 已重写当前协议层，用于选择模型族、数据、evaluator 和配置入口；
其后的 2026-08-23 实现快照只作历史附录。指标、checkpoint 和完成状态仍只从总登记表引用。

| 任务 | 当前依据 | 必须单列的协议差异 |
|---|---|---|
| Document PPL | [`Evaluation.md`](Evaluation.md) §4 + 总登记表 §3；Pushdown 另见 [`docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`](docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md) | terminal、tree marginalization、Tree-Shuffle terminal、Pushdown native top-K 不混写 |
| SG / BLiMP | [`Evaluation.md`](Evaluation.md) §5 + 总登记表对应小节 | Tree/TG、Pause、Tree-Shuffle、Pushdown 各自的候选与计分方式 |
| XSum / BoolQ | [`Evaluation.md`](Evaluation.md) §6.1 + 总登记表对应小节 | Pause 专用生成流程、Tree-Shuffle masked checkpoint、Pushdown gold-span 流程 |
| FineWeb-Edu OLMES | [`Evaluation.md`](Evaluation.md) §6.2 + 总登记表 §4 | task shots、Benepar 1-best、terminal score 与 full score 分开登记 |

每条准备进入论文表格的结果至少记录：

- corpus、split 与数据版本；
- 模型族、规模和唯一 checkpoint 路径；
- evaluator protocol、候选 support、归一化/分母；
- seed、run id 或 Slurm job、artifact/log 路径和完成状态；
- 主协议结果或 alternative-protocol diagnostic，二者不可混列。

错误实现产生的数值不进入 README 或主结果表；如果错误揭示了有用的协议差异，只保留
错误原因，并把可复核结果迁移到相应专题协议文档。

## 6. 仓库地图

| 路径 | 内容 |
|---|---|
| `datatools/parse_pretrain_data/` | BBC 与 FineWeb-Edu 数据下载、解析、tokenization、组装和验证 |
| `train_configs/` | 预训练配置与论文模型 manifest |
| `scripts/` | campaign 生成、提交和专用评测入口 |
| `olmo/` | 模型、训练器、数据集和 evaluator 实现 |
| `tests/` | 数据与预训练复现的回归测试 |
| [`docs/`](docs/README.md) | 当前协议、实现记录和诊断材料的分类索引；引用前检查纠错记忆中的可信度分级 |
| `reports/` | 带日期的审计与专题结果，不替代总登记表 |
| `artifacts/`、`saved_models/` | 本机运行产物；路径存在不等于已发布或可再生资产 |

评测层本轮已按“核对总登记表与纠错记忆 → 整理数据与配置入口 → 重写
`Evaluation.md` 当前协议层”完成。后续清理历史报告和本机 artifacts 时，删除前先把仍有
解释力的错误原因或替代协议迁移到纠错记忆或对应专题文档。
