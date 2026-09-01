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
| evaluator 的实现背景 | [`Evaluation.md`](Evaluation.md) | 目前是 2026-08-23 历史快照，不是当前协议或结果真源 |

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

python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus bbc --jobs 16
python -m datatools.parse_pretrain_data.build_pretrain_data assemble  --corpus bbc
python -m datatools.parse_pretrain_data.build_pretrain_data variants  --corpus bbc --jobs 4
python -m datatools.parse_pretrain_data.build_pretrain_data baselines --corpus bbc --workers 8
python -m datatools.parse_pretrain_data.build_pretrain_data validate  --corpus bbc
```

字节级复现应使用发布的 `dataset/bbc-news/{dev,test}_index.json`。若没有，可先运行
`make-split-indices --corpus bbc` 生成 seed 42 的确定性重建 split，但它不冒充历史 split。
`assemble` 使用 mmap 两遍写出标准 `.npy`，不要用 `cat` 拼接 NumPy 文件。
数据阶段的 `validate` 检查 shard 数量、dtype 与 tokenizer；最终模型输入由下文
`--validate-data` 逐路径门禁。

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

python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus fineweb-edu --jobs 16
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus fineweb-edu
```

FineWeb-Edu 使用历史 `benepar_en3` 与 Qwen3 扩展 tokenizer；后者应为 vocab 151732、
`<|SEP|>=151673`，因此必须保存为 `uint32`。预训练直接消费 246 个 shard pattern；
terminal/LIN1/LIN2 约为 98B/233B/301B token。

## 3. 生成全部论文预训练配置

机器可读清单
[`train_configs/paper_pretraining_manifest.json`](train_configs/paper_pretraining_manifest.json)
由复现登记表 §2A 固化，并逐项绑定已核对 checkpoint config。先查看模型，再生成包含
27 个已登记 run 的 dry-run campaign：

```bash
python scripts/prepare_paper_pretraining.py --list
python scripts/prepare_paper_pretraining.py \
  --campaign-dir artifacts/experiment/paper_pretraining_reproduction
```

| 实验组 | 生成的模型 |
|---|---|
| BBC 100M | Terminal；Tree、TGTree、TG、TGNomask、TGNomask-Aug、Tree-NoONT、Tree-Compress、Tree-TripleCNT、Tree-Shuffle、TGNomask-Mix-TG、TGTree-Mix-TG；Pause-1/补充 Pause-2 |
| BBC 100M 架构基线 | TreeReg-L9、Pushdown |
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

100M 历史 Pause checkpoint 是 `pause_token_id=null` 的 repeat-token control；FineWeb-Edu 1B
才使用 SEP 151673。独立 SEP 的 100M 重训属于补充实验，入口为
[`scripts/submit_pause_sep_pretrain.py`](scripts/submit_pause_sep_pretrain.py)，不能覆盖历史模型身份。

## 4. 验证

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_pretraining_reproduction_pipeline.py \
  tests/test_parse_pipeline_smoke.py \
  tests/test_make_tree_variant.py \
  tests/test_submit_pause_sep_pretrain.py \
  tests/test_step_law.py
```

## 5. 评测与结果登记

`Evaluation.md` 尚未完成当前协议重写，因此不要把其中的历史行号、旧结果速查表或统一分支
描述直接当作论文复现入口。现阶段按下表选择依据：

| 任务 | 当前依据 | 必须单列的协议差异 |
|---|---|---|
| Document PPL | 总登记表 §3；Pushdown 另见 [`docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md`](docs/pushdown_word_atom_strict_binary_document_ppl_protocol.md) | terminal、tree marginalization、Tree-Shuffle terminal、Pushdown native top-K 不混写 |
| SG / BLiMP | 论文方法 + 总登记表对应小节 | Tree/TG、Pause、Tree-Shuffle、Pushdown 各自的候选与计分方式 |
| XSum / BoolQ | 论文方法 + 总登记表对应小节 | Pause 专用生成流程、Tree-Shuffle masked checkpoint、Pushdown gold-span 流程 |
| FineWeb-Edu OLMES | 论文 terminal-format 方法 + 总登记表 §4 | task shots、Benepar 1-best、terminal score 与 full score 分开登记 |

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

建议的整理顺序是：先核对总登记表与纠错记忆，再整理数据入口和配置，随后重写
`Evaluation.md` 的当前协议层，最后才清理历史报告和本机 artifacts。删除文档前，先把仍有
解释力的错误原因或替代协议迁移到纠错记忆或对应专题文档。
