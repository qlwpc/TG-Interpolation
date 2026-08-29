# TG-Interpolation

本仓库是论文从语料构建、预训练到评测的复现总入口。预训练参数与 checkpoint 身份以
[`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md) 为唯一人工登记；
数据与训练的完整自动化细则见
[`docs/pretraining_reproduction.md`](docs/pretraining_reproduction.md)，评测见
[`Evaluation.md`](Evaluation.md)。

## 环境

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

## 1. 构建预训练数据

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

## 2. 生成全部论文预训练配置

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

## 3. 验证与后续评测

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_pretraining_reproduction_pipeline.py \
  tests/test_parse_pipeline_smoke.py \
  tests/test_make_tree_variant.py \
  tests/test_submit_pause_sep_pretrain.py \
  tests/test_step_law.py
```

预训练完成后按 [`Evaluation.md`](Evaluation.md) 运行 Doc-PPL、SG、BLiMP、XSum、BoolQ 和
FineWeb-Edu OLMES 评测，并将 checkpoint、协议、任务状态与结果回填到
[`EXPERIMENT_REPRODUCTION_RECORD.md`](EXPERIMENT_REPRODUCTION_RECORD.md)。
