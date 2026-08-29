# 论文预训练复现细则

本文保存 README 背后的完整自动化协议。模型身份与超参数以
[`EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md) §2/§2A
为人工核对来源；[`train_configs/paper_pretraining_manifest.json`](../train_configs/paper_pretraining_manifest.json)
是机器可读映射。若二者与 checkpoint config 不一致，campaign 生成会直接失败。

## 1. 数据构建统一入口

所有命令均从仓库根目录执行。先检查解析依赖：

```bash
python -m datatools.parse_pretrain_data.setup_parse_deps --check
# 缺少模型时再执行；会访问网络。
python -m datatools.parse_pretrain_data.setup_parse_deps
```

稳定入口是：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data <stage> --corpus <bbc|fineweb-edu>
```

`plan` 只生成阶段、数组范围、路径和命令清单，不下载数据或启动计算：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data plan --corpus bbc
python -m datatools.parse_pretrain_data.build_pretrain_data plan --corpus fineweb-edu
```

数组任务可显式传 `--task-index N`，也可从 `SLURM_ARRAY_TASK_ID` 读取。下载、解析、
tokenization 都可重复执行：下载会复用完整文件，解析按已写文档行数续跑，三路 tokenization
只有在 `tree/terminal/tg` 同名输出都存在时才跳过。

### 1.1 BBC News（FineWeb BBC 子集）

固定顺序的 94 个 Common Crawl config 保存在
[`bbc_configs.txt`](../datatools/parse_pretrain_data/bbc_configs.txt)。完整链路为：

```text
permutans/fineweb-bbc-news parquet
  -> benepar_en3_large + spaCy 分句
  -> dataset/bbc-news-parsed/<config>.txt（每文档一行）
  -> GPT-2 扩展 tokenizer
  -> dataset/bbc-news-shards/{tree,terminal,tg}/<config>.npy（uint16）
  -> 固定文档 split 的 train/dev/test.npy
  -> Tree-NoONT / Tree-Compress / Tree-TripleCNT
  -> TreeReg / Pushdown parse-aligned 数据
```

阶段命令：

```bash
# tokenizer；论文文件应得到 vocab=50320、SEP=50261。
python -m datatools.parse_pretrain_data.build_pretrain_data tokenizer --corpus bbc

# 调度器数组 0-93：每项分别下载、解析同一个 config。
python -m datatools.parse_pretrain_data.build_pretrain_data download --corpus bbc --task-index "$TASK_ID"
python -m datatools.parse_pretrain_data.build_pretrain_data parse    --corpus bbc --task-index "$TASK_ID"

# 全部解析完成后，一次生成 LIN1 tree / terminal / LIN2 tg 三路 shard。
python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus bbc --jobs 16
```

字节级复现论文语料时，应把发布的 `dev_index.json`、`test_index.json` 放入
`dataset/bbc-news/`。若只有原始语料，可生成 seed 42 的确定性重建 split；该 split 可复跑，
但不宣称与历史发布 split 逐字节相同：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data make-split-indices --corpus bbc
python -m datatools.parse_pretrain_data.build_pretrain_data assemble --corpus bbc
```

`assemble` 对每种表示做两遍 mmap 扫描，以标准 `.npy` header 写出最终流，不把 20–80 GB
数组装入内存，也不使用会拼坏 `.npy` header 的 `cat`。每个 tokenized 文档显式包含 BOS/EOS，
三种表示使用相同的文档索引。

生成消融数据与两个架构基线：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data variants --corpus bbc --jobs 4
python -m datatools.parse_pretrain_data.build_pretrain_data baselines --corpus bbc --workers 8
```

`variants` 从 LIN1 构建 `tree_noont/`、`tree_compress/`、`tree_triplecnt/`；Tree-Shuffle
不落盘，由 collator 在线打乱 NT。`baselines` 对 train/dev/test 分别生成
`parse_aligned/*_treereg` 和 `*_pushdown_unary_terminals`，使用 unary collapse、right CNF
和 terminal 对齐。

验证三路 tokenized shard 的数量、dtype 与 tokenizer checksum；最终 train/dev/test、消融和
parse-aligned 路径会在预训练 campaign 的 `--validate-data` 门禁中继续检查：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus bbc
```

已发布 BBC terminal `train.npy` 的逻辑长度为 `10,061,025,584` 个 `uint16` token；
LIN1/LIN2 约为 24.7B/32B。逻辑长度必须从 NumPy header 读取，不能用物理字节数除以 2。

### 1.2 FineWeb-Edu-100BT

链路为：

```text
HuggingFaceFW/fineweb-edu:sample-100BT
  -> datasets Arrow cache 的可追踪 staging 目录
  -> 984 Arrow shards / 246 个四分片解析任务
  -> benepar_en3 + spaCy 分句
  -> Qwen3 扩展 tokenizer
  -> dataset/fineweb-edu-v2/{tree,terminal,tg}/246 shards（uint32）
```

阶段命令：

```bash
# 论文文件应得到 vocab=151732、SEP=151673；token id 超过 uint16。
python -m datatools.parse_pretrain_data.build_pretrain_data tokenizer --corpus fineweb-edu

# 下载 sample-100BT，并把 datasets cache Arrow 文件以 symlink 固化到 staging 目录。
python -m datatools.parse_pretrain_data.build_pretrain_data download --corpus fineweb-edu

# 调度器数组 0-245；任务 i 总是处理 [4i, 4i+1, 4i+2, 4i+3] 四个 Arrow shard。
python -m datatools.parse_pretrain_data.build_pretrain_data parse --corpus fineweb-edu --task-index "$TASK_ID"

python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus fineweb-edu --jobs 16
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus fineweb-edu
```

若需让 staging 不依赖 Hugging Face cache，可在下载阶段加 `--copy-cache-files`，但会额外占用
接近整套语料的磁盘空间。FineWeb-Edu 预训练直接消费 246 个 shard pattern，不再拼成一个
train.npy。论文 token 量约为 terminal 98B、LIN1 233B、LIN2 301B。

BBC 使用 `benepar_en3_large`，FineWeb-Edu 的历史流程使用 `benepar_en3`；这是已登记的历史
差异，复现旧数据时不静默统一模型。

## 2. 全部论文模型的预训练 campaign

列出机器可读清单：

```bash
python scripts/prepare_paper_pretraining.py --list
```

生成全部 27 个已登记 run（默认只写文件、不提交、不要求数据已在本机）：

```bash
python scripts/prepare_paper_pretraining.py \
  --campaign-dir artifacts/experiment/paper_pretraining_reproduction
```

按实验组生成：

```bash
python scripts/prepare_paper_pretraining.py --groups bbc-100m
python scripts/prepare_paper_pretraining.py --groups bbc-100m-baselines bbc-500m
python scripts/prepare_paper_pretraining.py --groups bbc-1b-supplementary
python scripts/prepare_paper_pretraining.py --groups fineweb-edu-1b
```

每个 run 目录包含：

```text
runs/<model-id>/
├── config.yaml       # 从已核对 checkpoint config 清洗出的重训配置
├── launch.sh         # 运行前强制 schema 与全部本地输入检查的 torchrun 入口
└── protocol.json     # 论文身份、数据表示、LR/batch/step、来源与输出 SHA-256
```

生成器执行以下门禁：

1. checkpoint config 的规模、seed、precision、grammar、LR、global batch、microbatch、dtype、
   pause id/sequence length、mixing heads 必须与 §2A 登记一致；
2. 清除 checkpoint 的绝对旧路径、被评测污染的 `data.paths`、旧 evaluator、`load_path` 与
   trainer/optimizer restore state；
3. BBC 路由到标准单流或 parse-aligned 目录；FineWeb-Edu 重建完整 246 个 shard pattern；
4. 默认不覆盖已有 save folder；启动前既做 `TrainConfig` schema 检查，也显式展开并检查所有
   普通字符串形式的训练、parse-tree、评测与 tokenizer 路径（历史 YAML 并未使用 path resolver）。

正式启动前应加 `--validate-data` 重新生成一次。BBC/500M 默认论文 world size 为 4；
FineWeb-Edu 1B 为 64。多节点 `launch.sh` 需要调度器在每个节点设置 `NNODES`、
`NPROC_PER_NODE`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`，且前两者乘积必须等于登记的
world size。

### 2.1 模型覆盖

- BBC 100M：Terminal；论文 12 个句法变体中的 Tree、TGTree、TG、TGNomask、
  TGNomask-Aug、Tree-NoONT、Tree-Compress、Tree-TripleCNT、Tree-Shuffle、
  TGNomask-Mix-TG、TGTree-Mix-TG；以及 Pause-1。登记表中的补充 Pause-2 也保留为独立 run。
- BBC 100M 架构基线：TreeReg-L9、Pushdown。
- BBC 500M：Terminal、Tree、TGTree、TGNomask-Aug。
- BBC 1B 补充 scaling：Terminal、Tree；二者是 GPT-2/BBC 模型，不能当作 FineWeb-Edu 主实验。
- FineWeb-Edu 1B 主实验：Terminal、Tree、TGTree、Pause-1、Pause-2。

100M 历史 Pause-1/2 checkpoint 的 `pause_token_id=null`，实际是 repeat-token compute
control；FineWeb-Edu 1B 使用专用 SEP id 151673。不要把二者改成同一种协议。独立 SEP 的
100M 重训属于补充实验，继续使用
[`scripts/submit_pause_sep_pretrain.py`](../scripts/submit_pause_sep_pretrain.py)，不覆盖历史 run。

## 3. 验证

不需要下载全量语料即可验证任务映射、真实 checkpoint config 清洗、合成流式组装、
线性化变体和 Step Law：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_pretraining_reproduction_pipeline.py \
  tests/test_parse_pipeline_smoke.py \
  tests/test_make_tree_variant.py \
  tests/test_submit_pause_sep_pretrain.py \
  tests/test_step_law.py
```

预训练结果和 checkpoint 身份仍统一回填到
[`EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md)；不要在 README
另建一份会漂移的人工超参数表。
