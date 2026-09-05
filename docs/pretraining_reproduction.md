# 论文预训练复现细则

本文保存 README 背后的完整自动化协议。模型身份与超参数以
[`EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md) §2/§2A
为人工核对来源；[`train_configs/paper_pretraining_manifest.json`](../train_configs/paper_pretraining_manifest.json)
是机器可读映射。若二者与 checkpoint config 不一致，campaign 生成会直接失败。
BBC SEP Pause 使用固定 SHA-256 的原始提交配置副本，并单独登记最终 checkpoint 身份；
详见 [`pause_protocol.md`](pause_protocol.md)。

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

数组任务可显式传 `--task-index N`，也可从 `SLURM_ARRAY_TASK_ID` 读取。`plan` 生成的每条
命令均保留自定义 raw/parsed/tokenized/final/tokenizer 路径，并提供可复制的 `shell_command`。
它不是作业提交器：必须等下载/解析数组全部成功，才能运行后续阶段。

下载采用临时文件和完整长度检查后替换；解析按完整文档行续跑，并写 `.parse.json` 完成记录。
三路 tokenization 只在源文本、tokenizer 和三个输出的 SHA-256 均匹配完成记录时复用，
不再仅凭文件存在跳过。已有 tokenizer 默认校验布局并复用，不重复联网构建。

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
python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus bbc --jobs 1
```

仓库已包含原始 `dataset/bbc-news/dev_index.json` 和 `test_index.json`，不需要重新生成。
默认 `plan` 不包含 `make-split-indices`，`assemble` 从仓库的发布位置读取这两个文件
（即使 `--final-dir` 指向其他目录），并按
[`bbc_split_manifest.json`](../datatools/parse_pretrain_data/bbc_split_manifest.json) 校验 SHA-256：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data validate-splits --corpus bbc
python -m datatools.parse_pretrain_data.build_pretrain_data assemble --corpus bbc
```

索引协议如下：

- dev 为 4,980 个文档，test 为 5,025 个文档；各覆盖 89 个 config，无重复或交叉选择。
- config 顺序使用 `bbc_configs.txt`；每个 config 内的 dev/test 文档保留 JSON 列表的原顺序，
  不排序、不转集合；train 保留源文档顺序，排除 dev/test 的并集。
- `CC-MAIN-2023-{06,14,23,40,50}` 没有留出索引，五个 config 全部进入 train。
- 未知 config、重复 ID、负数/非整数、dev/test 交叉、越界 ID 都直接报错；不跳过缺失 shard。

`assemble` 在写任何输出前，检查全部源/目标、dtype、逐 shard 文档数及 split 范围。
分块扫描 BOS、以 mmap 写标准 `.npy` header，不把 20–80 GB 数组装入内存，也不使用 `cat`
拼接 `.npy`。三种表示使用同一份索引，各输出完成后原子替换，并最终写
`assembly_manifest.json`，保存 tokenizer/索引身份、文档数和输出哈希。

注意：旧 `gen_final_train.py` 会原地给索引列表追加哨兵文档，且跨表示累积；新流程不复刻
这一副作用。提供的索引锁定“选哪些文档及其顺序”，不能单独证明重新下载/解析后的 token 流
与历史训练流逐字节一致。现有 4,966 文档 DocPPL 语料是另一个评测契约，不能用本次生成的
5,025 文档 `test.npy` 自动替换；历史边界见 [`tree300_vs_test_boundary_report.md`](tree300_vs_test_boundary_report.md)。

若要做新的 split 实验，必须显式指定独立目录，再同时传入两个自定义索引：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data make-split-indices --corpus bbc \
  --output-dir artifacts/data/custom_bbc_split --seed 42
python -m datatools.parse_pretrain_data.build_pretrain_data assemble --corpus bbc \
  --dev-index artifacts/data/custom_bbc_split/dev_index.json \
  --test-index artifacts/data/custom_bbc_split/test_index.json \
  --final-dir artifacts/data/custom_bbc_streams
```

新索引不得写入发布/final 目录，也不得覆盖已有 split 文件；自定义结果不标为论文发布 split。

生成消融数据与两个架构基线：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data variants --corpus bbc --jobs 4
python -m datatools.parse_pretrain_data.build_pretrain_data baselines --corpus bbc --workers 8
```

`variants` 从 LIN1 构建 `tree_noont/`、`tree_compress/`、`tree_triplecnt/`；Tree-Shuffle
不落盘，由 collator 在线打乱 NT。`baselines` 对 train/dev/test 分别生成
`parse_aligned/*_treereg` 和 `*_pushdown_unary_terminals`，使用 unary collapse、right CNF
和 terminal 对齐。

校验阶段区分 shard 和最终三路组装流，避免“验证了中间数据”被误读成“验证了最终输出”：

```bash
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus bbc --scope shards
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus bbc --scope assembled
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus bbc --scope all
```

shard 校验要求精确文件名集合、论文 tokenizer 布局、完成记录、输出哈希、dtype、合法 token
范围、BOS/EOS 与文档数；assembled 校验读取完成 manifest 并核对九个输出的形状、dtype、哈希。
`variants`/`baselines` 执行前也检查组装结果。`--scope all` 不验证派生消融或 TreeReg/Pushdown
监督语义；它们由各自预计算程序生成，并由 campaign 的 `--validate-data` 继续做路径检查。

历史 BBC terminal `train.npy` 的逻辑长度为 `10,061,025,584` 个 `uint16` token；
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

python -m datatools.parse_pretrain_data.build_pretrain_data tokenize --corpus fineweb-edu --jobs 1
python -m datatools.parse_pretrain_data.build_pretrain_data validate --corpus fineweb-edu
```

若需让 staging 不依赖 Hugging Face cache，可在下载阶段加 `--copy-cache-files`，但会额外占用
接近整套语料的磁盘空间。FineWeb-Edu 预训练直接消费 246 个 shard pattern，不再拼成一个
train.npy。论文 token 量约为 terminal 98B、LIN1 233B、LIN2 301B。

BBC 使用 `benepar_en3_large`，FineWeb-Edu 的历史流程使用 `benepar_en3`；这是已登记的历史
差异，复现旧数据时不静默统一模型。

### 1.3 失败、恢复和复现边界

- 两套 parser 均兼容标准 Benepar 的单棵 `Tree` 与历史 top-k 候选列表，检查句子数与完整
  terminal yield。标准接口不能直接取 `tree[0]`。FineWeb Arrow 文件名排序后再加载。
- `parse --max-docs N` 对两套语料均生效，只用于小样本检查；完成记录为 partial 的 shard
  不能进入统一入口的 `tokenize`。去掉限制续跑至全部文档完成后再 tokenization。
- 空/坏树行、未结束的最后一行、嵌入 BOS/EOS 都会使 tokenization 失败，不静默删行。
  BBC 空文档分句也会报错；不得通过过滤来“修复”，否则发布索引会偏移。
- tokenization 内存按单文档和固定复制块使用；临时磁盘需容纳一个 shard 的三路 raw + NPY
  输出，约为该 shard 三路总大小的两倍。`--jobs` 默认 1，按磁盘空间和 I/O 预算增加。
- 没有完成记录的旧 shard/半成品不自动复用；明确核对路径后，可用 `tokenize --overwrite`
  重建。旧 parsed `.txt` 可重新执行对应 `parse` 任务检查行数并补完成记录，但这不能证明
  旧行使用了同一源数据/解析器版本。需要严格的新构建时使用新的 parsed/tokenized/final 目录。
- `assemble` 与 `variants` 默认拒绝覆盖；恢复半成品需显式 `--overwrite`。覆盖授权只应给
  重建目录，不要指向仍被 checkpoint/评测使用的历史语料。不要并发写同一 shard 或 final 目录。
- Hub revision、上游 Arrow cache 版本及 parser/model 权重尚未全部固定。因此这里验证的是
  数据协议、已提供索引和本次构建身份，不宣称已经复现整套历史训练数据或论文指标。

本次修复的离线验证范围及命令见 [`pretraining_data_pipeline_repair.md`](pretraining_data_pipeline_repair.md)。

## 2. 全部论文模型的预训练 campaign

列出机器可读清单：

```bash
python scripts/prepare_paper_pretraining.py --list
```

默认生成 27 个已登记 run，BBC Pause-1/2 对应最终论文的 dedicated-SEP 重训；
清单另外保留两个默认不选中的历史 repeat-token 对照。生成只写文件、不提交、不要求数据已在本机：

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
├── config.yaml       # 从 checkpoint 或固定哈希的原始提交配置清洗出的重训配置
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

正式启动前应加 `--validate-data` 重新生成一次。BBC 100M dedicated-SEP Pause 的实际
world size 为 8，其余 BBC 100M/500M 默认为 4；FineWeb-Edu 1B 为 64。
逐 run 的 `gpu_count` 优先于规模默认值。多节点 `launch.sh` 需要调度器在每个节点设置 `NNODES`、
`NPROC_PER_NODE`、`NODE_RANK`、`MASTER_ADDR` 和 `MASTER_PORT`，且前两者乘积必须等于登记的
world size。

### 2.1 模型覆盖

- BBC 100M：Terminal；论文 12 个句法变体中的 Tree、TGTree、TG、TGNomask、
  TGNomask-Aug、Tree-NoONT、Tree-Compress、Tree-TripleCNT、Tree-Shuffle、
  TGNomask-Mix-TG、TGTree-Mix-TG；以及论文 dedicated-SEP Pause-1、Pause-2。
- BBC 100M 架构基线：TreeReg-L9、Pushdown。
- BBC 500M：Terminal、Tree、TGTree、TGNomask-Aug。
- BBC 1B 补充 scaling：Terminal、Tree；二者是 GPT-2/BBC 模型，不能当作 FineWeb-Edu 主实验。
- FineWeb-Edu 1B 主实验：Terminal、Tree、TGTree、Pause-1、Pause-2。

BBC 100M 论文 Pause-1/2 的 ID 为 `bbc_100m_pause1_sep` / `bbc_100m_pause2_sep`，
使用专用 SEP id 50261。两份提交配置保存在 `train_configs/paper_sources/`，生成时校验
原始 SHA-256，不要求本机已有 SEP 权重。FineWeb-Edu 1B 继续使用专用 SEP id 151673。

历史 `pause_token_id=null` 的 `bbc_100m_pause1_repeat` / `bbc_100m_pause2_repeat`
是 repeat-token 对照，不对应最终论文 Pause 行；通过以下命令显式生成：

```bash
python scripts/prepare_paper_pretraining.py --groups bbc-100m-historical
```

[`scripts/submit_pause_sep_pretrain.py`](../scripts/submit_pause_sep_pretrain.py) 保留为原始
集群提交与 Step Law 工具。复现已登记参数使用上面的固定配置入口；不同 GPU 数下重新
运行 Step Law 工具可能改变舍入后的 global batch。

## 3. 验证

不需要下载全量语料即可验证任务映射、真实 checkpoint config 清洗、合成流式组装、
线性化变体和 Step Law：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_pretraining_reproduction_pipeline.py \
  tests/test_paper_pause_protocol.py \
  tests/test_parse_pipeline_smoke.py \
  tests/test_make_tree_variant.py \
  tests/test_submit_pause_sep_pretrain.py \
  tests/test_step_law.py
```

预训练结果和 checkpoint 身份仍统一回填到
[`EXPERIMENT_REPRODUCTION_RECORD.md`](../EXPERIMENT_REPRODUCTION_RECORD.md)；不要在 README
另建一份会漂移的人工超参数表。
