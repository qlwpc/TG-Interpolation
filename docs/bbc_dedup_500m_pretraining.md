# 去重 BBC 数据的 500M 预训练配置

正式开展预训练必须遵循 [预训练工作流](pretraining_workflow.md)。公共生成器
`scripts/prepare_bbc_dedup_500m.py` 包含八种模型配方和工作流配置。
默认仍选择 `terminal/tree/tgtree/tgnomask_aug`，新增模型须通过 `--models` 显式选择：

| CLI 模型 | 数据表示 | grammar / attention |
|---|---|---|
| `terminal` / `tree` / `tgtree` | terminal / tree / tg | 同名 grammar，causal FlashAttention |
| `tgnomask_aug` | tg | 保留原 aug + Flex 配方，区别于非 aug |
| `tg` / `tgnomask` | tg | 同名 grammar，typed TG / TGNomask |
| `tgnomask_mix_tg` | tg | `mixing`，8 TG + 8 TGNomask heads，typed |
| `tree_noont` | tree_noont | `tree_noont`，causal FlashAttention |

从仓库根目录、在训练环境中生成全新的 campaign：

```bash
python scripts/prepare_bbc_dedup_500m.py \
  --models tg tgnomask tgnomask_mix_tg tree_noont \
  --gpus 2 --hardware RTX5090 --fixed-microbatch 4 \
  --output-dir artifacts/experiment/NEW_UNIQUE_CAMPAIGN
```

训练默认读取 `dataset/bbc-news-parsed-dedup/{format}/train.npy`；dev/test 默认读取
`dataset/bbc-news-reserved-clean-v1/{format}/` 的完整基础数组（1000/5000 篇）。
Tree-NoONT 的 clean dev/test 默认在 `--data-dir/tree_noont/`，也可通过
`--noont-eval-dir ROOT` 指定 `ROOT/tree_noont/`；不能替换为旧测试集。
统一目录可同时传给 `--data-dir`、`--eval-dir`、`--noont-eval-dir`。

每模型输出 `base.yaml`、`config.yaml`、`smoke.yaml`、`protocol.json`、`launch.sh`、
`smoke.sh`。正式配置包含每 3000 步完整 clean dev/test 评估、W&B offline、DDP/BF16、
1 epoch、checkpoint 保留策略、44h Trainer 时间预算和独立的 run/save 路径；smoke 单独
执行 3 步及短评估，`--smoke-timeout` 默认 1800 秒，不改正式训练预算。launcher 设置 `WANDB_MODE=offline` 和短 TMPDIR，
使用真实 world size 的 torchrun；仅在已获分配的 GPU 作业内运行，不负责申请资源或提交。

- `--max-microbatch` 默认 1，仅生成初始整除候选；protocol 标为 `search_max/pending_gpu_probe`。
- `--fixed-microbatch N` 固定候选可不整除 local batch，记录实际 ceil 累积次数和末尾 batch；
  不能改变 global batch，仍须在目标机器验证。两项均不是已通过显存探测的声明。
- `--runtime-root /absolute/target/repo` 将仓库内输入、tokenizer、输出和 launcher 路径映射
  到目标仓库；此模式要求这些本地路径位于仓库内。生成时检查本地资产，远端存在性另行验证。
- `--campaign` 默认输出目录名，配合 `--hardware`、GPU 数生成 run 名和 W&B group；
  `--wandb-entity` 可调整归属。输出目录必须全新。
- `--tokenizer-reference` 默认使用已公开的冻结 JSON；字节哈希不同须通过参考哈希和完整
  JSON 等价性校验，不再仅报警后继续生成。
- 默认检查 train NPY header/payload 和 manifest，复用 manifest 中的 train SHA-256；
  `--verify-train-hash` 可重新计算完整训练数组哈希。dev/test 重新计算完整哈希，检查
  manifest、文档数和 uint64 offsets；兼容 clean 总 manifest 和逐 split manifest。

生成器对所有模型执行 TrainConfig schema 检查，并逐一调用 `use_typed_tg` 核验 train、
dev、test 路由；protocol 保存数据与配置身份、Step Law 推导、运行设置及待验证状态。
生成后仍须完成工作流的远端路径、扩展/数值、IPC/DDP、microbatch 和提交回执验证。
训练期间的 LM loss 评估不等同于 top-300 document PPL。

协议依据 [论文训练设置](pretraining_workflow.md#3-参数)：16 层、hidden size 1408、
16 heads、MLP ratio 8、sequence length 2048、seed 6198、从零训练 1 epoch，
AdamW (0.9, 0.95)、weight decay 0.1、gradient clipping 1.0、cosine schedule、
2000 步 warmup、最低 LR 1e-5。非 embedding 参数量从 meta 模型实际计算。
学习率和 batch 根据各表示的新训练 token 数使用现有 Step Law 实现重算；global batch
取最近 GPU 数倍数，microbatch 取上限以内能整除每卡 batch 的最大整数。
`--fixed-microbatch` 则保留指定值和末尾小 batch；生成阶段没有验证 GPU 峰值显存。

当前 clean manifest 的排除范围仍标记为历史训练集，本次没有重新进行新训练集与 clean
dev/test 的完整重叠审计。配置生成和 CPU 校验不代表训练已运行或论文指标已复现。
