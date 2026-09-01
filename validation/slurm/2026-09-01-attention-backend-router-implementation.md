# Flash/Flex/SDPA 场景路由实现与验证（2026-09-01）

## 已实现策略

路由在原 `LLM` 环境（torch 2.7.1+cu126、Triton 3.3.1、
FlashAttention 2.8.2）内完成，不依赖环境升级。

| 场景 | 默认 backend | 规则 |
|---|---|---|
| terminal/tree/tgtree/pause/treereg 普通 causal | FlashAttention | 无 TG bias 时保持原有 Flash 路径；flash-attn 不可用时才回退 SDPA |
| 带 `doc_lens` 的 causal 文档评测 | FlashAttention varlen | 路由优先级高于普通 Flash/Flex/SDPA |
| TG/tgproximal/tgnomask 训练 | SDPA 或 Flex | 原始 `N < 1024` 用同语义 additive-mask SDPA；`N >= 1024` 用 Flex |
| TG/tgproximal/tgnomask 模型内评测 | SDPA 或 Flex | `N < 2048` 用 SDPA；`N >= 2048` 才允许 Flex |
| 独立 TG/tgnomask 下游评测配置 | SDPA | 配置直接关闭 Flex，避免短/动态任务的约 4 秒冷编译 |
| mixing | SDPA | 真实逐头 mask A/B 完成前维持关闭 |
| Pushdown 解析训练/PPL | 专用 Flex score-mod | 不套用 TG 长度阈值；保留已有高速路径 |
| Pushdown KV-cache/显式关闭的短任务 | fallback | 固定 BlockMask 不用于 KV-cache；BoolQ 每指标配置继续使用 `pushdown_use_flex: false` |

配置接口位于 `ModelConfig`：

- `flex_attention_train_min_sequence_length: 1024`
- `flex_attention_eval_min_sequence_length: 2048`
- `flex_attention_pad_to_multiple: 128`

旧的 `OLMO_FLEX_ATTENTION_PAD_TO_MULTIPLE` 环境变量仍可覆盖配置。
`OLMO_ATTENTION_ROUTER_DIAGNOSTICS=1` 会记录 grammar、训练/评测状态、原始/
有效长度、backend 和选择原因。

## 128 反传安全约束

1. 只有路由选中 Flex 后才构造 BlockMask。
2. 默认把所有非 128 整数倍的 TG 类 Flex 输入补齐；原 token 不能看到补齐
   key，补齐 query 只看到自身，logits 前恢复原长度。
3. 如果显式设置 `flex_attention_pad_to_multiple: null`，任何启用 autograd
   且非 128 整数倍的形状会安全回退 SDPA，不允许进入已知故障 kernel。
4. padding multiple 必须是正的 128 整数倍，否则模型初始化直接报配置错误。
5. `use_cache=True` 的 prefill、已有 KV-cache 的 decoding、非方形 bias 均
   强制使用 SDPA/既有生成路径，避免把补齐 token 写入 cache。
6. TG bias 与 `attention_mask` 现在先以 batch-aware 位运算合并；删除了原先
   vmap 不支持的 Python `and` 和错误的无 batch 索引。

## 阈值证据

真实 BBC TG mask、RTX 3090、BF16、12 层 attention-only 估算均计入一次
稳定态 BlockMask 构建。完整 step 的收益会小于 attention-only 收益。

### 实际 TG 训练 microbatch=10

| N | SDPA (ms) | Flex+mask (ms) | Flex/SDPA | 决策 |
|---:|---:|---:|---:|---|
| 512 | 12.12 | 12.84 | 1.06x | SDPA；Flex 稳定态也没有优势，另有约 10.1 秒冷编译 |
| 1024 | 43.05 | 33.25 | 0.77x | Flex；attention 部分快约 23% |
| 2048 | 160.48 | 94.63 | 0.59x | Flex；attention 部分快约 41% |

因此训练阈值取 1024。

### 独立 TG 评测 batch=100

| N | SDPA (ms) | Flex+mask (ms) | Flex/SDPA | Flex cold (s) |
|---:|---:|---:|---:|---:|
| 128 | 1.72 | 5.62 | 3.26x | 4.09 |
| 512 | 22.19 | 18.34 | 0.83x | 4.11 |
| 1024 | 85.51 | 57.63 | 0.67x | 4.12 |

大 batch 会让 Flex 的稳定态交叉点下移，但冷编译改变总任务最优解：`N=512`
约需 1060 个同 shape batch、`N=1024` 约需 148 个 batch 才能摊销。独立
BoolQ/XSum/PPL 配置通常达不到该复用次数，因此直接关闭 Flex；模型内长序列
评测保留 2048 的保守阈值，以便训练已产生可复用 kernel 时获得收益。

普通 causal 控制在 `N=16–2048` 上 Flex 始终比 SDPA 慢 1.4–2.8 倍；原
环境还有第三方 FlashAttention，因此 causal 模型明确不进入 Flex。

## 代码与配置变化

- `olmo/model.py`：结构化路由、自动 padding、unsafe fallback、KV-cache
  guard、batch-aware 组合 mask、可选诊断日志；SDPA additive mask 在 block
  循环前只转换一次低精度，避免 12 层重复分配同一副本。
- `olmo/config.py`：三个有默认值且向后兼容的新路由字段与校验。
- `train_configs/TG.yaml`：显式固化 1024/2048/128 策略。
- `terminal-500M.yaml`、`tree-500M.yaml`、`tgtree-500M.yaml`：清理无效 Flex
  标志，保持 FlashAttention。
- `evaluation/eval_configs/{tg,nomask}.yaml`：独立评测关闭 Flex。
- `scripts/init_cfg_and_sbatch.py`：只让 TG/tgproximal/tgnomask 进入结构化
  长度路由；terminal/tree/tgtree/mixing 不再被统一强开 Flex。

## 验证

- CPU 路由/配置单测：15 passed。
- 较宽 CPU 回归：222 passed、26 skipped。
- GPU job 3929：19 passed，包括：
  - 默认 `N=127` 确认不调用 Flex、SDPA 反传有限；
  - 强制短 Flex 时确认调用 Flex、自动 `127 -> 128`、输出恢复 127；
  - 同时存在 batch-broadcast TG bias 和 batch-specific `attention_mask`；
  - padding 显式关闭时确认 unsafe `N=127` 回退 SDPA；
  - Pushdown `N=16` 专用 Flex 反传仍通过。
- 真实 mask job 3930：TG/tgnomask 的算子和完整一层 OLMo 共 4 个 case 全部
  通过。完整模型输出长度为 127；参数梯度有限，相对 SDPA cosine 至少
  0.999995、relative L2 至多 0.00316。
- 配置加载：TG、三个 causal 500M、两个独立结构化评测配置均成功解析新增
  默认字段。

## 适用边界

1024/2048 是针对本地 RTX 3090、100M 架构和当前 batch 场景的证据化默认
值，不宣称是所有 GPU/模型的永久常数。迁移到 A40/H800/5090、改变 head
dimension、启用真实 mixing，或显著改变 batch/reuse 次数时，应使用同一
benchmark 重新标定阈值；128 padding/fallback 属于正确性约束，不随性能
阈值一起放宽。

## 证据文件

- 路由单测：`tests/test_attention_backend_router.py`
- GPU 路由入口：`validation/slurm/verify_attention_router.sbatch`
- GPU JUnit：`validation/slurm/results/attention-router_3929.xml`
- 真实 mask validator：`validation/slurm/verify_real_tg_pad128.py`
- 真实 mask 结果：`validation/slurm/results/real-tg-pad128_3930.json`
- 代表 batch 基准：jobs 3920–3925，对应
  `validation/slurm/results/flex-vs-sdpa-tg-b*.json`
