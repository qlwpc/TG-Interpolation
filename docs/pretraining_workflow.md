# 预训练复现工作流

新 BBC 实验默认 **dedup train + reserved-clean dev/test**。复现论文历史设置时明确选择
历史生成器及对应语料；两者不是同一次实验。模型与结果说明见[论文结果](paper_results.md)，
外部资产状态见[公开材料](../reproducibility/README.md)。

## 1. 环境与源码

按根目录 [README](../README.md#environment) 创建 `LLM` 环境并构建扩展。
使用目标 Python 构建 `tg_mask`、`cppbackend`；typed layout 还要求 C++17 编译器。
保存源码 commit 及未提交 patch/新增源码、配置、依赖包版本、GPU/driver、Torch/CUDA/ABI、
实际加载动态库和扩展的哈希。不能用包版本号替代实际动态库身份。
公共环境由 [environment.yml](../environment.yml) 定义；本地 5090 的专用安装和库修复
不是公共安装前提。新硬件需要独立 kernel、NCCL collective 和 DDP 验证。

## 2. 先冻结数据与模型身份

| 数据 | 默认位置 | 文档数 |
|---|---|---:|
| dedup train | `dataset/bbc-news-parsed-dedup/{format}/train.npy` | 1,466,132 |
| clean dev/test | `dataset/bbc-news-reserved-clean-v1/{format}/{dev,test}.npy` | 1,000 / 5,000 |
| NoONT clean dev/test | `dataset/bbc-news-parsed-dedup/tree_noont/` | 1,000 / 5,000 |

NoONT 必须由同版 clean Tree 删除 ONT 得到；不能因为路径在 dedup 目录就使用旧 dev/test。
每种表示保留 manifest：来源、split、dtype、shape、tokens、bytes、文档数、SHA-256。
NPY 必须读 header；BBC 为 `uint16`，Qwen3/FineWeb 为 `uint32`。dev/test offsets 为
严格递增的 `uint64`，长度 N+1，首项 0、末项等于 token 数。

BBC tokenizer 词表为 50320，EOS/PAD/SEP=50256/50258/50261。公共生成器用
[冻结 tokenizer reference](../reproducibility/tokenizer-reference.json) 核验字节身份或完整
JSON 等价性。词表大小相同不证明 token-ID 一致。新传输输入计算完整 hash；复用既有可信
train hash 时记录本次未重算，dev/test 仍完整验证。
数据污染结论的范围见[数据说明](bbc_data_provenance.md)。

<a id="22-四模型的实际路由合同"></a>
### TG 路由

TGTree 是 LIN2 + causal；TG 是 LIN2 + STACK/COMPOSE。非 aug TGNomask 与 aug 是不同模型。
公共生成器的 `tg/tgnomask/tgnomask_mix_tg` 使用 typed kernels：
`tg_typed_attention=true`，Flash/Flex=false；train/dev/test 的
`generate_attention_mask=false`、`generate_doc_lengths=false`，native layout、
`cuda_prefetch=false`。Mix 为 8 TG + 8 TGNomask heads。NoONT 为独立 causal 模型。
分别检查 train/dev/test 的 `use_typed_tg` 路由，并在真实数据上验证 mask、label 和梯度。
详见 [TG 输入管线](tg_input_pipeline.md)。

## 3. 参数

500M：16 层、hidden 1408、16 heads、MLP ratio 8、RoPE、pre-RMSNorm、SwiGLU、
共享 embedding、sequence length 2048、seed 6198、BF16、从零训练 1 epoch。
AdamW betas=(0.9,0.95)、eps=1e-8、weight decay=0.1、梯度裁剪 1.0；cosine，
2000 步 warmup、最低 LR=1e-5。

```text
peak_lr = 1.79 × N^(-0.713) × D^(0.307)
token_batch = 0.58 × D^(0.571)
```

N 为非 embedding 参数量；500M 实测 N=507,896,576。D 为该表示含结构/special token 的
实际 train 数组长度。global batch 按 sequence length 和 world size 舍入，详见
[Step Law 实现](../scripts/step_law.py)。microbatch 仅调整显存与梯度累积，不能静默改变
数据、模型或 global batch；保存实际累积次数和末尾小 batch。

## 4. 选择生成器

| 范围 | 命令 / 指南 |
|---|---|
| 新 BBC dedup 500M | [公共生成器参数](bbc_dedup_500m_pretraining.md) |
| 论文历史 27 个默认 run | `python scripts/prepare_paper_pretraining.py --campaign-dir artifacts/experiment/NEW_PAPER_RUN`；[细则](pretraining_reproduction.md) |
| SEP Pause | `scripts/submit_pause_sep_pretrain.py`；[Pause 协议](pause_protocol.md) |

```bash
python scripts/prepare_bbc_dedup_500m.py \
  --models tg tgnomask tgnomask_mix_tg tree_noont \
  --gpus 2 --fixed-microbatch 4 \
  --output-dir artifacts/experiment/NEW_CAMPAIGN
```

## 5. 检查生成配置

输出必须是新目录。检查 base/config/smoke、protocol 和 launcher：路径可达，grammar、
tokenizer、表示一致；正式运行每 3000 步完整 clean dev/test、W&B offline、独立 run/save
目录。训练 LM loss 与 top-300 document PPL 不同。生成器的 probe 状态仍为 pending。
用 `--runtime-root` 映射目标仓库时，远端仍需验证真实文件、链接目标、hash 和 schema。
Pause-1/2、TGTree-Mix-TG 新增实验的独立冻结配方范围见[新增结果](bbc_dedup_500m_evaluation_results_20260925.md)；
不要用相近模型的 CLI key 代替。

## 6. 远端传输、环境验证与 Slurm 提交

先做文件/hash 检查，再构建扩展、导入/schema、真实 shape 的 kernel 前向/梯度，
然后以正式 world size、BF16 autocast、batch 和 workers 做完整训练/短评估 smoke。
单卡 kernel 成功不能替代真实多进程、多卡训练。验证 wrapper 必须传递失败退出码。
GPU 和重型计算在 `sbatch` allocation 中执行；正文直接运行 Python/torchrun，不用
`srun/salloc`，不在登录节点编译或试跑。account、partition、GPU/CPU、时限和日志路径
根据目标集群设置，先 `sbatch --test-only` 再提交，保存 Job ID 与实际资源。

启动 Python 前设置短且独享的 TMPDIR，避免 DataLoader 的 pathname socket 超限：

```bash
set -euo pipefail
export TMPDIR=$(mktemp -d /tmp/tg.XXXXXX)
python scripts/pretraining/check_ipc.py --output "$record/ipc.json"
# 然后运行本次生成的 smoke.sh / launch.sh；record 必须是本次新目录。
```

`check_ipc.py` 检查 filesystem socket、spawn worker 和张量传输，并拒绝覆盖旧凭据。
缓存、日志可在长路径，IPC 临时目录不能复用长 campaign 路径。native builder 的缓存
可能随 allocation 变化，训练节点也须具备编译器。不要并发修改运行任务的共享环境。

## 7. microbatch 验证与正式训练

- `search_max`：在保持 global batch 下搜索可用 microbatch；每候选从零做 3 个 optimizer
  steps 和 dev/test 短评估，最大候选做 10 步确认。只有明确 CUDA OOM 才缩小 batch；
  环境异常、超时和非有限值直接失败。
- `fixed`：按明确指定值做真实训练验证，可有末尾小 batch。记录 `fixed_value_validated`，
  不记作最大值。外层 timeout 和允许成本写入 protocol。

smoke 不保存正式 checkpoint，不继承其权重；通过后重新初始化正式训练，恢复完整评估、
保存和停止策略。短 smoke 不能保证任意后续 batch 的容量；调整 microbatch 后记录原因，
维持 global batch/LR/数据和进度。

## 8. 续训和完成

队列消失后核验 `sacct`、退出码、日志、optimizer step 和 checkpoint。
零步失败可以修复后新 attempt 从零重提，原证据本地保留；已有 checkpoint 的续训保持
seed、LR、batch、scheduler、数据与进度，使用 `restore_dataloader=true`、
`reset_optimizer_state=false`、`reset_trainer_state=false`，新 continuation 输出目录，
不重走从零探测入口。不要用覆盖标志掩盖冲突。

完成需要 epoch/步数证据、成功退出、可恢复 checkpoint 和最终评估。
提交成功、probe 成功和时限到达都不是完成。成本累计包括 setup、失败、probe 和续训。

## 9. 结果登记

本地原 campaign 保存配置、数据/源码/环境身份、各阶段回执、checkpoint 和日志。
公开结果仅提取有效最终指标、必要配置/身份和完成范围；失败过程、排队状态留在本地。
新结果回填现有结果页及其机器可读证据，避免第二份人工滚动表。

## 10. 公开复现交付与验收

[公开材料清单](../reproducibility/README.md) 区分：已包含的源代码/配置/结果证据、
仍需下载的数据/权重，以及实际执行过的验证。hash 不能替代下载或构建入口。
在不含本地归档的公开文件副本中检查链接、配置来源和 CPU 测试；从外部新环境完成
构建、GPU smoke 与完整重训是另一个验收层级，不能由上述检查推断。
