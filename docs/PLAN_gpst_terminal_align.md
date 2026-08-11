# GPST Pretrain 对标 terminal.yaml (OLMo-300M)

## 目标
配置 GPST unsupervised pretrain,与 `train_configs/terminal.yaml` 的 OLMo-300M 干净对标:相同数据(BBC terminal)、相同 tokenizer(vocab 50320)、相同骨架规模(768/12/12)、相同 token budget(~4.26B)、对齐 norm/激活/flash(RoPE 因 GPST 内在约束必须关)。对比只隔离"结构化 composition"这一个变量。

## 对齐表

| 轴 | terminal.yaml | GPST 默认 | 动作 |
|---|---|---|---|
| tokenizer/vocab | BBC TG 50320 | gpt2-small 50264 | 数据已 50320;模型 vocab 设 50320 |
| d_model/n_head/n_layer | 768/12/12 | 768/12/12 | ✅ |
| activation | swiglu | gelu | OLMoStack→swiglu |
| norm | rms | default | OLMoStack→rms |
| RoPE | true | false | 保持 false(GPST 必须用 learned wpe) |
| flash_attn | true | false | OLMoStack→flash=True |
| weight_tying | true | true | ✅ |
| max_seq_len | 2048 | 1024 | 2048 |
| precision | amp_bf16 | fp16+GradScaler | trainer→bf16(去 GradScaler) |
| 数据 | terminal 流,doc 边界 | 我之前的(整篇=1段,截断1024) | **重生成:tree.npy 提 terminal+句子边界,不截断** |
| token budget | 14425×144×2048≈4.26B | num_samples | num_samples≈2.08M |
| LR/sched | 5.3e-3 cosine | 5e-5 线性 | 保留 GPST 自有 LR(标准做法) |

## 步骤

### 1. 重写数据转换:tree.npy → lazy(terminal + 句子边界)
替换 `scripts/gpst/terminal_npy_to_lazy.py` 逻辑。原版从 terminal.npy 整篇切、无句子边界,会撑爆 chart(parser_max_len=1024)。
- 流式 mmap 遍历 `dataset/bbc-news/tree/train.npy`(247亿 tok,~47GB)
- `<|beginoftext|>`(50257) 开新 doc
- 跳过所有非终结符(50268-50319)
- `<(S>`(50282)/`<S)>`(50308) 切句:每句 terminal 拼成一个 entry
- doc 内多句→多个非零长度 entry;doc 末写 0(LazyLoader doc 分隔符)
- 不截断;输出 `corpus/bbc-tree.lazy/{data,data.len.pkl}`
- dev.npy 先验证:句子数、token 数、LazyLoader+collate 加载、单句长度中位~10

### 2. 新建 gpt config(vocab 50320)
`olmo/gpst/data/gpt2-bbc/config.json`:复制 gpt2-small,改 `vocab_size:50320`、`n_ctx:2048`、`bos_token_id:50257`。

### 3. OLMoStack backbone 对齐(model_factory.py:41)
`_gpt_config_to_olmo_model_config`:`activation_type=swiglu`、`layer_norm_type=rms`、`flash_attention=True`、`weight_tying=True`、`layer_norm_eps=1e-6`;`rope=False` 保持。

### 4. trainer 精度 bf16(trainer.py)
autocast→`dtype=torch.bfloat16`;去掉 GradScaler(scale/unscale/step/update→直接 backward/step);保留 no_sync+two-backward+all_reduce。可选:scheduler 改 cosine。

### 5. sbatch + wrapper
- 修 `pretrain_gpst_small_unsupervised.sh` 第17行 `}` 笔误
- 参数:`--corpus_path corpus/bbc-tree.lazy --gpt_config_path olmo/gpst/data/gpt2-bbc/config.json --max_seq_len 2048 --batch_size 32 --num_samples 2079744 --gradient_checkpoint`
- token 核算:8×32×2048=524288 tok/step;2.08M/(256)=8124 步;×524288≈4.26B ✓(0.42 epoch,同 baseline)
- sbatch:`-c 1 --mem-per-cpu=1M --gres=gpu:8`,不手设 CUDA_VISIBLE_DEVICES

### 6. 单卡冒烟→8 卡正式
先 `--nproc-per-node=1 --num_samples 256 --batch_size 4 --max_seq_len 2048` 验证 hard-EM 前向/反向;通过后 8 卡。

## 验证清单
- [ ] dev 转换:句子数>0、单句≤1024、collate 产 batch 无错
- [ ] gpt-bbc vocab=50320,embedding shape 对
- [ ] OLMoStack swiglu/rms/flash 前向 shape 对
- [ ] trainer bf16 单步 backward 无 GradScaler 报错
- [ ] 单卡冒烟 256 sample 跑通,loss 下降
- [ ] 8 卡正式 8124 步≈4.26B token

## 风险
- chart 显存@2048:按句子段算但 batch×段数×段长² 约 2×;已开 gradient_checkpoint,OOM 则降 batch 16 或 seq 1024。
- parser_max_len=1024:单句超 1024 被裁;BBC 句中位~10,风险低。
- bf16 改 trainer:保留 all_reduce 时序,勿破坏 two-backward。
