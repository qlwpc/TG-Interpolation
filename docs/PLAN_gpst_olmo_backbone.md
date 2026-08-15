# Plan: GPST 主 Transformer 改用 OLMo 架构

## 目标

GPST 当前生成模型的 type/token 两段 transformer 是 HuggingFace `GPT2Model` 的薄包装
(`olmo/gpst/model/gpt2_flash_attn.py`)。目标是让 GPST 能选用 **OLMo 架构**(`olmo/model.py`
的 `OLMoBlock` 栈)作为主干，而 `FastGenerativeR2D2`、trainer、reader、C++ 后端等
**全部不动**。HF GPT2 路径作为默认继续可用，OLMo 作为新增可选 backbone。

## 关键约束(决定方案取舍)

1. **GPST 的 transformer 是"两段式可分别调用的子栈"**: `FastGenerativeR2D2` 实例化
   两个独立 transformer(`action_layers` 浅 / `generation_layers` 深)，各自接收
   `inputs_embeds`(surrogate 表示)并独立 forward，中间还插一次 `gather(next_token_indices)`
   (`generative_r2d2_fast.py:135,150`)。因此 **不能直接调 `OLMo.forward`**(它把所有层
   一次跑完、内部处理 RoPE/wpe、无中途 gather 入口)。必须复用 **`OLMoBlock`** 而非
   `OLMo` 顶层。

2. **position_ids 是 tree-ordered 的非连续整数**(来自 C++ `prepare_generation`，
   post-order 遍历索引)。HF GPT2 用 *learned* `wpe` 按这些 id 查表，任意整数都能用。
   而 `OLMoBlock` 的 RoPE 在 `attention()` 内部按 `key_len/query_len` 自算位置
   (`model.py:317-318,644-646`)，**不接受外部 position_ids**。因此直接套 RoPE 会丢失
   GPST 的位置语义。

3. **past_key_values**: GPST 用 HF 风格 per-layer `(k,v)` tuple 列表；`OLMoBlock.forward`
   返回 `(present_k, present_v)` per layer。两者结构一致，仅做列表组织即可适配，无需改 OLMo。

## 设计取舍: 位置编码

选 **方案 A(learned wpe, 关闭 RoPE)** — 精确复刻 GPT2 语义:

- `OLMoStack` 配 `rope=False`、`alibi=False`，自带一个 `nn.Embedding(max_seq, d_model)`
  的 `wpe`，在 `forward` 入口处 `x = x + wpe(position_ids)`，再喂给 block 栈。
- 这与 HF `GPT2Model` 内部行为一致，position_ids 仍是 tree-ordered 任意整数 → 语义不变。
- 完全不动 `olmo/model.py` 主干；RoPE/position 改造全在 `olmo/gpst/model/` 内部。

否决方案 B(给 RoPE 注入外部 positions): 需改 `RotaryEmbedding.forward` 与
`OLMoBlock.attention` 签名，侵入核心模型，风险高且偏离 GPST 原意。可在未来作为可选项再加。

## 实现内容

### 1. 新建 `olmo/gpst/model/backbone_common.py`

抽出共享类型 `_ModelOutput`(dataclass: `last_hidden_state`, `past_key_values`)，供
`gpt2_flash_attn.py` 与新 `olmo_stack.py` 共用，避免循环导入。`gpt2_flash_attn.py` 改为
从此 import。

### 2. 新建 `olmo/gpst/model/olmo_stack.py`

`OLMoStack(nn.Module)`，对外暴露与现有 `GPT2Model` 完全相同的接口(让
`FastGenerativeR2D2` 无感切换):

```python
class OLMoStack(nn.Module):
    def __init__(self, olmo_model_config, n_layers, no_embedding=True,
                 no_layer_norm=False, add_position=True, init_device=None):
        # 1. 从 olmo_model_config 构造 BufferCache
        # 2. self.blocks = nn.ModuleList([OLMoBlock.build(i, cfg, cache) for i in range(n_layers)])
        # 3. if add_position: self.wpe = nn.Embedding(max_seq, d_model)
        # 4. self.ln_f = Identity() if no_layer_norm else LayerNorm.build(cfg)

    @property
    def gradient_checkpointing(self): ...
    @gradient_checkpointing.setter
    def gradient_checkpointing(self, v): ...  # 透传 block.set_activation_checkpointing

    def forward(self, inputs_embeds=None, position_ids=None,
                past_key_values=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        if self.add_position and position_ids is not None:
            x = x + self.wpe(position_ids)
        bias = self._build_causal_pad_bias(attention_mask, x.shape[1])  # None if no mask
        present_kvs = []
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, attention_bias=bias, layer_past=layer_past,
                               use_cache=past_key_values is not None or kwargs.get("use_cache", False))
            present_kvs.append(present)
        x = self.ln_f(x)
        return _ModelOutput(last_hidden_state=x, past_key_values=present_kvs)
```

- `attention_bias`: 无 mask 时传 `None` → block 走 `is_causal=True` 标准因果 SDPA
  (`model.py:684`)，与 GPT2 因果掩码等价。有 padding 时 `_build_causal_pad_bias` 把
  `(B,T)` mask 编成 `(B,1,T,T)` 加性 bias(因果下三角 ∩ padding)，传给 `attention_bias`
  并令 OLMoBlock 走非 causal SDPA 路径。

### 3. `olmo/gpst/model/model_factory.py` — 增加 backbone 选择

`create_model(...)` 增参 `backbone: str = "gpt2"`:

```python
def create_model(model_type, r2d2_config_path, gpt_config_path,
                 fix_embeddings=False, gradient_checkpoint=False, backbone="gpt2"):
    if model_type == 'r2d2-gen-fast':
        ...
        if backbone == "gpt2":
            action_transformers = GPT2Model(...); gpt_transformers = GPT2Model(...)
        elif backbone == "olmo":
            from olmo.gpst.model.olmo_stack import OLMoStack
            olmo_cfg = _gpt_config_to_olmo_model_config(gpt_config, init_device)
            action_transformers = OLMoStack(olmo_cfg, n_layers=action_layer_num,
                                           no_embedding=True, no_layer_norm=True)
            gpt_transformers = OLMoStack(olmo_cfg, n_layers=total-action_layer_num,
                                         no_embedding=True, no_extra_embedding=True)
```

新增 `_gpt_config_to_olmo_model_config(gpt_config, init_device) -> ModelConfig`:
把 HF `GPT2Config`(n_embd, n_head, n_layer, vocab_size, n_positions)映射到
`ModelConfig`(d_model, n_heads, n_layers, vocab_size, max_sequence_length)，
`rope=False`、`block_type=sequential`、`init_fn=normal`、`init_device` 透传。
`activation_type` 默认 swiglu(OLMo 原生)，可选 gelu 贴近 GPT2 做对照。

### 4. `scripts/gpst/run_gpst.py` — 暴露 CLI

加 `--backbone {gpt2,olmo}`(默认 `gpt2`)，透传给 `create_model`。

## TDD 测试计划

新增 `tests/test_gpst_olmo_backbone.py`(CPU, 与现有 GPST 测试同风格 AAA):

1. **`test_olmo_stack_forward_shape`** — 小 `OLMoStack`(d_model=64,n_heads=4,
   n_layers=2)，随机 `inputs_embeds` + `position_ids`，断言 `last_hidden_state`
   形状 `(B,T,d_model)` 且有限。
2. **`test_olmo_stack_position_ids_honored`** — 不同 `position_ids` → 不同输出
   (证明 wpe 真用了外部 positions)。
3. **`test_olmo_stack_past_key_values`** — 两次 forward(无 past / 有 past)输出一致，
   断言 `past_key_values` 长度=n_layers。
4. **`test_olmo_stack_causal`** — 改变未来 token 不影响过去 token 输出(因果性)。
5. **`test_olmo_stack_padding`** — 带 padding 的 batch，padding 位不影响有效位。
6. **`test_create_model_olmo_backbone`** — `create_model('r2d2-gen-fast', ...,
   backbone='olmo')` 返回的 `FastGenerativeR2D2` 其 `action_layers`/`generation_layers`
   是 `OLMoStack` 实例。
7. **`test_trainer_olmo_backbone_cpu`** — 复用 `test_gpst_trainer.py` 的 batch，
   `backbone='olmo'` 跑 2 步 hard-EM，断言 loss 有限、`a_ij_require_grad` 切换正常。
8. **`test_grad_stop_toggle_olmo`** — grad-stop 在 OLMo backbone 下仍生效。

现有 6 个 GPST 测试(`tests/test_gpst_*.py`)必须继续全绿(backbone 默认 gpt2 不变)。

## 执行顺序

1. 写 `tests/test_gpst_olmo_backbone.py`(RED)。
2. 实现 `backbone_common.py` + `olmo_stack.py`(GREEN: 1-5 过)。
3. 改 `model_factory.py`(加 backbone 分支 + converter)(GREEN: 6-8 过)。
4. 改 `run_gpst.py` CLI。
5. 跑全套 GPST 测试确认无回归。

## 不改动的文件(明确边界)

- `olmo/model.py`、`olmo/config.py`(主干 — 零修改)
- `olmo/gpst/model/generative_r2d2_fast.py`(消费方接口不变)
- `olmo/gpst/trainer/*`、`olmo/gpst/reader/*`、`olmo/gpst/data_structure/*`、C++ 后端
- 现有 `GPT2Model`(保留为默认 backbone, 仅抽出共享类型)

## 风险与回退

- **风险**: `OLMoSequentialBlock` 默认 `activation_type=swiglu` + GQA 等，参数量/行为与
  GPT2(gelu, MHA)不同 → 这是"换架构"的预期差异。converter 可选 `activation_type='gelu'`、
  `n_kv_heads=None`(MHA)以贴近 GPT2 做对照实验。
- **回退**: backbone 默认 `gpt2`，任何 OLMo 路径异常都不影响现有训练/测试。
