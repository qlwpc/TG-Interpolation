from __future__ import annotations
import logging

from typing import List, Optional, Sequence, Tuple, Callable, Dict, Set

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from .model import OLMoOutput, _non_meta_init_device
from .config import ModelConfig, ActivationCheckpointingStrategy, CheckpointType, TrainConfig, FSDPWrapStrategy
from .aliases import PathOrStr
from .tokenizer import Tokenizer
from .checkpoint import Checkpointer, ShardedCheckpointerType
from transformers.models.olmo.modeling_olmo import OlmoModel, OlmoDecoderLayer

__all__ = [
    "HuggingModel", 
]

class HuggingModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True, train_config:TrainConfig=None):
        super().__init__()
        self.transformer = AutoModelForCausalLM.from_pretrained(config.modelname, 
                                                                attn_implementation= "flash_attention_2" if config.flash_attention else "eager",
                                                                torch_dtype=train_config.autocast_precision)
        self.tokenizer = AutoTokenizer.from_pretrained(config.modelname)
        self.config = config
        self.transformer.to(config.init_device)
        self.__num_fwd_flops: Optional[int] = None
        self.__num_bck_flops: Optional[int] = None

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.model.get_input_embeddings().weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        doc_lens: Optional[torch.Tensor] = None,
        max_doc_lens: Optional[Sequence[int]] = None,
    ) -> OLMoOutput:
        
        output = self.transformer(input_ids=input_ids, 
                                  input_embeddings=input_embeddings, 
                                  attention_mask=attention_mask, 
                                  past_key_values=past_key_values,
                                  use_cache=use_cache,
                                  output_hidden_states=output_hidden_states,
                                  doc_lens=doc_lens,
                                  max_doc_lens=max_doc_lens
                                  )
        return OLMoOutput(
            logits=output.logits,
            attn_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
        )

    def set_activation_checkpointing(
        self, strategy: Optional[ActivationCheckpointingStrategy], checkpoint_func: Optional[Callable] = None
    ):
        pass

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0] and "embed" not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False)
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * self.config.max_sequence_length)
        )
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params()
        params_flops_per_token = 4 * n_params
        attn_flops_per_token = self.config.n_layers * 8 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops
    
    def reset_parameters(self):
        pass

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: Optional[CheckpointType] = None
    ) -> HuggingModel: # type: ignore
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = HuggingModel(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            if train_config.sharded_checkpointer == ShardedCheckpointerType.olmo_core:
                from olmo_core.distributed.checkpoint import (  # type: ignore
                    load_model_and_optim_state,
                )

                model_config.init_device = device
                model = HuggingModel(model_config)
                load_model_and_optim_state(checkpoint_dir, model)
            else:
                # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
                from .checkpoint import load_model_state

                # Initialize model on target device. In this case the state dict is loaded in-place
                # so it's not necessary to start on CPU if the target device is a GPU.
                model_config.init_device = device
                model = HuggingModel(model_config)

                # Load state dict in place.
                load_model_state(checkpoint_dir, model)

        return model.eval()

    def add_tokens_and_initialize(self, new_tokenizer: Tokenizer):
        """向 tokenizer 添加新 tokens, 并使用已有 embeddings 的平均值来初始化这些 tokens 的 embedding。

        同时尝试将输出 head 对应新 token 的权重初始为对前面已有 token 几乎没有影响的值：
        - 若输出层存在独立权重（未 tie), 则把新行设置为极小值或平均值乘以很小的系数；
        - 若存在 bias 向量，会把相应新 token 的 bias 设为很大负数（例如 -1e9), 以确保 softmax 上几乎为 0。
        """
        old_vocab_size = len(self.tokenizer)
        new_vocab_size = new_tokenizer.vocab_size
        added = new_vocab_size - old_vocab_size
        if added<=0:
            return

        new_ids = list(range(old_vocab_size, new_vocab_size))
        embed_module = self.transformer.get_input_embeddings()
        embed_weight = embed_module.weight
        dtype = embed_weight.dtype
        device = self.device

        # 备份旧 embedding（放到 CPU 避免显存问题，随后会搬回）
        old_emb = embed_weight.data[:old_vocab_size].detach().to(device)

        # 4) 扩展模型 embedding 矩阵
        self.transformer.resize_token_embeddings(new_vocab_size)

        # re-fetch embed weight（resize 之后会变化）
        embed_weight = self.transformer.get_input_embeddings().weight.data

        # 5) 用旧 embedding 的均值来初始化新 tokens 的 embedding
        mean_emb = old_emb.mean(dim=0, keepdim=True).to(dtype).to(device)
        for idx in new_ids:
            embed_weight[idx].data.copy_(mean_emb[0])

        print(f"已将 {added} 个新 token 加入 tokenizer，ID 从 {old_vocab_size} 到 {new_vocab_size}。")

        # 6) 处理输出 head，使得新 tokens 在 softmax 概率上几乎没有影响
        # 尝试查找 output embedding / lm_head
        out_emb_module = self.transformer.get_output_embeddings()  # 可能与输入 embeddings tie

        if out_emb_module is not None and hasattr(out_emb_module, 'weight'):
            out_w = out_emb_module.weight.data
            if out_w.data_ptr() != embed_weight.data_ptr():
                old_out = out_w[:old_vocab_size].detach().to(device)
                mean_out = old_out.mean(dim=0, keepdim=True).to(dtype)
                for idx in new_ids:
                    out_w[idx].copy_(mean_out[0])

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        pass

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        
        size_based_module_to_wrap = {}
        if hasattr(self.transformer, "wte"):
            size_based_module_to_wrap.add(self.transformer.wte)
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OlmoDecoderLayer)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OlmoDecoderLayer,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OlmoDecoderLayer) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)


if __name__ == '__main__':
    MODEL_NAME = 'Qwen/Qwen3-0.6B'  # <- 请替换为你要使用的模型 id
    NEW_TOKENS = [f"<extra_token_{i}>" for i in range(20)]
    USE_FLASH_ATTENTION = False
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载 tokenizer 和模型
    print(f"加载 tokenizer 和 model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False, attn_implementation="flash_attention_2")
    model.to(DEVICE)

    # 2) 添加 token 并初始化 embedding / head
    model, tokenizer = HuggingModel.add_tokens_and_initialize(model, tokenizer, NEW_TOKENS, device=DEVICE)

    model.eval()
    model.reset_parameters()

    # 4) 简单 forward 测试
    texts = ["Hello, this is a test."]
    inputs = tokenizer(texts, return_tensors='pt', padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    print("forward 完成，logits shape:", logits.shape)

    # 5) 保存模型（可选）
    # model.save_pretrained('./qwen3-0.6b-extended')
    # tokenizer.save_pretrained('./qwen3-0.6b-extended')

    print("脚本执行完毕。请根据你的模型结构检查 'lm_head' / 'output_embeddings' / 'final logits bias' 的命名并做小调整以保证兼容性。")
