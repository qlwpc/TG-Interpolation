from __future__ import annotations
import logging

from typing import List, Optional, Sequence, Tuple, Callable

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'Qwen/Qwen3-0.6B-Base'  # <- 请替换为你要使用的模型 id
NEW_TOKENS = [f"<extra_token_{i}>" for i in range(20)]
USE_FLASH_ATTENTION = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 加载 tokenizer 和模型
print(f"加载 tokenizer 和 model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False)
# 2. 导出到一个干净的目录
# save_directory = "/home/wangpch/TG-Interpolation/Qwen3-0.6B-Base"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)