import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.nn.functional as F

import torch.nn as nn
import time
import statistics
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from flash_attn import (  # type: ignore
                    flash_attn_func,
                    flash_attn_varlen_func,
                )
from olmo.data.tg_mask import SentencepieceVocab, TG_attention_bias

dropout_p = 0.0
attn_mask = None
is_causal = True

Q = torch.ones(1, 2, 3, 4)
K = Q
V = Q
res = flex_attention(Q,K,V)

F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )



class AttentionProfiler:
    def __init__(self, device='cuda', dtype=torch.bfloat16):
        """
        注意力分析器初始化
        :param device: 计算设备 (cuda/cpu)
        :param dtype: 数据类型 (float32/float16)
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.results = {}
        
    def generate_data(self, batch_size, n_heads, seq_len, embed_dim):
        """
        生成测试数据
        :return: Q, K, V 张量
        """
        return (
            torch.randn(batch_size,  n_heads, seq_len, embed_dim, 
                       device=self.device, dtype=self.dtype),
            torch.randn(batch_size,  n_heads, seq_len, embed_dim, 
                       device=self.device, dtype=self.dtype),
            torch.randn(batch_size,  n_heads, seq_len, embed_dim, 
                       device=self.device, dtype=self.dtype)
        )

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        手动实现的缩放点积注意力
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=self.dtype))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)

    def pytorch_attention(self, Q, K, V, mask=None):
        """
        使用PyTorch内置的MultiheadAttention
        """
        attn = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        return attn
    
    def flash_attn(self, Q, K, V, mask=None):
        return flash_attn_func(  #(batch_size, seqlen, nheads, headdim)
                Q.transpose(1,2),
                K.transpose(1,2),
                V.transpose(1,2),
                dropout_p=0.0,
                causal=False
        )
    
    def flex_attn(self, Q, K, V, mask=None):
        return self.flex_attention(Q,K,V, block_mask=self.block_mask)

    def profile(self, attention_fn, Q, K, V, mask=None, warmup=50, repeats=500):
        """
        性能测试函数
        :param attention_fn: 注意力函数
        :param warmup: 预热次数
        :param repeats: 正式测试重复次数
        :return: 平均运行时间(ms), 标准差
        """
        # 预热
        for _ in range(warmup):
            _ = attention_fn(Q, K, V)
            torch.cuda.synchronize()  # 确保CUDA操作完成
        
        # 正式计时
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            _ = attention_fn(Q, K, V, mask)
            torch.cuda.synchronize()  # 等待GPU操作完成
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # 转换为毫秒
        
        return statistics.mean(timings), statistics.stdev(timings)

    def run_tests(self, configs, attn_mask=None):
        """
        运行多组测试配置
        :param configs: 测试配置列表
        """
        for config in tqdm(configs, desc="Running tests"):
            Q, K, V = self.generate_data(**config)
            
            # 测试手动实现
            manual_time, manual_std = self.profile(
                self.scaled_dot_product_attention, Q, K, V, attn_mask)
            
            # 测试PyTorch实现
            pytorch_time, pytorch_std = self.profile(
                self.pytorch_attention, Q, K, V, attn_mask)

            flash_time, flash_std = self.profile(
                self.flash_attn, Q, K, V)
            
            flex_time, flex_std = self.profile(
                self.flex_attn, Q, K, V, attn_mask)

            # 保存结果
            key = f"bs{config['batch_size']}_seq{config['seq_len']}_emb{config['embed_dim']}"
            self.results[key] = {
                'manual_attention': manual_time,
                'pytorch_attention': pytorch_time,
                'flash_attention': flash_time,
                'flex_attention': flex_time,
                # 'manual_std': manual_std,
                # 'pytorch_std': pytorch_std,
                'config': config.copy()
            }

    def plot_results(self, save_path=None):
        """
        可视化测试结果
        :param save_path: 图片保存路径
        """
        if not self.results:
            print("No results to plot")
            return
        
        labels = list(self.results.keys())
        manual_times = [v['manual_attention'] for v in self.results.values()]
        pytorch_times = [v['pytorch_attention'] for v in self.results.values()]

        x = range(len(labels))
        width = 0.35

        plt.figure(figsize=(14, 8))
        plt.bar(x, manual_times, width, label='Manual Attention', yerr=[v['manual_std'] for v in self.results.values()])
        plt.bar([i + width for i in x], pytorch_times, width, label='PyTorch Attention', 
               yerr=[v['pytorch_std'] for v in self.results.values()])

        plt.xlabel('Test Configurations')
        plt.ylabel('Execution Time (ms)')
        plt.title('Attention Mechanism Performance Comparison')
        plt.xticks([i + width/2 for i in x], labels, rotation=45)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved plot to {save_path}")
        plt.show()

    def print_results(self):
        """打印测试结果表格"""
        print("\n" + "="*70)
        print(f"{'Configuration':^30} | {'Manual (ms)':^12} | {'PyTorch (ms)':^12} | {'Flash (ms)':^12} | {'Flex (ms)':^12}")
        print("="*70)
        for key, res in self.results.items():
            speedup = res['manual_attention'] / res['pytorch_attention']
            print(f"{key:30} | {res['manual_attention']:12.4f} | "
                  f"{res['pytorch_attention']:12.4f} | {res['flash_attention']:12.4f} | {res['flex_attention']:12.4f} |")
        print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Attention Profiler')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['float32', 'float16', 'bf16'],
                       help='Data precision')
    parser.add_argument('--plot', type=str, default='attention_perf.png',
                       help='Path to save performance plot')
    args = parser.parse_args()

    # 配置测试参数
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bf16': torch.bfloat16}
    profiler = AttentionProfiler(
        device=args.device,
        dtype=dtype_map[args.dtype]
    )

    seq_len = 2048
    test_configs = [
        # {'batch_size': 16, 'seq_len': 128, 'embed_dim': 256},
        # {'batch_size': 16, 'seq_len': 512, 'embed_dim': 256},
        # {'batch_size': 64, 'seq_len': 128, 'embed_dim': 512},
        # {'batch_size': 64, 'seq_len': 512, 'embed_dim': 512},
        # {'batch_size': 4, 'n_heads': 2, 'seq_len': 1024, 'embed_dim': 512},
        {'batch_size': 8, 'n_heads': 12, 'seq_len': seq_len, 'embed_dim': 64},
    ]

    data = np.load("../dataset/bbc-news/tg/test.npy")
    vocab_path = "../dataset/bbc-news/TG_GPT2_tokenizer.json"
    tokenizer = Tokenizer.from_file(vocab_path)
    input_ids = torch.tensor(data[:seq_len], dtype=torch.long)
    TG_mask = TG_attention_bias("../dataset/bbc-news/TG_GPT2_tokenizer.json", seq_len)
    print(tokenizer.decode(data[:seq_len], skip_special_tokens=False))
    attn_mask, label_mask = TG_mask(input_ids)
    attn_mask = attn_mask.cuda()
    # print(attn_mask)
    # exit(0)

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def TG_causal(b, h, q_idx, kv_idx):
        return attn_mask[q_idx, kv_idx]
    
    profiler.block_mask = create_block_mask(TG_causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)
    # flex_attention(Q,K,V, block_mask=block_mask)
    profiler.flex_attention = torch.compile(flex_attention)

    Q, K, V = profiler.generate_data(**test_configs[0])
    # flash = profiler.flash_attn(Q, K, V).transpose(1,2)
    flex = profiler.flex_attn(Q, K, V)
    # print(flex.shape)
    Fnn = profiler.pytorch_attention(Q, K, V, attn_mask)
    # print(flash) 
    # print(flex)
    # assert torch.all(torch.isclose(flash, flex, rtol=1e-2, atol=1e-1))
    assert torch.all(torch.isclose(flex, Fnn, rtol=1e-2, atol=1e-1))
    # 运行测试
    profiler.run_tests(test_configs, attn_mask)
    
    # 输出结果
    profiler.print_results()
    # profiler.plot_results(save_path=args.plot)