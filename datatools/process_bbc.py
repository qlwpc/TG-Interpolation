import os
import numpy as np
import torch
import multiprocessing as mp
from functools import partial
from pathlib import Path

# 假设来自本地模块
from olmo.data.tg_mask import SentencepieceVocab

def process_chunk(chunk_data, vocab, mode, fixed_token=None):
    """
    处理数组的一个分段
    返回: (处理后的数组, 原始分段是否以Closing开头, 原始分段是否以Closing结尾)
    """
    if len(chunk_data) == 0:
        return np.array([], dtype=np.uint16), False, False

    # 预识别类型
    # 注意：在子进程中频繁调用 vocab 方法可能受限于 Python 序列化性能
    # 建议将 vocab 的判断逻辑向量化或缓存
    vocab = SentencepieceVocab.from_vocab_file(vocab)
    is_opening = (chunk_data < vocab.opening_non_terminals[1]) & (chunk_data >= vocab.opening_non_terminals[0])
    is_closing = (chunk_data < vocab.closing_non_terminals[1]) & (chunk_data >= vocab.closing_non_terminals[0])
    print(chunk_data)
    
    starts_with_closing = bool(is_closing[0])
    ends_with_closing = bool(is_closing[-1])
    processed = None

    if mode == 1:
        processed = chunk_data[~is_opening]
    
    elif mode == 2:
        res = []
        i = 0
        while i < len(chunk_data):
            if is_closing[i]:
                res.append(fixed_token)
                while i < len(chunk_data) and is_closing[i]:
                    i += 1
            else:
                res.append(chunk_data[i])
                i += 1
        processed = np.array(res, dtype=np.uint16)

    elif mode == 3:
        res = []
        for i in range(len(chunk_data)):
            if is_closing[i]:
                res.extend([chunk_data[i]] * 3)
            else:
                res.append(chunk_data[i])
        processed = np.array(res, dtype=np.uint16)
    print(processed)
    return processed, starts_with_closing, ends_with_closing

def parallel_process_file(file_path, output_dir, vocab, mode, fixed_token, n_workers):
    print(f"正在处理: {os.path.basename(file_path)} (使用 {n_workers} 线程/进程)")
    
    # 1. 使用 mmap 模式加载，避免一次性将大文件读入 RAM
    data = np.load(file_path, mmap_mode='r')
    
    # 2. 切分索引
    chunks = np.array_split(data, n_workers)
    
    # 3. 并行执行
    worker_func = partial(process_chunk, vocab=vocab, mode=mode, fixed_token=fixed_token)
    with mp.Pool(n_workers) as pool:
        results = pool.map(worker_func, chunks)

    # 4. 合并结果 (处理边界问题)
    final_list = []
    for i in range(len(results)):
        curr_processed, curr_starts_closing, curr_ends_closing = results[i]
        
        if mode == 2 and i > 0:
            # 边界检查：如果上一个 Chunk 结尾是 Closing 且当前开头也是 Closing
            prev_processed, prev_starts, prev_ends = results[i-1]
            if prev_ends and curr_starts_closing:
                # 方式二下，当前分段开头的第一个 token 必定是 fixed_token，需剔除
                curr_processed = curr_processed[1:]
        
        final_list.append(curr_processed)

    # 拼接并保存
    final_data = np.concatenate(final_list).astype(np.uint16)
    save_path = os.path.join(output_dir, os.path.basename(file_path))
    final_data.tofile(Path(save_path).with_suffix('.npy'))
    print(f"保存完毕: {save_path}, 最终长度: {len(final_data)}")

def main():
    input_dir = "./dataset/bbc-news/tree/"
    output_dir = "./dataset/bbc-news/processed/"
    mode = 2
    fixed_token_id = 50319
    vocab = "./dataset/bbc-news/TG_GPT2_tokenizer.json"
    
    # 自动根据系统配置
    total_cpus = 1
    
    # 获取文件列表
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('dev.npy')]
    

    # # 模拟 Vocab (实际请使用你的 tg_mask 模块)
    # class DummyVocab:
    #      def is_opening_non_terminal(self, t): return 500 <= t < 600
    #      def is_closing_non_terminal(self, t): return 600 <= t < 700
    # ;    def is_terminal(self, t): return t < 500

    for mode in range(1, 4):
        cur_output = os.path.join(output_dir, str(mode))
        if not os.path.exists(cur_output):
            os.makedirs(cur_output)
        for f_path in files:
            # 根据单个文件大小动态分配 Worker (例如每 100MB 分配一个核心，最多不超过总核心数)
            file_size_gb = os.path.getsize(f_path) / (1024**3)
            # 假设 1GB 以上的文件使用全部核心，小的减半
            n_workers = total_cpus if file_size_gb > 0.5 else max(1, total_cpus // 2)
            
            parallel_process_file(f_path, cur_output, vocab, mode, fixed_token_id, n_workers)

if __name__ == "__main__":
    main()
