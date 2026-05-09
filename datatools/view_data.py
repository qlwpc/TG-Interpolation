import os
import numpy as np
import random
# 假设你的 tokenizer 是 SentencePiece
from tokenizers import Tokenizer
from olmo.data.tg_mask import SentencepieceVocab

def verify_bin_data(bin_path, tokenizer_path, sample_num=3, seq_len=100):
    """
    加载二进制文件并采样验证内容
    """
    # 1. 加载 Tokenizer 和 Vocab 判别器
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab = SentencepieceVocab.from_vocab_file(tokenizer_path)

    # 2. 从二进制文件读取数据 (uint16)
    if not os.path.exists(bin_path):
        print(f"错误：文件 {bin_path} 不存在")
        return

    # 使用 memmap 或 fromfile
    # tofile 保存的数据没有 header，直接按类型读取即可
    data = np.fromfile(bin_path, dtype=np.uint16)
    total_tokens = len(data)
    print(f"--- 文件信息 ---")
    print(f"路径: {bin_path}")
    print(f"总 Token 数: {total_tokens}")
    print(f"数据类型: {data.dtype}")
    print("-" * 30)

    # 3. 随机采样
    for i in range(sample_num):
        start_idx = random.randint(0, max(0, total_tokens - seq_len))
        end_idx = start_idx + seq_len
        sample_ids = data[start_idx:end_idx].tolist()

        print(f"\n[采样 {i+1}] (索引 {start_idx} 到 {end_idx}):")
        
        # --- 可视化 Token 类型 ---
        # 帮助肉眼观察：[O] 代表 Opening, [C] 代表 Closing, [T] 代表 Terminal
        visual_parts = []
        for tid in sample_ids:
            token_str = tokenizer.decode([tid], skip_special_tokens=False)
            # 处理不可见字符
            if not token_str.strip():
                token_str = f"<{tid}>"
            
            if vocab.is_opening_non_terminal(tid):
                visual_parts.append(f"\033[94m{token_str}(O)\033[0m") # 蓝色
            elif vocab.is_closing_non_terminal(tid):
                visual_parts.append(f"\033[92m{token_str}(C)\033[0m") # 绿色
            else:
                visual_parts.append(token_str)

        print("类型标注视图:")
        print(" ".join(visual_parts))

        print("\n完整 Decode 文本:")
        print(tokenizer.decode(sample_ids, skip_special_tokens=False))
        print("-" * 20)

def main():
    # 配置路径
    BIN_FILE = "../dataset/bbc-news/processed/1/dev.bin"
    TOKENIZER_MODEL = "../dataset/bbc-news/TG_GPT2_tokenizer.json"
    
    # 执行验证
    verify_bin_data(
        bin_path=BIN_FILE, 
        tokenizer_path=TOKENIZER_MODEL,
        sample_num=2,  # 采样几段
        seq_len=150    # 每段看多长
    )

if __name__ == "__main__":
    main()
