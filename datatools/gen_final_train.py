import json
import numpy as np
import os
from tqdm import tqdm

def process_files(remove_dict, bos_token_id, base_dir:str):
    """
    处理JSON文件中指定的所有.npy文件，删除指定文章后保存结果
    """
    train_list = []
    for file_prefix, del_indices in tqdm(remove_dict.items(), desc="处理文件中"):
        npy_path = os.path.join(base_dir, f"{file_prefix}.npy")
        
        if not os.path.exists(npy_path):
            print(f"警告: 文件 {npy_path} 不存在，跳过")
            continue
        tokens = np.load(npy_path)
        
        bos_positions = np.where(tokens == bos_token_id)[0]
        bos_positions = np.append(bos_positions, len(tokens))
        
        current_start = 0
        del_indices.append(len(del_indices))
        keep_segments = []
        # 遍历所有文章
        for dindex in del_indices:
            start_idx = bos_positions[current_start]
            end_idx = bos_positions[dindex]
            if end_idx>start_idx:
                keep_segments.append(tokens[start_idx:end_idx])
            current_start = dindex + 1
        
        if keep_segments:
            new_tokens = np.concatenate(keep_segments, axis=0)
        else:
            new_tokens = np.array([], dtype=tokens.dtype)
        print(f"{file_prefix} (原始长度: {len(tokens)}, 新长度: {len(new_tokens)})")
        train_list.append(new_tokens)

    train_tokens = np.concatenate(train_list, axis=0)
    output_path = os.path.join(base_dir, f"train.npy")
    np.save(output_path, train_tokens)

if __name__ == "__main__":
    JSON_FILE_PATH = "./bbc_tokenized/"  # JSON文件路径
    BOS_TOKEN_ID = 50257  # 文章分隔符的token ID
    
    with open(JSON_FILE_PATH + "test.json", 'r') as f:
        test = json.load(f)
    with open(JSON_FILE_PATH + "dev.json", 'r') as f:
        dev = json.load(f)
    for key in test.keys():
        test[key].extend(dev[key])
        test[key].sort()

    for format in ["terminal", "tree", "tg"]:
        process_files(test, BOS_TOKEN_ID, os.path.join(JSON_FILE_PATH, format))