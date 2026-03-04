import json
import numpy as np
import os
from tqdm import tqdm
import argparse
from tokenizers import Tokenizer

def process_files(remove_dict, bos_token_id, base_dir:str):
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
        del tokens

    train_tokens = np.concatenate(train_list, axis=0)
    del train_list
    output_path = os.path.join(base_dir, f"train.npy")
    np.save(output_path, train_tokens)

def extract_test_split(extract_dict, bos_token_id, base_dir, filename):
    train_list = []
    for file_prefix, retain_indices in tqdm(extract_dict.items(), desc="处理文件中"):
        npy_path = os.path.join(base_dir, f"{file_prefix}.npy")
        if not os.path.exists(npy_path):
            print(f"警告: 文件 {npy_path} 不存在，跳过")
            continue
        tokens = np.load(npy_path)
        
        bos_positions = np.where(tokens == bos_token_id)[0]
        bos_positions = np.append(bos_positions, len(tokens))
        
        retain_indices.append(len(retain_indices))
        keep_segments = []
        for dindex in retain_indices:
            start_idx = bos_positions[dindex]
            end_idx = bos_positions[dindex+1]
            if end_idx>start_idx:
                keep_segments.append(tokens[start_idx:end_idx])
        
        if keep_segments:
            new_tokens = np.concatenate(keep_segments, axis=0)
        else:
            new_tokens = np.array([], dtype=tokens.dtype)
        print(f"{file_prefix} (原始长度: {len(tokens)}, 新长度: {len(new_tokens)})")
        train_list.append(new_tokens)
        del tokens

    train_tokens = np.concatenate(train_list, axis=0)
    del train_list
    output_path = os.path.join(base_dir, f"{filename}.npy")
    np.save(output_path, train_tokens)

if __name__ == "__main__":
    JSON_FILE_PATH = "../dataset/bbc-news/"  # JSON文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default="../dataset/TG_GPT2_tokenizer.json")
    parser.add_argument('--data_dir', type=str, default="../dataset/bbc-news/")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    BOS_TOKEN_ID = tokenizer.token_to_id("<|beginoftext|>")  # 文章分隔符的token ID
    
    with open(JSON_FILE_PATH + "test_index.json", 'r') as f:
        test = json.load(f)
        for format in ["terminal", "tree", "tg"]:
            extract_test_split(test, BOS_TOKEN_ID, os.path.join(args.data_dir, format), "test")
    with open(JSON_FILE_PATH + "dev_index.json", 'r') as f:
        dev = json.load(f)
        for format in ["terminal", "tree", "tg"]:
            extract_test_split(dev, BOS_TOKEN_ID, os.path.join(args.data_dir, format), "dev")
    for key in test.keys():
        test[key].extend(dev[key])
        test[key].sort()

    # for format in ["terminal", "tree", "tg"]:
    #     process_files(test, BOS_TOKEN_ID, os.path.join(args.data_dir, format))