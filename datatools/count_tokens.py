import os
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count

def process_single_file(filepath):
    """处理单个文件的核心逻辑"""
    try:
        # 注意：np.fromfile 用于读取原始二进制，如果是标准的 .npy 文件，
        # 通常建议使用 np.load(filepath, mmap_mode='r') 来加速并节省内存
        data = np.fromfile(filepath, dtype=np.uint32)
        if data.ndim == 1:
            return data.size
    except Exception:
        pass
    return 0

def count_npy_data_parallel(directory, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()

    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_paths.append(os.path.join(root, file))
    
    print(f"正在使用 {num_processes} 个进程处理 {len(file_paths)} 个文件...")
    
    with Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(process_single_file, file_paths)
        total = sum(results)
    
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/storage/")
    parser.add_argument('--jobs', type=int, default=None, help="进程数量，默认使用全部CPU")
    args = parser.parse_args()

    total_count = count_npy_data_parallel(args.input_dir, args.jobs)
    print(f"所有 npy 文件数据总量: {total_count}")