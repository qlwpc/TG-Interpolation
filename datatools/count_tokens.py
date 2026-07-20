import os
import numpy as np
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

from olmo.memmap_utils import inspect_memmap_file

def process_single_file(filepath, dtype="uint32", file_format="auto"):
    """处理单个文件的核心逻辑"""
    try:
        info = inspect_memmap_file(filepath, np.dtype(dtype), file_format)
        return info.element_count, None
    except Exception as exc:
        return 0, f"{filepath}: {exc}"

def count_npy_data_parallel(directory, num_processes=None, dtype="uint32", file_format="auto"):
    if num_processes is None:
        num_processes = cpu_count()

    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_paths.append(os.path.join(root, file))
    
    print(f"正在使用 {num_processes} 个进程处理 {len(file_paths)} 个文件...")
    
    with Pool(processes=num_processes) as pool:
        worker = partial(process_single_file, dtype=dtype, file_format=file_format)
        results = pool.imap_unordered(worker, file_paths)
        total = 0
        errors = []
        for count, error in results:
            total += count
            if error is not None:
                errors.append(error)

    if errors:
        raise ValueError("Failed to inspect data files:\n" + "\n".join(errors))
    
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/storage/")
    parser.add_argument('--jobs', type=int, default=None, help="进程数量，默认使用全部CPU")
    parser.add_argument('--memmap_dtype', default="uint32")
    parser.add_argument('--memmap_format', choices=("auto", "npy", "raw"), default="auto")
    args = parser.parse_args()

    total_count = count_npy_data_parallel(
        args.input_dir, args.jobs, args.memmap_dtype, args.memmap_format
    )
    print(f"所有 npy 文件数据总量: {total_count}")
