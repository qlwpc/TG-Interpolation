import numpy as np
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List

def convert_npy_to_bin(npy_path: str, delete_original: bool = False):
    """
    将单个 .npy 文件转换为纯二进制文件 (.bin)
    """
    try:
        # 使用 mmap_mode='r' 可以避免将整个大文件加载到内存中
        # 这样即使文件比内存大，转换也能平稳进行
        data = np.load(npy_path, mmap_mode='r')
        
        # 构造新的文件名 (例如: data.npy -> data.bin)
        bin_path = str(Path(npy_path).with_suffix('.bin'))
        
        # 核心步骤：直接将原始二进制数据导出
        data.tofile(bin_path)
        
        if delete_original:
            os.remove(npy_path)
            return f"✓ 已转换并删除: {npy_path}"
        return f"✓ 已转换: {bin_path}"
    
    except Exception as e:
        return f"✗ 转换失败 {npy_path}: {str(e)}"

def batch_convert(file_list: List[str], max_workers: int = 4):
    print(f"开始转换 {len(file_list)} 个文件，线程数: {max_workers}...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        results = list(executor.map(convert_npy_to_bin, file_list))
        
        # 打印部分结果
        for res in results:
            print(res)

# 使用示例
if __name__ == "__main__":
    # 填入你的文件路径列表
    dir = ["terminal", "tree", "tg"]
    base = "/storage/wangpch"
    process_list = []
    for form in dir:
        cwd = os.path.join(base, form)
        for file in sorted(os.listdir(cwd)):
            filename = os.path.join(cwd, file)
            process_list.append(filename)
    batch_convert(process_list, max_workers=16)