from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import numpy as np


def plot_line_length_histogram(filename):
    # 读取文件并计算每行长度
    line_lengths = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行尾换行符并计算长度
                line_lengths.append(line.rstrip('\n'))
    except FileNotFoundError:
        print(f"错误：文件 '{filename}' 未找到")
        return
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return
    
    line_lengths = sorted(line_lengths, key=lambda x:len(x), reverse=True)
    cnt = 0
    for x in line_lengths:
        print(x)
        cnt += 1
        if cnt>=50: break
    if not line_lengths:
        print("文件为空")
        return
    return
    # 创建直方图
    plt.figure(figsize=(10, 6))
    
    # 计算直方图数据
    counts, bins, patches = plt.hist(line_lengths, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
    
    # 添加标签和标题
    plt.xlabel('字符串长度')
    plt.ylabel('频数')
    plt.title(f'文件 "{filename}" 中每行字符串长度的频数分布')
    plt.grid(axis='y', alpha=0.75)
    
    # 添加数值标签
    for count, bin_edge, patch in zip(counts, bins, patches):
        if count > 0:
            plt.text(patch.get_x() + patch.get_width()/2, count + 0.01,
                     f'{int(count)}', ha='center', va='bottom')
    
    # 显示图表
    # plt.tight_layout()
    plt.savefig("train.png")
    
    # 打印基本统计信息
    # print(f"总行数: {len(line_lengths)}")
    # print(f"长度最小值: {min(line_lengths)}")
    # print(f"长度最大值: {max(line_lengths)}")
    # print(f"长度平均值: {np.mean(line_lengths):.2f}")
    # print(f"长度中位数: {np.median(line_lengths)}")

from datasets import load_dataset
import json
def prepare_gold_summary():
    ds = load_dataset("EdinburghNLP/xsum")
    for split in ["validation", "test"]:
        with open(f"../dataset/Xsum/gold_{split}_summary.jsonl", "w+") as file:
            for doc in ds[split]:
                if doc['document']!='':
                    summary = {
                        'summary': doc['summary'],
                        'id': doc['id']
                    }
                    file.write(json.dumps(summary) + '\n')
                    # prepared_ds.append(doc[text])

# 使用示例
if __name__ == "__main__":
    filename = "/public/home/wangpch/TG-Interpolation/dataset/Xsum/xsum_train_summary.txt"
    plot_line_length_histogram(filename)
    # prepare_gold_summary()