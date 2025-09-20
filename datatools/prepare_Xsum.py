from nltk import Tree
from tqdm import tqdm
import argparse
from tokenizers import Tokenizer
import re
import numpy as np
from numpy import ndarray
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import json

def pformat_flat(self, nodesep="", parens="()", quotes=False):
    childstrs = []
    for child in self:
        if isinstance(child, Tree):
            childstrs.append(pformat_flat(child, nodesep, parens, quotes))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            return child if child!='Ċ' else "<|SEP|>"  # 50261
        else:
            childstrs.append(repr(child))
    # print(f"label {self._label} child {childstrs}")
    if isinstance(self._label, str):
        if self._label=="qlwpcRegen":
            return " ".join(childstrs)
        else:
            return "{}{}{} {} {}{}".format(
                parens[0],
                self._label,
                nodesep,
                " ".join(childstrs),
                self._label,
                parens[1],
            )
    else:
        return "{}{}{} {} {}{}".format(
            parens[0],
            repr(self._label),
            nodesep,
            " ".join(childstrs),
            repr(self._label),
            parens[1],
        )

def convert_TG_format(input:str) -> str:
    line = "(qlwpcRegen " + input.strip() + ")"
    tree = Tree.fromstring(line, remove_empty_top_bracketing=False)
    outputstr = pformat_flat(tree)
    outputstr = re.sub(" \n", "\n", outputstr)
    return outputstr

from olmo.data.tg_mask import SentencepieceVocab
if __name__=="__main__":
    
    split = "train"
    filename = "/public/home/wangpch/TG-Interpolation/dataset/Xsum/xsum_train.txt"
    summary = "/public/home/wangpch/TG-Interpolation/dataset/Xsum/xsum_train_summary.txt"
    gold_summary = "/public/home/wangpch/TG-Interpolation/dataset/Xsum/gold_train_summary.jsonl"
    tokenizer = Tokenizer.from_file("/public/home/wangpch/TG-Interpolation/dataset/bbc-news/TG_GPT2_tokenizer.json")
    vocab = SentencepieceVocab.from_vocab_file("/public/home/wangpch/TG-Interpolation/dataset/bbc-news/TG_GPT2_tokenizer.json")
    gold = []
    with open(filename, 'r') as file:
        train = file.readlines()
    with open(summary, 'r') as file:
        train_summary = file.readlines()
    with open(gold_summary, 'r') as file:
        for line in file:
            gold.append(json.loads(line.strip()))

    ds = []
    line_lengths = []
    terminal_lengths = []
    max_length = 0
    for x,y, ref in tqdm(zip(train, train_summary, gold)):
        try:
            # doc = convert_TG_format(x)
            summary = convert_TG_format(y)
        except:
            print(summary)
            continue
        # doc_ids = tokenizer.encode(doc).ids
        summary_ids = tokenizer.encode(summary).ids
        summary_ids = np.array(summary_ids)
        TG = vocab.convert_treenpy_to_TG(summary_ids)
        terminal = vocab.convert_treenpy_to_terminal(summary_ids)
        line_lengths.append(len(TG))
        terminal_lengths.append(len(terminal))
        ds.append(
            {
                # "doc": doc,
                "summary": summary,
            }
        )
    
    print(f"{max(line_lengths)} tokens summary max")
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
    plt.savefig("TG_summary.png")
    plt.close()
    counts, bins, patches = plt.hist(terminal_lengths, bins='auto', alpha=0.7, color='skyblue', edgecolor='black')
    
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
    plt.savefig("terminal_summary.png")