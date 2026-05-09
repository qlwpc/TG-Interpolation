# 输入数据（使用用户提供的例子）
tree_str = """(S (SBAR (WHADVP When WHADVP) (S (NP Robert NP) (VP brought (NP a cat NP) (ADVP home ADVP) (PP from (NP the shelter NP) PP) VP) S) SBAR) , (NP Steven NP) (VP was (ADJP thrilled ADJP) VP) . S) (S (NP Robert NP) (VP asked (SBAR if (S (NP he NP) (VP could (VP name (NP it NP) VP) VP) S) SBAR) VP) . S)"""

original_with_underscore = "When Robert brought a cat home from the shelter, Steven was thrilled. _ asked if he could name it."
option_replacement = "Robert"  # 用于替换下划线的option

import re

def tokenize_text(text):
    # 简单tokenizer：把字母/数字序列视为单词，把标点单独作为token
    tokens = re.findall(r"[A-Za-z0-9]+|[^\s\w]", text)
    return tokens

def extract_terminals_from_tree(tree):
    # 从括号结构中提取终端词（即不在括号名里的词）
    # 方法：删除括号和标签，保留标签后的terminal词
    # 更稳健的方法是直接从括号结构中匹配括号内可能的terminal序列
    # 这里用正则匹配单词和标点，顺序与句子一致。
    # tokens = re.findall(r"[A-Za-z0-9]+|[^\s\w]", tree)
    return tree.split(" ")

# 1. 用option替换下划线，得到完整句子（便于定位切分点）
filled_sentence = original_with_underscore.replace("_", option_replacement)

# 2. 分割为左右两部分（以下划线为切点）
left_text, right_text = original_with_underscore.split("_", 1)
left_text = left_text.strip()
right_text = right_text.strip()

# 3. tokenize 三者
tree_tokens = extract_terminals_from_tree(tree_str)
left_tokens = tokenize_text(left_text)
right_tokens = tokenize_text(right_text.replace(option_replacement, option_replacement))  # 保持一致性
filled_tokens = tokenize_text(filled_sentence)

# 4. 在tree_tokens中找到left_tokens连续出现的位置，作为切分索引
def find_subsequence_index(seq, subseq):
    """返回subseq在seq中首次出现的起始索引，找不到返回-1"""
    n, m = len(seq), len(subseq)
    i, j = 0, 0
    while j<len(subseq):
        if seq[i] == subseq[j]:
            j += 1
        i += 1
    if j<len(subseq):
        return -1
    else:
        return i

start_idx = find_subsequence_index(tree_tokens, left_tokens)
split_index = start_idx

front_seq = tree_tokens[:split_index]
back_seq = tree_tokens[split_index:]

# 输出结果
print("填入下划线后的完整句子:")
print(filled_sentence)
print("\n从语法树提取的终端词序列:")
print(tree_tokens)
print("\n左侧文本 token 序列 (下划线之前):")
print(left_tokens)
print("\n右侧文本 token 序列 (下划线之后, 未含替换词):")
print(right_tokens)
print("\n切分索引 (在 tree_tokens 中):", split_index)
print("\n前半部分序列 (front_seq):")
print(front_seq)
print("\n后半部分序列 (back_seq):")
print(back_seq)
