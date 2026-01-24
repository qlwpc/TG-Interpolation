from nltk import Tree
from tqdm import tqdm
import argparse
from tokenizers import Tokenizer
import re
import numpy as np
from numpy import ndarray
import os
from joblib import Parallel, delayed
import subprocess

def pformat_flat(self, nodesep="", parens="()", quotes=False):
    childstrs = []
    for child in self:
        if isinstance(child, Tree):
            childstrs.append(pformat_flat(child, nodesep, parens, quotes))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            mapping = {
                "-LRB-": "(",
                "-RRB-": ")",
                "-LCB-": "{",
                "-RCB-": "}",
                "-LSB-": "[",
                "-RSB-": "]",
                "Ċ" : "\n"
            }
            out = mapping[child] if child in mapping else child
            return out
        else:
            childstrs.append(repr(child))
    # print(f"label {self._label} child {childstrs}")
    if isinstance(self._label, str):
        if self._label=="qlwpcRegen":
            return " ".join(childstrs)
        else:
            return "<{}{}{}> {} <{}{}>".format(
                parens[0],
                self._label,
                nodesep,
                " ".join(childstrs),
                self._label,
                parens[1],
            )
    else:
        assert(0)
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
    try:
        tree = Tree.fromstring(line, remove_empty_top_bracketing=False)
        outputstr = pformat_flat(tree)
    except Exception as e:
        print("error occurs when processing data: ")
        print(line)
        print(e)
        outputstr = ""
    # outputstr = re.sub(" \n", "\n", outputstr)
    return outputstr

def count_lines_linux_style(filename):
    result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True)
    reslist = result.stdout.split()
    return int(reslist[0]) if reslist!=[] else 0

# def convert_treenpy_to_TG(tree: ndarray, vocab):
#     T = tree.shape[0]
#     for token in tree:
#         if vocab.is_closing_non_terminal(token):
#             T += 1
#     TG = np.zeros((T, ), tree.dtype)
#     i = 0
#     for token in tqdm(tree):
#         TG[i] = token
#         i += 1
#         if vocab.is_closing_non_terminal(token):
#             TG[i] = token
#             i += 1
#     assert i==T
#     return TG

if __name__=="__main__":

    print(convert_TG_format("(Ċ Ċ) (S (NP (NP (DT The) (NML (NNP United) (NNP Nations)) (NML (NNP Sustainable) (NNP Development)) (NNPS Goals)) (-LRB- -LRB-) (NP (NNPS SDGs)) (-RRB- -RRB-)) (VP (VBP consist) (PP (IN of) (NP (NP (CD 17) (JJ key) (NNS commitments)) (SBAR (WHNP (WDT that)) (S (VP (VBP strive) (S (VP (TO to) (VP (VP (VB eradicate) (NP (NN poverty))) (, ,) (VP (VB protect) (NP (DT the) (NN planet))) (CC and) (VP (VB ensure) (NP (NP (NN prosperity)) (PP (IN for) (NP (DT all)))) (PP (IN by) (NP (CD 2030))))))))))))) (. .)) (Ċ Ċ)"))
    parser = argparse.ArgumentParser()
    # parser.add_argument('--list_arg', type=str)  # 接收单个字符串
    parser.add_argument('--input_dir', type=str, default="~/TG-LLaMA/dataset/bbc-news-parsed-raw")
    parser.add_argument('--tokenizer', type=str, default="./dataset/TG_GPT2_tokenizer.json")
    parser.add_argument('--output_dir', type=str, default="./dataset/bbc_tokenized/")
    args = parser.parse_args()
    # result_list = args.list_arg.split(',') if args.list_arg else []
    
    tokenizer = Tokenizer.from_file(args.tokenizer)
    dtype = np.uint16 if tokenizer.get_vocab_size() < 65536 else np.uint32
    output_dir = args.output_dir
    
    from olmo.data.tg_mask import SentencepieceVocab
    vocab = SentencepieceVocab.from_vocab_file(args.tokenizer)
    
    tree_dir = os.path.join(output_dir, "tree")
    terminal_dir = os.path.join(output_dir, "terminal")
    TG_dir = os.path.join(output_dir, "tg")
    os.makedirs(tree_dir, exist_ok=True)
    os.makedirs(terminal_dir, exist_ok=True)
    os.makedirs(TG_dir, exist_ok=True)
    input_directory = os.path.expanduser(args.input_dir)


    def tokenize_file(input):
        pid = os.getpid()
        tree_seq = []
        terminal_seq = []
        tg_seq = []
        # print(f"start tokenizing {input}")
        filename = os.path.join(input_directory, input + '.txt')
        with open(filename, 'r') as file:
            with tqdm(total=count_lines_linux_style(filename), desc=f"Process {pid}", position=hash(pid) % 16) as pbar:
                for line in file:
                    TG_str = convert_TG_format(line)
                    if TG_str=="":
                        continue
                    outputid = tokenizer.encode(TG_str).ids + [vocab.eos]
                    outputid = np.array(outputid, dtype=dtype)
                    tree_seq.append(outputid)
                    terminal_ids = vocab.convert_treenpy_to_terminal(outputid)
                    terminal_seq.append(terminal_ids)
                    TG_ids = vocab.convert_treenpy_to_TG(outputid)
                    tg_seq.append(TG_ids)
                    pbar.update(1)
                    # print(tokenizer.decode(TG_ids, skip_special_tokens=False))
            
        terminal_data = np.concatenate(terminal_seq, axis=0)
        np.save(os.path.join(terminal_dir, input+".npy"), terminal_data)
        tree_data = np.concatenate(tree_seq, axis=0)
        np.save(os.path.join(tree_dir, input+".npy"), tree_data)
        TG_data = np.concatenate(tg_seq, axis=0)
        np.save(os.path.join(TG_dir, input+".npy"), TG_data)
        return


    all_files = os.listdir(input_directory)
    all_files = sorted(all_files)
    exclude_list = os.listdir(terminal_dir)
    exclude_list = [os.path.splitext(name)[0] for name in exclude_list]

    process_list = []
    for filename in all_files:
        full_path = os.path.join(input_directory, filename)
        main_name = os.path.splitext(filename)[0]
        if os.path.isfile(full_path):
            if main_name not in exclude_list:
                process_list.append(main_name)
                # tokenize_file(main_name)
    print(f"file to tokenize is {process_list}")
    Parallel(n_jobs=16)(delayed(tokenize_file)(name) for name in process_list)

    # tg_exclude_list = os.listdir(TG_dir)
    # tg_exclude_list = [os.path.splitext(name)[0] for name in tg_exclude_list]
    # for filename in all_files:
    #     full_path = os.path.join(input_directory, filename)
    #     main_name = os.path.splitext(filename)[0]
    #     if os.path.isfile(full_path):
    #         if main_name not in tg_exclude_list:
    #             print(f"start tokenizing TG {main_name}")
    #             TG_data = vocab.convert_treenpy_to_TG(np.load("./dataset/bbc_tokenized/tree/" + main_name + ".npy", mmap_mode='r'), vocab)
    #             np.save("./dataset/bbc_tokenized/tg/"+main_name+".npy", TG_data)