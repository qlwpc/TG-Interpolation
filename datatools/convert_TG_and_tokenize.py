from nltk import Tree
from tqdm import tqdm
import argparse
from tokenizers import Tokenizer
import re
import numpy as np
from numpy import ndarray
import os
from joblib import Parallel, delayed

def pformat_flat(self, nodesep="", parens="()", quotes=False):
    childstrs = []
    for child in self:
        if isinstance(child, Tree):
            childstrs.append(pformat_flat(child, nodesep, parens, quotes))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            return child if child!='Ċ' else "<|SEP|>"  # newline
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
    # outputstr = re.sub(" \n", "\n", outputstr)
    return outputstr
    
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

    # print(convert_TG_format("(Ċ Ċ) (S (VP (VB Summarize) (NP (DT the) (JJ above) (NN article)) (PP (IN in) (NP (CD 1) (NN sentence)))) (. .)) (Ċ Ċ)"))
    parser = argparse.ArgumentParser()
    # parser.add_argument('--list_arg', type=str)  # 接收单个字符串
    parser.add_argument('--input_dir', type=str, default="~/TG-LLaMA/dataset/bbc-news-parsed-raw")
    parser.add_argument('--tokenizer', type=str, default="../dataset/TG_GPT2_tokenizer.json")
    args = parser.parse_args()
    # result_list = args.list_arg.split(',') if args.list_arg else []
    
    tokenizer = Tokenizer.from_file(args.tokenizer)
    dtype = np.uint16 if tokenizer.get_vocab_size() < 65536 else np.uint32
    
    from olmo.data.tg_mask import SentencepieceVocab
    vocab = SentencepieceVocab.from_vocab_file(args.tokenizer)
    
    os.makedirs("./dataset/bbc_tokenized/tree", exist_ok=True)
    os.makedirs("./dataset/bbc_tokenized/terminal", exist_ok=True)
    os.makedirs("./dataset/bbc_tokenized/tg", exist_ok=True)
    input_directory = os.path.expanduser(args.input_dir)


    def tokenize_file(input):
        tree_seq = []
        terminal_seq = []
        tg_seq = []
        print(f"start tokenizing {input}")
        with open(os.path.join(input_directory, input + '.txt'), 'r') as file:
            for line in tqdm(file):
                TG_str = convert_TG_format(line)
                outputid = [vocab.bos] + tokenizer.encode(TG_str).ids + [vocab.eos]
                outputid = np.array(outputid, dtype=dtype)
                # subtitute special <|SEP|> to \n(198)
                outputid[outputid == vocab.newline] = 198
                tree_seq.append(outputid)
                terminal_ids = vocab.convert_treenpy_to_terminal(outputid)
                terminal_seq.append(terminal_ids)
                TG_ids = vocab.convert_treenpy_to_TG(outputid)
                tg_seq.append(TG_ids)
                # print(tokenizer.decode(TG_ids, skip_special_tokens=False))
            
        terminal_data = np.concatenate(terminal_seq, axis=0)
        np.save("./dataset/bbc_tokenized/terminal/"+input+".npy", terminal_data)
        tree_data = np.concatenate(tree_seq, axis=0)
        np.save("./dataset/bbc_tokenized/tree/"+input+".npy", tree_data)
        TG_data = np.concatenate(tg_seq, axis=0)
        np.save("./dataset/bbc_tokenized/tg/"+input+".npy", TG_data)
        return


    all_files = os.listdir(input_directory)
    all_files = sorted(all_files)
    terminal_dir = "./dataset/bbc_tokenized/terminal"
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
    Parallel(n_jobs=5)(delayed(tokenize_file)(name) for name in process_list)

    TG_dir = "./dataset/bbc_tokenized/tg"
    tg_exclude_list = os.listdir(TG_dir)
    tg_exclude_list = [os.path.splitext(name)[0] for name in tg_exclude_list]
    for filename in all_files:
        full_path = os.path.join(input_directory, filename)
        main_name = os.path.splitext(filename)[0]
        if os.path.isfile(full_path):
            if main_name not in tg_exclude_list:
                print(f"start tokenizing TG {main_name}")
                TG_data = vocab.convert_treenpy_to_TG(np.load("./dataset/bbc_tokenized/tree/" + main_name + ".npy", mmap_mode='r'), vocab)
                np.save("./dataset/bbc_tokenized/tg/"+main_name+".npy", TG_data)