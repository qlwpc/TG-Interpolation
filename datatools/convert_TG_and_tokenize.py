from nltk import Tree
from tqdm import tqdm
import argparse
from tokenizers import Tokenizer
import re
import numpy as np
import os


def pformat_flat(self, nodesep="", parens="()", quotes=False):
    childstrs = []
    for child in self:
        if isinstance(child, Tree):
            childstrs.append(pformat_flat(child, nodesep, parens, quotes))
        elif isinstance(child, tuple):
            childstrs.append("/".join(child))
        elif isinstance(child, str) and not quotes:
            return child if child!='Ċ' else "\n"
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
    line = "(qlwpcRegen" + input.strip() + ")"
    tree = Tree.fromstring(line, remove_empty_top_bracketing=False)
    outputstr = pformat_flat(tree)
    outputstr = re.sub(" \n", "\n", outputstr)
    return outputstr
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_arg', type=str)  # 接收单个字符串
    args = parser.parse_args()
    result_list = args.list_arg.split(',') if args.list_arg else []
    
    tokenizer = Tokenizer.from_file("TG_GPT2_tokenizer.json")
    
    from olmo.data.tg_mask import SentencepieceVocab
    vocab = SentencepieceVocab.from_vocab_file("TG_GPT2_tokenizer.json")
    
    os.makedirs("bbc_tokenized/tree", exist_ok=True)
    os.makedirs("bbc_tokenized/terminal", exist_ok=True)

    for input in result_list:
        tree_seq = []
        terminal_seq = []
        with open("bbc-news/" + input + '.txt', 'r') as file:
            for line in tqdm(file):
                TG_str = convert_TG_format(line)
                outputid = [vocab.bos] + tokenizer.encode(TG_str).ids + [vocab.eos]
                tree_seq.append(np.array(outputid, dtype=np.uint16))
                i = 1
                for j in range(1, len(outputid)):
                    if vocab.is_terminal(outputid[j]):
                        outputid[i] = outputid[j]
                        i += 1
                terminal_seq.append(np.array(outputid[:i], dtype=np.uint16))
            
        tree_data = np.concatenate(tree_seq, axis=0)
        np.save("bbc_tokenized/tree/"+input+".npy", tree_data)
        terminal_data = np.concatenate(terminal_seq, axis=0)
        np.save("bbc_tokenized/terminal/"+input+".npy", terminal_data)