from transformers import GPT2Tokenizer, T5TokenizerFast
import transformers
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from tokenizers import Tokenizer, AddedToken
from tokenizers.pre_tokenizers import Split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="gpt2")
args = parser.parse_args()

model = args.model_name
name = {"gpt2": "gpt2", "qwen3": "Qwen/Qwen3-0.6B"}

tokenizer = Tokenizer.from_pretrained(name[model])

# tokenizer.pre_tokenizer = Split(pattern='\n')


# no <unk>	
pad = AddedToken("<|pad|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
bos = AddedToken("<|beginoftext|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
tokenizer.add_special_tokens([bos, pad])

sum = AddedToken("<|SUM|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
cls = AddedToken("<|CLS|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
sep = AddedToken("<|SEP|>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)
tokenizer.add_special_tokens([sum, cls, sep])


# BLLIP special parentheses -LRB- : ( and -RRB- : ) 
# add in tokenizers but we don't use in data
LRB = AddedToken("<-LRB->", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False) # (
RRB = AddedToken("<-RRB->", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False) # )
LCB = AddedToken("<-LCB->", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False) # {
RCB = AddedToken("<-RCB->", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False) # }
LSB = AddedToken("<-LSB->", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False) # [ 
RSB = AddedToken("<-RSB->", single_word=False, lstrip=False, rstrip=False, normalized=False, special=False) # ]

tokenizer.add_tokens([LRB, RRB, LCB, RCB, LSB, RSB])

# do not contain "<(TOP>" or "<TOP)>"
# because Qwen3-tokenizer has terminal-tokens "<(S>" so we must set NT to "(S "
NT = [ 
    "<(ADJP>", "<(ADVP>", "<(CONJP>", "<(FRAG>", "<(INTJ>", "<(LST>", "<(NAC>", "<(NML>", "<(NP>", "<(PP>", "<(PRN>", "<(PRT>", "<(QP>", "<(RRC>", "<(S>", "<(SBAR>", "<(SBARQ>", "<(SINV>", "<(SQ>", "<(UCP>", "<(VP>", "<(WHADJP>", "<(WHADVP>", "<(WHNP>", "<(WHPP>", "<(X>", 
     "<ADJP)>", "<ADVP)>", "<CONJP)>", "<FRAG)>", "<INTJ)>", "<LST)>", "<NAC)>", "<NML)>", "<NP)>", "<PP)>", "<PRN)>", "<PRT)>", "<QP)>", "<RRC)>", "<S)>", "<SBAR)>", "<SBARQ)>", "<SINV)>", "<SQ)>", "<UCP)>", "<VP)>", "<WHADJP)>", "<WHADVP)>", "<WHNP)>", "<WHPP)>", "<X)>"
]
for non_terminal in NT:
    tokenizer.add_special_tokens(
        [AddedToken(non_terminal, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True)]
    )


tokenizer.save(f"../dataset/TG_{model.upper()}_tokenizer.json")