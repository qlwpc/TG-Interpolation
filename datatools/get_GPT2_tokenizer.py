from transformers import GPT2Tokenizer, T5TokenizerFast
import transformers
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from tokenizers import Tokenizer, AddedToken
from tokenizers.pre_tokenizers import Split


tokenizer = Tokenizer.from_pretrained("gpt2") # tokenizer = GPT2Tokenizer.from_pretrained("gpt2"")", # tokenizer.save_vocabulary("GPT2_tokenizer"")", 

# tokenizer.pre_tokenizer = Split(pattern='\n')


# no <unk>	
pad = AddedToken("<|pad|>", single_word=False, lstrip=False, rstrip=False, normalized=True, special=True)
bos = AddedToken("<|beginoftext|>", single_word=False, lstrip=False, rstrip=False, normalized=True, special=True)
tokenizer.add_special_tokens([bos, pad])

sum = AddedToken("<|SUM|>", single_word=False, lstrip=False, rstrip=False, normalized=True, special=True)
cls = AddedToken("<|CLS|>", single_word=False, lstrip=False, rstrip=False, normalized=True, special=True)
sep = AddedToken("<|SEP|>", single_word=False, lstrip=False, rstrip=False, normalized=True, special=True)
tokenizer.add_special_tokens([sum, cls, sep])


# BLLIP special parentheses -LRB- : ( and -RRB- : ) 
LRB = AddedToken("-LRB-", single_word=True, lstrip=True, rstrip=False, normalized=True, special=False) # (
RRB = AddedToken("-RRB-", single_word=True, lstrip=True, rstrip=False, normalized=True, special=False) # )
LCB = AddedToken("-LCB-", single_word=True, lstrip=True, rstrip=False, normalized=True, special=False) # {
RCB = AddedToken("-RCB-", single_word=True, lstrip=True, rstrip=False, normalized=True, special=False) # }
LSB = AddedToken("-LSB-", single_word=True, lstrip=True, rstrip=False, normalized=True, special=False) # [ 
RSB = AddedToken("-RSB-", single_word=True, lstrip=True, rstrip=False, normalized=True, special=False) # ]

tokenizer.add_tokens([LRB, RRB, LCB, RCB, LSB, RSB])

# do not contain "(TOP" or "TOP)"
NT = [ 
    "(ADJP", "(ADVP", "(CONJP", "(FRAG", "(INTJ", "(LST", "(NAC", "(NML", "(NP", "(PP", "(PRN", "(PRT", "(QP", "(RRC", "(S", "(SBAR", "(SBARQ", "(SINV", "(SQ", "(UCP", "(VP", "(WHADJP", "(WHADVP", "(WHNP", "(WHPP", "(X", 
     "ADJP)", "ADVP)", "CONJP)", "FRAG)", "INTJ)", "LST)", "NAC)", "NML)", "NP)", "PP)", "PRN)", "PRT)", "QP)", "RRC)", "S)", "SBAR)", "SBARQ)", "SINV)", "SQ)", "UCP)", "VP)", "WHADJP)", "WHADVP)", "WHNP)", "WHPP)", "X)"
]
for non_terminal in NT:
    tokenizer.add_special_tokens(
        [AddedToken(non_terminal, single_word=True, lstrip=True, rstrip=False, normalized=True, special=True)]
    )


tokenizer.save("TG_GPT2_tokenizer.json")