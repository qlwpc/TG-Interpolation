import torch
import re
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
default_vocab_path = "../TG-LLaMA/dataset/spm.vocab"

__all__ = ["Old_TG_attention_bias", "Old_TG_attention_bias_nomask", "Old_KProximal_TG_attention_bias"]

TokenID = int

@dataclasses.dataclass(frozen=True)
class SentencepieceVocab:
    pad: TokenID
    bos: TokenID
    eos: TokenID
    unk: TokenID
    whitespace: TokenID
    bosent : TokenID
    eosent : TokenID
    opening_non_terminals: Tuple[int, int]
    closing_non_terminals: Tuple[int, int]
    #terminals: Tuple[int, int]
    tokens: List[str]

    @classmethod
    def from_vocab_file(cls, vocab_file):
        pad, bos, eos, unk, bosent, eosent = None, None, None, None, None, None
        whitespace = None
        opening_non_terminals = [-1,-1]
        closing_non_terminals = [-1,-1]
        tokens_dic = []
        with open(vocab_file, "r") as f:
            for idx, line in enumerate(f):
                token, _ = line.rstrip().split("\t")
                tokens_dic.append(token)
                if token == "<pad>":
                    pad = idx
                elif token == "<s>":
                    bos = idx
                elif token == "</s>":
                    eos = idx
                elif token == "<unk>":
                    unk = idx
                elif token == "(S1":
                    bosent = idx
                elif token == "S1)":
                    eosent = idx
                elif re.fullmatch(r"\([A-Z]+", token):
                    if opening_non_terminals[0]==-1:
                        opening_non_terminals[0] = idx
                    opening_non_terminals[1] = idx
                elif re.fullmatch(r"[A-Z]+\)", token):
                    if closing_non_terminals[0]==-1:
                        closing_non_terminals[0] = idx
                    closing_non_terminals[1] = idx
                else:
                    # Terminal, or whitespace.
                    # NOTE: This is brittle, and valid only with SP models built with the
                    # default options.
                    if token == "▁":
                        whitespace = idx
        return cls(pad=pad, bos=bos, eos=eos, unk=unk, whitespace=whitespace, bosent=bosent, eosent=eosent,
                opening_non_terminals=tuple(opening_non_terminals),
                closing_non_terminals=tuple(closing_non_terminals),
                tokens=tokens_dic)
    
    def is_opening_non_terminal(self, id:TokenID):
        return (self.opening_non_terminals[0] <= id <= self.opening_non_terminals[1])
    
    def is_closing_non_terminal(self, id:TokenID):
        return (self.closing_non_terminals[0] <= id <= self.closing_non_terminals[1])
    
    def is_non_terminal(self,id:TokenID):
        return (self.opening_non_terminals[0] <= id <= self.closing_non_terminals[1])
    
    def is_terminal(self, id:TokenID):
        return id != self.pad and id != self.bos and id != self.eos \
                and not self.is_non_terminal(id)


class TG_Cache:
    '''
    The size of Cache is large enough.
    '''
    def __init__(self, max_length, dtype=torch.long) -> None:
        self.start = 0
        self.end = 0
        self.max_length = max_length
        self.buffer = torch.empty(max_length, dtype=dtype)

    def clear(self):
        self.start = self.end = 0

    def __getitem__(self, i):
        return self.buffer[(self.start + i) % self.max_length]

    def append(self, input_ids:torch.tensor, update_state=False) -> None:
        T = input_ids.shape[0]
        if self.end + T > self.max_length:
            fir_len = self.max_length - self.end
            self.buffer[self.end : self.max_length] = input_ids[:fir_len]
            self.buffer[0 : T - fir_len] = input_ids[fir_len : T]
            if update_state:
                self.end = T - fir_len
        else:
            self.buffer[self.end : self.end + T] = input_ids
            if update_state:
                self.end += T
    
    def pop_front(self, pop_length) -> None:
        self.start = (self.start + pop_length) % self.max_length    

class Old_TG_attention_bias:
    def __init__(self, vocab_path:str = None, max_token_length:int = 2048) -> None:
        vocab_path = vocab_path if vocab_path is not None else default_vocab_path
        self.vocab = SentencepieceVocab.from_vocab_file(vocab_path)
        self.stk = torch.zeros(max_token_length*2, dtype=torch.long) 
        self.cached_input = TG_Cache(max_token_length*2, torch.long)
        self.max_length = max_token_length
        self.reset_state()
    
    def reset_state(self) -> None:
        self.last_token = None
        self.top = -1
        self.cur_length = 0
        self.cached_input.clear()
    
    '''
    last token is None: NewStart
    last token is -1 and token is NT: compose
    last token is NT and token is same NT: no label, predict next
    '''
    def should_compose(self, token, last_token, input_ids, idx) -> bool:
        if self.vocab.is_closing_non_terminal(token):
            if last_token != token and last_token is not None:
                return True
            elif last_token == None:
                cnt=0
                while input_ids[idx]==token:
                    cnt += 1
                    idx += 1
                    if idx>=len(input_ids):
                        break
                return cnt % 2 == 0           # even means should compose
        return False
    
    def cache_concatenate(self, cache, input) -> torch.Tensor:
        T = input.shape[0]
        pastT = self.cur_length
        if self.cur_length + T > self.max_length:
            remove_len = self.cur_length + T - self.max_length
            pastT -= remove_len
            tmp = cache[remove_len: self.cur_length].clone()
            cache[:pastT].copy_(tmp)
        cache[pastT : pastT + T].copy_(input)
        return cache

    def stack_truncated_to_max_length(self, stk, top, add_len) -> Tuple[torch.Tensor, int]:
        pastT = self.cur_length
        if self.cur_length + add_len > self.max_length:
            remove_len = self.cur_length + add_len - self.max_length
            pastT -= remove_len
            stk[:top+1] -= remove_len
            stkgt0 = stk[:top+1] >= 0
            if not stkgt0.any(): 
                top = -1
            else:
                first_non_neg = stkgt0.nonzero()[0,0]
                if first_non_neg > 0:
                    non_negs = stk[first_non_neg:top+1].clone()
                    top = non_negs.shape[0] - 1
                    stk[:top+1] = non_negs
        
        return stk, top, pastT

    """
    Set update_state = True to update states
    when update_state == True, it is required that input_ids has no padding in the end
    """
    def __call__(self, input_ids:torch.Tensor, update_state=False) -> Tuple[torch.Tensor, torch.Tensor]:
        T = input_ids.shape[0]
        stk = self.stk
        top = self.top
        last_token = self.last_token
        self.cached_input.append(input_ids, update_state)
        remove_len = self.cur_length + T - self.max_length if self.cur_length + T > self.max_length else 0
        pastT = self.max_length - T  if self.cur_length + T > self.max_length else self.cur_length
        temp_stk_idx = stk[:top+1] >= remove_len
        stk_beg = temp_stk_idx.nonzero()[0,0] if temp_stk_idx.any() else top+1
        label_mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask = torch.zeros(T, pastT + T, dtype=torch.bool)
        for i in range(T):
            mask[i, pastT+i] = True
            token = input_ids[i]
            if self.should_compose(token, last_token, input_ids, i):
                j = self.cur_length + i
                while top>=stk_beg and not self.vocab.is_opening_non_terminal(self.cached_input[j]):
                    j = stk[top]
                    top -= 1
                    mask[i, j - remove_len] = True
                top += 1
                stk[top] = self.cur_length + i
            else:
                if not self.vocab.is_closing_non_terminal(token) and token != self.vocab.pad:
                    top += 1
                    stk[top] = self.cur_length + i
                else:
                    label_mask[i] = False

                if top>=0:
                    mask[i, stk[stk_beg:top+1] - remove_len] = True
            
            last_token = token
            if token == self.vocab.eos:
                top = -1
                stk_beg = 0
        
        if update_state:
            temp_stk = stk[stk_beg:top+1] - remove_len
            self.top = temp_stk.shape[0] - 1
            self.stk[:self.top+1] = temp_stk
            self.last_token = last_token
            self.cached_input.pop_front(remove_len)
            self.cur_length = min(self.max_length, self.cur_length + T)
        
        return mask, label_mask

class Old_TG_attention_bias_nomask(Old_TG_attention_bias):
    '''
    compose = TG
    otherwise = causal
    '''
    def __init__(self, vocab_path:str = None, max_token_length:int = 2048):
        super().__init__(vocab_path, max_token_length)
        self.hist_stk = torch.zeros(max_token_length+1, dtype=torch.long)
    
    def reset_state(self) -> None:
        self.last_token = None
        self.top = self.htop = -1
        self.cur_length = 0
    
    def __call__(self, input_ids:torch.Tensor, update_state=False) -> Tuple[torch.Tensor, torch.Tensor]:
        T = input_ids.shape[0]
        compose_stk = self.stk.clone()   if update_state==False else self.stk
        hist_stk = self.hist_stk.clone() if update_state==False else self.hist_stk
        cached_input = self.cached_input.clone() if update_state==False else self.cached_input
        ctop = self.top
        htop = self.htop
        last_token = self.last_token
        compose_stk, ctop, pastT = self.stack_truncated_to_max_length(compose_stk, ctop, T)
        hist_stk, htop, pastT = self.stack_truncated_to_max_length(hist_stk, htop, T)
        cached_input = self.cache_concatenate(cached_input, input_ids)

        mask = torch.zeros(T, pastT + T, dtype=torch.bool)
        label_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for i in range(T):
            mask[i, pastT + i] = True
            token = input_ids[i]
            if self.should_compose(token, last_token, input_ids, i):
                j = pastT + i
                while ctop>=0 and not self.vocab.is_opening_non_terminal(cached_input[j]):
                    j = compose_stk[ctop]
                    ctop -= 1
                    mask[i, j] = True  # compose
                ctop += 1
                htop += 1
                compose_stk[ctop] = pastT + i
                hist_stk[htop] = pastT + i
            else:
                if not self.vocab.is_closing_non_terminal(token) and token != self.vocab.pad:
                    ctop += 1
                    htop += 1
                    compose_stk[ctop] = pastT + i
                    hist_stk[htop] = pastT + i
                else:
                    label_mask[i] = False

                if htop>=0:
                    mask[i, hist_stk[:htop+1]] = True  # nomask

            last_token = token
            if token == self.vocab.eos:
                ctop = htop = -1
        
        if update_state:
            self.stk, self.top, self.last_token = compose_stk, ctop, last_token
            self.hist_stk, self.htop = hist_stk, htop
            self.cached_input = cached_input
            self.cur_length = min(self.max_length, self.cur_length + T)
        return mask, label_mask

class Old_KProximal_TG_attention_bias(Old_TG_attention_bias):
    def __init__(self, vocab_path:str = None, max_token_length:int = 2048, Proximal_lenK:int = 20):
        super().__init__(vocab_path, max_token_length)
        self._k = Proximal_lenK
        self.cached_label = torch.zeros(max_token_length*2, dtype=torch.bool)
        self.cached_input = torch.zeros(max_token_length*2, dtype=torch.long)
    
    def __call__(self, input_ids:torch.Tensor, update_state=False) -> Tuple[torch.Tensor, torch.Tensor]:
        T = input_ids.shape[0]
        stk = self.stk.clone() if update_state==False else self.stk
        cached_input = self.cached_input.clone() if update_state==False else self.cached_input
        cached_label = self.cached_label.clone() if update_state==False else self.cached_label
        top = self.top
        last_token = self.last_token
        stk, top, pastT = self.stack_truncated_to_max_length(stk, top, T)
        cached_input = self.cache_concatenate(cached_input, input_ids)

        label_mask = torch.ones_like(input_ids, dtype=torch.bool)
        cached_label = self.cache_concatenate(cached_label, label_mask)
        mask = torch.zeros(T, pastT + T, dtype=torch.bool)
        for i in range(T):
            mask[i, pastT+i] = True
            token = input_ids[i]
            if self.should_compose(token, last_token, input_ids, i):
                j = pastT + i
                while top>=0 and not self.vocab.is_opening_non_terminal(cached_input[j]):
                    j = stk[top]
                    top -= 1
                    mask[i, j] = True
                top += 1
                stk[top] = pastT + i
            else:
                if not self.vocab.is_closing_non_terminal(token) and token != self.vocab.pad:
                    top += 1
                    stk[top] = pastT + i
                else:
                    label_mask[i] = False
                    cached_label[pastT + i] = False

                mask[i, max(0, pastT + i - self._k):pastT + i] = cached_label[max(0, pastT + i - self._k):pastT + i]
                if top>=0:
                    mask[i, stk[:top+1]] = True
            
            last_token = token
            if token == self.vocab.eos:
                top = -1
        
        if update_state:
            self.stk, self.top, self.last_token = stk, top, last_token
            self.cached_input = cached_input
            self.cached_label = cached_label
            self.cur_length = min(self.max_length, self.cur_length + T)
        
        return mask, label_mask
    
if __name__ == "__main__":
    from olmo.tokenizer import Tokenizer
    from olmo.data import ChangeHead_bias, Soft_Alibilike_bias
    #import tg_mask as test
    inputs = [60,19,18,18,11,76,170,38,38,25,25,334,64,18,25,67,25,2991,11,11,68,1181,3446,2769,38,38,13,86,11,437,930,38,38,40,40,38,38,52,52,52,52,45,45,52,52,128,25,99,25,2781,18,25,67,25,1784,11,11,68,379,2769,38,38,25,3516,77,11,11,63,472,38,38,13,66,11,1181,1756,38,38,40,40,38,38,13,75,11,1461,2259,38,38,40,40,52,52,38,38,52,52,52,52,45,45,52,52,52,52,52,52,45,45,62,11,224,38,38,25,80,52,52,60,61,45,45,46,46,19,18,18,11,11,14083,9335,1141,60,445,1387,11141,60,811,10751,105,38,38,62,11,11,285,1580,38,38,13,66,11,295,60,1550,38,38,40,40,38,38,62,38,38,25,80,20,18,11,1459,38,38,25,94,4,60,70,70,270,82,60,13039,60,65,65,18,25,67,25,472,11,1181,1696,38,38,13,5,3960,82,32,32,75,11,8727,2259,38,38,40,40,52,52,52,52,45,45,31,31,52,52,45,45,47,47,52,52,45,45,62,128,18,11,103,38,38,25,1458,11,63,717,2313,6962,6692,72,5134,1759,6557,6030,117,38,38,20,165,18,11,83,38,38,60,70,70,25,124,11,1433,2712,8810,38,38,52,52,45,45,47,47,52,52,45,45,60,61,60,65,65,45,45,46,46,19,18,60,70,70,18,11,317,2769,38,38,25,1110,64,11,11,68,204,38,38,13,66,11,11,662,38,38,13,86,11,11,60,6375,64,38,38,69,11,11,350,204,38,38,13,86,11,2024,64,38,38,40,40,38,38,38,38,40,40,38,38,40,40,38,38,52,52,45,45,62,60,65,65,11,63,285,1580,38,38,25,652,11,1929,64,38,38,52,52,60,61,45,45,46,46,19,18,11,60,347,2509,119,279,38,38,62,11,11,1459,38,38,69,11,437,930,38,38,38,38,25,847,11,11,68,151,72,252,1222,38,38,13,66,11,123,4,3545,69,6075,31,31,3446,38,38,40,40,38,38,52,52,60,61,45,45,46,46,19,18,11,76,780,6459,38,38,25,94,11,11,68,4,5578,8396,31,31,1181,269,38,38,20,28,74,55,55,18,25,456,64,11,1459,38,38,11,60,70,70,2838,72,4627,60,65,65,1696,38,38,114,11,11,68,2695,1453,38,38,13,989,18,25,1361,11,109,60,13575,60,6332,38,38,52,52,45,45,40,40,38,38,52,52,45,45,47,47,38,38,52,52,60,61,45,45,46,46,19,18,11,84,61,60,445,1387,11141,60,811,10751,105,38,38,25,1972,13,86,11,11,883,4359,38,38,69,11,3009,1626,12660,60,2001,2693,197,38,38,38,38,40,40,52,52,60,61,45,45,46,46,19,18,11,76,3493,285,1580,38,38,25,80,20,18,11,358,662,38,38,25,1757,18,25,67,25,2991,11,63,1497,38,38,52,52,52,52,45,45,52,52,45,45,47,47,52,52,60,61,45,45,46,46,19,18,11,1174,38,38,25,78,11,11,63,3504,38,38,13,75,11,678,10293,38,38,40,40,38,38,52,52,60,61,45,45,46,46]
    # inputs =  [    1,    76,   855,    74,   706,  4426,    80,   126,   117,  3445,
    #      1242,    73,   350,    60, 13006,    91,    63,  1027,  2848,   706,
    #      4984,   183,  3015]
    tokenizer = Tokenizer.from_file("../TG-LLaMA/dataset/TG_spm_uni.json", 
                    eos_token_id=2,
                    pad_token_id=0)
    print(tokenizer.decode(inputs))
    END_SENT = 46
    sents = []
    sent_deco = []
    begin = 0
    for i in range(len(inputs)):
        if i%5==0:
            sents.append(inputs[begin:i+1])
            begin = i+1
    
    torch.set_printoptions(
        precision=4,    # 小数位数
        threshold=10000000, # 触发缩略显示的阈值（元素数量）
        edgeitems=3,    # 缩略时显示的首尾元素数量
        linewidth=100000,  # 每行的字符宽度
        sci_mode=False  # 是否禁用科学计数法
    )

    TG_mask = Old_KProximal_TG_attention_bias("../TG-LLaMA/dataset/spm.vocab", max_token_length=40, Proximal_lenK=40)
    Prox_mask = Soft_Alibilike_bias("../TG-LLaMA/dataset/spm.vocab", max_token_length=40, type="mixB")
    # Prox_mask = Old_KProximal_TG_attention_bias(max_token_length=40, Proximal_lenK=40)
    #TG_mask = test.Height_TG_attention_bias("../TG-LLaMA/dataset/spm.vocab", max_token_length=40, Height_H=1)
    # import pickle
    # data = pickle.dumps(TG_mask)
    # # print(TG_mask.__getstate__())
    # # print(TG_mask.__setstate__(TG_mask.__getstate__()))
    # # print(tmp = x.__setstate__())
    # # print(data)
    # reloaded = pickle.loads(data) 
    # print(type(reloaded)) # 输出: <class 'your_module.Derived'>
    # exit(0)
    for sent in sents:
        print(tokenizer.decode(sent))
        TGmask, TGlabel = TG_mask(torch.tensor(sent), True)
        Kmask, Klabel = Prox_mask(torch.tensor(sent), True)
        print(TGmask)
        # print(TGmask.shape)
        print(Kmask)
        print(TGlabel)
        print(Klabel)
        # assert(torch.equal(TGmask, Kmask))
        # assert(torch.equal(TGlabel, Klabel))