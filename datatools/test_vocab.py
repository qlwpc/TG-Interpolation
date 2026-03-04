from olmo.data.tg_mask import SentencepieceVocab, TG_attention_bias
from tokenizers import Tokenizer
import torch
import numpy as np

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("./dataset/TG_QWEN3_tokenizer.json")
    vocab = SentencepieceVocab.from_vocab_file("./dataset/TG_QWEN3_tokenizer.json")
    # file = "./dataset/finewebedu_tokenized/tree/*-00(004|005|006|007)*arrow.npy"
    # train_data = np.fromfile(file, dtype=np.uint32)
    # input_ids = train_data[:200]
    # input = tokenizer.decode(input_ids,  skip_special_tokens=False)
    # print(tokenizer.all_special_tokens)
    # print(tokenizer.all_special_ids)
    test_input = "<(S><(NP> Fencing<NP)><(VP> is<(NP><(NP> a category<NP)><(PP> of<(NP> three affiliated combat sports<NP)><PP)><NP)><VP)> .<S)>\n<(S><(NP><(NP> The three classifications<NP)><(PP> in<(NP> modern fencing<NP)><PP)><NP)><(VP> include<(NP><(NP> the sabre<NP)> ,<(NP> the epee<NP)> , and<(NP> the foil<NP)><NP)><VP)> .<S)><(S><(ADVP> Basically<ADVP)> ,<(NP> winning points<NP)><(VP> are<(ADVP> usually<ADVP)><(VP> made<(PP> via<(NP><(NP> the touch<NP)><(PP> with<(NP> a rival<NP)><PP)><NP)><PP)><VP)><VP)> .<S)>\n<(S><(NP><(NP> Singlestick<NP)> ,<(SBAR><(WHNP> which<WHNP)><(S><(VP> is<(NP><(NP> the fourth discipline<NP)><(PP> in<(NP> fencing<NP)><PP)><NP)><VP)><S)><SBAR)><NP)><(VP> appeared<(PP>"
    
    # input = "<|beginoftext|> (S <(NP> <(NP> A police officer <NP)> <(PP> in <(NP> Louisiana <NP)> PP) <NP)> <(VP> has <(VP> resigned <(PP> after (S <(VP> sparking <(NP> online outrage <NP)> <(PP> with <(NP> an inflammatory meme <NP)> PP) VP) S) PP) VP) VP) . S) <|SEP|>"
    # input = "<(S> <(NP> Scientists <NP)> <(VP> have <(VP> \" entangled \" <(NP> <(NP> the <(FRAG> <(WHPP> motions <WHPP)> <FRAG)> <NP)> <(PP> of <(NP> <(NP> pairs <NP)> <(PP> of <(NP> atoms <NP)> PP) <NP)>"
    # ids = torch.tensor([1,2,3,4,5,50268,50268, 50290, 50297, 50297])
    ids = tokenizer.encode(test_input).ids
    print(ids)
    for id in ids:
        print(tokenizer.decode([id], skip_special_tokens=False))
    # gen = TG_attention_bias("TG_GPT2_tokenizer.json", 2048)
    # processed_id = gen.convert_input_to_TG_format(ids)
    print(tokenizer.decode(ids,  skip_special_tokens=False))
    # ids = np.array(ids)
    print(ids)
    print(tokenizer.decode(vocab.convert_treenpy_to_terminal(np.array(ids, dtype=np.uint32)),  skip_special_tokens=False))
    # shuffle_tree = vocab.random_shuffle_tree(ids)
    # print(shuffle_tree)
    # print(tokenizer.decode(shuffle_tree, skip_special_tokens=False))
    # print(processed_id)