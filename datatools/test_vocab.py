from tg_mask import SentencepieceVocab, TG_attention_bias
from tokenizers import Tokenizer
import torch

if __name__ == "__main__":
    tokenizer = Tokenizer.from_file("TG_GPT2_tokenizer.json")
    vocab = SentencepieceVocab.from_vocab_file("TG_GPT2_tokenizer.json")
    ids = torch.tensor([1,2,3,4,5,50268,50268, 50290, 50297, 50297])
    gen = TG_attention_bias("TG_GPT2_tokenizer.json", 2048)
    processed_id = gen.convert_input_to_TG_format(ids)
    print(processed_id)