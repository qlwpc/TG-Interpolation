import argparse
from olmo.data.tg_mask import SentencepieceVocab
import numpy as np
import os
from joblib import Parallel, delayed

if __name__=="__main__":
    vocab = SentencepieceVocab.from_vocab_file("TG_GPT2_tokenizer.json")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_prefix', type=str, default="tree")
    parser.add_argument('--directory', type=str, default="../dataset/bbc-news/testppl_tree")
    parser.add_argument('--output_dir', type=str, default="../dataset/bbc-news/testppl_tg")
    parser.add_argument('--output_prefix', type=str, default="tg")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = np.load(os.path.join(args.directory, f"{args.input_prefix}_300.npy"))
    sent_index = np.load(os.path.join(args.directory, f"{args.input_prefix}_sent_index.npy"))
    doc_index = np.load(os.path.join(args.directory, f"{args.input_prefix}_doc_index.npy"))
    np.save(os.path.join(args.output_dir, f"{args.output_prefix}_doc_index.npy"), doc_index)

    sent_cumindex = np.concatenate([np.array([0], dtype=np.uint32), np.cumsum(sent_index, axis=0)])
    def count_NT(i):
        cnt = 0
        for x in range(sent_cumindex[i], sent_cumindex[i+1]):
            cnt += vocab.is_closing_non_terminal(data[x])
        return i, cnt
    
    results_with_indices = Parallel(n_jobs=2)(delayed(count_NT)(i) for i in range(len(sent_index)))
    for i, add in results_with_indices:
        sent_index[i] += add
    np.save(os.path.join(args.output_dir, f"{args.output_prefix}_sent_index.npy"), sent_index)

    tg_data = vocab.convert_treenpy_to_TG(data)
    np.save(os.path.join(args.output_dir, f"{args.output_prefix}_300.npy"), tg_data)