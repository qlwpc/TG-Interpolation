from tokenizers import Tokenizer
from tqdm import tqdm
from tokenizers import Tokenizer
import numpy as np
from numpy import ndarray
import os
from joblib import Parallel, delayed
import argparse
from datasets import load_dataset
import json
import spacy
# from olmo.data.tg_mask import SentencepieceVocab

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "parse_pretrain_data"))
from benepar_parse import set_custom_boundaries, preprocess_text, split_long_sentence

def process_text(text, max_len=256):
    text = preprocess_text(text)
    doc = sentparser(text)
    final_sentences = []
    for sent in doc.sents:
        tokens = [token.text for token in sent]
        if len(sent) <= max_len:
            final_sentences.append(tokens)
        else:
            segments = split_long_sentence(tokens, max_len)
            final_sentences.extend(segments)
    return final_sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_OutDomain', action='store_true')  # 接收单个字符串
    parser.add_argument("--output_dir", type=str, default="../dataset/bbc-news/test300/")
    parser.add_argument("--eos", type=int, default=50256)
    parser.add_argument("--bos", type=int, default=50257)
    args = parser.parse_args()

    # vocab = SentencepieceVocab.from_vocab_file("TG_GPT2_tokenizer.json")
    tokenizer = Tokenizer.from_file("TG_GPT2_tokenizer.json")
    sentparser = spacy.load('en_core_web_md')
    sentparser.add_pipe("set_custom_boundaries", before="parser")
    InDomain = ['CC-MAIN-2013-48', 'CC-MAIN-2014-10', 'CC-MAIN-2014-15', 'CC-MAIN-2014-23', 'CC-MAIN-2014-35', 'CC-MAIN-2014-41', 'CC-MAIN-2014-42', 'CC-MAIN-2014-49', 'CC-MAIN-2014-52', 'CC-MAIN-2015-06', 'CC-MAIN-2015-11', 'CC-MAIN-2015-14', 'CC-MAIN-2015-18', 'CC-MAIN-2015-22', 'CC-MAIN-2015-27', 'CC-MAIN-2015-32', 'CC-MAIN-2015-35', 'CC-MAIN-2015-40', 'CC-MAIN-2015-48', 'CC-MAIN-2016-07', 'CC-MAIN-2016-18', 'CC-MAIN-2016-22', 'CC-MAIN-2016-26', 'CC-MAIN-2016-30', 'CC-MAIN-2016-36', 'CC-MAIN-2016-40', 'CC-MAIN-2016-44', 'CC-MAIN-2016-50', 'CC-MAIN-2017-04', 'CC-MAIN-2017-09', 'CC-MAIN-2017-13', 'CC-MAIN-2017-17', 'CC-MAIN-2017-22', 'CC-MAIN-2017-26', 'CC-MAIN-2017-30', 'CC-MAIN-2017-34', 'CC-MAIN-2017-39', 'CC-MAIN-2017-43', 'CC-MAIN-2017-47', 'CC-MAIN-2017-51', 'CC-MAIN-2018-05', 'CC-MAIN-2018-09', 'CC-MAIN-2018-13', 'CC-MAIN-2018-17', 'CC-MAIN-2018-22', 'CC-MAIN-2018-26', 'CC-MAIN-2018-30', 'CC-MAIN-2018-34', 'CC-MAIN-2018-39', 'CC-MAIN-2018-43', 'CC-MAIN-2018-47', 'CC-MAIN-2018-51', 'CC-MAIN-2019-04', 'CC-MAIN-2019-09', 'CC-MAIN-2019-13', 'CC-MAIN-2019-18', 'CC-MAIN-2019-22', 'CC-MAIN-2019-26', 'CC-MAIN-2019-30', 'CC-MAIN-2019-35', 'CC-MAIN-2019-39', 'CC-MAIN-2019-43', 'CC-MAIN-2019-47', 'CC-MAIN-2019-51', 'CC-MAIN-2020-05', 'CC-MAIN-2020-10', 'CC-MAIN-2020-16', 'CC-MAIN-2020-24', 'CC-MAIN-2020-29', 'CC-MAIN-2020-34', 'CC-MAIN-2020-40', 'CC-MAIN-2020-45', 'CC-MAIN-2020-50', 'CC-MAIN-2021-04', 'CC-MAIN-2021-10', 'CC-MAIN-2021-17', 'CC-MAIN-2021-21', 'CC-MAIN-2021-25', 'CC-MAIN-2021-31', 'CC-MAIN-2021-39', 'CC-MAIN-2021-43', 'CC-MAIN-2021-49', 'CC-MAIN-2022-05', 'CC-MAIN-2022-21', 'CC-MAIN-2022-27', 'CC-MAIN-2022-33', 'CC-MAIN-2022-40', 'CC-MAIN-2022-49']
    OutDomain = ['CC-MAIN-2023-06', 'CC-MAIN-2023-14', 'CC-MAIN-2023-23', 'CC-MAIN-2023-40', 'CC-MAIN-2023-50']

    with open("../dataset/bbc-news/test_index.json", 'r') as file:
        all_test_indexes = json.load(file)
    
    all_files = InDomain if not args.list_OutDomain else OutDomain

    exclude_list = os.listdir("../dataset/bbc-news/test300")
    # exclude_list = [os.path.splitext(name)[0] for name in exclude_list]
    exclude_list = sorted(exclude_list)
    # print(exclude_list)
    tokenize_list = []
    for filename in all_files:
        full_path = "tree_300_" + filename + ".npy"
        if full_path not in exclude_list:
            tokenize_list.append(filename)
    
    def tokenize_testppl_file(input):
        print(f"start tokenizing {input}")
        ds = load_dataset("permutans/fineweb-bbc-news", input)
        ds = ds["train"]
        sents_cnt = []
        for index in all_test_indexes[input]:
            line = ds[index]["text"]
            inputstr = line.strip()
            split_sents = process_text(inputstr, max_len=256)
            # print(split_sents)
            cnt = 0
            for sent in split_sents:
                if sent!=['\n']:
                    cnt += 1
            sents_cnt.append(cnt)
        
        np.save(os.path.join(args.output_dir, f"doc_index_{input}.npy"), np.array(sents_cnt, dtype=np.uint32))
        for i in range(1, len(sents_cnt)):
            sents_cnt[i] += sents_cnt[i-1]

        doc_id = 1
        line_num = 0
        sent_length = []
        tree_ids = []
        with open(args.output_dir + input + '.txt', 'r') as file:
            for line in tqdm(file):
                if line_num % 300 == 0:
                    if line_num // 300 == sents_cnt[doc_id]:
                        doc_id += 1

                inputstr = line.strip()
                outputid = tokenizer.encode(inputstr).ids
                # subtitute special <|SEP|>(50261) to \n(198)
                if outputid[-1] == 50261:
                    outputid[-1] = 198
                if line_num // 300 == 0 or line_num // 300 == sents_cnt[doc_id]:
                    outputid = [args.bos] + outputid
                elif line_num // 300 == sents_cnt[doc_id] - 1:
                    outputid += [args.eos]
                outputid = np.array(outputid, dtype=np.uint16)
                # doc_list.append({
                #     "doc_id": doc_id,
                #     "sent_id": line_num//300 + 1, # index start from 1
                #     "input_ids": outputid
                # })
                sent_length.append(outputid.shape[0])
                tree_ids.append(outputid)
                line_num += 1
                # print(tokenizer.decode(TG_ids, skip_special_tokens=False))
        
        # with open(args.output_dir + f"test_tree_{input}.json", "w+") as output:
        #     json.dump(doc_list, output, indent=None)
        final_ids = np.concatenate(tree_ids, axis=0)
        np.save(os.path.join(args.output_dir, f"tree_300_{input}.npy"), final_ids)
        np.save(os.path.join(args.output_dir, f"sent_index_{input}.npy"), np.array(sent_length, dtype=np.uint16))
    
    # Parallel(n_jobs=5)(delayed(tokenize_testppl_file)(name) for name in tokenize_list)
    for name in tokenize_list:
        tokenize_testppl_file(name)

    data = []
    sent = []
    doc = []
    for name in all_files:
        data.append(np.load(os.path.join(args.output_dir, f"tree_300_{name}.npy")))
        sent.append(np.load(os.path.join(args.output_dir, f"sent_index_{name}.npy")))
        doc.append(np.load(os.path.join(args.output_dir, f"doc_index_{name}.npy")))
    
    data = np.concatenate(data, axis=0)
    np.save(os.path.join(args.output_dir, f"test_tree_300.npy"), data)
    sent = np.concatenate(sent, axis=0)
    np.save(os.path.join(args.output_dir, f"tree_sent_index.npy"), sent)
    doc = np.concatenate(doc, axis=0)
    np.save(os.path.join(args.output_dir, f"tree_doc_index.npy"), doc)