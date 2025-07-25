import benepar, spacy
from spacy.language import Language
from tqdm import tqdm
from benepar_parse import preprocess_text, split_long_sentence, set_custom_boundaries
from tokenizers import Tokenizer
from convert_TG_and_tokenize import convert_TG_format
import argparse
import json
from datasets import load_dataset

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

from benepar.integrations.spacy_plugin import PartialConstituentData

def doc_parse_prepare(doc):
    parse = doc._.sample_data
    constituent_data = [PartialConstituentData() for j in range(300)]
    for id,sent in enumerate(doc.sents):
        for j in range(300):
            constituent_data[j].starts.append(parse[id][j].starts + sent.start)
            constituent_data[j].ends.append(parse[id][j].ends + sent.start)
            constituent_data[j].labels.append(parse[id][j].labels)


    constituent_data_list = [constituent_data[j].finalize(doc, doc._.label_index) for j in range(300)]
    return constituent_data_list


def get_constituent(span, topk, constituent):
    constituent_data = constituent[topk]

    search_start = constituent_data.loc_to_constituent[span.start]
    if span.start + 1 < len(constituent_data.loc_to_constituent):
        search_end = constituent_data.loc_to_constituent[span.start + 1]
    else:
        search_end = len(constituent_data.ends)
    found_position = None
    for position in range(search_start, search_end):
        if constituent_data.ends[position] <= span.end:
            if constituent_data.ends[position] == span.end:
                found_position = position
            break

    if found_position is None:
        raise Exception("Span is not a constituent: {}".format(span))
    return constituent_data, found_position

def parse_string(span, kth, constituent):
    constituent_data, position = get_constituent(span, kth, constituent)
    label_vocab = constituent_data.label_vocab
    doc = span.doc

    idx = position - 1

    def make_str():
        nonlocal idx
        idx += 1
        i, j, label_idx = (
            constituent_data.starts[idx],
            constituent_data.ends[idx],
            constituent_data.labels[idx],
        )
        label = label_vocab[label_idx]
        if (i + 1) >= j:
            token = doc[i]
            s = (
                "("
                + u"{} {}".format(token.tag_, token.text)
                .replace("(", "-LRB-")
                .replace(")", "-RRB-")
                .replace("{", "-LCB-")
                .replace("}", "-RCB-")
                .replace("[", "-LSB-")
                .replace("]", "-RSB-")
                + ")"
            )
        else:
            children = []
            while (
                (idx + 1) < len(constituent_data.starts)
                and i <= constituent_data.starts[idx + 1]
                and constituent_data.ends[idx + 1] <= j
            ):
                children.append(make_str())

            s = u" ".join(children)

        for sublabel in reversed(label):
            s = u"({} {})".format(sublabel, s)
        return s

    return make_str()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eos", type=int, default=50257)
    parser.add_argument("--bos", type=int, default=50256)
    parser.add_argument("--input_dir", type=str, default="../dataset/bbc-news/")
    parser.add_argument("--output_dir", type=str, default="../dataset/bbc-news/")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file("TG_GPT2_tokenizer.json")
    # vocab = SentencepieceVocab.from_vocab_file("TG_GPT2_tokenizer.json")

    sentparser = spacy.load('en_core_web_md')
    sentparser.add_pipe("set_custom_boundaries", before="parser")
        
    beneparser = benepar.Parser("benepar_en3_large")
    # nlp = spacy.load('en_core_web_md')
    # if spacy.__version__.startswith('2'):
        # nlp.add_pipe(benepar.BeneparComponent("benepar_en3_large"))
    # else:
    #     nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})

    ds = load_dataset("permutans/fineweb-bbc-news", "CC-MAIN-2017-04")
    ds = ds["train"]

    test_docs = []
    # with open("~/TG-Interpolation/dataset/bbc-news/test_tree.txt", "w+") as file:
    for i in range(5):
        test_docs.append(ds[i]["text"])

    with open(args.input_dir + "test_tree.txt", "w+") as input:
        doc_id = 0
        sent_id = 0
        doc_list = []
        for line in tqdm(test_docs):
            doc_id += 1  # count from 1
            split_sents = process_text(line.strip(), max_len=256)
            sents = []
            end = []
            for sent in split_sents:
                if sent==['\n']:
                    end[-1] = True
                else:
                    sents.append(sent)
                    end.append(False)
            # print(sents)
            cur_cnt = 0
            for sent, if_endline in zip(sents, end):
                sent_id += 1
                cur_cnt += 1
                # print(sent.text)
                # text = " ".join(sent)
                # doc = nlp(text)
                # constituent_data = doc_parse_prepare(doc)
                # sent = list(doc.sents)[0]
                # for j in range(300):
                #     tree = parse_string(sent, j, constituent_data)
                #     output.write(tree + "\n")
                input_sentence1 = benepar.InputSentence(words=sent)
                trees = beneparser.parse(input_sentence1)
                for tree in trees:
                    tree = tree[0]
                    parsed_string = tree.pformat(margin=100000) if tree.leaves() != ['\n'] else "(Ċ Ċ)"
                    if if_endline:
                        parsed_string += " (Ċ Ċ)"
                    TG_string = convert_TG_format(parsed_string)
                    input.write(TG_string + "\n")
                    input_ids = tokenizer.encode(TG_string).ids
                    if cur_cnt == 1:
                        input_ids = [args.bos] + input_ids
                    elif cur_cnt == len(split_sents):
                        input_ids += [args.eos]
                    doc_list.append({
                        "doc_id": doc_id,
                        "sent_id": sent_id,
                        "input_ids": input_ids
                    })
        with open(args.output_dir + "test_tree.json", "w+") as output:
            json.dump(doc_list, output, indent=None)
