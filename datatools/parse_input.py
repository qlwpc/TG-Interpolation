import benepar
import spacy
from datasets import load_dataset
from transformers import T5TokenizerFast
from spacy.language import Language
from tqdm import tqdm
from nltk import Tree
import nltk
import logging
import argparse
import re
import subprocess
import os
import json
import torch
import numpy as np
import gc

from benepar import retokenization
nltk.data.path.append("/2024233198/nltk_data")


def my_retokenize(
    tokenizer,
    words,
    space_after,
    return_attention_mask=True,
    return_offsets_mapping=False,
    return_tensors=None,
    **kwargs
):
    """Re-tokenize into subwords.

    Args:
        tokenizer: An instance of transformers.PreTrainedTokenizerFast
        words: List of words
        space_after: A list of the same length as `words`, indicating whether
            whitespace follows each word.
        **kwargs: all remaining arguments are passed on to tokenizer.__call__

    Returns:
        The output of tokenizer.__call__, with one additional dictionary field:
        - **words_from_tokens** -- List of the same length as `words`, where
          each entry is the index of the *last* subword that overlaps the
          corresponding word.
    """
    s = "".join([w + (" " if sp else "") for w, sp in zip(words, space_after)])
    word_offset_starts = np.cumsum(
        [0] + [len(w) + (1 if sp else 0) for w, sp in zip(words, space_after)]
    )[:-1]
    word_offset_ends = word_offset_starts + np.asarray([len(w) for w in words])

    tokenized = tokenizer(
        s,
        return_attention_mask=return_attention_mask,
        return_offsets_mapping=True,
        return_tensors=return_tensors,
        **kwargs
    )
    if return_offsets_mapping:
        token_offset_mapping = tokenized["offset_mapping"]
    else:
        token_offset_mapping = tokenized.pop("offset_mapping")
    if return_tensors is not None:
        token_offset_mapping = np.asarray(token_offset_mapping)[0].tolist()

    offset_mapping_list = [
            (i, (start, end))
            for (i, (start, end)) in enumerate(token_offset_mapping)
            if start != end
        ]
    offset_mapping_iter = iter(offset_mapping_list)
    # print(words)
    # print(space_after)
    # print(s)
    # print(token_offset_mapping)
    # print(word_offset_starts)
    # print(word_offset_ends)
    # print("Tokens:", tokenizer.convert_ids_to_tokens(tokenized.input_ids))
    # print(f"tokens len is {len(tokenized.input_ids)}")

    if len(offset_mapping_list) > 0:
        token_idx, (token_start, token_end) = next(offset_mapping_iter)
        words_from_tokens = [-100] * len(words)
        for word_idx, (word_start, word_end) in enumerate(
            zip(word_offset_starts, word_offset_ends)
        ):
            # print(f"word = {words[word_idx]}")
            while token_end <= word_start:
                # print(f"token_idx = {token_idx} token_start = {token_start}, end = {token_end}")
                token_idx, (token_start, token_end) = next(offset_mapping_iter)
            if token_end > word_end:
                words_from_tokens[word_idx] = token_idx
            while token_end <= word_end:
                words_from_tokens[word_idx] = token_idx
                try:
                    token_idx, (token_start, token_end) = next(offset_mapping_iter)
                except StopIteration:
                    #assert word_idx == len(words) - 1
                    break
    else:
        token_idx, (token_start, token_end) = 0, (0,0)
        words_from_tokens = [0]
    
    if return_tensors == "np":
        words_from_tokens = np.asarray(words_from_tokens, dtype=int)
    elif return_tensors == "pt":
        words_from_tokens = torch.tensor(words_from_tokens, dtype=torch.long)
    elif return_tensors == "tf":
        raise NotImplementedError("Returning tf tensors is not implemented")
    tokenized["words_from_tokens"] = words_from_tokens
    return tokenized

retokenization.retokenize = my_retokenize

@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    if len(doc)==0:
        return doc
    for token in doc[:-1]:
        if re.match(r"\n+", token.text) or token.text == 'Ċ':
            doc[token.i].is_sent_start = True
            doc[token.i + 1].is_sent_start = True
    if doc[-1].text == "Ċ" or re.match(r"\n+", doc[-1].text):
         doc[-1].is_sent_start = True
    return doc

def count_lines_linux_style(filename):
    result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True)
    reslist = result.stdout.split()
    return int(reslist[0]) if reslist!=[] else 0

tokenizer = T5TokenizerFast.from_pretrained("t5-large")
sentparser = spacy.load('en_core_web_md', disable=['tagger', 'attribute_ruler', 'lemmatizer', 'ner'])
sentparser.add_pipe("set_custom_boundaries", before="parser")

beneparser = benepar.Parser("benepar_en3_large")
print(f"parser batch size is {beneparser.batch_size}")

# benepar.download('benepar_en3_wsj')
# nlp = spacy.load('en_core_web_md')
# if spacy.__version__.startswith('2'):
#         nlp.add_pipe(benepar.BeneparComponent("benepar_en3_large"))
# else:
#         nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
# nlp.add_pipe("set_custom_boundaries", before="parser")

def preprocess_text(text:str) -> str:
    text = text.replace("�", "")
    return re.sub(r' $', '', re.sub(r'( )?\n +', '\n', re.sub(r'[^\S\n]+', ' ', text)))

def split_long_sentence(tokens, max_len=256):
    if len(tokens) <= max_len:
        return [tokens]

    punctuations = ['.', '?', '!', ';', ':', ',']
    
    split_index = -1
    for i in range(max_len - 1, -1, -1):
        if tokens[i] in punctuations:
            split_index = i
            break
    
    if split_index != -1:
        part1 = tokens[:split_index + 1]
        part2 = tokens[split_index + 1:]
        # print(f"lenpart1 = {len(part1)}, 2 = {len(part2)}")
        return [part1] + split_long_sentence(part2, max_len)
    else:
        part1 = tokens[:max_len]
        part2 = tokens[max_len:]
        return [part1] + split_long_sentence(part2, max_len)


def split_text_into_sents(text:str):
    text = preprocess_text(text)
    doc = sentparser(text)
    sentences = [[str(token.text) for token in sent] for sent in doc.sents] # 提取字符串
    del doc
    return sentences

def split_list_limit(sub_list, max_tokens=512):
    punctuations = {',', '.', '!', '?', ';', ':', '，', '。', '！', '？', '；', '：', '-'}
    final_output = []
    charlen = sum([len(x) for x in sub_list])
    # print(sub_list)
    # print(f"charlen is {charlen}")
    if charlen <= max_tokens - 20:
        return [sub_list]

    new_list = []
    for words in sub_list:
        if len(words)>max_tokens:
            chunks = [words[i : i + max_tokens-5] for i in range(0, len(words), max_tokens-5)]
            new_list.extend(chunks)
        else:
            new_list.append(words)
    sub_list = new_list
    
    encoding = tokenizer(
        sub_list, 
        is_split_into_words=True, 
        return_attention_mask=False,   
        return_token_type_ids=False,   
        return_tensors=None,           
        return_offsets_mapping=False,  
        return_length=False,           
        add_special_tokens=False
    )
    input_ids = encoding.input_ids
    word_ids = encoding.word_ids()
    num_idx = np.zeros((len(sub_list), ), dtype=np.int32)
    for id in word_ids:
        num_idx[id] += 1
    num_idx = np.concatenate([np.zeros((1,), dtype=np.int32), np.cumsum(num_idx)])
    start_idx = 0
    print(sub_list)
    cnt = 0
    while len(input_ids) - num_idx[start_idx] > max_tokens:
        limit_word_idx = max(word_ids[num_idx[start_idx] + max_tokens] - 1, start_idx)
        split_at_word = limit_word_idx + 1 # 默认切分位置
        
        for i in range(limit_word_idx, start_idx, -1):
            if any(p in sub_list[i] for p in punctuations):
                split_at_word = i + 1
                break
        print(f"split_at_word at {split_at_word} nextword is {sub_list[split_at_word]}")
        final_output.append(sub_list[start_idx:split_at_word])
        start_idx = split_at_word
        cnt += 1
        if cnt >= 100000:
            break
    
    if len(input_ids) - num_idx[start_idx] > 0:
        final_output.append(sub_list[start_idx:])
    recover = []
    for split in final_output:
        recover += split
    assert (recover==sub_list)
    del encoding
    del input_ids
    del word_ids
    return final_output

def process_doc_into_maxlen(sents, max_len=512):
    final_sentences = []
    for tokens in sents:
        final_sentences.extend(split_list_limit(tokens, max_tokens=max_len))
    return final_sentences

# text = "Adam Afriyie (Windsor), Peter Aldous (Waveney), David Amess (Southend West), Stuart Andrew (Pudsey), Richard Bacon (Norfolk South), Steven Baker (Wycombe), Stephen Barclay (Cambridgeshire North East), John Baron (Basildon & Billericay), Gavin Barwell (Croydon Central), Guto Bebb (Aberconwy), Andrew Bingham (High Peak), Brian Binley (Northampton South), Crispin Blunt (Reigate), Graham Brady (Altrincham & Sale West), Andrew Bridgen (Leicestershire North West), Steve Brine (Winchester), Fiona Bruce (Congleton), Aidan Burley (Cannock Chase), Conor Burns (Bournemouth West), David Burrowes (Enfield Southgate), Dan Byles (Warwickshire North), Alun Cairns (Vale of Glamorgan), Bill Cash (Stone), Rehman Chishti (Gillingham & Rainham), Christopher Chope (Christchurch), James Clappison (Hertsmere), Geoffrey Cox (Devon West & Torridge), Tracey Crouch (Chatham & Aylesford), David Davies (Monmouth), Philip Davies (Shipley), David Davis (Haltemprice & Howden), Nick de Bois (Enfield North), Caroline Dinenage (Gosport), Nadine Dorries (Bedfordshire Mid),Richard Drax (Dorset South), James Duddridge (Rochford & Southend East), Graham Evans (Weaver Vale), Lorraine Fullbrook (South Ribble), Roger Gale (Thanet North), James Gray (Wiltshire North), Robert Halfon (Harlow), Simon Hart (Carmarthen West & Pembrokeshire South), Gordon Henderson (Sittingbourne & Sheppey), Sir Gerald Howarth (Aldershot), Stewart Jackson (Peterborough), Bernard Jenkin (Harwich & Essex North), Gareth Johnson (Dartford), Marcus Jones (Nuneaton), Daniel Kawczynski (Shrewsbury & Atcham), Chris Kelly (Dudley South), Simon Kirby (Brighton Kemptown), Andrea Leadsom (Northamptonshire South), Jessica Lee (Erewash), Phillip Lee (Bracknell), Edward Leigh (Gainsborough), Charlotte Leslie (Bristol North West), Julian Lewis (New Forest East), Ian Liddell-Grainger (Bridgwater & Somerset West), Jonathan Lord (Woking), Tim Loughton (Worthing East & Shoreham), Karen Lumley (Redditch), Jason McCartney (Colne Valley), Karl McCartney (Lincoln), Stephen McPartland (Stevenage), Anne Main (St Albans), Paul Maynard (Blackpool North & Cleveleys), Mark Menzies (Fylde), Patrick Mercer (Newark), Stephen Metcalfe (Basildon South & Thurrock East), Nigel Mills (Amber Valley), David Morris (Morecambe & Lunesdale), James Morris (Halesowen & Rowley Regis), Caroline Nokes (Romsey & Southampton North), David Nuttall (Bury North), Matthew Offord (Hendon), Eric Ollerenshaw (Lancaster & Fleetwood), Priti Patel (Witham), John Penrose (Weston-Super-Mare), Andrew Percy (Brigg & Goole), Stephen Phillips (Sleaford & North Hykeham), Chris Pincher (Tamworth), Dominic Raab (Esher & Walton), Mark Reckless (Rochester & Strood), John Redwood (Wokingham), Jacob Rees-Mogg (Somerset North East), Laurence Robertson (Tewkesbury), Andrew Rosindell (Romford), David Ruffley (Bury St Edmunds), Andrew Selous (Bedfordshire South West), Alec Shelbrooke (Elmet & Rothwell), Sir Richard Shepherd (Aldridge-Brownhills), Henry Smith (Crawley), Mark Spencer (Sherwood), Andrew Stephenson (Pendle), John Stevenson (Carlisle), Iain Stewart (Milton Keynes South), Gary Streeter (Devon South West), Mel Stride (Devon Central), Julian Sturdy (York Outer), Sir Peter Tapsell (Louth & Horncastle), Justin Tomlinson (Swindon North), David Tredinnick (Bosworth), Andrew Turner (Isle of Wight), Martin Vickers (Cleethorpes), Charles Walker (Broxbourne), Robin Walker (Worcester), James Wharton (Stockton South), Heather Wheeler (Derbyshire South), Chris White (Warwick & Leamington), Craig Whittaker (Calder Valley), John Whittingdale (Maldon), Bill Wiggin (Herefordshire North), Dr Sarah Wollaston (Totnes), Nadhim Zahawi (Stratford-on-Avon). The two Tory tellers were Peter Bone (Wellingborough) and Philip Hollobone (Kettering)."
# sents = process_text(text)
# print(sents)
# input_sentence1 = benepar.InputSentence(
#     words=['The', 'time', 'for', 'action', 'is', 'now'] * 25,
# )
# gene = beneparser.parse_sents([input_sentence1] * 64) # [130, 200]


# length more than 70? -> 15 sents per batch
# length smaller than 70 -> 64 sents per batch
shortlen = 70
short_batchsize = 64
long_batchsize = 15

class batch_buffer:
    def __init__(self, output_file, pbar):
        self.init_batch()
        self.doc_to_write = ""
        self.file = output_file
        self.pbar = pbar
        
    def init_batch(self):
        self.batches = []
        self.is_short = True
        self.document_end = []
    
    def write(self, sent:str, is_end:bool):
        self.doc_to_write += sent + ("\n" if is_end else " ") 
        if is_end:
            self.file.write(self.doc_to_write)
            self.doc_to_write = ""
            self.pbar.update(1)

    def parse_batch(self):
        if len(self.batches)==0:
            return
        TreeGen = beneparser.parse_sents(self.batches)
        for tree, DocEnd in zip(TreeGen, self.document_end):
            tree = tree[0]
            leaves = tree.leaves()
            if len(leaves) == 1 and re.match(r"\n+", leaves[0]):
                parsed_string = leaves[0].replace("\n", "(Ċ Ċ) ").rstrip()
            else:
                parsed_string = tree.pformat(margin=100000) if tree.leaves() != ['\n'] else "(Ċ Ċ)"
            #print(parsed_string)
            self.write(parsed_string, DocEnd)
        self.init_batch()
    
    def append_batch(self, sents):
        if len(sents)==0:
            self.batches.append(benepar.InputSentence(words='\n'))
            self.document_end.append(True)
        for i, sent in enumerate(sents):
            input_sent = benepar.InputSentence(words=sent)
            if len(sent)>70 and self.is_short:
                self.parse_batch()
                self.is_short = False
            
            self.batches.append(input_sent)
            self.document_end.append( i == len(sents)-1 )
            if self.is_short and len(self.batches)==short_batchsize or \
               not self.is_short and len(self.batches)==long_batchsize:
                self.parse_batch()


def load_shrunk_dataset(directory_path, file_pattern:str = None):
    file_pattern = file_pattern or "*.arrow"
    data_files = []
    regex = re.compile(file_pattern)
    for filename in os.listdir(directory_path):
        if regex.match(filename):
            data_files.append(os.path.join(directory_path,filename))
    print(data_files)
    if not data_files:
        print(f"错误：在路径 {directory_path} 下没找到任何 .arrow 文件！")
        return None
    print(f"找到 {len(data_files)} 个分片文件，正在加载...")
    dataset = load_dataset(
        "arrow", data_files=data_files, split="train",
    )
    return dataset

def prepare_dataset(config:str):
    prepared_ds = []
    if config[0:3]=="str":
        prepared_ds.append(config[3:])
        filename = "tmp_parse.txt"
    elif config[0:4]=="file":
        with open(config[4:], 'r') as file:
            prepared_ds.append("".join(file.readlines()))
        filename = f"tmp_out.txt"
    elif config[0:4]=="xsum":
        ds = load_dataset("EdinburghNLP/xsum")
        filename = os.path.join(os.path.expanduser("~/TG-Interpolation/dataset/Xsum"), config + ".txt")
        if "train" in config:
            ds = ds["train"]
        elif "validation" in config:
            ds = ds["validation"]
        else:
            ds = ds["test"]
        if "summary" in config:
            text = "summary"
        else:
            text = "document"
        for doc in ds:
            if doc['document']!='':
                prepared_ds.append(doc[text])
    elif config[0:4]=="AX-b":
        filename = f"../dataset/SuperGLUE/AX-b/sentence{config[-1]}.txt"
        key = f"sentence{config[-1]}"
        with open('../dataset/SuperGLUE/AX-b/AX-b.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                prepared_ds.append(data[key])
    elif config[0:4]=="AX-g":
        key = "premise" if config[-1]=='1' else "hypothesis"
        filename = f"../dataset/SuperGLUE/AX-g/{key}.txt"
        with open('../dataset/SuperGLUE/AX-g/AX-g.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                prepared_ds.append(data[key])
    elif config[0:6]=="ReCoRD":
        split = config.split("_")
        key = split[2]
        filename = f"../dataset/SuperGLUE/{split[0]}/{split[1]}_{split[2]}.txt"
        with open(f'../dataset/SuperGLUE/{split[0]}/{split[1]}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                if key=="text":
                    prepared_ds.append(data["passage"]["text"])
                elif key=='query':
                    prepared_ds.append(data["qas"][0]["query"])
    elif config[:9] == "hellaswag":
        split = config.split("_")
        key = split[1]
        filename = f"../dataset/hellaswag/{config}.txt"
        def swag_preprocess(text):
            text = text.strip()
            text = text.replace(" [title]", ". ")
            text = re.sub("\\[.*?\\] ", "", text)
            text = re.sub(r"^\.+", "", text)
            text = text.replace("..", ".")
            text = text.replace("  ", " ")
            return text
        with open(f"../dataset/hellaswag/{config}.jsonl", 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                prepared_ds.append(swag_preprocess(data["ctx_a"]))
                for endings in data["endings"]:
                    prepared_ds.append(data["ctx_b"] + endings)
                prepared_ds.append(swag_preprocess(data["ctx_b"].capitalize() + " " + endings))
    elif config[:10] == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl")
        filename = os.path.join("../dataset/winogrande/", config + ".txt")
        if "train" in config:
            ds = ds["train"]
        elif "val" in config:
            ds = ds["validation"]
        else:
            ds = ds["test"]
        for doc in ds:
            prepared_ds.append(doc["sentence"].replace("_", doc["option1"]))
            prepared_ds.append(doc["sentence"].replace("_", doc["option2"]))
    elif config[:10] == "finewebedu":
        edupath = "~/.cache/huggingface/datasets/HuggingFaceFW___fineweb-edu/sample-100BT/0.0.0/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
        file_pattern = config[10:] if len(config)>10 else None
        ds = load_shrunk_dataset(os.path.expanduser(edupath), file_pattern=file_pattern)
        os.makedirs("../dataset/finewebedu-100BT/", exist_ok=True)
        filename = f"../dataset/finewebedu-100BT/{file_pattern}.txt"
        for doc in ds:
            prepared_ds.append(doc['text'])
    else: # file_split_key
        split = config.split("_")
        key = split[2]
        filename = f"../dataset/SuperGLUE/{split[0]}/{split[1]}_{split[2]}.txt"
        with open(f'../dataset/SuperGLUE/{split[0]}/{split[1]}.jsonl', 'r', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line.strip())
                prepared_ds.append(data[key])

    return filename, prepared_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_list', type=str)
    parser.add_argument('--start_index', type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    print(args.input_list)
    result_list = args.input_list.split(',') if args.input_list else []
    print("Received list:", result_list) # [ "CC-MAIN-2014-41"]
    config = result_list
    for split in config:
        filename, ds = prepare_dataset(split)
        print(f"len dataset is {len(ds)}")
        totallen = len(ds)
        print(totallen)
        logging.info(f"start parsing {split}")
        pbar = tqdm(total=totallen)

        # filename = "bbc-news/" + split + ".txt"
        index = count_lines_linux_style(filename)
        pbar.update(index)
        with open(filename, "a+") as output:
            Buffer = batch_buffer(output, pbar)
            while index < len(ds):
                document = ds[index]
                # text = document['text']
                # doc = sentparser(document)
                # print(f"doc is {document}")
                max_len = 450
                doc = split_text_into_sents(document)
                split_sents = process_doc_into_maxlen(doc, max_len=max_len)
                # print(split_sents)
                Buffer.append_batch(split_sents)
                # print(f"input is {split_sents}")
                index += 1
                if index % 200 == 0:
                    gc.collect()
                # if index % 10000 == 0:
                #     torch.cuda.empty_cache()
            Buffer.parse_batch()
            print("end parse")
        index = 0
                
    logging.info("finished")

if __name__=="__main__":
    main()
