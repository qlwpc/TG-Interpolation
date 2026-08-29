import os
# benepar's Retokenizer converts the slow t5 sentencepiece tokenizer, whose
# _pb2 generated code is incompatible with protobuf>=4 (see parse_input.py).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import benepar
import spacy
from datasets import load_dataset
from spacy.language import Language
from tqdm import tqdm
from nltk import Tree
import logging
import argparse
import os
import re
import subprocess
from pathlib import Path

HUB_DATASET = "permutans/fineweb-bbc-news"


@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "\n" or token.text == 'Ċ':
            doc[token.i].is_sent_start = True
            doc[token.i + 1].is_sent_start = True
    if doc[-1].text == "Ċ" or doc[-1].text == '\n':
         doc[-1].is_sent_start = True
    return doc

# benepar.download('benepar_en3_wsj')
# nlp = spacy.load('en_core_web_md')
# if spacy.__version__.startswith('2'):
#         nlp.add_pipe(benepar.BeneparComponent("benepar_en3_large"))
# else:
#         nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
# nlp.add_pipe("set_custom_boundaries", before="parser")

def preprocess_text(text:str) -> str:
    return re.sub(r'\n +', '\n', re.sub(r' +', ' ', text))

def split_long_sentence(tokens, max_len=512):
    """
    递归分割超过 max_len 的句子，优先按标点分割，否则硬切分
    """
    if len(tokens) <= max_len:
        return [tokens]
    
    # 优先级从高到低的切分标点
    punctuations = ['.', '?', '!', ';', ':', ',']
    
    # 在 max_len 位置向前查找最近的标点
    split_index = -1
    for i in range(max_len - 1, -1, -1):
        if tokens[i] in punctuations:
            split_index = i
            break
    
    # 找到标点则按标点分割
    if split_index != -1:
        part1 = tokens[:split_index + 1]
        part2 = tokens[split_index + 1:]
        # print(f"lenpart1 = {len(part1)}, 2 = {len(part2)}")
        return [part1] + split_long_sentence(part2, max_len)
    
    # 找不到标点则硬切分
    else:
        part1 = tokens[:max_len]
        part2 = tokens[max_len:]
        return [part1] + split_long_sentence(part2, max_len)


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

# text = "Adam Afriyie (Windsor), Peter Aldous (Waveney), David Amess (Southend West), Stuart Andrew (Pudsey), Richard Bacon (Norfolk South), Steven Baker (Wycombe), Stephen Barclay (Cambridgeshire North East), John Baron (Basildon & Billericay), Gavin Barwell (Croydon Central), Guto Bebb (Aberconwy), Andrew Bingham (High Peak), Brian Binley (Northampton South), Crispin Blunt (Reigate), Graham Brady (Altrincham & Sale West), Andrew Bridgen (Leicestershire North West), Steve Brine (Winchester), Fiona Bruce (Congleton), Aidan Burley (Cannock Chase), Conor Burns (Bournemouth West), David Burrowes (Enfield Southgate), Dan Byles (Warwickshire North), Alun Cairns (Vale of Glamorgan), Bill Cash (Stone), Rehman Chishti (Gillingham & Rainham), Christopher Chope (Christchurch), James Clappison (Hertsmere), Geoffrey Cox (Devon West & Torridge), Tracey Crouch (Chatham & Aylesford), David Davies (Monmouth), Philip Davies (Shipley), David Davis (Haltemprice & Howden), Nick de Bois (Enfield North), Caroline Dinenage (Gosport), Nadine Dorries (Bedfordshire Mid),Richard Drax (Dorset South), James Duddridge (Rochford & Southend East), Graham Evans (Weaver Vale), Lorraine Fullbrook (South Ribble), Roger Gale (Thanet North), James Gray (Wiltshire North), Robert Halfon (Harlow), Simon Hart (Carmarthen West & Pembrokeshire South), Gordon Henderson (Sittingbourne & Sheppey), Sir Gerald Howarth (Aldershot), Stewart Jackson (Peterborough), Bernard Jenkin (Harwich & Essex North), Gareth Johnson (Dartford), Marcus Jones (Nuneaton), Daniel Kawczynski (Shrewsbury & Atcham), Chris Kelly (Dudley South), Simon Kirby (Brighton Kemptown), Andrea Leadsom (Northamptonshire South), Jessica Lee (Erewash), Phillip Lee (Bracknell), Edward Leigh (Gainsborough), Charlotte Leslie (Bristol North West), Julian Lewis (New Forest East), Ian Liddell-Grainger (Bridgwater & Somerset West), Jonathan Lord (Woking), Tim Loughton (Worthing East & Shoreham), Karen Lumley (Redditch), Jason McCartney (Colne Valley), Karl McCartney (Lincoln), Stephen McPartland (Stevenage), Anne Main (St Albans), Paul Maynard (Blackpool North & Cleveleys), Mark Menzies (Fylde), Patrick Mercer (Newark), Stephen Metcalfe (Basildon South & Thurrock East), Nigel Mills (Amber Valley), David Morris (Morecambe & Lunesdale), James Morris (Halesowen & Rowley Regis), Caroline Nokes (Romsey & Southampton North), David Nuttall (Bury North), Matthew Offord (Hendon), Eric Ollerenshaw (Lancaster & Fleetwood), Priti Patel (Witham), John Penrose (Weston-Super-Mare), Andrew Percy (Brigg & Goole), Stephen Phillips (Sleaford & North Hykeham), Chris Pincher (Tamworth), Dominic Raab (Esher & Walton), Mark Reckless (Rochester & Strood), John Redwood (Wokingham), Jacob Rees-Mogg (Somerset North East), Laurence Robertson (Tewkesbury), Andrew Rosindell (Romford), David Ruffley (Bury St Edmunds), Andrew Selous (Bedfordshire South West), Alec Shelbrooke (Elmet & Rothwell), Sir Richard Shepherd (Aldridge-Brownhills), Henry Smith (Crawley), Mark Spencer (Sherwood), Andrew Stephenson (Pendle), John Stevenson (Carlisle), Iain Stewart (Milton Keynes South), Gary Streeter (Devon South West), Mel Stride (Devon Central), Julian Sturdy (York Outer), Sir Peter Tapsell (Louth & Horncastle), Justin Tomlinson (Swindon North), David Tredinnick (Bosworth), Andrew Turner (Isle of Wight), Martin Vickers (Cleethorpes), Charles Walker (Broxbourne), Robin Walker (Worcester), James Wharton (Stockton South), Heather Wheeler (Derbyshire South), Chris White (Warwick & Leamington), Craig Whittaker (Calder Valley), John Whittingdale (Maldon), Bill Wiggin (Herefordshire North), Dr Sarah Wollaston (Totnes), Nadhim Zahawi (Stratford-on-Avon). The two Tory tellers were Peter Bone (Wellingborough) and Philip Hollobone (Kettering)."
# sents = process_text(text)
# print(sents)
# input_sentence1 = benepar.InputSentence(
#     words=['The', 'time', 'for', 'action', 'is', 'now'] * 25,
# )
# gene = beneparser.parse_sents([input_sentence1] * 64) # [130, 200]

def count_lines_linux_style(filename):
    result = subprocess.run(['wc', '-l', filename], capture_output=True, text=True)
    reslist = result.stdout.split()
    return int(reslist[0]) if reslist!=[] else 0

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
            parsed_string = tree.pformat(margin=100000) if tree.leaves() != ['\n'] else "(Ċ Ċ)"
            #print(parsed_string)
            self.write(parsed_string, DocEnd)
        self.init_batch()
    
    def append_batch(self, sents):
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


#[, 'CC-MAIN-2013-48', 'CC-MAIN-2014-10', 'CC-MAIN-2014-15', 'CC-MAIN-2014-23', 'CC-MAIN-2014-35', 'CC-MAIN-2014-41', 'CC-MAIN-2014-42', 'CC-MAIN-2014-49', 'CC-MAIN-2014-52', 'CC-MAIN-2015-06', 'CC-MAIN-2015-11', 'CC-MAIN-2015-14', 'CC-MAIN-2015-18', 'CC-MAIN-2015-22', 'CC-MAIN-2015-27', 'CC-MAIN-2015-32', 'CC-MAIN-2015-35', 'CC-MAIN-2015-40', 'CC-MAIN-2015-48', 'CC-MAIN-2016-07', 'CC-MAIN-2016-18', 'CC-MAIN-2016-22', 'CC-MAIN-2016-26', 'CC-MAIN-2016-30', 'CC-MAIN-2016-36', 'CC-MAIN-2016-40', 'CC-MAIN-2016-44', 'CC-MAIN-2016-50', 'CC-MAIN-2017-04', 'CC-MAIN-2017-09', 'CC-MAIN-2017-13', 'CC-MAIN-2017-17', 'CC-MAIN-2017-22', 'CC-MAIN-2017-26', 'CC-MAIN-2017-30', 'CC-MAIN-2017-34', 'CC-MAIN-2017-39', 'CC-MAIN-2017-43', 'CC-MAIN-2017-47', 'CC-MAIN-2017-51', 'CC-MAIN-2018-05', 'CC-MAIN-2018-09', 'CC-MAIN-2018-13', 'CC-MAIN-2018-17', 'CC-MAIN-2018-22', 'CC-MAIN-2018-26', 'CC-MAIN-2018-30', 'CC-MAIN-2018-34', 'CC-MAIN-2018-39', 'CC-MAIN-2018-43', 'CC-MAIN-2018-47', 'CC-MAIN-2018-51', 'CC-MAIN-2019-04', 'CC-MAIN-2019-09', 'CC-MAIN-2019-13', 'CC-MAIN-2019-18', 'CC-MAIN-2019-22', 'CC-MAIN-2019-26', 'CC-MAIN-2019-30', 'CC-MAIN-2019-35', 'CC-MAIN-2019-39', 'CC-MAIN-2019-43', 'CC-MAIN-2019-47', 'CC-MAIN-2019-51', 'CC-MAIN-2020-05', 'CC-MAIN-2020-10', 'CC-MAIN-2020-16', 'CC-MAIN-2020-24', 'CC-MAIN-2020-29', 'CC-MAIN-2020-34', 'CC-MAIN-2020-40', 'CC-MAIN-2020-45', 'CC-MAIN-2020-50', 'CC-MAIN-2021-04', 'CC-MAIN-2021-10', 'CC-MAIN-2021-17', 'CC-MAIN-2021-21', 'CC-MAIN-2021-25', 'CC-MAIN-2021-31', 'CC-MAIN-2021-39', 'CC-MAIN-2021-43', 'CC-MAIN-2021-49', 'CC-MAIN-2022-05', 'CC-MAIN-2022-21', 'CC-MAIN-2022-27', 'CC-MAIN-2022-33', 'CC-MAIN-2022-40', 'CC-MAIN-2022-49', 'CC-MAIN-2023-06', 'CC-MAIN-2023-14', 'CC-MAIN-2023-23', 'CC-MAIN-2023-40', 'CC-MAIN-2023-50']
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_arg', type=str)  # 接收单个字符串
    parser.add_argument('--data_files', type=str, default=None,
                        help='comma-separated local parquet/arrow files (or glob) to parse '
                             'instead of resolving the Hub dataset (useful when the Hub '
                             'resolve endpoint is unreachable/blocked)')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--max-docs', type=int, default=None,
                        help='stop after this many documents (smoke tests); '
                             'default: parse the whole split')
    parser.add_argument('--output-dir', type=Path, default=Path('dataset/bbc-news-parsed'),
                        help='directory for one-document-per-line parsed text shards')
    parser.add_argument('--skip-deps', action='store_true',
                        help='skip the dependency bootstrap (models/tokenizers)')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_deps:
        try:
            from setup_parse_deps import ensure_all  # sibling-script import
        except ImportError:
            from datatools.parse_pretrain_data.setup_parse_deps import ensure_all
        ensure_all()
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    sentparser = spacy.load('en_core_web_md')
    sentparser.add_pipe("set_custom_boundaries", before="parser")

    beneparser = benepar.Parser("benepar_en3_large")
    print(f"parser batch size is {beneparser.batch_size}")

    print(args.list_arg)
    result_list = args.list_arg.split(',') if args.list_arg else []
    print("Received list:", result_list) # [ "CC-MAIN-2014-41"]
    config = result_list

    if args.data_files:
        # local-parquet mode: group files by their parent config dir name
        import glob as _glob
        files = []
        for pattern in args.data_files.split(','):
            files.extend(sorted(_glob.glob(os.path.expanduser(pattern.strip()))))
        if not files:
            raise SystemExit(f"--data_files matched nothing: {args.data_files}")
        by_config = {}
        for f in files:
            by_config.setdefault(Path(f).parent.name, []).append(f)
        config = sorted(by_config)
        datasets_map = {
            c: load_dataset("parquet", data_files=by_config[c], split="train")
            for c in config
        }
    else:
        datasets_map = {c: load_dataset(HUB_DATASET, c) for c in config}

    for split in config:
        ds = datasets_map[split]
        if hasattr(ds, "keys"):  # DatasetDict (hub path) → take the train split
            ds = ds["train"]
        totallen = len(ds)
        print(totallen)
        logging.info(f"start parsing {split}")

        pbar = tqdm(total=totallen)
        filename = str(args.output_dir / f"{split}.txt")
        index = count_lines_linux_style(filename)
        pbar.update(index)
        stop_at = len(ds) if args.max_docs is None else min(index + args.max_docs, len(ds))
        with open(filename, "a+") as output:
            Buffer = batch_buffer(output, pbar)
            while index < stop_at:
                document = ds[index]
                text = document['text']
                split_sents = process_text(text, max_len=256)
                Buffer.append_batch(split_sents)
                # for sent in split_sents:
                #     doc = nlp(sent)
                #     sents = list(doc.sents)
                #     print(sents)
                #     for parsed in sents:
                #         parse_string = parsed._.parse_string
                #         if parsed.text == 'Ċ':
                #             parse_string = '(Ċ Ċ)'
                #         if not is_first:
                #             parse_string = " " + parse_string 
                #         print(parse_string)
                index += 1
            Buffer.parse_batch()
        index = 0
                
    logging.info("finished")
