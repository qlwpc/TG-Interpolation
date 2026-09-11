"""Validate the published base arrays, then scan actual train once for both splits."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer

from datatools.reserved_clean.common import atomic_json, normalize, records, sha_file
from datatools.reserved_clean.scan_arrays import documents
from diagnostics.audit_bbc_test_train_overlap import make_targets, scan
from datatools.parse_test_docppl_data.generate_native_topk import CanonicalPPLCorpus, audit_alignment


def run(args):
    root=args.dataset;manifest=json.loads((root/'manifest.json').read_text())
    for name,value in manifest['files'].items():
        if sha_file(root/name)!=value: raise ValueError(f'published file changed: {name}')
    audit=root/'audit';audit.mkdir(exist_ok=True)
    tokenizer=Tokenizer.from_file(str(args.tokenizer));chunks=[];target_labels=[];seen=set();results={}
    for split in ('dev','test'):
        selected=list(records(root/(split+'_selected.jsonl.gz')))
        arrays={fmt:np.load(root/fmt/(split+'.npy'),mmap_mode='r') for fmt in ('terminal','tree','tg')}
        offsets={fmt:np.load(root/fmt/(split+'_doc_offsets.npy')) for fmt in arrays}
        for fmt,a in arrays.items():
            actual=list(documents(a))
            if len(actual)!=len(selected):raise ValueError(f'{split}/{fmt}: document count mismatch')
            if [off for off,_ in actual]+[len(a)]!=offsets[fmt].tolist():
                raise ValueError(f'{split}/{fmt}: document offset mismatch')
        for d,r in enumerate(selected):
            if r['group_id'] in seen:raise ValueError('content group crosses selected splits')
            seen.add(r['group_id'])
            parts={fmt:np.asarray(a[int(offsets[fmt][d]):int(offsets[fmt][d+1])]) for fmt,a in arrays.items()}
            for fmt in ('tree','tg'):
                a=parts[fmt];projected=a[(a<50268)|(a>50319)]
                if not np.array_equal(projected,parts['terminal']):raise ValueError('format terminal mismatch')
            text=tokenizer.decode(parts['terminal'][1:-1].tolist(),skip_special_tokens=False)
            if normalize(text)!=r['body']:raise ValueError('scoring text differs from audited source body')
            target_labels.append({'combined_doc':len(target_labels),'split':split,'doc_id':d,
                                  'source':[r['source_shard'],r['source_row']],'group_id':r['group_id']})
        chunks.append(np.asarray(arrays['terminal']))
        native_output=root/'native_model_topk_300_v2'/split
        audit_alignment(SimpleNamespace(ppl_dir=root/'canonical'/split,tokenizer=args.tokenizer,
            test_tree=root/'tree'/(split+'.npy'),output=native_output))
        corpus=CanonicalPPLCorpus(root/'canonical'/split,args.tokenizer)
        # Exercise the real parser-word/BPE adapter on every sentence before GPU work.
        max_words=0
        for s in range(len(corpus.lengths)):
            sentence=corpus.sentence_input(s)
            max_words=max(max_words,len(sentence.words))
        results[split]={'documents':len(selected),'sentences':len(corpus.lengths),'max_parser_words':max_words,
                        'all_sentence_adapters_verified':True}
    targets_path=audit/'dev_test_targets.npy';np.save(targets_path,np.concatenate(chunks))
    result=scan(make_targets(targets_path),args.train)
    # The historical diagnostic has four legacy article annotations; those
    # fields have no meaning for this new corpus and are never published here.
    for key in ('other_test_documents','other_matched_test_documents','known_four_copies'):
        result.pop(key,None)
    atomic_json(audit/'final_train_scan.json',result)
    atomic_json(audit/'target_labels.json',target_labels)
    if not result['scan_complete'] or result['matched_test_documents'] or result['train_documents_without_final_eos']:
        raise ValueError('final arrays did not pass the complete actual-train exact audit')
    if result['train_file_sha256']!=args.expected_train_sha256:
        raise ValueError('actual train differs from declared frozen training scope')
    atomic_json(audit/'validation.json',{'complete':True,'status':'base_arrays_validated',
        'manifest_sha256':sha_file(root/'manifest.json'),'train_sha256':result['train_file_sha256'],
        'strict_final_train_matches':0,'splits':results,
        'protocol_limit':'Declared normalization/5-gram near/13-gram local rules; no claim of zero semantic overlap.',
        'topk_status':'pending','outputs':{name:sha_file(audit/name) for name in
            ('final_train_scan.json','target_labels.json','dev_test_targets.npy')}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,default=Path('dataset/bbc-news/TG_GPT2_tokenizer.json'))
    p.add_argument('--train',type=Path,default=Path('dataset/bbc-news/terminal/train.npy'))
    p.add_argument('--expected-train-sha256',required=True)
    run(p.parse_args())
