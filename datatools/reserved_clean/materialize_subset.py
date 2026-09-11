"""Recreate canonical inputs from existing remote candidates and ID-only selection.

This allows GPU construction beside the source corpus without transferring
local dev/test reference text. The candidate receipt hash must match the
locally audited candidate pool.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from datatools.reserved_clean.build import encode_document
from datatools.reserved_clean.common import atomic_json, emit, records, sha_file
from datatools.reserved_clean.filter import validate_receipt
from datatools.parse_test_docppl_data.reproduce_bbc_test import legacy_tokenizer


def run(args):
    validate_receipt(args.candidates)
    selection=json.loads(args.selection.read_text())
    if selection['candidate_receipt_sha256']!=sha_file(args.candidates/'complete.json'):
        raise ValueError('selection was derived from another candidate pool')
    ids=selection['candidate_ids']
    if not ids or len(ids)!=len(set(ids)) or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('invalid selection IDs')
    wanted=set(ids)
    rows={r['candidate_id']:r for r in records(args.candidates/'candidates.jsonl.gz') if r['candidate_id'] in wanted}
    if set(rows)!=wanted:raise ValueError('missing selected candidates')
    args.output.mkdir(parents=True,exist_ok=False)
    tok=legacy_tokenizer(Tokenizer.from_file(str(args.tokenizer)))
    parts=[];offsets=[0];counts=[]
    with gzip.open(args.output/'selected.jsonl.gz','wt',encoding='utf-8') as f:
        for i in ids:
            arrays,bounds,repairs=encode_document(rows[i]['parsed'],tok)
            parts.append(arrays['tree']);base=offsets[-1]
            offsets.extend(base+int(v) for v in bounds[1:]);counts.append(len(bounds)-1)
            emit(f,{'candidate_id':i,'source':[rows[i]['source_shard'],rows[i]['source_row']],
                    'strict_sha256':rows[i]['strict_sha256'],'repairs':repairs})
    np.save(args.output/'tree.npy',np.concatenate(parts))
    np.save(args.output/'sentence_offsets.npy',np.asarray(offsets,dtype=np.uint64))
    np.save(args.output/'document_sentence_counts.npy',np.asarray(counts,dtype=np.uint32))
    atomic_json(args.output/'manifest.json',{'format':'canonical-tree-sentences-v1','tree_path':'tree.npy',
        'document_count':len(ids),'sentence_count':sum(counts),'selection_sha256':sha_file(args.selection),
        'candidate_receipt_sha256':selection['candidate_receipt_sha256'],
        'tree_sha256':sha_file(args.output/'tree.npy'),
        'sentence_offsets_sha256':sha_file(args.output/'sentence_offsets.npy'),
        'document_sentence_counts_sha256':sha_file(args.output/'document_sentence_counts.npy')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
