"""Freeze group-disjoint splits and aligned scoring arrays after audit completion."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from nltk import Tree
from tokenizers import Tokenizer

from datatools.reserved_clean.common import RESERVED, atomic_json, body_from_parse, emit, normalize, records, sha_file
from datatools.reserved_clean.filter import validate_receipt
from datatools.parse_test_docppl_data.reproduce_bbc_test import legacy_tokenizer


def encode_document(parsed, tokenizer):
    root = Tree.fromstring('(ROOT '+parsed.strip()+')')
    vocab = tokenizer.get_vocab()
    repairs = Counter()
    def render(node):
        if not isinstance(node,Tree):
            raise ValueError('non-tree child')
        if not node:
            repairs['empty_node_removed'] += 1
            return ''
        if isinstance(node[0],str):
            if len(node)!=1:
                raise ValueError('multiple leaves in preterminal')
            return ' '+ ('\n' if node[0]=='Ċ' else node[0])
        label=node.label()
        if label=='ADJ':
            label='ADJP'; repairs['ADJ_to_ADJP'] += 1
        a,b=f'<({label}>',f'<{label})>'
        if a not in vocab or b not in vocab:
            raise ValueError(f'unknown constituent label {label}')
        return a+''.join(render(c) for c in node)+b
    text=''.join(render(c) for c in root)
    tree=np.asarray([50257,*tokenizer.encode(text).ids,50256],dtype='<u2')
    opens={v for k,v in vocab.items() if k.startswith('<(') and k.endswith('>')}
    closes={v for k,v in vocab.items() if k.startswith('<') and k.endswith(')>')}
    op=np.isin(tree,list(opens)); cl=np.isin(tree,list(closes)); nt=op|cl
    terminal=tree[~nt]
    if np.count_nonzero(terminal==50257)!=1 or np.count_nonzero(terminal==50256)!=1:
        raise ValueError('embedded document boundary token')
    if np.any(np.isin(terminal,[50258,50259,50260,50261])):
        raise ValueError('reserved special token in scoring text')
    decoded=tokenizer.decode(terminal[1:-1].tolist(),skip_special_tokens=False)
    if normalize(decoded)!=body_from_parse(parsed):
        raise ValueError('encoded terminal changes normalized leaf text')
    depth=np.cumsum(op.astype(int)-cl.astype(int))
    if depth.min()<0 or depth[-1]!=0:
        raise ValueError('unbalanced encoded tree')
    roots=np.flatnonzero(op & (depth==1))
    if not len(roots):
        raise ValueError('document has no constituent tree')
    bounds=np.r_[0,roots[1:],len(tree)].astype(np.uint64)
    # A sentence record includes its adjacent separators/BOS/EOS exactly once.
    for a,b in zip(bounds[:-1],bounds[1:]):
        if not np.any(~nt[int(a):int(b)] & (tree[int(a):int(b)]<50256)):
            raise ValueError('sentence contains no scored text')
    return {'terminal':terminal,'tree':tree,'tg':np.repeat(tree,1+cl.astype(int))}, bounds, dict(repairs)


class UnionFind:
    def __init__(self,n): self.parent=list(range(n))
    def find(self,a):
        while a!=self.parent[a]:
            self.parent[a]=self.parent[self.parent[a]]; a=self.parent[a]
        return a
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b: self.parent[max(a,b)]=min(a,b)


def allocate(counts,total):
    n=sum(counts.values())
    if not 0<=total<=n: raise ValueError('invalid allocation size')
    out={k:total*v//n for k,v in counts.items()} if n else {k:0 for k in counts}
    order=sorted(counts,key=lambda k:(-(total*counts[k]%n),k)) if n else []
    for k in order[:total-sum(out.values())]: out[k]+=1
    return out


def stratified_partition(items,seed=20260906):
    bins=defaultdict(list)
    for item in items:
        tokens=item['terminal_tokens']
        length=0 if tokens<=256 else 1 if tokens<=512 else 2 if tokens<=1024 else 3
        bins[(item['source_shard'],length)].append(item)
    rng=random.Random(seed)
    for k in sorted(bins):
        bins[k].sort(key=lambda x:x['group_id']); rng.shuffle(bins[k])
    n=len(items)
    nd,nt=(1000,5000) if n>=6000 else (round(n/6),n-round(n/6))
    dev=allocate({k:len(v) for k,v in bins.items()},nd)
    test=allocate({k:len(v)-dev[k] for k,v in bins.items()},nt)
    out={'dev':[],'test':[],'reserve':[]}
    for k in sorted(bins):
        a,b=dev[k],dev[k]+test[k]
        out['dev']+=bins[k][:a]; out['test']+=bins[k][a:b]; out['reserve']+=bins[k][b:]
    for split,rows in out.items():
        rows.sort(key=lambda r:(RESERVED.index(r['source_shard']),r['source_row']))
    return out


def read_tsv(path):
    with path.open() as f: return list(csv.DictReader(f,delimiter='\t'))


def local_references(args):
    from datatools.reserved_clean.scan_arrays import documents
    args.output.mkdir(parents=True,exist_ok=False)
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    sources=[]
    with gzip.open(args.output/'bodies.jsonl.gz','wt',compresslevel=1,encoding='utf-8') as f:
        for path in args.inputs:
            before=sha_file(path); count=0
            for d,(offset,part) in enumerate(documents(np.load(path,mmap_mode='r'))):
                body=normalize(tokenizer.decode(part[1:-1].tolist(),skip_special_tokens=False))
                emit(f,{'source':[str(path.resolve()),d,offset],'body':body,
                        'body_sha256':hashlib.sha256(body.encode()).hexdigest(),
                        'strict_sha256':hashlib.sha256(part.tobytes()).hexdigest()})
                count=d+1
            if before!=sha_file(path): raise ValueError('local reference changed')
            sources.append({'path':str(path.resolve()),'sha256':before,'documents':count})
    atomic_json(args.output/'complete.json',{'complete':True,'sources':sources,
        'outputs':{'bodies.jsonl.gz':sha_file(args.output/'bodies.jsonl.gz')}})


def build(args):
    candidate_receipt=validate_receipt(args.candidates)
    match_receipts=[validate_receipt(p) for p in args.matches]
    if len(args.matches)<2: raise ValueError('both remote training and local dev/test audits are required')
    candidate_hash=sha_file(args.candidates/'complete.json')
    if any(r.get('candidate_receipt_sha256')!=candidate_hash for r in match_receipts):
        raise ValueError('matching results belong to a different candidate pool')
    array_receipt_path=args.array_matches.parent/'complete.json'
    array_receipt=json.loads(array_receipt_path.read_text())
    if (not array_receipt.get('complete') or
            array_receipt['outputs'].get(args.array_matches.name)!=sha_file(args.array_matches)):
        raise ValueError('actual-array match list does not match its completion receipt')
    if not any(r.get('prerequisite_receipts',{}).get('arrays')==sha_file(array_receipt_path) for r in match_receipts):
        raise ValueError('no matching receipt is bound to the actual training scan')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=list(records(args.candidates/'candidates.jsonl.gz')); n=len(rows)
    if n!=candidate_receipt['candidates']: raise ValueError('candidate count mismatch')
    if [r['candidate_id'] for r in rows]!=list(range(n)): raise ValueError('nonsequential candidate records')
    uf=UnionFind(n); reasons=defaultdict(set); cover=[0]*n
    for folder in args.matches:
        for edge in read_tsv(folder/'internal_edges.tsv'):
            uf.union(int(edge['candidate']),int(edge['source']))
        external=read_tsv(folder/'external.tsv')
        if len(external)!=n or [int(r['candidate']) for r in external]!=list(range(n)):
            raise ValueError('external results do not cover ordered candidates')
        for d,r in enumerate(external):
            if int(r['exact']): reasons[d].add('normalized_reference_exact')
            if int(r['near']): reasons[d].add('reference_near_duplicate')
            if len(r['covered_bitmap'])!=len(rows[d]['body'].split()): raise ValueError('coverage length mismatch')
            cover[d] |= int(r['covered_bitmap'] or '0',2)
    for d in range(n):
        words=len(rows[d]['body'].split())
        if words<10: reasons[d].add('too_short_for_overlap_audit')
        if words and cover[d].bit_count()/words>=0.3: reasons[d].add('reference_local_coverage_ge_30pct')
    # Direct actual-train token matches are additional evidence, independent of
    # decoder normalization and raw-to-train reconstruction.
    strict_hits={r['strict_sha256'] for r in records(args.array_matches)}
    local_hits={r['strict_sha256'] for r in records(args.local_references)}
    for d,r in enumerate(rows):
        if r['strict_sha256'] in strict_hits|local_hits: reasons[d].add('actual_array_exact')
    groups=defaultdict(list)
    for d in range(n): groups[uf.find(d)].append(d)
    tokenizer=legacy_tokenizer(Tokenizer.from_file(str(args.tokenizer)))
    encoded,eligible={},[]
    with (args.output/'groups.jsonl').open('w') as groupfile:
        for representative,members in sorted(groups.items()):
            inherited=set().union(*(reasons[d] for d in members))
            if len(members)>100: inherited.add('large_content_group_requires_review')
            if not inherited:
                try:
                    arrays,bounds,repairs=encode_document(rows[representative]['parsed'],tokenizer)
                    encoded[representative]=(arrays,bounds,repairs)
                except (ValueError,RecursionError) as exc:
                    inherited.add('format_quality: '+str(exc))
            group_id=rows[representative]['strict_sha256']
            for d in members:
                rows[d]['group_id']=group_id
                reasons[d].update(inherited)
                if d!=representative: reasons[d].add('content_group_nonrepresentative')
            emit(groupfile,{'group_id':group_id,'representative':representative,'members':members,
                            'excluded_reasons':sorted(inherited)})
            if not inherited:
                r=rows[representative]
                r['terminal_tokens']=len(encoded[representative][0]['terminal'])
                eligible.append(r)
    splits=stratified_partition(eligible,args.seed)
    assignment={r['candidate_id']:split for split,rs in splits.items() for r in rs}
    with (args.output/'candidates.jsonl').open('w') as f, (args.output/'exclusions.jsonl').open('w') as exclusions, \
            (args.output/'quarantine.jsonl').open('w') as quarantine:
        for d,r in enumerate(rows):
            meta={k:v for k,v in r.items() if k not in ('parsed','terminal_b64','body')}
            meta.update(split=assignment.get(d),exclusion_reasons=sorted(reasons[d]),
                        matched_word_coverage=cover[d].bit_count()/max(1,len(r['body'].split())))
            emit(f,meta)
            if reasons[d]: emit(exclusions,meta)
            if any(reason.startswith(('format_quality','too_short','large_content','reference_local')) for reason in reasons[d]):
                emit(quarantine,meta)
    shutil.copyfile(args.candidates/'raw_exclusions.jsonl',args.output/'raw_exclusions.jsonl')
    outputs={}
    for split,selection in splits.items():
        index=defaultdict(list)
        for r in selection: index[r['source_shard']].append(r['source_row'])
        atomic_json(args.output/(split+'_index.json'),dict(index))
        with gzip.open(args.output/(split+'_selected.jsonl.gz'),'wt',compresslevel=1,encoding='utf-8') as f:
            for doc_id,r in enumerate(selection): emit(f,{**r,'doc_id':doc_id})
        if split=='reserve' or not selection: continue
        arrays_by_format={fmt:[] for fmt in ('terminal','tree','tg')}
        doc_offsets={fmt:[0] for fmt in arrays_by_format}
        sent_offsets=[0]; doc_sent=[0]; repair_counts=Counter()
        for r in selection:
            arrays,bounds,repairs=encoded[r['candidate_id']]; repair_counts.update(repairs)
            base=doc_offsets['tree'][-1]
            for fmt in arrays:
                arrays_by_format[fmt].append(arrays[fmt]);doc_offsets[fmt].append(doc_offsets[fmt][-1]+len(arrays[fmt]))
            sent_offsets.extend(int(b)+base for b in bounds[1:]);doc_sent.append(len(sent_offsets)-1)
        for fmt,parts in arrays_by_format.items():
            folder=args.output/fmt;folder.mkdir(exist_ok=True)
            np.save(folder/(split+'.npy'),np.concatenate(parts))
            np.save(folder/(split+'_doc_offsets.npy'),np.asarray(doc_offsets[fmt],dtype=np.uint64))
        canonical=args.output/'canonical'/split;canonical.mkdir(parents=True)
        np.save(canonical/'sentence_offsets.npy',np.asarray(sent_offsets,dtype=np.uint64))
        np.save(canonical/'document_sentence_counts.npy',np.diff(doc_sent).astype(np.uint32))
        # Path is relative to this manifest; canonical input holds ONE archived
        # tree per sentence, never pretends to be a 300-candidate proposal.
        atomic_json(canonical/'manifest.json',{'format':'canonical-tree-sentences-v1',
            'tree_path':f'../../tree/{split}.npy','document_count':len(selection),'sentence_count':doc_sent[-1],
            'tree_sha256':sha_file(args.output/'tree'/(split+'.npy')),
            'sentence_offsets_sha256':sha_file(canonical/'sentence_offsets.npy'),
            'document_sentence_counts_sha256':sha_file(canonical/'document_sentence_counts.npy')})
        outputs[split]={'documents':len(selection),'sentences':doc_sent[-1],
                        'tokens':{fmt:offset[-1] for fmt,offset in doc_offsets.items()},'repairs':dict(repair_counts)}
    # Smoke selection is drawn ONLY from dev, before any model scores exist.
    smoke=sorted(splits['dev'],key=lambda r:hashlib.sha256(f"{args.seed}:{r['group_id']}".encode()).digest())[:100]
    atomic_json(args.output/'smoke_dev_candidate_ids.json',[r['candidate_id'] for r in smoke])
    costly_test=sorted(splits['test'],key=lambda r:hashlib.sha256(
        f"{args.seed}:docppl:{r['group_id']}".encode()).digest())[:1000]
    atomic_json(args.output/'test_docppl_1000_candidate_ids.json',[r['candidate_id'] for r in costly_test])
    counts=Counter(reason for rs in reasons.values() for reason in rs)
    manifest={'schema_version':1,'status':'base_arrays_complete_topk_pending','claim_type':'computed',
        'name':'bbc_reserved_clean_v1','seed':args.seed,'strict_candidates':n,'content_groups':len(groups),
        'eligible_groups':len(eligible),'split_documents':{k:len(v) for k,v in splits.items()},
        'fixed_subsets':{'dev_smoke_documents':len(smoke),'test_docppl_documents':len(costly_test)},
        'exclusion_reason_counts_nonexclusive':dict(counts),'outputs':outputs,
        'training_scope':'Frozen historical BBC terminal/train.npy; other checkpoint-specific corpora not certified.',
        'candidate_receipt_sha256':sha_file(args.candidates/'complete.json'),
        'matching_receipts_sha256':[sha_file(p/'complete.json') for p in args.matches],
        'array_matches_sha256':sha_file(args.array_matches),'local_references_sha256':sha_file(args.local_references),
        'array_receipt_sha256':sha_file(array_receipt_path),
        'tokenizer_sha256':sha_file(args.tokenizer),'effective_tokenizer_sha256':hashlib.sha256(tokenizer.to_str().encode()).hexdigest(),
        'builder_sha256':sha_file(__file__),'evaluation_run':False,
        'files':{str(p.relative_to(args.output)):sha_file(p) for p in args.output.rglob('*') if p.is_file()}}
    atomic_json(args.output/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    l=sub.add_parser('local-references');l.add_argument('--inputs',type=Path,nargs='+',required=True)
    b=sub.add_parser('build');b.add_argument('--candidates',type=Path,required=True)
    b.add_argument('--matches',type=Path,nargs='+',required=True)
    b.add_argument('--array-matches',type=Path,required=True);b.add_argument('--local-references',type=Path,required=True)
    b.add_argument('--seed',type=int,default=20260906)
    for s in (l,b):
        s.add_argument('--tokenizer',type=Path,default=Path('dataset/bbc-news/TG_GPT2_tokenizer.json'))
        s.add_argument('--output',type=Path,required=True)
    args=p.parse_args();local_references(args) if args.command=='local-references' else build(args)
