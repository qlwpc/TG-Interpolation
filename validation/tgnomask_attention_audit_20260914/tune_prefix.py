"""Bounded offline prefix tile sweep; records every candidate and parity."""
import hashlib,json,os,statistics
from pathlib import Path
import numpy as np
import torch
import triton
from olmo.attention_kernels.tgnomask import build_tgnomask_layout
from olmo.data.tg_mask import SentencepieceVocab
import olmo.attention_kernels.kernels as t
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
vocab=SentencepieceVocab.from_vocab_file(str(ROOT/'dataset/bbc-news/TG_GPT2_tokenizer.json'))
data=np.load(ROOT/'dataset/bbc-news/tg/test.npy',mmap_mode='r')
groups=json.loads((HERE/'benchmark_4451.json').read_text())['groups'][:3]
candidates=[(bm,bn,4,ns) for bm,bn in ((32,32),(32,64),(32,128),(64,32),(64,64),(64,128),(128,32),(128,64)) for ns in (2,3)]
result={'job':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),'groups':[],
        'sha256':hashlib.sha256((ROOT/'olmo/attention_kernels/kernels.py').read_bytes()).hexdigest()}
path=HERE/f"prefix_tuning_{result['job']}.json"
torch.set_num_threads(2)
for gi,group in enumerate(groups):
    tokens=torch.stack([torch.from_numpy(np.array(data[c*2048:(c+1)*2048],dtype=np.int64)) for c in group['chunk_indices']])
    layout=build_tgnomask_layout(tokens,vocab,device='cuda')
    torch.manual_seed(6198+gi)
    q,k,v=[torch.randn(4,2048,12,64,device='cuda',dtype=torch.bfloat16).transpose(1,2) for _ in range(3)]
    params=t._parameters(q); moves={k:v for k,v in params.items() if k!='SCALE'}
    strides=t._strides(q,k,v); kvstrides={k:v for k,v in strides.items() if not k.startswith('Q')}
    pq,pk,pv,out,cdo,pdo,dq,dsk,dsv,dk,dv=t._empty_qkv(q,11)
    plse,slse,pd,sd=[torch.empty((4,12,2048),device='cuda',dtype=torch.float32) for _ in range(4)]
    t._pack_qkv[(64,48)](q,k,v,layout.q_inverse,layout.k_inverse,pq,pk,pv,**strides,**moves,BM=32,num_warps=4)
    do=torch.randn_like(q)
    def run(kind,config):
        bm,bn,nw,ns=config
        extra=dict(**params,BM=bm,BN=bn,num_warps=nw,num_stages=ns)
        if kind=='fwd':
            t._prefix_fwd[(triton.cdiv(2048,bm),48)](pq,pk,pv,k,v,layout.q_index,layout.prefix,layout.self_mask,
                layout.q_count,layout.k_count,out,plse,**kvstrides,**extra)
        elif kind=='dq':
            t._prefix_dq[(triton.cdiv(2048,bm),48)](pq,pk,pv,k,v,layout.q_index,layout.prefix,layout.self_mask,
                layout.q_count,layout.k_count,pdo,plse,pd,dq,dsk,dsv,**kvstrides,**extra)
        else:
            t._prefix_dkv[(triton.cdiv(2048,bn),48)](pq,pk,pv,layout.prefix,layout.self_mask,layout.q_count,
                layout.k_count,layout.q_start,pdo,plse,pd,dk,dv,**extra)
    row={'group':gi,'results':{}}
    for kind,default in zip(('fwd','dq','dkv'),t._PREFIX_CONFIGS):
        # Only ordinary output rows are used for prefix delta. Initialize other
        # rows before preparing so this diagnostic never reads uninitialized data.
        out.zero_()
        run('fwd',t._PREFIX_CONFIGS[0])
        t._prepare_backward[(64,48)](do,out,layout.q_inverse,cdo,pdo,pd,sd,
            DSB=do.stride(0),DSH=do.stride(1),DSN=do.stride(2),DSD=do.stride(3),**moves,BM=32,num_warps=4)
        outputs={'fwd':[out],'dq':[dq,dsk,dsv],'dkv':[dk,dv]}[kind]
        for a in outputs:a.zero_()
        run(kind,default)
        refs=[x.clone() for x in outputs]
        records=[]
        for cfg in candidates:
            rec={'config':cfg}
            try:
                for a in outputs:a.zero_()
                run(kind,cfg)
                errors=[float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12)) for a,r in zip(outputs,refs)]
                assert all(x<.012 for x in errors),errors
                rec['relative_l2']=errors
                for _ in range(4):run(kind,cfg)
                samples=[]
                for _ in range(3):
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(20):run(kind,cfg)
                    b.record();b.synchronize();samples.append(a.elapsed_time(b)/20)
                rec['ms']=statistics.mean(samples)
            except Exception as e:rec['error']=repr(e)
            records.append(rec)
        row['results'][kind]=records
        print(json.dumps({'group':gi,'kind':kind,'best':min((r for r in records if 'ms' in r),key=lambda r:r['ms'])}),flush=True)
    result['groups'].append(row);path.write_text(json.dumps(result,indent=2)+'\n')
result['best']={}
for kind in ('fwd','dq','dkv'):
    scores=[]
    for cfg in candidates:
        records=[next(r for r in g['results'][kind] if tuple(r['config'])==cfg) for g in result['groups']]
        if all('ms' in r for r in records):scores.append({'config':cfg,'ms':statistics.mean(r['ms'] for r in records)})
    result['best'][kind]=sorted(scores,key=lambda r:r['ms'])
result['status']='passed';path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['best'],indent=2),flush=True)
