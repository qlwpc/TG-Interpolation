"""Tune row grouping without changing edges or reduction semantics."""
import hashlib,json,os,random,statistics
from pathlib import Path
import numpy as np
import torch
import triton
from olmo.attention_kernels.tgnomask import build_tgnomask_layout
from olmo.data.tg_mask import SentencepieceVocab
import olmo.attention_kernels.kernels as t
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
vocab=SentencepieceVocab.from_vocab_file(str(ROOT/'dataset/bbc-news/TG_GPT2_tokenizer.json'))
data=np.load(ROOT/'dataset/bbc-news/tg/test.npy',mmap_mode='r')
groups=json.loads((HERE/'benchmark_4451.json').read_text())['groups'][:3]
result={'job':os.environ['SLURM_JOB_ID'],'groups':[],
        'sha256':hashlib.sha256((ROOT/'olmo/attention_kernels/kernels.py').read_bytes()).hexdigest()}
path=HERE/f"compose_tuning_{result['job']}.json"
torch.set_num_threads(2)
for gi,group in enumerate(groups):
    tokens=torch.stack([torch.from_numpy(np.array(data[c*2048:(c+1)*2048],dtype=np.int64)) for c in group['chunk_indices']])
    layout=build_tgnomask_layout(tokens,vocab,device='cuda')
    torch.manual_seed(6198+gi)
    q,k,v=[torch.randn(4,2048,12,64,device='cuda',dtype=torch.bfloat16).transpose(1,2) for _ in range(3)]
    params=t._parameters(q);strides=t._strides(q,k,v)
    out,do,dq,pdk,pdv,dsk,dsv,dk,dv=t._empty_qkv(q,9)
    for x in (out,dq,pdk,pdv,dsk,dsv,dk,dv):x.zero_()
    do.normal_()
    lse,delta=[torch.zeros((4,12,2048),device='cuda',dtype=torch.float32) for _ in range(2)]
    def run(kind,cfg):
        br,be,nw=cfg
        if kind=='merge':
            t._merge_dkv[(triton.cdiv(2048,br),48)](q,k,v,layout.reverse,layout.offsets,do,lse,delta,
                pdk,pdv,dsk,dsv,layout.q_inverse,layout.k_inverse,dk,dv,**strides,**params,BM=br,num_warps=nw)
        else:
            t._sparse_rows[(triton.cdiv(layout.sparse_capacity,br),48)](q,k,v,layout.sparse_index,
                layout.sparse_count,layout.offsets,layout.edges,out,lse,do,delta,dq,
                **strides,**params,BR=br,BE=be,BACKWARD=kind=='dq',num_warps=nw)
    run('fwd',(4,8,4));delta.copy_((out.float()*do.float()).sum(-1))
    row={'group':gi,'results':{}}
    for kind in ('fwd','dq','merge'):
        cfgs=[(br,be,4) for br in (4,8,16,32) for be in (4,8,16)] if kind!='merge' else [(br,0,4) for br in (4,8,16,32,64)]
        run(kind,(4,8,4) if kind!='merge' else (8,0,4))
        outputs={'fwd':[out],'dq':[dq],'merge':[dk,dv]}[kind]
        refs=[x.clone() for x in outputs]
        records=[]
        # Compile and validate all candidates before timing; randomize rounds.
        for cfg in cfgs:
            rec={'config':cfg}
            try:
                run(kind,cfg)
                errors=[float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12)) for a,r in zip(outputs,refs)]
                assert all(x<.012 for x in errors),errors
                rec['relative_l2']=errors;rec['samples']=[]
            except Exception as e:rec['error']=repr(e)
            records.append(rec)
        valid=[r for r in records if 'samples' in r]
        for rec in valid:
            for _ in range(5):run(kind,rec['config'])
        rng=random.Random(901+gi)
        for _ in range(5):
            rng.shuffle(valid)
            for rec in valid:
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(30):run(kind,rec['config'])
                b.record();b.synchronize();rec['samples'].append(a.elapsed_time(b)/30)
        for rec in valid:rec['ms']=statistics.mean(rec['samples'])
        row['results'][kind]=records
        print(json.dumps({'group':gi,'kind':kind,'best':min(valid,key=lambda r:r['ms'])}),flush=True)
    result['groups'].append(row);path.write_text(json.dumps(result,indent=2)+'\n')
result['best']={}
for kind in ('fwd','dq','merge'):
    scores=[]
    for rec in result['groups'][0]['results'][kind]:
        cfg=rec['config'];rs=[next(r for r in g['results'][kind] if r['config']==cfg) for g in result['groups']]
        if all('ms' in r for r in rs):scores.append({'config':cfg,'ms':statistics.mean(r['ms'] for r in rs)})
    result['best'][kind]=sorted(scores,key=lambda r:r['ms'])
result['status']='passed';path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['best'],indent=2),flush=True)
