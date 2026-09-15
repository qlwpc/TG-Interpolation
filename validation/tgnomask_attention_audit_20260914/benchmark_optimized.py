"""Same 10 groups as benchmark 4451; typed timing includes all packing/backward."""
import contextlib
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from olmo.data.tg_mask import SentencepieceVocab, KProximal_TG_attention_bias
from olmo.attention_kernels.tgnomask import build_tgnomask_layout, tgnomask_attention

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def stats(xs):
    return {"mean":statistics.mean(xs),"std":statistics.stdev(xs) if len(xs)>1 else 0.,"samples":xs}


def main():
    torch.set_num_threads(2)
    timing_seed=int(os.environ.get("TIMING_SEED", "6198"))
    vocab_path = str(ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json")
    vocab = SentencepieceVocab.from_vocab_file(vocab_path)
    data = np.load(ROOT / "dataset/bbc-news/tg/test.npy", mmap_mode="r")
    groups = json.loads((HERE / "benchmark_4451.json").read_text())["groups"]
    snapshot = HERE / "fused_snapshot_kernels.py"
    spec = importlib.util.spec_from_file_location("_tgnomask_fused_snapshot", snapshot)
    v1 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v1
    spec.loader.exec_module(v1)
    path=HERE/f"optimized_benchmark_{os.environ.get('SLURM_JOB_ID')}.json"
    result={"status":"running","groups":[],"torch":torch.__version__,"timing_seed":timing_seed,
            "gpu":torch.cuda.get_device_name(),"job":os.environ.get("SLURM_JOB_ID"),
            "protocol":"B4 H12 N2048 D64 BF16; production layout; 10 groups from 4451; 5 rounds, 20 eval / 10 train iterations; randomized order; 5 warmups. Compare exact 4491 fused snapshot and three optimizations; typed times include all layout work and backward.",
            "fused_snapshot_sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in (
                "olmo/attention_kernels/tgnomask.py","olmo/attention_kernels/kernels.py","olmo/model.py")}}
    def save():path.write_text(json.dumps(result,indent=2)+"\n")
    compiled=torch.compile(flex_attention,dynamic=False)
    methods=("causal_flash","tgnomask_sdpa","tgnomask_flex","tgnomask_fused_baseline","tgnomask_typed")
    try:
        for index,old in enumerate(groups):
            tokens=torch.stack([torch.from_numpy(np.array(data[c*2048:(c+1)*2048],dtype=np.int64)) for c in old["chunk_indices"]])
            began=time.perf_counter()
            cpu_layout=build_tgnomask_layout(tokens,vocab)
            cpu_layout_ms=(time.perf_counter()-began)*1000
            began=time.perf_counter();layout=cpu_layout.to("cuda");torch.cuda.synchronize()
            layout_h2d_ms=(time.perf_counter()-began)*1000
            # Restore chronological sparse scheduling for the exact 4491 baseline.
            old_si=cpu_layout.sparse_index.clone()
            for batch,count in enumerate(cpu_layout.sparse_count.tolist()):
                old_si[batch,:count]=old_si[batch,:count].sort().values
            baseline_layout=replace(cpu_layout,sparse_index=old_si).to('cuda')
            masks=[]
            for t in tokens:
                gen=KProximal_TG_attention_bias(vocab_path,2048,2048,False)
                masks.append(gen(t)[0])
            dense=torch.stack(masks).cuda()
            assert torch.equal(layout.dense_mask()[:,0],dense)
            additive=torch.zeros((4,1,2048,2048),device="cuda",dtype=torch.bfloat16).masked_fill_(~dense[:,None],torch.finfo(torch.bfloat16).min)
            def mask_mod(b,h,q,k):return dense[b,q,k]
            def build():return create_block_mask(mask_mod,B=4,H=None,Q_LEN=2048,KV_LEN=2048,device="cuda")
            block=build();torch.cuda.synchronize();build_times=[]
            for _ in range(5):
                began=time.perf_counter();block=build();torch.cuda.synchronize();build_times.append((time.perf_counter()-began)*1000)
            torch.manual_seed(6198+index)
            q,k,v=[torch.randn(4,2048,12,64,device="cuda",dtype=torch.bfloat16).transpose(1,2).detach().requires_grad_() for _ in range(3)]
            upstream=torch.randn_like(q)
            def clear():q.grad=k.grad=v.grad=None
            def forward(method):
                if method=="causal_flash":return F.scaled_dot_product_attention(q,k,v,is_causal=True)
                if method=="tgnomask_sdpa":return F.scaled_dot_product_attention(q,k,v,attn_mask=additive)
                if method=="tgnomask_flex":return compiled(q,k,v,block_mask=block)
                if method=="tgnomask_fused_baseline":return v1.typed_attention(q,k,v,baseline_layout)
                return tgnomask_attention(q,k,v,layout)
            with torch.enable_grad():
                reference=forward("tgnomask_sdpa");reference.backward(upstream)
                ref_grads=[x.grad.detach().clone() for x in (q,k,v)];clear()
                actual=forward("tgnomask_typed");actual.backward(upstream)
                parity=[]
                for a,r in [(actual,reference)]+[(x.grad,g) for x,g in zip((q,k,v),ref_grads)]:
                    error=float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12))
                    assert a.isfinite().all() and error<.012,error
                    parity.append(error)
                clear();del reference,actual,ref_grads
            row={"index":index,"chunks":old["chunk_indices"],"cpu_layout_ms":cpu_layout_ms,
                 "layout_h2d_ms":layout_h2d_ms,"block_mask_ms":stats(build_times),"parity_relative_l2":parity,"workloads":{}}
            for workload,iterations in (("eval",20),("train",10)):
                with torch.set_grad_enabled(workload=="train"):
                    def step(method):
                        output=forward(method)
                        if workload=="train":output.backward(upstream);clear()
                    for method in methods:
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if method=="causal_flash" else contextlib.nullcontext():
                            for _ in range(5):step(method)
                    torch.cuda.synchronize()
                    samples={m:[] for m in methods};order_rng=random.Random(timing_seed+index+1000*(workload=="train"))
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    for _ in range(5):
                        order=list(methods);order_rng.shuffle(order)
                        for method in order:
                            with sdpa_kernel(SDPBackend.FLASH_ATTENTION) if method=="causal_flash" else contextlib.nullcontext():
                                start.record()
                                for _ in range(iterations):step(method)
                                end.record();end.synchronize()
                                samples[method].append(start.elapsed_time(end)/iterations)
                    row["workloads"][workload]={m:stats(xs) for m,xs in samples.items()}
                    if index == 0:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                            step("tgnomask_typed")
                            torch.cuda.synchronize()
                        result.setdefault("fused_profiles",{})[workload]={
                            "aten_ops":[e.key for e in prof.key_averages() if e.key.startswith("aten::")],
                            "cuda_total_us":sum(e.time_range.elapsed_us() for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA),
                            "cuda_kernels":sorted({e.name for e in prof.events()
                                if e.device_type==torch.autograd.DeviceType.CUDA})}
            result["groups"].append(row);save()
            print(json.dumps({"group":index,"times":{w:{m:s["mean"] for m,s in x.items()} for w,x in row["workloads"].items()}}),flush=True)
        result["aggregate"]={w:{m:stats([g["workloads"][w][m]["mean"] for g in result["groups"]]) for m in methods} for w in ("eval","train")}
        result["status"]="passed";print(json.dumps(result["aggregate"],indent=2),flush=True)
    except Exception:
        result["status"]="failed";result["error"]=traceback.format_exc();raise
    finally:save();print(path,flush=True)


if __name__=="__main__":main()
