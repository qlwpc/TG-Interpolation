"""Triton kernels for typed TGnomask. Imported lazily by olmo.attention_kernels.tgnomask."""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _prefix_fwd(Q, K, V, SK, SV, QI, Prefix, Self, QC, KC, O, LSE,
                KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
                VSB: tl.constexpr, VSH: tl.constexpr, VSN: tl.constexpr, VSD: tl.constexpr,
                N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                BD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch = bh // H
    count = tl.load(QC + batch)
    key_count = tl.load(KC + batch)
    if block * BM < count:
        i = block * BM + tl.arange(0, BM)
        d = tl.arange(0, BD)
        base = bh * N * D
        active = i < count
        q = tl.load(Q + base + i[:, None] * D + d[None, :], active[:, None] & (d[None, :] < D), 0)
        prefix = tl.load(Prefix + batch * N + i, active, 0)
        special = tl.load(Self + batch * N + i, active, 0)
        original_i = tl.load(QI + batch * N + i, active, 0)
        self_rows = active[:, None] & special[:, None] & (d[None, :] < D)
        sk = tl.load(SK + batch * KSB + (bh % H) * KSH + original_i[:, None] * KSN + d[None, :] * KSD, self_rows, 0)
        sv = tl.load(SV + batch * VSB + (bh % H) * VSH + original_i[:, None] * VSN + d[None, :] * VSD, self_rows, 0)
        self_score = tl.sum(q.to(tl.float32) * sk.to(tl.float32), 1) * (SCALE * 1.4426950408889634)
        m = tl.where(special, self_score, float("-inf"))
        m = tl.where(active, m, 0.)
        l = special.to(tl.float32)
        acc = tl.where(special[:, None], sv.to(tl.float32), 0.)
        end = tl.max(prefix, 0)
        full_end = tl.min(tl.where(active, prefix, 2147483647), 0)
        for stage in tl.static_range(2):
            lo = 0 if stage == 0 else full_end // BN * BN
            hi = full_end // BN * BN if stage == 0 else end
            for start in range(lo, hi, BN):
                j = start + tl.arange(0, BN)
                k = tl.load(K + base + j[:, None] * D + d[None, :], (j[:, None] < key_count) & (d[None, :] < D), 0)
                v = tl.load(V + base + j[:, None] * D + d[None, :], (j[:, None] < key_count) & (d[None, :] < D), 0)
                scores = tl.dot(q, tl.trans(k), input_precision="ieee") * (SCALE * 1.4426950408889634)
                if stage == 1:
                    scores = tl.where(j[None, :] < prefix[:, None], scores, float("-inf"))
                scores = tl.where(active[:, None], scores, float("-inf"))
                new_m = tl.maximum(m, tl.max(scores, 1))
                alpha = tl.exp2(m - new_m)
                p = tl.exp2(scores - new_m[:, None])
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
                l = l * alpha + tl.sum(p, 1)
                m = new_m
        out = acc / tl.where(l > 0., l, 1.)[:, None]
        tl.store(O + base + original_i[:, None] * D + d[None, :], out, active[:, None] & (d[None, :] < D))
        # Keep the public backward contract in natural-log units.
        tl.store(LSE + bh * N + i, m * 0.6931471805599453 + tl.log(l), active)


@triton.jit
def _prefix_dq(Q, K, V, SK, SV, QI, Prefix, Self, QC, KC, DO, LSE, Delta, DQ, DSK, DSV,
                KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
                VSB: tl.constexpr, VSH: tl.constexpr, VSN: tl.constexpr, VSD: tl.constexpr,
               N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
               BD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch = bh // H
    count = tl.load(QC + batch)
    key_count = tl.load(KC + batch)
    if block * BM < count:
        i = block * BM + tl.arange(0, BM)
        d = tl.arange(0, BD)
        base = bh * N * D
        active = i < count
        mask = active[:, None] & (d[None, :] < D)
        q = tl.load(Q + base + i[:, None] * D + d[None, :], mask, 0)
        do = tl.load(DO + base + i[:, None] * D + d[None, :], mask, 0)
        lse = tl.load(LSE + bh * N + i, active, 0)
        delta = tl.load(Delta + bh * N + i, active, 0)
        prefix = tl.load(Prefix + batch * N + i, active, 0)
        special = tl.load(Self + batch * N + i, active, 0)
        original_i = tl.load(QI + batch * N + i, active, 0)
        multiple = prefix + special.to(tl.int32) > 1
        dq = tl.full((BM, BD), 0., tl.float32)
        end = tl.max(prefix, 0)
        for start in range(0, end, BN):
            j = start + tl.arange(0, BN)
            k = tl.load(K + base + j[:, None] * D + d[None, :], (j[:, None] < key_count) & (d[None, :] < D), 0)
            v = tl.load(V + base + j[:, None] * D + d[None, :], (j[:, None] < key_count) & (d[None, :] < D), 0)
            score = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
            p = tl.exp(score - lse[:, None])
            p = tl.where(active[:, None] & (j[None, :] < prefix[:, None]), p, 0.)
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None]) * SCALE
            ds = tl.where(multiple[:, None], ds, 0.)
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")
        special = tl.load(Self + batch * N + i, active, 0)
        sk = tl.load(SK + batch * KSB + (bh % H) * KSH + original_i[:, None] * KSN + d[None, :] * KSD, mask & special[:, None], 0).to(tl.float32)
        sv = tl.load(SV + batch * VSB + (bh % H) * VSH + original_i[:, None] * VSN + d[None, :] * VSD, mask & special[:, None], 0).to(tl.float32)
        ps = tl.exp(tl.sum(q.to(tl.float32) * sk, 1) * SCALE - lse)
        ps = tl.where(special, ps, 0.)
        ds = ps * (tl.sum(do.to(tl.float32) * sv, 1) - delta) * SCALE
        ds = tl.where(multiple, ds, 0.)
        dq += ds[:, None] * sk
        tl.store(DQ + base + original_i[:, None] * D + d[None, :], dq, mask)
        tl.store(DSK + base + i[:, None] * D + d[None, :], ds[:, None] * q, mask)
        tl.store(DSV + base + i[:, None] * D + d[None, :], ps[:, None] * do, mask)


@triton.jit
def _prefix_dkv(Q, K, V, Prefix, Self, QC, KC, QStart, DO, LSE, Delta, DK, DV,
                N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                BD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch = bh // H
    kc, qc = tl.load(KC + batch), tl.load(QC + batch)
    if block * BN < kc:
        j = block * BN + tl.arange(0, BN)
        d = tl.arange(0, BD)
        base = bh * N * D
        mask = (j[:, None] < kc) & (d[None, :] < D)
        k = tl.load(K + base + j[:, None] * D + d[None, :], mask, 0)
        v = tl.load(V + base + j[:, None] * D + d[None, :], mask, 0)
        dk = tl.full((BN, BD), 0., tl.float32)
        dv = tl.full((BN, BD), 0., tl.float32)
        first_q = tl.load(QStart + batch * N + block * BN)
        for start in range(first_q // BM * BM, qc, BM):
            i = start + tl.arange(0, BM)
            prefix = tl.load(Prefix + batch * N + i, i < qc, 0)
            if tl.max(prefix, 0) > block * BN:
                q = tl.load(Q + base + i[:, None] * D + d[None, :], (i[:, None] < qc) & (d[None, :] < D), 0)
                do = tl.load(DO + base + i[:, None] * D + d[None, :], (i[:, None] < qc) & (d[None, :] < D), 0)
                lse = tl.load(LSE + bh * N + i, i < qc, 0)
                delta = tl.load(Delta + bh * N + i, i < qc, 0)
                score = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
                p = tl.exp(score - lse[:, None])
                p = tl.where((i[:, None] < qc) & (j[None, :] < prefix[:, None]), p, 0.)
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = p * (dp - delta[:, None]) * SCALE
                special = tl.load(Self + batch * N + i, i < qc, 0)
                ds = tl.where((prefix + special.to(tl.int32) > 1)[:, None], ds, 0.)
                dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee")
                dv += tl.dot(tl.trans(p.to(do.dtype)), do, input_precision="ieee")
        tl.store(DK + base + j[:, None] * D + d[None, :], dk, mask)
        tl.store(DV + base + j[:, None] * D + d[None, :], dv, mask)


@triton.jit
def _load_rows(X, rows, d, bh, H: tl.constexpr, SB: tl.constexpr,
               SH: tl.constexpr, SN: tl.constexpr, SD: tl.constexpr, mask):
    return tl.load(X + (bh // H) * SB + (bh % H) * SH + rows[:, None] * SN + d[None, :] * SD, mask, 0)


@triton.jit
def _sparse_rows(Q, K, V, SI, SC, Off, Edges, O, LSE, DO, Delta, DQ,
                 QSB: tl.constexpr, QSH: tl.constexpr, QSN: tl.constexpr, QSD: tl.constexpr,
                 KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
                 VSB: tl.constexpr, VSH: tl.constexpr, VSN: tl.constexpr, VSD: tl.constexpr,
                 N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 BD: tl.constexpr, BR: tl.constexpr, BE: tl.constexpr,
                 SCALE: tl.constexpr, BACKWARD: tl.constexpr):
    bh = tl.program_id(1)
    batch = bh // H
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    active = rows < tl.load(SC + batch)
    i = tl.load(SI + batch * N + rows, active, 0)
    begin = tl.load(Off + batch * (N + 1) + i, active, 0)
    end = tl.load(Off + batch * (N + 1) + i + 1, active, 0)
    degree = end - begin
    d = tl.arange(0, BD)
    mask = active[:, None] & (d[None, :] < D)
    q = _load_rows(Q, i, d, bh, H, QSB, QSH, QSN, QSD, mask).to(tl.float32)
    base = bh * N * D
    if BACKWARD:
        do = tl.load(DO + base + i[:, None] * D + d[None, :], mask, 0).to(tl.float32)
        lse = tl.load(LSE + bh * N + i, active, 0)
        delta = tl.load(Delta + bh * N + i, active, 0)
        dq = tl.full((BR, BD), 0., tl.float32)
    else:
        m = tl.where(active, float('-inf'), 0.)
        l = tl.full((BR,), 0., tl.float32)
        acc = tl.full((BR, BD), 0., tl.float32)
    for start in range(0, tl.max(degree, 0), BE):
        e = begin[:, None] + start + tl.arange(0, BE)[None, :]
        emask = active[:, None] & (e < end[:, None])
        j = tl.load(Edges + batch * 2 * N + e, emask, 0)
        kmask = emask[:, :, None] & (d[None, None, :] < D)
        k = tl.load(K + batch*KSB + (bh%H)*KSH + j[:, :, None]*KSN + d[None,None,:]*KSD, kmask, 0).to(tl.float32)
        v = tl.load(V + batch*VSB + (bh%H)*VSH + j[:, :, None]*VSN + d[None,None,:]*VSD, kmask, 0).to(tl.float32)
        score = tl.sum(k * q[:, None, :], 2) * SCALE
        if BACKWARD:
            p = tl.exp(score - lse[:, None])
            p = tl.where(emask, p, 0.)
            ds = p * (tl.sum(v * do[:, None, :], 2) - delta[:, None]) * SCALE
            ds = tl.where((degree > 1)[:, None], ds, 0.)
            dq += tl.sum(ds[:, :, None] * k, 1)
        else:
            score = tl.where(emask, score, float('-inf'))
            new_m = tl.maximum(m, tl.max(score, 1))
            alpha = tl.exp(m - new_m)
            p = tl.exp(score - new_m[:, None])
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v, 1)
            l = l * alpha + tl.sum(p, 1)
            m = new_m
    if BACKWARD:
        tl.store(DQ + base + i[:,None]*D + d[None,:], dq, mask)
    else:
        tl.store(O + base + i[:,None]*D + d[None,:], acc / tl.where(l > 0, l, 1.)[:,None], mask)
        tl.store(LSE + bh*N + i, m + tl.log(l), active)


@triton.jit
def _pack_qkv(Q, K, V, QInv, KInv, PQ, PK, PV,
              QSB: tl.constexpr, QSH: tl.constexpr, QSN: tl.constexpr, QSD: tl.constexpr,
              KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
              VSB: tl.constexpr, VSH: tl.constexpr, VSN: tl.constexpr, VSD: tl.constexpr,
              N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              BD: tl.constexpr, BM: tl.constexpr):
    bh = tl.program_id(1)
    i = tl.program_id(0)*BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    qi = tl.load(QInv + (bh//H)*N + i, i < N, -1)
    ki = tl.load(KInv + (bh//H)*N + i, i < N, -1)
    qmask = (i[:,None] < N) & (qi[:,None] >= 0) & (d[None,:] < D)
    kmask = (i[:,None] < N) & (ki[:,None] >= 0) & (d[None,:] < D)
    q = _load_rows(Q, i, d, bh, H, QSB, QSH, QSN, QSD, qmask)
    k = _load_rows(K, i, d, bh, H, KSB, KSH, KSN, KSD, kmask)
    v = _load_rows(V, i, d, bh, H, VSB, VSH, VSN, VSD, kmask)
    base = bh*N*D
    tl.store(PQ + base + qi[:,None]*D + d[None,:], q, qmask)
    tl.store(PK + base + ki[:,None]*D + d[None,:], k, kmask)
    tl.store(PV + base + ki[:,None]*D + d[None,:], v, kmask)


@triton.jit
def _prepare_backward(DO, Out, QInv, CDO, PDO, PDelta, SDelta,
                      DSB: tl.constexpr, DSH: tl.constexpr, DSN: tl.constexpr, DSD: tl.constexpr,
                      N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr):
    bh = tl.program_id(1)
    i = tl.program_id(0)*BM + tl.arange(0,BM)
    d = tl.arange(0,BD)
    mask = (i[:,None] < N) & (d[None,:] < D)
    qi = tl.load(QInv + (bh//H)*N + i, i < N, -1)
    base = bh*N*D
    do = _load_rows(DO, i, d, bh, H, DSB, DSH, DSN, DSD, mask)
    out = tl.load(Out + base + i[:,None]*D + d[None,:], mask, 0)
    delta = tl.sum(do.to(tl.float32)*out.to(tl.float32),1)
    tl.store(CDO + base + i[:,None]*D + d[None,:], do, mask)
    tl.store(PDO + base + qi[:,None]*D + d[None,:], do, mask & (qi[:,None]>=0))
    tl.store(PDelta + bh*N + qi, delta, (i<N)&(qi>=0))
    tl.store(SDelta + bh*N + i, delta, i<N)


@triton.jit
def _merge_dkv(Q, K, V, Reverse, Off, DO, LSE, Delta,
               PDK, PDV, DSK, DSV, QInv, KInv, DK, DV,
               QSB: tl.constexpr, QSH: tl.constexpr, QSN: tl.constexpr, QSD: tl.constexpr,
               KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
               VSB: tl.constexpr, VSH: tl.constexpr, VSN: tl.constexpr, VSD: tl.constexpr,
               N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
               BM: tl.constexpr, SCALE: tl.constexpr):
    """Multiple original keys per CTA; fuse reverse compose and final summation."""
    bh = tl.program_id(1)
    batch = bh//H
    j = tl.program_id(0)*BM + tl.arange(0,BM)
    d = tl.arange(0,BD)
    mask = (j[:,None]<N)&(d[None,:]<D)
    base = bh*N*D
    k = _load_rows(K, j, d, bh, H, KSB, KSH, KSN, KSD, mask).to(tl.float32)
    v = _load_rows(V, j, d, bh, H, VSB, VSH, VSN, VSD, mask).to(tl.float32)
    dk, dv = tl.full((BM,BD),0.,tl.float32), tl.full((BM,BD),0.,tl.float32)
    for slot in tl.static_range(2):
        i = tl.load(Reverse + (batch*N+j)*2 + slot, j<N, -1)
        active = (j<N)&(i>=0)
        imask = active[:,None] & (d[None,:]<D)
        q = _load_rows(Q, i, d, bh, H, QSB, QSH, QSN, QSD, imask).to(tl.float32)
        do = tl.load(DO + base + i[:,None]*D + d[None,:], imask, 0).to(tl.float32)
        lse = tl.load(LSE + bh*N + i, active, 0)
        delta = tl.load(Delta + bh*N + i, active, 0)
        p = tl.exp(tl.sum(q*k,1)*SCALE-lse)
        p = tl.where(active,p,0.)
        ds = p*(tl.sum(do*v,1)-delta)*SCALE
        lo = tl.load(Off + batch*(N+1)+i, active, 0)
        hi = tl.load(Off + batch*(N+1)+i+1, active, 0)
        ds = tl.where(hi-lo>1,ds,0.)
        dk += ds[:,None]*q
        dv += p[:,None]*do
    qi = tl.load(QInv+batch*N+j,j<N,-1)
    ki = tl.load(KInv+batch*N+j,j<N,-1)
    # Preserve float32 accumulation across all sources, then cast once.
    dk += tl.load(PDK+base+ki[:,None]*D+d[None,:],mask&(ki[:,None]>=0),0).to(tl.float32)
    dk += tl.load(DSK+base+qi[:,None]*D+d[None,:],mask&(qi[:,None]>=0),0).to(tl.float32)
    dv += tl.load(PDV+base+ki[:,None]*D+d[None,:],mask&(ki[:,None]>=0),0).to(tl.float32)
    dv += tl.load(DSV+base+qi[:,None]*D+d[None,:],mask&(qi[:,None]>=0),0).to(tl.float32)
    tl.store(DK+base+j[:,None]*D+d[None,:],dk,mask)
    tl.store(DV+base+j[:,None]*D+d[None,:],dv,mask)


def _parameters(q):
    b,h,n,d = q.shape
    return dict(N=n,H=h,D=d,BD=triton.next_power_of_2(d),SCALE=d**-0.5)


def _strides(q,k,v):
    return {f'{name}S{axis}': stride for name,t in (('Q',q),('K',k),('V',v))
            for axis,stride in zip('BHND',t.stride())}


def _empty_qkv(q, number):
    return [torch.empty(q.shape,device=q.device,dtype=q.dtype) for _ in range(number)]


# Separate choices for forward, dQ and dKV. Tuned on the target Ampere shape.
_PREFIX_CONFIGS = ((128,64,4,3),(32,64,4,2),(64,32,4,2))


def _configs_for(q):
    # The large forward tile was measured for D64 half precision. FP32 needs
    # more shared memory and must not inherit this Ampere-specific choice.
    if q.dtype in (torch.float16, torch.bfloat16) and q.shape[-1] == 64 and q.shape[-2] >= 512:
        return _PREFIX_CONFIGS
    return ((32,64,4,2),(32,64,4,2),(32,32,4,2))


class _FusedTyped(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,layout):
        b,h,n,d = q.shape
        params = _parameters(q)
        moves = {key:value for key,value in params.items() if key!='SCALE'}
        strides = _strides(q,k,v)
        kvstrides = {key:value for key,value in strides.items() if not key.startswith('Q')}
        pq,pk,pv,out = _empty_qkv(q,4)
        _pack_qkv[(triton.cdiv(n,32),b*h)](q,k,v,layout.q_inverse,layout.k_inverse,pq,pk,pv,
                                         **strides,**moves,BM=32,num_warps=4)
        plse,slse = [torch.empty((b,h,n),device=q.device,dtype=torch.float32) for _ in range(2)]
        configs = _configs_for(q)
        bm,bn,nw,ns = configs[0]
        _prefix_fwd[(triton.cdiv(n,bm),b*h)](pq,pk,pv,k,v,layout.q_index,layout.prefix,layout.self_mask,
            layout.q_count,layout.k_count,out,plse,**kvstrides,**params,BM=bm,BN=bn,num_warps=nw,num_stages=ns)
        if layout.sparse_capacity:
            _sparse_rows[(triton.cdiv(layout.sparse_capacity,16),b*h)](q,k,v,layout.sparse_index,
                layout.sparse_count,layout.offsets,layout.edges,out,slse,out,slse,out,
                **strides,**params,BR=16,BE=8,BACKWARD=False,num_warps=4)
        ctx.sparse_capacity = layout.sparse_capacity
        ctx.configs = configs
        ctx.save_for_backward(q,k,v,pq,pk,pv,out,plse,slse,layout.q_index,layout.prefix,
            layout.self_mask,layout.q_count,layout.k_count,layout.q_start,layout.sparse_index,
            layout.sparse_count,layout.offsets,layout.edges,layout.reverse,layout.q_inverse,layout.k_inverse)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,do):
        q,k,v,pq,pk,pv,out,plse,slse,qidx,prefix,self_mask,qc,kc,qstart,si,sc,offsets,edges,reverse,qi,ki = ctx.saved_tensors
        b,h,n,d = q.shape
        params = _parameters(q)
        moves = {key:value for key,value in params.items() if key!='SCALE'}
        strides = _strides(q,k,v)
        kvstrides = {key:value for key,value in strides.items() if not key.startswith('Q')}
        cdo,pdo,pdk,pdv,dsk,dsv,dq,dk,dv = _empty_qkv(q,9)
        pd,sd = torch.empty_like(plse),torch.empty_like(slse)
        _prepare_backward[(triton.cdiv(n,32),b*h)](do,out,qi,cdo,pdo,pd,sd,
            DSB=do.stride(0),DSH=do.stride(1),DSN=do.stride(2),DSD=do.stride(3),**moves,BM=32,num_warps=4)
        bm,bn,nw,ns = ctx.configs[1]
        _prefix_dq[(triton.cdiv(n,bm),b*h)](pq,pk,pv,k,v,qidx,prefix,self_mask,qc,kc,pdo,plse,pd,dq,dsk,dsv,
            **kvstrides,**params,BM=bm,BN=bn,num_warps=nw,num_stages=ns)
        bm,bn,nw,ns = ctx.configs[2]
        _prefix_dkv[(triton.cdiv(n,bn),b*h)](pq,pk,pv,prefix,self_mask,qc,kc,qstart,pdo,plse,pd,pdk,pdv,
            **params,BM=bm,BN=bn,num_warps=nw,num_stages=ns)
        if ctx.sparse_capacity:
            _sparse_rows[(triton.cdiv(ctx.sparse_capacity,16),b*h)](q,k,v,si,sc,offsets,edges,out,slse,cdo,sd,dq,
                **strides,**params,BR=16,BE=4,BACKWARD=True,num_warps=4)
        _merge_dkv[(triton.cdiv(n,8),b*h)](q,k,v,reverse,offsets,cdo,slse,sd,pdk,pdv,dsk,dsv,qi,ki,dk,dv,
            **strides,**params,BM=8,num_warps=4)
        return dq,dk,dv,None


def typed_attention(q,k,v,layout):
    """Prefix + grouped compose with direct original-order output and dQ."""
    return _FusedTyped.apply(q,k,v,layout)
