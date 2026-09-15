"""TG union-gather attention. No NxN masks, probabilities or atomic gradients."""
import torch
from dataclasses import fields
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .tgnomask_kernels import _load_rows, _parameters, _sparse_rows, _strides


@triton.jit(do_not_specialize=['KS'])
def _ordinary(Q, K, V, QI, QC, Lo, Hi, Degree, Off, Keys, O, LSE, DO, Delta, DQ,
              QSB: tl.constexpr, QSH: tl.constexpr, QSN: tl.constexpr, QSD: tl.constexpr,
              KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
              VSB: tl.constexpr, VSH: tl.constexpr, VSN: tl.constexpr, VSD: tl.constexpr,
              N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
              OS: tl.constexpr, KS, BM: tl.constexpr, BN: tl.constexpr,
              SCALE: tl.constexpr, BACKWARD: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch = bh // H
    count = tl.load(QC + batch)
    if block * BM < count:
        rank = block * BM + tl.arange(0, BM)
        active = rank < count
        i = tl.load(QI + batch * N + rank, active, 0)
        d = tl.arange(0, BD)
        mask = active[:, None] & (d[None, :] < D)
        q = _load_rows(Q, i, d, bh, H, QSB, QSH, QSN, QSD, mask)
        begin = tl.load(Off + batch * OS + block)
        end = tl.load(Off + batch * OS + block + 1)
        base = bh * N * D
        if BACKWARD:
            do = tl.load(DO + base + i[:, None] * D + d[None, :], mask, 0)
            lse = tl.load(LSE + bh * N + i, active, 0)
            delta = tl.load(Delta + bh * N + i, active, 0)
            multiple = tl.load(Degree + batch * N + i, active, 0) > 1
            dq = tl.full((BM, BD), 0., tl.float32)
        else:
            m = tl.full((BM,), float('-inf'), tl.float32)
            normalizer = tl.full((BM,), 0., tl.float32)
            acc = tl.full((BM, BD), 0., tl.float32)
        for start in range(begin, end, BN):
            slot = start + tl.arange(0, BN)
            key_active = slot < end
            j = tl.load(Keys + batch * KS + slot, key_active, 0)
            low = tl.load(Lo + batch * N + j, key_active, 0)
            high = tl.load(Hi + batch * N + j, key_active, 0)
            visible = active[:, None] & key_active[None, :] & (rank[:, None] >= low[None, :]) & (rank[:, None] < high[None, :])
            kmask = key_active[:, None] & (d[None, :] < D)
            k = _load_rows(K, j, d, bh, H, KSB, KSH, KSN, KSD, kmask)
            v = _load_rows(V, j, d, bh, H, VSB, VSH, VSN, VSD, kmask)
            score = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
            if BACKWARD:
                p = tl.where(visible, tl.exp(score - lse[:, None]), 0.)
                dp = tl.dot(do, tl.trans(v), input_precision="ieee")
                ds = tl.where(multiple[:, None], p * (dp - delta[:, None]) * SCALE, 0.)
                dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")
            else:
                score = tl.where(visible, score * 1.4426950408889634, float('-inf'))
                new_m = tl.maximum(m, tl.max(score, 1))
                # Some rows see no keys in the first union tile. Avoid inf-inf.
                safe_m = tl.where(new_m == float('-inf'), 0., new_m)
                alpha = tl.exp2(m - safe_m)
                p = tl.exp2(score - safe_m[:, None])
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
                normalizer = normalizer * alpha + tl.sum(p, 1)
                m = safe_m
        if BACKWARD:
            tl.store(DQ + base + i[:, None] * D + d[None, :], dq, mask)
        else:
            tl.store(O + base + i[:, None] * D + d[None, :], acc / tl.where(normalizer > 0., normalizer, 1.)[:, None], mask)
            tl.store(LSE + bh * N + i, m * 0.6931471805599453 + tl.log(normalizer), active)


@triton.jit
def _prepare(DO, O, CDO, Delta,
             DSB: tl.constexpr, DSH: tl.constexpr, DSN: tl.constexpr, DSD: tl.constexpr,
             N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr):
    bh = tl.program_id(1)
    i = tl.program_id(0) * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    mask = (i[:, None] < N) & (d[None, :] < D)
    do = _load_rows(DO, i, d, bh, H, DSB, DSH, DSN, DSD, mask)
    out = tl.load(O + bh * N * D + i[:, None] * D + d[None, :], mask, 0)
    tl.store(CDO + bh * N * D + i[:, None] * D + d[None, :], do, mask)
    delta = tl.sum(do.to(tl.float32) * out.to(tl.float32), 1)
    tl.store(Delta + bh * N + i, delta, i < N)


@triton.jit(do_not_specialize=['QS'])
def _dkv(Q, K, V, QI, Lo, Hi, Degree, KeyOrder, Off, Queries, Reverse, SOff,
         DO, LSE, Delta, DK, DV,
         QSB: tl.constexpr, QSH: tl.constexpr, QSN: tl.constexpr, QSD: tl.constexpr,
         KSB: tl.constexpr, KSH: tl.constexpr, KSN: tl.constexpr, KSD: tl.constexpr,
         VSB: tl.constexpr, VSH: tl.constexpr, VSN: tl.constexpr, VSD: tl.constexpr,
         N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
         OS: tl.constexpr, QS, BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch = bh // H
    slot = block * BN + tl.arange(0, BN)
    active = slot < N
    j = tl.load(KeyOrder + batch * N + slot, active, 0)
    d = tl.arange(0, BD)
    mask = active[:, None] & (d[None, :] < D)
    k = _load_rows(K, j, d, bh, H, KSB, KSH, KSN, KSD, mask)
    v = _load_rows(V, j, d, bh, H, VSB, VSH, VSN, VSD, mask)
    lo = tl.load(Lo + batch * N + j, active, 0)
    hi = tl.load(Hi + batch * N + j, active, 0)
    begin = tl.load(Off + batch * OS + block)
    end = tl.load(Off + batch * OS + block + 1)
    dk = tl.full((BN, BD), 0., tl.float32)
    dv = tl.full((BN, BD), 0., tl.float32)
    base = bh * N * D
    for start in range(begin, end, BM):
        entry = start + tl.arange(0, BM)
        qa = entry < end
        rank = tl.load(Queries + batch * QS + entry, qa, 0)
        i = tl.load(QI + batch * N + rank, qa, 0)
        qmask = qa[:, None] & (d[None, :] < D)
        q = _load_rows(Q, i, d, bh, H, QSB, QSH, QSN, QSD, qmask)
        do = tl.load(DO + base + i[:, None] * D + d[None, :], qmask, 0)
        lse = tl.load(LSE + bh * N + i, qa, 0)
        delta = tl.load(Delta + bh * N + i, qa, 0)
        degree = tl.load(Degree + batch * N + i, qa, 0)
        visible = qa[:, None] & active[None, :] & (rank[:, None] >= lo[None, :]) & (rank[:, None] < hi[None, :])
        score = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
        p = tl.where(visible, tl.exp(score - lse[:, None]), 0.)
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = tl.where((degree > 1)[:, None], p * (dp - delta[:, None]) * SCALE, 0.)
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q, input_precision="ieee")
        dv += tl.dot(tl.trans(p.to(do.dtype)), do, input_precision="ieee")
    # Compose/self consumers are bounded by two; fuse their contribution before
    # the sole final dK/dV store. Ordinary consumers are handled above, unbounded.
    for reverse_slot in tl.static_range(2):
        i = tl.load(Reverse + (batch * N + j) * 2 + reverse_slot, active, -1)
        qa = active & (i >= 0)
        qmask = qa[:, None] & (d[None, :] < D)
        q = _load_rows(Q, i, d, bh, H, QSB, QSH, QSN, QSD, qmask).to(tl.float32)
        do = tl.load(DO + base + i[:, None] * D + d[None, :], qmask, 0).to(tl.float32)
        lse = tl.load(LSE + bh * N + i, qa, 0)
        delta = tl.load(Delta + bh * N + i, qa, 0)
        left = tl.load(SOff + batch * (N + 1) + i, qa, 0)
        right = tl.load(SOff + batch * (N + 1) + i + 1, qa, 0)
        p = tl.where(qa, tl.exp(tl.sum(q * k.to(tl.float32), 1) * SCALE - lse), 0.)
        ds = tl.where(right - left > 1, p * (tl.sum(do * v.to(tl.float32), 1) - delta) * SCALE, 0.)
        dk += ds[:, None] * q
        dv += p[:, None] * do
    tl.store(DK + base + j[:, None] * D + d[None, :], dk, mask)
    tl.store(DV + base + j[:, None] * D + d[None, :], dv, mask)


# Runtime tree data never enters the specialization key. These tile constants
# are exposed for offline paired tuning; layouts carry matching Q/K group sizes.
_CONFIG = dict(fwd_bn=32, dq_bn=32, dkv_bm=32, warps=4, stages=2)


class _TG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, layout):
        b, h, n, d = q.shape
        base = layout.base
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        lse = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        params, strides = _parameters(q), _strides(q, k, v)
        config = dict(_CONFIG)
        _ordinary[(triton.cdiv(n, layout.q_tile), b * h)](q, k, v, base.q_index, base.q_count,
            layout.lo, layout.hi, layout.degree, layout.fwd_offsets, layout.fwd_keys, out, lse, out, lse, out,
            **params, **strides, OS=layout.fwd_offsets.shape[1], KS=layout.fwd_keys.shape[1],
            BM=layout.q_tile, BN=config['fwd_bn'], BACKWARD=False,
            num_warps=config['warps'], num_stages=config['stages'])
        if base.sparse_capacity:
            _sparse_rows[(triton.cdiv(base.sparse_capacity, 16), b * h)](q, k, v,
                base.sparse_index, base.sparse_count, base.offsets, base.edges, out, lse, out, lse, out,
                **params, **strides, BR=16, BE=8, BACKWARD=False, num_warps=4)
        metadata = [getattr(layout, f.name) for f in fields(layout)
                    if isinstance(getattr(layout, f.name), torch.Tensor)]
        metadata += [getattr(base, f.name) for f in fields(base)
                     if isinstance(getattr(base, f.name), torch.Tensor)]
        ctx.save_for_backward(q, k, v, out, lse, *metadata)
        ctx.layout, ctx.config = layout, config
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        q, k, v, out, lse = ctx.saved_tensors[:5]
        layout, config = ctx.layout, ctx.config
        base = layout.base
        b, h, n, d = q.shape
        params, strides = _parameters(q), _strides(q, k, v)
        cdo, dq, dk, dv = [torch.empty(q.shape, device=q.device, dtype=q.dtype) for _ in range(4)]
        delta = torch.empty_like(lse)
        _prepare[(triton.cdiv(n, 32), b * h)](do, out, cdo, delta,
            DSB=do.stride(0), DSH=do.stride(1), DSN=do.stride(2), DSD=do.stride(3),
            **{key: value for key, value in params.items() if key != 'SCALE'}, BM=32, num_warps=4)
        _ordinary[(triton.cdiv(n, layout.q_tile), b * h)](q, k, v, base.q_index, base.q_count,
            layout.lo, layout.hi, layout.degree, layout.fwd_offsets, layout.fwd_keys, out, lse, cdo, delta, dq,
            **params, **strides, OS=layout.fwd_offsets.shape[1], KS=layout.fwd_keys.shape[1],
            BM=layout.q_tile, BN=config['dq_bn'], BACKWARD=True,
            num_warps=config['warps'], num_stages=config['stages'])
        if base.sparse_capacity:
            _sparse_rows[(triton.cdiv(base.sparse_capacity, 16), b * h)](q, k, v,
                base.sparse_index, base.sparse_count, base.offsets, base.edges, out, lse, cdo, delta, dq,
                **params, **strides, BR=16, BE=4, BACKWARD=True, num_warps=4)
        _dkv[(triton.cdiv(n, layout.k_tile), b * h)](q, k, v, base.q_index, layout.lo, layout.hi,
            layout.degree, layout.key_order, layout.bwd_offsets, layout.bwd_queries, base.reverse, base.offsets,
            cdo, lse, delta, dk, dv, **params, **strides, OS=layout.bwd_offsets.shape[1], QS=layout.bwd_queries.shape[1],
            BM=config['dkv_bm'], BN=layout.k_tile, num_warps=config['warps'], num_stages=config['stages'])
        return dq, dk, dv, None


def typed_tg_attention(q, k, v, layout):
    return _TG.apply(q, k, v, layout)


@triton.jit
def _join_gradients(GQ, GK, GV, DQ, DK, DV, GQS, GKS, GVS,
                    DQS, DKS, DVS, HEADS,
                    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < B * H * N * D
    d = index % D
    n = index // D % N
    h = index // (D * N) % H
    b = index // (D * N * H)
    start = 0
    for group in tl.static_range(len(HEADS)):
        mask = active & (h >= start) & (h < start + HEADS[group])
        local = h - start
        q = tl.load(GQ[group] + b*GQS[group][0] + local*GQS[group][1] + n*GQS[group][2] + d*GQS[group][3], mask, 0)
        k = tl.load(GK[group] + b*GKS[group][0] + local*GKS[group][1] + n*GKS[group][2] + d*GKS[group][3], mask, 0)
        v = tl.load(GV[group] + b*GVS[group][0] + local*GVS[group][1] + n*GVS[group][2] + d*GVS[group][3], mask, 0)
        tl.store(DQ + b*DQS[0] + h*DQS[1] + n*DQS[2] + d*DQS[3], q, mask)
        tl.store(DK + b*DKS[0] + h*DKS[1] + n*DKS[2] + d*DKS[3], k, mask)
        tl.store(DV + b*DVS[0] + h*DVS[1] + n*DVS[2] + d*DVS[3], v, mask)
        start += HEADS[group]


def join_gradients(grads, originals, heads):
    outputs = tuple(torch.empty_like(x) for x in originals)
    groups = tuple(tuple(g[axis] for g in grads) for axis in range(3))
    b, h, n, d = originals[0].shape
    _join_gradients[(triton.cdiv(b*h*n*d, 256),)](*groups, *outputs,
        GQS=tuple(x.stride() for x in groups[0]), GKS=tuple(x.stride() for x in groups[1]),
        GVS=tuple(x.stride() for x in groups[2]), DQS=outputs[0].stride(), DKS=outputs[1].stride(),
        DVS=outputs[2].stride(), HEADS=heads, B=b, H=h, N=n, D=d, BLOCK=256, num_warps=4)
    return outputs
