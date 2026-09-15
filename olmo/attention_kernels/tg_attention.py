"""Exact fresh-segment TG and head-mixed attention with compact shared layouts.

Build layouts alongside CPU batches, transfer once, and reuse across layers.
No dense attention mask is allocated by the layout builder or CUDA operators.
"""
from dataclasses import dataclass, fields
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .tgnomask import TGNoMaskLayout, tgnomask_attention


@dataclass(frozen=True)
class TGLayout:
    base: TGNoMaskLayout
    lo: torch.Tensor
    hi: torch.Tensor
    degree: torch.Tensor
    fwd_offsets: torch.Tensor
    fwd_keys: torch.Tensor
    bwd_offsets: torch.Tensor
    bwd_queries: torch.Tensor
    key_order: torch.Tensor
    q_tile: int
    k_tile: int

    @property
    def shape(self):
        return self.base.shape

    @property
    def device(self):
        return self.base.device

    def to(self, device):
        if self.device == torch.device(device):
            return self
        return TGLayout(**{f.name: (getattr(self, f.name).to(device)
            if hasattr(getattr(self, f.name), "to") else getattr(self, f.name)) for f in fields(self)})

    def slice_batch(self, start, end):
        base = TGNoMaskLayout(**{f.name: (getattr(self.base, f.name)[start:end]
            if isinstance(getattr(self.base, f.name), torch.Tensor) else getattr(self.base, f.name)) for f in fields(self.base)})
        return TGLayout(**{f.name: (base if f.name == 'base' else getattr(self, f.name)[start:end]
            if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name)) for f in fields(self)})

    def pin_memory(self):
        base = TGNoMaskLayout(**{f.name: (getattr(self.base, f.name).pin_memory()
            if isinstance(getattr(self.base, f.name), torch.Tensor) else getattr(self.base, f.name)) for f in fields(self.base)})
        return TGLayout(**{f.name: (base if f.name == 'base' else getattr(self, f.name).pin_memory()
            if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name)) for f in fields(self)})

    def dense_mask(self):
        """Debug/reference only; CUDA kernels never call this method."""
        cpu = self.to("cpu")
        b, n = cpu.shape
        out = torch.zeros(b, 1, n, n, dtype=torch.bool)
        for batch in range(b):
            qi = cpu.base.q_index[batch, :int(cpu.base.q_count[batch])].long()
            rank = torch.arange(len(qi))[:, None]
            out[batch, 0, qi] = (rank >= cpu.lo[batch]) & (rank < cpu.hi[batch])
            for query in cpu.base.sparse_index[batch, :int(cpu.base.sparse_count[batch])]:
                low, high = cpu.base.offsets[batch, query:query + 2]
                out[batch, 0, query, cpu.base.edges[batch, low:high].long()] = True
        return out.to(self.device)


def _padded(rows):
    out = np.zeros((len(rows), max(1, max(map(len, rows)))), dtype=np.int32)
    for i, row in enumerate(rows):
        out[i, :len(row)] = row
    return torch.from_numpy(out)


def _build_base(input_ids, vocab):
    """Same parser/actions as TGNoMaskLayout, using bulk NumPy tensor creation.

    Avoid thousands of individual torch tensor writes in dataloader workers.
    The resulting layout remains compatible with the existing TGnomask kernels.
    """
    ids = input_ids.detach().cpu()
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or not ids.shape[0] or not ids.shape[1] or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('input_ids must be a nonempty 1-D or 2-D integer tensor')
    b, n = ids.shape
    op, oe = vocab.opening_non_terminals
    cl, ce = vocab.closing_non_terminals
    pad = vocab.pad
    a = {name: np.zeros((b, n), dtype=np.int32) for name in
         ('q_index', 'k_index', 'prefix', 'sparse_index', 'q_start')}
    a.update(q_count=np.zeros(b, dtype=np.int32), k_count=np.zeros(b, dtype=np.int32),
             sparse_count=np.zeros(b, dtype=np.int32), self_mask=np.zeros((b, n), dtype=bool),
             offsets=np.zeros((b, n+1), dtype=np.int32), edges=np.zeros((b, 2*n), dtype=np.int32),
             reverse=np.full((b, n, 2), -1, dtype=np.int32), label_mask=np.ones((b, n), dtype=bool),
             q_inverse=np.full((b, n), -1, dtype=np.int32), k_inverse=np.full((b, n), -1, dtype=np.int32))
    for batch, tokens in enumerate(ids.tolist()):
        stack, ordinary, keys, sparse, edges, prefixes, specials = ([] for _ in range(7))
        last = -1
        offsets = [0]
        reverse = [[] for _ in tokens]
        for i, token in enumerate(tokens):
            closing = cl <= token < ce
            compose = closing and last != token and last != -1
            if closing and last == -1:
                end = i
                while end < n and tokens[end] == token:
                    end += 1
                compose = (end-i) % 2 == 0
            allowed = []
            if token == pad:
                sparse.append(i)
                allowed = [i]
            elif compose:
                popped, j = [], i
                while stack and not op <= tokens[j] < oe:
                    j = stack.pop()
                    popped.append(j)
                stack.append(i)
                keys.append(i)
                sparse.append(i)
                allowed = popped[::-1] + [i]
                last = token
            else:
                if not closing:
                    stack.append(i)
                    keys.append(i)
                else:
                    a['label_mask'][batch, i] = False
                ordinary.append(i)
                prefixes.append(len(keys))
                specials.append(closing)
                last = token
            for key in allowed:
                reverse[key].append(i)
            edges.extend(allowed)
            offsets.append(len(edges))
        sparse.sort(key=lambda i: offsets[i+1]-offsets[i])
        for name, values in (('q_index', ordinary), ('k_index', keys), ('prefix', prefixes),
                             ('sparse_index', sparse), ('edges', edges), ('self_mask', specials)):
            a[name][batch, :len(values)] = values
        a['offsets'][batch] = offsets
        a['q_start'][batch] = np.searchsorted(prefixes, np.arange(n), side='right')
        for key, consumers in enumerate(reverse):
            assert len(consumers) <= 2
            if consumers:
                a['reverse'][batch, key, :len(consumers)] = consumers
        a['q_count'][batch], a['k_count'][batch], a['sparse_count'][batch] = len(ordinary), len(keys), len(sparse)
        a['q_inverse'][batch, ordinary] = np.arange(len(ordinary), dtype=np.int32)
        a['k_inverse'][batch, keys] = np.arange(len(keys), dtype=np.int32)
    return TGNoMaskLayout(**{name: torch.from_numpy(values) for name, values in a.items()},
                          sparse_capacity=int(a['sparse_count'].max()))


def build_tg_layout(input_ids: torch.Tensor, vocab: Any, device=None,
                    q_tile: int = 32, k_tile: int = 16) -> TGLayout:
    """Build lifetime intervals and tile unions; supports truncated/repeated closes.

    Base metadata is linear in sequence length. Union schedules can be quadratic
    for degenerate trees; their storage is proportional to scheduled key/query
    lists, not head count. CPU tokens avoid a device synchronization here.
    """
    if q_tile not in (16, 32, 64) or k_tile not in (16, 32, 64):
        raise ValueError("TG tile sizes must be 16, 32 or 64")
    base = _build_base(input_ids, vocab)
    b, n = base.shape
    lows, highs, degrees, orders, foffs, fkeys, boffs, bqueries = ([] for _ in range(8))
    for batch in range(b):
        qi = base.q_index[batch, :int(base.q_count[batch])].numpy()
        death = np.full(n, n, dtype=np.int32)
        offsets, edges = base.offsets[batch].numpy(), base.edges[batch].numpy()
        for query in base.sparse_index[batch, :int(base.sparse_count[batch])].tolist():
            left, right = offsets[query:query + 2]
            popped = edges[left:right - 1]
            death[popped] = query
        valid = base.k_inverse[batch].numpy() >= 0
        lo = np.searchsorted(qi, np.arange(n)).astype(np.int32)
        hi = np.searchsorted(qi, death).astype(np.int32)
        inv = base.q_inverse[batch].numpy()
        # Non-stack ordinary keys (repeated closes) have only their self edge.
        lo[~valid] = np.maximum(inv[~valid], 0)
        hi[~valid] = np.maximum(inv[~valid] + 1, 0)
        delta = np.zeros(len(qi) + 1, dtype=np.int32)
        np.add.at(delta, lo, 1)
        np.add.at(delta, hi, -1)
        degree = np.zeros(n, dtype=np.int32)
        degree[qi] = np.cumsum(delta)[:-1]
        off, keys = [0], []
        for start in range(0, n, q_tile):
            keys.extend(np.flatnonzero((lo < min(start + q_tile, len(qi))) & (hi > start)).tolist())
            off.append(len(keys))
        # Group keys with similar consumer counts; keep neighboring lifetimes
        # together within each bucket to reduce wasted query-union arithmetic.
        order = sorted(range(n), key=lambda k: (int(hi[k] - lo[k]).bit_length(), int(lo[k]), k))
        backoff, queries = [0], []
        for start in range(0, n, k_tile):
            intervals = sorted((int(lo[k]), int(hi[k])) for k in order[start:start + k_tile] if hi[k] > lo[k])
            end = 0
            for left, right in intervals:
                queries.extend(range(max(left, end), right))
                end = max(end, right)
            backoff.append(len(queries))
        lows.append(lo); highs.append(hi); degrees.append(degree); orders.append(order)
        foffs.append(off); fkeys.append(keys); boffs.append(backoff); bqueries.append(queries)
    result = TGLayout(base, _padded(lows), _padded(highs), _padded(degrees),
                      _padded(foffs), _padded(fkeys), _padded(boffs), _padded(bqueries),
                      _padded(orders), q_tile, k_tile)
    return result.to(device or "cpu")


@dataclass(frozen=True)
class MixTGLayout:
    tg: TGLayout
    head_groups: Tuple[Tuple[str, int], ...]

    @property
    def shape(self):
        return self.tg.shape

    @property
    def device(self):
        return self.tg.device

    @property
    def label_mask(self):
        return self.tg.base.label_mask

    def to(self, device):
        return MixTGLayout(self.tg.to(device), self.head_groups)

    def slice_batch(self, start, end):
        return MixTGLayout(self.tg.slice_batch(start, end), self.head_groups)

    def pin_memory(self):
        return MixTGLayout(self.tg.pin_memory(), self.head_groups)

    def dense_mask(self):
        b, n = self.shape
        masks = {"tg": self.tg.dense_mask(), "tgnomask": self.tg.base.dense_mask()}
        masks["tgtree"] = torch.ones((b, 1, n, n), device=self.device, dtype=torch.bool).tril()
        return torch.cat([masks[kind].expand(b, heads, n, n) for kind, heads in self.head_groups], 1)


def build_mix_tg_layout(input_ids, vocab, head_groups: Sequence, device=None, **kwargs):
    groups = tuple((g.grammar_type, g.n_heads) if hasattr(g, "grammar_type") else tuple(g) for g in head_groups)
    if not groups or any(kind not in ("tg", "tgnomask", "tgtree") or not isinstance(h, int) or h <= 0 for kind, h in groups):
        raise ValueError("Mixed TG requires positive head groups of tg/tgnomask/tgtree")
    return MixTGLayout(build_tg_layout(input_ids, vocab, device, **kwargs), groups)


def _validate(q, k, v, layout):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("TG requires matching (B,H,N,D) Q/K/V (MHA)")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("TG requires Q/K/V on the same CUDA device")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("TG requires matching fp32/fp16/bf16 Q/K/V")
    if not 16 <= q.shape[-1] <= 128:
        raise ValueError("TG supports head dimensions 16 through 128")
    if layout.shape != (q.shape[0], q.shape[2]) or layout.device != q.device:
        raise ValueError("TG layout shape/device must match Q/K/V")


def tg_attention(q, k, v, layout: TGLayout):
    """Exact TG visibility with first-order backward; no dropout/cache/GQA."""
    _validate(q, k, v, layout)
    from .tg_kernels import typed_tg_attention
    return typed_tg_attention(q, k, v, layout)


def mix_tg_attention(q, k, v, layout: MixTGLayout):
    """Preserve head order and delegate each group to its specialized operator."""
    _validate(q, k, v, layout)
    if sum(h for _, h in layout.head_groups) != q.shape[1]:
        raise ValueError("Mixed head allocation must equal Q/K/V head count")
    need_grad = torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v))
    return _MixedTG.apply(q, k, v, layout, need_grad)


def _group_attention(xs, kind, layout):
    if kind == "tg":
        return tg_attention(*xs, layout.tg)
    if kind == "tgnomask":
        return tgnomask_attention(*xs, layout.tg.base)
    return F.scaled_dot_product_attention(*xs, is_causal=True)


class _KernelState:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class _MixedTG(torch.autograd.Function):
    """One autograd node for grouped kernels, with one fused gradient assembly."""
    @staticmethod
    def forward(ctx, q, k, v, layout, need_grad):
        from .tg_kernels import _TG
        from .tgnomask_kernels import _FusedTyped
        outputs, tensors, states, start = [], [], [], 0
        for kind, heads in layout.head_groups:
            xs = [x[:, start:start + heads].detach() for x in (q, k, v)]
            state = _KernelState()
            if kind == 'tg':
                out = _TG.forward(state, *xs, layout.tg)
            elif kind == 'tgnomask':
                out = _FusedTyped.forward(state, *xs, layout.tg.base)
            elif need_grad:
                with torch.enable_grad():
                    xs = [x.requires_grad_() for x in xs]
                    out = _group_attention(xs, kind, layout)
                state.save_for_backward(*xs, out)
            else:
                out = _group_attention(xs, kind, layout)
            outputs.append(out)
            if need_grad:
                states.append((kind, state, len(state.saved_tensors)))
                tensors.extend(state.saved_tensors)
                state.saved_tensors = ()
            start += heads
        if need_grad:
            ctx.save_for_backward(q, k, v, *tensors)
            ctx.states = states
            ctx.heads = tuple(h for _, h in layout.head_groups)
        # A new output tensor also keeps the private graph intact for a
        # single-group layout (Function must not replace its inner grad_fn).
        return torch.cat(outputs, dim=1)

    @staticmethod
    @once_differentiable
    def backward(ctx, do):
        from .tg_kernels import _TG, join_gradients
        from .tgnomask_kernels import _FusedTyped
        saved = ctx.saved_tensors
        offset, grads = 3, []
        for (kind, state, count), upstream in zip(ctx.states, do.split(ctx.heads, dim=1)):
            state.saved_tensors = saved[offset:offset+count]
            if kind == 'tg':
                grad = _TG.backward(state, upstream)[:3]
            elif kind == 'tgnomask':
                grad = _FusedTyped.backward(state, upstream)[:3]
            else:
                grad = torch.autograd.grad(state.saved_tensors[-1], state.saved_tensors[:3], upstream, retain_graph=True)
            grads.append(grad)
            state.saved_tensors = ()
            offset += count
        return join_gradients(grads, saved[:3], ctx.heads) + (None, None)
