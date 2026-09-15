"""CPU metadata for fresh TG, TGnomask/aug and mixed-head segments.

The native builder is the default. Python is an explicit semantic reference.
All layouts share packed storage, worker IPC and asynchronous transfer support.
"""

from dataclasses import dataclass, fields
from math import prod
from typing import Any, Sequence, Tuple

import numpy as np
import torch


class _PackedLayout:
    def _tensors(self):
        result = []
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, _PackedLayout):
                result.extend(value._tensors())
            elif isinstance(value, torch.Tensor):
                result.append(value)
        return result

    def _replace_tensors(self, tensors, buffers=()):
        return self._rebuild(iter(tensors), buffers)

    def _rebuild(self, tensors, buffers=()):
        # Use method recursion: a recursive local closure forms a reference
        # cycle and defers tensor destruction to GC. Forked CPU workers must
        # never inherit unreachable CUDA tensors from an earlier transfer.
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, _PackedLayout):
                value = value._rebuild(tensors)
            elif isinstance(value, torch.Tensor):
                value = next(tensors)
            elif f.name == "_buffers":
                value = buffers
            values[f.name] = value
        return type(self)(**values)

    def __reduce__(self):
        if not self._buffers:
            return (type(self), tuple(getattr(self, f.name) for f in fields(self)))
        base = self.base if isinstance(self, TGLayout) else self
        metadata = (base.sparse_capacity, base.augmented)
        if isinstance(self, TGLayout):
            metadata += (self.q_tile, self.k_tile)
        return (
            _restore_packed_layout,
            (type(self), self._buffers, tuple(t.shape for t in self._tensors()), metadata),
        )

    def pack(self):
        """Two shared storages reduce worker IPC, pinning and H2D launch overhead."""
        if self._buffers:
            return self
        tensors = self._tensors()
        buffers = tuple(
            torch.cat([t.reshape(-1) for t in tensors if t.dtype == dtype])
            for dtype in (torch.int32, torch.bool)
        )
        offsets = [0, 0]
        views = []
        for t in tensors:
            i = int(t.dtype == torch.bool)
            views.append(buffers[i][offsets[i] : offsets[i] + t.numel()].view(t.shape))
            offsets[i] += t.numel()
        return self._replace_tensors(views, buffers)

    def _map_buffers(self, fn):
        if not self._buffers:
            return self._replace_tensors([fn(t) for t in self._tensors()])
        buffers = tuple(fn(t) for t in self._buffers)
        return self._replace_tensors(
            [
                buffers[int(t.dtype == torch.bool)].as_strided(t.shape, t.stride(), t.storage_offset())
                for t in self._tensors()
            ],
            buffers,
        )

    def to(self, device, non_blocking=False):
        if self.device == torch.device(device):
            return self
        return self._map_buffers(lambda t: t.to(device, non_blocking=non_blocking))

    def slice_batch(self, start, end):
        return self._replace_tensors([t[start:end] for t in self._tensors()])

    def pin_memory(self):
        return self.pack()._map_buffers(lambda t: t.pin_memory())

    def record_stream(self, stream):
        for tensor in self._buffers or self._tensors():
            tensor.record_stream(stream)


@dataclass(frozen=True)
class TGNoMaskLayout(_PackedLayout):
    q_index: torch.Tensor
    k_index: torch.Tensor
    q_count: torch.Tensor
    k_count: torch.Tensor
    prefix: torch.Tensor
    self_mask: torch.Tensor
    sparse_index: torch.Tensor
    sparse_count: torch.Tensor
    offsets: torch.Tensor
    edges: torch.Tensor
    reverse: torch.Tensor
    label_mask: torch.Tensor
    q_inverse: torch.Tensor
    k_inverse: torch.Tensor
    q_start: torch.Tensor
    sparse_capacity: int
    augmented: bool = False
    _buffers: Tuple[torch.Tensor, ...] = ()

    @property
    def shape(self):
        return tuple(self.q_index.shape)

    @property
    def device(self):
        return self.q_index.device

    def dense_mask(self):
        """Diagnostic reconstruction; never used by the attention kernels."""
        layout = self.to("cpu")
        b, n = layout.shape
        mask = torch.zeros(b, 1, n, n, dtype=torch.bool)
        for batch in range(b):
            for i in range(int(layout.q_count[batch])):
                query = int(layout.q_index[batch, i])
                keys = layout.k_index[batch, : int(layout.prefix[batch, i])].long()
                mask[batch, 0, query, keys] = True
                mask[batch, 0, query, query] = True
            for i in range(int(layout.sparse_count[batch])):
                query = int(layout.sparse_index[batch, i])
                lo, hi = map(int, layout.offsets[batch, query : query + 2])
                mask[batch, 0, query, layout.edges[batch, lo:hi].long()] = True
        return mask.to(self.device)


@dataclass(frozen=True)
class TGLayout(_PackedLayout):
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
    _buffers: Tuple[torch.Tensor, ...] = ()

    @property
    def shape(self):
        return self.base.shape

    @property
    def device(self):
        return self.base.device

    @property
    def label_mask(self):
        return self.base.label_mask

    def dense_mask(self):
        """Debug/reference only; CUDA kernels never call this method."""
        cpu = self.to("cpu")
        b, n = cpu.shape
        out = torch.zeros(b, 1, n, n, dtype=torch.bool)
        for batch in range(b):
            qi = cpu.base.q_index[batch, : int(cpu.base.q_count[batch])].long()
            rank = torch.arange(len(qi))[:, None]
            out[batch, 0, qi] = (rank >= cpu.lo[batch]) & (rank < cpu.hi[batch])
            for query in cpu.base.sparse_index[batch, : int(cpu.base.sparse_count[batch])]:
                low, high = cpu.base.offsets[batch, query : query + 2]
                out[batch, 0, query, cpu.base.edges[batch, low:high].long()] = True
        return out.to(self.device)


def _restore_packed_layout(cls, buffers, shapes, metadata):
    """Rebuild tensor views after sending only two storage owners through IPC."""
    base_fields = [
        f.name for f in fields(TGNoMaskLayout) if f.name not in ("sparse_capacity", "augmented", "_buffers")
    ]
    tg_fields = [f.name for f in fields(TGLayout) if f.name not in ("base", "q_tile", "k_tile", "_buffers")]
    names = base_fields + (tg_fields if cls is TGLayout else [])
    offsets = [0, 0]
    tensors = {}
    for name, shape in zip(names, shapes):
        i = int(name in ("self_mask", "label_mask"))
        count = prod(shape)
        tensors[name] = buffers[i][offsets[i] : offsets[i] + count].view(shape)
        offsets[i] += count
    base = TGNoMaskLayout(
        **{name: tensors.pop(name) for name in base_fields},
        sparse_capacity=metadata[0],
        augmented=metadata[1],
        _buffers=buffers if cls is TGNoMaskLayout else (),
    )
    if cls is TGNoMaskLayout:
        return base
    return TGLayout(base=base, **tensors, q_tile=metadata[2], k_tile=metadata[3], _buffers=buffers)


def _padded(rows):
    out = np.zeros((len(rows), max(1, max(map(len, rows)))), dtype=np.int32)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return torch.from_numpy(out)


def _build_base(input_ids, vocab, augmented=False):
    """Same parser/actions as TGNoMaskLayout, using bulk NumPy tensor creation.

    Avoid thousands of individual torch tensor writes in dataloader workers.
    The resulting layout remains compatible with the existing TGnomask kernels.
    """
    ids = input_ids.detach().cpu()
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or not ids.shape[0] or not ids.shape[1] or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a nonempty 1-D or 2-D integer tensor")
    b, n = ids.shape
    op, oe = vocab.opening_non_terminals
    cl, ce = vocab.closing_non_terminals
    pad = vocab.pad
    a = {
        name: np.zeros((b, n), dtype=np.int32)
        for name in ("q_index", "k_index", "prefix", "sparse_index", "q_start")
    }
    a.update(
        q_count=np.zeros(b, dtype=np.int32),
        k_count=np.zeros(b, dtype=np.int32),
        sparse_count=np.zeros(b, dtype=np.int32),
        self_mask=np.zeros((b, n), dtype=bool),
        offsets=np.zeros((b, n + 1), dtype=np.int32),
        edges=np.zeros((b, 2 * n), dtype=np.int32),
        reverse=np.full((b, n, 2), -1, dtype=np.int32),
        label_mask=np.ones((b, n), dtype=bool),
        q_inverse=np.full((b, n), -1, dtype=np.int32),
        k_inverse=np.full((b, n), -1, dtype=np.int32),
    )
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
                compose = (end - i) % 2 == 0
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
                    a["label_mask"][batch, i] = False
                ordinary.append(i)
                prefixes.append(len(keys))
                specials.append(closing)
                last = token
            for key in allowed:
                reverse[key].append(i)
            edges.extend(allowed)
            offsets.append(len(edges))
        sparse.sort(key=lambda i: offsets[i + 1] - offsets[i])
        for name, values in (
            ("q_index", ordinary),
            ("k_index", keys),
            ("prefix", prefixes),
            ("sparse_index", sparse),
            ("edges", edges),
            ("self_mask", specials),
        ):
            a[name][batch, : len(values)] = values
        if augmented:
            # Augmented ordinary rows use every preceding position, including
            # padding and repeated closes. Compose rows retain their tree edges.
            keys = list(range(n))
            prefixes = [query + 1 for query in ordinary]
            a["k_index"][batch] = keys
            a["prefix"][batch, : len(prefixes)] = prefixes
            a["self_mask"][batch] = False
        a["offsets"][batch] = offsets
        a["q_start"][batch] = np.searchsorted(prefixes, np.arange(n), side="right")
        for key, consumers in enumerate(reverse):
            assert len(consumers) <= 2
            if consumers:
                a["reverse"][batch, key, : len(consumers)] = consumers
        a["q_count"][batch], a["k_count"][batch], a["sparse_count"][batch] = (
            len(ordinary),
            len(keys),
            len(sparse),
        )
        a["q_inverse"][batch, ordinary] = np.arange(len(ordinary), dtype=np.int32)
        a["k_inverse"][batch, keys] = np.arange(len(keys), dtype=np.int32)
    return TGNoMaskLayout(
        **{name: torch.from_numpy(values) for name, values in a.items()},
        sparse_capacity=int(a["sparse_count"].max()),
        augmented=augmented,
    )


def _build_tg_layout_python(
    input_ids: torch.Tensor, vocab: Any, device=None, q_tile: int = 32, k_tile: int = 16
) -> TGLayout:
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
        qi = base.q_index[batch, : int(base.q_count[batch])].numpy()
        death = np.full(n, n, dtype=np.int32)
        offsets, edges = base.offsets[batch].numpy(), base.edges[batch].numpy()
        for query in base.sparse_index[batch, : int(base.sparse_count[batch])].tolist():
            left, right = offsets[query : query + 2]
            popped = edges[left : right - 1]
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
            intervals = sorted(
                (int(lo[k]), int(hi[k])) for k in order[start : start + k_tile] if hi[k] > lo[k]
            )
            end = 0
            for left, right in intervals:
                queries.extend(range(max(left, end), right))
                end = max(end, right)
            backoff.append(len(queries))
        lows.append(lo)
        highs.append(hi)
        degrees.append(degree)
        orders.append(order)
        foffs.append(off)
        fkeys.append(keys)
        boffs.append(backoff)
        bqueries.append(queries)
    result = TGLayout(
        base,
        _padded(lows),
        _padded(highs),
        _padded(degrees),
        _padded(foffs),
        _padded(fkeys),
        _padded(boffs),
        _padded(bqueries),
        _padded(orders),
        q_tile,
        k_tile,
    )
    return result.to(device or "cpu")


def _build_layout(
    input_ids,
    vocab,
    *,
    include_tg,
    augmented=False,
    device=None,
    q_tile=32,
    k_tile=16,
    backend="native",
    pack=True,
):
    if backend not in ("auto", "native", "python"):
        raise ValueError("TG layout backend must be auto, native or python")
    if q_tile not in (16, 32, 64) or k_tile not in (16, 32, 64):
        raise ValueError("TG tile sizes must be 16, 32 or 64")
    if (
        input_ids.ndim not in (1, 2)
        or input_ids.numel() == 0
        or input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("input_ids must be a nonempty 1-D or 2-D integer tensor")
    from ._tg_layout_native import native_builder

    build = native_builder(backend == "native") if backend != "python" else None
    if build is None:
        result = (
            _build_tg_layout_python(input_ids, vocab, q_tile=q_tile, k_tile=k_tile)
            if include_tg
            else _build_base(input_ids, vocab, augmented=augmented)
        )
    else:
        ids = input_ids.detach().cpu().reshape(-1, input_ids.shape[-1]).numpy()
        arrays = build(
            ids,
            *vocab.opening_non_terminals,
            *vocab.closing_non_terminals,
            vocab.pad,
            q_tile,
            k_tile,
            include_tg,
            augmented,
        )
        capacity = arrays.pop("sparse_capacity")
        tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
        base = TGNoMaskLayout(
            **{
                f.name: tensors.pop(f.name)
                for f in fields(TGNoMaskLayout)
                if f.name not in ("sparse_capacity", "augmented", "_buffers")
            },
            sparse_capacity=capacity,
            augmented=augmented,
        )
        result = TGLayout(base=base, **tensors, q_tile=q_tile, k_tile=k_tile) if include_tg else base
    return (result.pack() if pack else result).to(device or "cpu")


def build_tg_layout(
    input_ids: torch.Tensor, vocab: Any, device=None, q_tile=32, k_tile=16, backend="native", pack=True
) -> TGLayout:
    """Build TG lifetimes and tile schedules once per CPU batch."""
    return _build_layout(
        input_ids,
        vocab,
        include_tg=True,
        device=device,
        q_tile=q_tile,
        k_tile=k_tile,
        backend=backend,
        pack=pack,
    )


def build_tgnomask_layout(
    input_ids: torch.Tensor, vocab: Any, device=None, *, augmented=False, backend="native", pack=True
) -> TGNoMaskLayout:
    """Build linear prefix/compose metadata, without TG's tile schedules."""
    return _build_layout(
        input_ids, vocab, include_tg=False, augmented=augmented, device=device, backend=backend, pack=pack
    )


@dataclass(frozen=True)
class MixTGLayout:
    tg: TGLayout
    head_groups: Tuple[Tuple[str, int], ...]
    aug: TGNoMaskLayout | None = None

    @property
    def shape(self):
        return self.tg.shape

    @property
    def device(self):
        return self.tg.device

    @property
    def label_mask(self):
        return (
            self.tg.base.label_mask
            if any(kind != "tgtree" for kind, _ in self.head_groups)
            else torch.ones_like(self.tg.base.label_mask)
        )

    def to(self, device, non_blocking=False):
        return MixTGLayout(
            self.tg.to(device, non_blocking=non_blocking),
            self.head_groups,
            self.aug.to(device, non_blocking=non_blocking) if self.aug is not None else None,
        )

    def record_stream(self, stream):
        self.tg.record_stream(stream)
        if self.aug is not None:
            self.aug.record_stream(stream)

    def slice_batch(self, start, end):
        return MixTGLayout(
            self.tg.slice_batch(start, end),
            self.head_groups,
            self.aug.slice_batch(start, end) if self.aug is not None else None,
        )

    def pin_memory(self):
        return MixTGLayout(
            self.tg.pin_memory(), self.head_groups, self.aug.pin_memory() if self.aug is not None else None
        )

    def dense_mask(self):
        b, n = self.shape
        masks = {"tg": self.tg.dense_mask(), "tgnomask": self.tg.base.dense_mask()}
        masks["tgtree"] = torch.ones((b, 1, n, n), device=self.device, dtype=torch.bool).tril()
        if self.aug is not None:
            masks["tgnomaskaug"] = self.aug.dense_mask()
        return torch.cat([masks[kind].expand(b, heads, n, n) for kind, heads in self.head_groups], 1)


def build_mix_tg_layout(input_ids, vocab, head_groups: Sequence, device=None, **kwargs):
    groups = tuple(
        (g.grammar_type, g.n_heads) if hasattr(g, "grammar_type") else tuple(g) for g in head_groups
    )
    if not groups or any(
        kind not in ("tg", "tgnomask", "tgnomaskaug", "tgtree") or not isinstance(h, int) or h <= 0
        for kind, h in groups
    ):
        raise ValueError("Mixed TG requires positive head groups of tg/tgnomask/tgnomaskaug/tgtree")
    aug = (
        build_tgnomask_layout(
            input_ids,
            vocab,
            device,
            augmented=True,
            **{k: v for k, v in kwargs.items() if k in ("backend", "pack")},
        )
        if any(kind == "tgnomaskaug" for kind, _ in groups)
        else None
    )
    return MixTGLayout(build_tg_layout(input_ids, vocab, device, **kwargs), groups, aug)
