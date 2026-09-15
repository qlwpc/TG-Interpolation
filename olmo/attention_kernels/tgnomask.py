"""Typed, full-context, non-augmented TGnomask attention.

Metadata is prepared once on CPU and shared across layers. It reproduces the
fresh-state C++ KProximal_TG_attention_bias operator (including its treatment of
truncated trees and repeated closing tokens). Incremental/cache use is excluded.
"""
from dataclasses import dataclass, fields
from bisect import bisect_right
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class TGNoMaskLayout:
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

    @property
    def shape(self):
        return tuple(self.q_index.shape)

    @property
    def device(self):
        return self.q_index.device

    def to(self, device):
        if self.device == torch.device(device):
            return self
        return TGNoMaskLayout(**{f.name: (getattr(self, f.name).to(device)
            if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name))
            for f in fields(self)})

    def dense_mask(self):
        """Diagnostic reconstruction; never used by the attention kernels."""
        layout = self.to("cpu")
        b, n = layout.shape
        mask = torch.zeros(b, 1, n, n, dtype=torch.bool)
        for batch in range(b):
            for i in range(int(layout.q_count[batch])):
                query = int(layout.q_index[batch, i])
                keys = layout.k_index[batch, :int(layout.prefix[batch, i])].long()
                mask[batch, 0, query, keys] = True
                mask[batch, 0, query, query] = True
            for i in range(int(layout.sparse_count[batch])):
                query = int(layout.sparse_index[batch, i])
                lo, hi = map(int, layout.offsets[batch, query:query + 2])
                mask[batch, 0, query, layout.edges[batch, lo:hi].long()] = True
        return mask.to(self.device)


def build_tgnomask_layout(input_ids: torch.Tensor, vocab: Any,
                         device: Optional[torch.device] = None) -> TGNoMaskLayout:
    """Build O(B*N) metadata from CPU tokens and a SentencepieceVocab-like object.

The caller should use the same vocabulary as its C++ mask generator. GPU input
is accepted but copied to CPU; prepare this alongside CPU data loading instead.
The input is a fresh full-context segment, not a continuation of parser state.
"""
    ids = input_ids.detach().cpu()
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[1] == 0 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids must be a nonempty 1-D or 2-D integer tensor")
    b, n = ids.shape
    oi, oe = vocab.opening_non_terminals
    ci, ce = vocab.closing_non_terminals
    values = {name: torch.zeros((b, n), dtype=torch.int32) for name in
              ("q_index", "k_index", "prefix", "sparse_index", "q_start")}
    values.update(q_count=torch.zeros(b, dtype=torch.int32), k_count=torch.zeros(b, dtype=torch.int32),
                  sparse_count=torch.zeros(b, dtype=torch.int32),
                  self_mask=torch.zeros(b, n, dtype=torch.bool),
                  offsets=torch.zeros(b, n + 1, dtype=torch.int32),
                  edges=torch.zeros(b, 2 * n, dtype=torch.int32),
                  reverse=torch.full((b, n, 2), -1, dtype=torch.int32),
                  q_inverse=torch.full((b, n), -1, dtype=torch.int32),
                  k_inverse=torch.full((b, n), -1, dtype=torch.int32),
                  label_mask=torch.ones(b, n, dtype=torch.bool))
    for batch, tokens in enumerate(ids.tolist()):
        stack, last, valid_keys, ordinary, sparse, edge_values = [], -1, [], [], [], []
        reverse_rows = [[] for _ in tokens]
        prefixes, exceptions = [], []
        offsets = [0]
        for i, token in enumerate(tokens):
            closing = ci <= token < ce
            compose = closing and last != token and last != -1
            if closing and last == -1:
                end = i
                while end < n and tokens[end] == token:
                    end += 1
                compose = (end - i) % 2 == 0
            allowed = []
            if token == vocab.pad:
                sparse.append(i)
                allowed = [i]
            elif compose:
                popped, j = [], i
                while stack and not oi <= tokens[j] < oe:
                    j = stack.pop()
                    popped.append(j)
                stack.append(i)
                valid_keys.append(i)
                sparse.append(i)
                allowed = popped[::-1] + [i]
                last = token
            else:
                if not closing:
                    stack.append(i)
                    valid_keys.append(i)
                else:
                    values["label_mask"][batch, i] = False
                ordinary.append(i)
                prefixes.append(len(valid_keys))
                exceptions.append(closing)
                last = token
            for key in allowed:
                # The reverse list stores query positions, not edge offsets.
                reverse_rows[key].append(i)
                if len(reverse_rows[key]) > 2:
                    raise AssertionError("A key cannot have more than two compose/self consumers")
            edge_values.extend(allowed)
            offsets.append(len(edge_values))
        assert len(edge_values) <= 2 * n
        # Nearby compose rows have similar work; wide nodes retain full edges.
        sparse.sort(key=lambda i: offsets[i + 1] - offsets[i])
        values["q_start"][batch] = torch.tensor(
            [bisect_right(prefixes, key) for key in range(n)], dtype=torch.int32)
        for name, array in (("q_index", ordinary), ("k_index", valid_keys),
                            ("sparse_index", sparse), ("prefix", prefixes), ("edges", edge_values)):
            values[name][batch, :len(array)] = torch.tensor(array, dtype=torch.int32)
        values["self_mask"][batch, :len(exceptions)] = torch.tensor(exceptions, dtype=torch.bool)
        values["offsets"][batch] = torch.tensor(offsets, dtype=torch.int32)
        for key, consumers in enumerate(reverse_rows):
            if consumers:
                values["reverse"][batch, key, :len(consumers)] = torch.tensor(consumers, dtype=torch.int32)
        values["q_count"][batch], values["k_count"][batch] = len(ordinary), len(valid_keys)
        values["sparse_count"][batch] = len(sparse)
        if ordinary:
            values["q_inverse"][batch, torch.tensor(ordinary)] = torch.arange(len(ordinary), dtype=torch.int32)
        if valid_keys:
            values["k_inverse"][batch, torch.tensor(valid_keys)] = torch.arange(len(valid_keys), dtype=torch.int32)
    values["sparse_capacity"] = int(values["sparse_count"].max())
    return TGNoMaskLayout(**values).to(device or "cpu")


def tgnomask_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      layout: TGNoMaskLayout) -> torch.Tensor:
    """Exact TGnomask visibility, Flash-style prefix + compose kernels, with backward.

Q/K/V: (B,H,N,D), CUDA fp16/bf16/fp32, equal heads, 16 <= D <= 128.
Floating-point reductions are not bit-identical to dense SDPA.
"""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Typed TGnomask requires equal (B,H,N,D) Q/K/V shapes")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("Typed TGnomask requires Q/K/V on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Typed TGnomask supports matching fp16/bf16/fp32 Q/K/V")
    if not 16 <= q.shape[-1] <= 128:
        raise ValueError("Typed TGnomask supports head dimensions 16 through 128")
    if layout.shape != (q.shape[0], q.shape[2]) or layout.device != q.device:
        raise ValueError("Layout shape/device must match Q/K/V; call layout.to(q.device)")
    from .tgnomask_kernels import typed_attention
    return typed_attention(q, k, v, layout)
