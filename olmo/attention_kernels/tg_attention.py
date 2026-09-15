"""TG-family attention entry points; importing this module does not load Triton."""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from .layouts import (
    TGLayout,
    TGNoMaskLayout,
    MixTGLayout,
    build_tg_layout,
    build_tgnomask_layout,
    build_mix_tg_layout,
)


__all__ = [
    "TGLayout",
    "TGNoMaskLayout",
    "MixTGLayout",
    "build_tg_layout",
    "build_tgnomask_layout",
    "build_mix_tg_layout",
    "tg_attention",
    "tgnomask_attention",
    "mix_tg_attention",
]


def _validate(q, k, v, layout):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("TG requires matching (B,H,N,D) Q/K/V (MHA)")
    if not q.is_cuda or q.device != k.device or q.device != v.device:
        raise ValueError("TG requires Q/K/V on the same CUDA device")
    if (
        q.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("TG requires matching fp32/fp16/bf16 Q/K/V")
    if not 16 <= q.shape[-1] <= 128:
        raise ValueError("TG supports head dimensions 16 through 128")
    if layout.shape != (q.shape[0], q.shape[2]) or layout.device != q.device:
        raise ValueError("TG layout shape/device must match Q/K/V")


def tg_attention(q, k, v, layout: TGLayout):
    """Exact TG visibility with first-order backward; no dropout/cache/GQA."""
    _validate(q, k, v, layout)
    from .kernels import typed_tg_attention

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
    if kind in ("tgnomask", "tgnomaskaug"):
        return tgnomask_attention(*xs, layout.aug if kind == "tgnomaskaug" else layout.tg.base)
    return F.scaled_dot_product_attention(*xs, is_causal=True)


class _KernelState:
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class _MixedTG(torch.autograd.Function):
    """One autograd node for grouped kernels, with one fused gradient assembly."""

    @staticmethod
    def forward(ctx, q, k, v, layout, need_grad):
        from .kernels import _TG
        from .kernels import _FusedTyped

        outputs, tensors, states, start = [], [], [], 0
        for kind, heads in layout.head_groups:
            xs = [x[:, start : start + heads].detach() for x in (q, k, v)]
            state = _KernelState()
            if kind == "tg":
                out = _TG.forward(state, *xs, layout.tg)
            elif kind in ("tgnomask", "tgnomaskaug"):
                out = _FusedTyped.forward(state, *xs, layout.aug if kind == "tgnomaskaug" else layout.tg.base)
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
        from .kernels import _TG, join_gradients
        from .kernels import _FusedTyped

        saved = ctx.saved_tensors
        offset, grads = 3, []
        for (kind, state, count), upstream in zip(ctx.states, do.split(ctx.heads, dim=1)):
            state.saved_tensors = saved[offset : offset + count]
            if kind == "tg":
                grad = _TG.backward(state, upstream)[:3]
            elif kind in ("tgnomask", "tgnomaskaug"):
                grad = _FusedTyped.backward(state, upstream)[:3]
            else:
                grad = torch.autograd.grad(
                    state.saved_tensors[-1], state.saved_tensors[:3], upstream, retain_graph=True
                )
            grads.append(grad)
            state.saved_tensors = ()
            offset += count
        return join_gradients(grads, saved[:3], ctx.heads) + (None, None)


def tgnomask_attention(q, k, v, layout: TGNoMaskLayout):
    """Exact TGnomask/aug prefix and compose visibility, with first-order backward."""
    _validate(q, k, v, layout)
    from .kernels import typed_attention

    return typed_attention(q, k, v, layout)
