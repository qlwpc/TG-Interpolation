"""Exact metadata and GPU forward/backward regressions for typed TGnomask."""
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from olmo.config import ModelConfig, BlockType, LayerNormType, ActivationType, InitFnType
from olmo.data.tg_mask import SentencepieceVocab, KProximal_TG_attention_bias
from olmo.exceptions import OLMoConfigurationError
from olmo.model import OLMo
from olmo.attention_kernels.tgnomask import build_tgnomask_layout, tgnomask_attention

ROOT = Path(__file__).resolve().parents[1]
VOCAB = str(ROOT / "dataset/bbc-news/TG_GPT2_tokenizer.json")
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA allocation")


@pytest.fixture(scope="module")
def vocab():
    return SentencepieceVocab.from_vocab_file(VOCAB)


def reference(tokens, augmented=False):
    masks, labels = [], []
    for row in tokens:
        gen = KProximal_TG_attention_bias(VOCAB, max(2048, row.numel()), max(2048, row.numel()), augmented)
        m, l = gen(row)
        masks.append(m); labels.append(l)
    return torch.stack(masks)[:, None], torch.stack(labels)


def real_tokens(n, start=0, batch=2):
    data = np.load(ROOT / "dataset/bbc-news/tg/test.npy", mmap_mode="r")
    return torch.stack([torch.from_numpy(np.array(data[start+i*2048:start+i*2048+n], dtype=np.int64)) for i in range(batch)])


@pytest.mark.parametrize("n,start", [(1,0),(2,79),(17,13),(127,0),(128,79),(129,13),(511,4096),(2048,8192)])
@pytest.mark.parametrize("augmented", [False, True])
def test_real_metadata(vocab, n, start, augmented):
    tokens = real_tokens(n,start)
    if n > 7:
        tokens[0,-7:] = vocab.pad
    layout = build_tgnomask_layout(tokens,vocab, augmented=augmented)
    mask, labels = reference(tokens, augmented)
    assert torch.equal(layout.dense_mask(),mask)
    assert torch.equal(layout.label_mask,labels)


@pytest.mark.parametrize("case", ["causal","pads","internal_pad","nested","repeated_close","unbalanced","wide"])
@pytest.mark.parametrize("augmented", [False, True])
def test_edge_metadata(vocab, case, augmented):
    op, cl, pad = vocab.opening_non_terminals[0], vocab.closing_non_terminals[0], vocab.pad
    examples = {"causal":[1,2,3,4],"pads":[pad]*17,
        "internal_pad":[op,1,pad,2,cl,cl,3,pad],
        "nested":[op,1,op+1,2,cl+1,cl+1,cl,cl],
        "repeated_close":[cl]*7+[1,op,2,cl,cl],
        "unbalanced":[cl,1,cl+1,cl+1,op,op+1,3,cl,cl],
        "wide":[op]+[1]*70+[cl,cl]}
    tokens = torch.tensor([examples[case]])
    layout = build_tgnomask_layout(tokens,vocab, augmented=augmented)
    mask,labels=reference(tokens, augmented)
    assert torch.equal(layout.dense_mask(),mask)
    assert torch.equal(layout.label_mask,labels)


@pytest.mark.parametrize("augmented", [False, True])
def test_random_unbalanced_metadata(vocab, augmented):
    rng = np.random.default_rng(101)
    choices = [1,2,3,vocab.pad,vocab.opening_non_terminals[0],vocab.opening_non_terminals[0]+1,
               vocab.closing_non_terminals[0],vocab.closing_non_terminals[0]+1,vocab.eos]
    for _ in range(20):
        tokens = torch.tensor(rng.choice(choices,size=(2,31)),dtype=torch.long)
        layout = build_tgnomask_layout(tokens,vocab, augmented=augmented)
        mask,labels=reference(tokens, augmented)
        assert torch.equal(layout.dense_mask(),mask)
    assert torch.equal(layout.label_mask,labels)


def test_inverse_indices(vocab):
    tokens=real_tokens(129,13)
    tokens[0,-7:]=vocab.pad
    layout=build_tgnomask_layout(tokens,vocab)
    for b in range(len(tokens)):
        for forward,inverse,count in ((layout.q_index,layout.q_inverse,layout.q_count),
                                       (layout.k_index,layout.k_inverse,layout.k_count)):
            positions=forward[b,:int(count[b])].long()
            assert torch.equal(inverse[b,positions],torch.arange(int(count[b]),dtype=torch.int32))
            excluded=torch.ones(tokens.shape[1],dtype=torch.bool)
            excluded[positions]=False
            assert (inverse[b,excluded]==-1).all()


def test_compose_scheduling_and_prefix_start(vocab):
    tokens=real_tokens(129,13)
    tokens[0,-7:]=vocab.pad
    layout=build_tgnomask_layout(tokens,vocab)
    assert layout.sparse_capacity == int(layout.sparse_count.max())
    for b in range(len(tokens)):
        si=layout.sparse_index[b,:int(layout.sparse_count[b])].long()
        degree=layout.offsets[b,si+1]-layout.offsets[b,si]
        assert (degree[1:]>=degree[:-1]).all()
        assert len(si.unique()) == len(si)
        prefix=layout.prefix[b,:int(layout.q_count[b])]
        for key in range(tokens.shape[1]):
            start=int(layout.q_start[b,key])
            assert (prefix[:start]<=key).all()
            assert (prefix[start:]>key).all()


def model_config(**overrides):
    kwargs = dict(d_model=64,n_heads=4,n_layers=2,mlp_hidden_size=128,
        vocab_size=50320,embedding_size=50320,max_sequence_length=2048,
        block_type=BlockType.sequential,layer_norm_type=LayerNormType.rms,
        activation_type=ActivationType.swiglu,rope=True,flash_attention=False,
        flex_attention=False,attention_dropout=0.,residual_dropout=0.,embedding_dropout=0.,
        init_device="cpu",init_fn=InitFnType.normal,init_std=0.02,
        transformer_grammar_type="tgnomask",weight_tying=True)
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


@pytest.mark.parametrize("extra", ["bias","padding_mask","cache","aug","dropout","alibi","llama"])
def test_model_rejects_ambiguous_layout(vocab,extra):
    overrides={"transformer_grammar_type":"tgnomaskaug"} if extra=="aug" else {"attention_dropout":.1} if extra=="dropout" else {"alibi":True,"rope":False} if extra=="alibi" else {"block_type":BlockType.llama} if extra=="llama" else {}
    tokens=torch.tensor([[1,2,3]])
    layout=build_tgnomask_layout(tokens,vocab)
    kwargs={"attention_bias":layout.dense_mask()} if extra=="bias" else {"attention_mask":torch.ones_like(tokens)} if extra=="padding_mask" else {"use_cache":True} if extra=="cache" else {}
    with pytest.raises(OLMoConfigurationError):
        model=OLMo(model_config(**overrides),init_params=False)
        model(tokens,tgnomask_layout=layout,**kwargs)


@GPU
@pytest.mark.parametrize("kind", ["causal","causal_single","pads","self_only","mixed","wide"])
@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("augmented", [False, True])
def test_gpu_edge_gradients(vocab,kind,dtype, augmented):
    op,cl,pad=vocab.opening_non_terminals[0],vocab.closing_non_terminals[0],vocab.pad
    cases={"causal":[1,2,3,4],"causal_single":[1],"pads":[pad]*17,"self_only":[cl],
           "mixed":[op,1,op+1,2,cl+1,cl+1,cl,cl,pad],
           "wide":[op]+[1]*70+[cl,cl]}
    tokens=torch.tensor([cases[kind]])
    layout=build_tgnomask_layout(tokens,vocab,device="cuda", augmented=augmented)
    mask,_=reference(tokens, augmented)
    torch.manual_seed(92)
    inputs=[torch.randn(1,2,tokens.numel(),32,device="cuda",dtype=dtype,requires_grad=True) for _ in range(3)]
    refs=[x.detach().clone().requires_grad_() for x in inputs]
    upstream=torch.randn_like(inputs[0])
    with sdpa_kernel(SDPBackend.MATH):
        expected=F.scaled_dot_product_attention(*refs,attn_mask=mask.cuda())
    actual=tgnomask_attention(*inputs,layout)
    actual.backward(upstream);expected.backward(upstream)
    if kind in ("causal_single", "pads", "self_only"):
        assert torch.count_nonzero(inputs[0].grad) == 0
        assert torch.count_nonzero(inputs[1].grad) == 0
    atol,rtol=(3e-5,3e-4) if dtype==torch.float32 else (.03,.03)
    for a,b in [(actual,expected)]+[(x.grad,r.grad) for x,r in zip(inputs,refs)]:
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


@GPU
@pytest.mark.parametrize("dtype", [torch.float32,torch.bfloat16])
@pytest.mark.parametrize("group_size", [1,2])
@pytest.mark.parametrize("augmented", [False, True])
def test_full_model_logits_loss_gradients(vocab,dtype,group_size, augmented):
    torch.manual_seed(203)
    model=OLMo(model_config(block_group_size=group_size, transformer_grammar_type="tgnomaskaug" if augmented else "tgnomask")).cuda().to(dtype).train()
    tokens=real_tokens(127,13).cuda()
    tokens[0,-5:]=vocab.pad
    mask,label=reference(tokens.cpu(), augmented)
    layout=build_tgnomask_layout(tokens,vocab,device="cuda", augmented=augmented)
    target=tokens[:,1:].reshape(-1)
    def run(typed):
        model.zero_grad(set_to_none=True)
        output=model(tokens,**({"tgnomask_layout":layout} if typed else {"attention_bias":mask.cuda()})).logits
        loss_values=F.cross_entropy(output[:,:-1].float().reshape(-1,output.shape[-1]),target,reduction="none")
        keep=label[:,1:].reshape(-1).cuda() & (target!=vocab.pad)
        loss=loss_values[keep].mean()
        loss.backward()
        return output.detach().float(),loss.detach(),{name:p.grad.detach().float().clone() for name,p in model.named_parameters() if p.grad is not None}
    expected,ref_loss,ref_grads=run(False)
    actual,loss,grads=run(True)
    atol,rtol=(2e-5,5e-4) if dtype==torch.float32 else (.015,.04)
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    torch.testing.assert_close(loss,ref_loss,atol=atol,rtol=rtol)
    assert grads.keys()==ref_grads.keys()
    squared_error=squared_norm=0.
    for name,g in grads.items():
        assert g.isfinite().all(),name
        squared_error+=float((g-ref_grads[name]).square().sum())
        squared_norm+=float(ref_grads[name].square().sum())
        torch.testing.assert_close(g,ref_grads[name],atol=atol,rtol=rtol,msg=name)
    assert (squared_error/max(squared_norm,1e-20))**.5 < (0.012 if dtype==torch.bfloat16 else 0.0002)


@GPU
@pytest.mark.parametrize("stride_kind", ["qkv_projection", "feature_stride", "broadcast_grad"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("augmented", [False, True])
def test_fused_strides_and_gradient_accumulation(vocab,stride_kind,dtype, augmented):
    """Exercise real fused-QKV views, unusual strides, dO broadcasts and accumulation."""
    tokens=real_tokens(129,79)
    tokens[0,-7:]=vocab.pad
    layout=build_tgnomask_layout(tokens,vocab,device="cuda", augmented=augmented)
    mask,_=reference(tokens, augmented)
    b,n=tokens.shape
    h,d=2,32
    torch.manual_seed(419)
    if stride_kind=="qkv_projection":
        storage=torch.randn(b,n,3,h,d,device="cuda",dtype=dtype)
        inputs=[storage[:,:,i].transpose(1,2).detach().requires_grad_() for i in range(3)]
    elif stride_kind=="feature_stride":
        inputs=[torch.randn(b,h,n,d*2,device="cuda",dtype=dtype)[...,::2].detach().requires_grad_() for _ in range(3)]
    else:
        inputs=[torch.randn(b,h,n,d,device="cuda",dtype=dtype,requires_grad=True) for _ in range(3)]
    refs=[x.detach().contiguous().requires_grad_() for x in inputs]
    if stride_kind=="broadcast_grad":
        upstream=torch.randn(b,h,1,d,device="cuda",dtype=dtype).expand(b,h,n,d)
    else:
        upstream=torch.randn(b,n,h,d,device="cuda",dtype=dtype).transpose(1,2)
    for _ in range(2):
        actual=tgnomask_attention(*inputs,layout)
        with sdpa_kernel(SDPBackend.MATH):
            expected=F.scaled_dot_product_attention(*refs,attn_mask=mask.cuda())
        actual.backward(upstream);expected.backward(upstream)
    tol=0.012 if dtype==torch.bfloat16 else .0002
    for a,r in [(actual,expected)]+[(x.grad,y.grad) for x,y in zip(inputs,refs)]:
        assert a.isfinite().all()
        assert float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12))<tol


@GPU
@pytest.mark.parametrize("augmented", [False, True])
def test_fused_unwritten_tails_are_never_read(vocab,monkeypatch, augmented):
    # Fused packing leaves unused rows unwritten. Poison every work buffer to
    # ensure zero-count batches and rounded tile tails cannot leak those rows.
    import olmo.attention_kernels.kernels as kernels
    monkeypatch.setattr(kernels,"_empty_qkv",lambda q,number:[
        torch.full(q.shape,float("nan"),device=q.device,dtype=q.dtype) for _ in range(number)])
    tokens=real_tokens(129,13)
    tokens[0,:]=vocab.pad
    layout=build_tgnomask_layout(tokens,vocab,device="cuda", augmented=augmented)
    mask,_=reference(tokens, augmented)
    torch.manual_seed(602)
    inputs=[torch.randn(2,2,129,80,device="cuda",dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    refs=[x.detach().clone().requires_grad_() for x in inputs]
    upstream=torch.randn_like(inputs[0])
    actual=tgnomask_attention(*inputs,layout)
    with sdpa_kernel(SDPBackend.MATH):
        expected=F.scaled_dot_product_attention(*refs,attn_mask=mask.cuda())
    actual.backward(upstream);expected.backward(upstream)
    for a,r in [(actual,expected)]+[(x.grad,y.grad) for x,y in zip(inputs,refs)]:
        assert a.isfinite().all()
        assert float((a.float()-r.float()).norm()/r.float().norm().clamp_min(1e-12))<.012
