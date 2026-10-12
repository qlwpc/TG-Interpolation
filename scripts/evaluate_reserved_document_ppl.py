"""Resumable complete-document runner using the current Trainer scoring path.

The physical candidate axis is kept on disk; only distinct projected valid
candidates are scored. Lossy projections require a complete input audit.
Sentence-wide padding makes context cropping independent of microbatch size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from olmo.config import TrainConfig
from olmo.data import get_TG_generate_bias_func
from olmo.eval.downstream import TGPerplexityApproximationDataset, TerminalDocumentPerplexityDataset
from olmo.model import OLMo
from olmo.train import Trainer
from scripts.native_document_results import input_identity
from scripts.projected_candidate_utils import deduplicate_samples


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


class LazyTreeDataset(TGPerplexityApproximationDataset):
    def prep_examples(self):
        # Avoid converting candidate0 of the entire corpus in every shard.
        self.sent_index = torch.cat((torch.zeros(1, dtype=torch.long), self.sent_index.cumsum(0)))
        self.doc_index = self.doc_index.cumsum(0)
        self.sent_doc_id = torch.repeat_interleave(
            torch.arange(1, len(self.document_ends) + 1),
            torch.from_numpy(self.document_ends - self.document_starts))
        self.term_len = []


class Capture:
    def update_metrics(self, batch, ce_loss, logits):
        self.losses.extend(ce_loss.detach().double().cpu().tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--format', choices=['terminal', 'tree', 'tg'], required=True)
    p.add_argument('--tokenizer', default='dataset/bbc-news/TG_GPT2_tokenizer.json')
    p.add_argument('--output', required=True)
    p.add_argument('--start-document', type=int, default=0)
    p.add_argument('--end-document', type=int)
    p.add_argument('--batch-size', type=int, default=60)
    p.add_argument('--max-batch-tokens', type=int, default=12000)
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--attention', choices=['config', 'sdpa'], default='config')
    p.add_argument('--document-ids', help='comma-separated complete documents for validation')
    p.add_argument('--projection-audit', type=Path, help='complete audit_projected_docppl output directory')
    args = p.parse_args()
    torch.set_num_threads(min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))))
    torch.manual_seed(6198)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = TrainConfig.load(Path(args.checkpoint) / 'config.yaml', validate_paths=False)
    cfg.model.transformer_grammar_type = cfg.model.transformer_grammar_type or 'terminal'
    grammar = cfg.model.transformer_grammar_type
    audit = None
    if grammar in ('tree_noont', 'tree_compress'):
        if args.projection_audit is None or args.format != 'tree':
            raise ValueError('NoONT/Compress require tree data and --projection-audit')
    if args.projection_audit:
        audit = json.loads((args.projection_audit / 'audit.json').read_text())
        if audit['status'] != 'complete' or audit['scope'] != 'full_clean_test':
            raise ValueError('projection audit is incomplete')
        if audit['tokenizer_sha256'] != digest(args.tokenizer):
            raise ValueError('tokenizer differs from projection audit')
        for code in ('scripts/projected_candidate_utils.py', 'olmo/data/tgmasking/mask.cpp',
                     'olmo/data/tg_mask.cpython-310-x86_64-linux-gnu.so'):
            if digest(ROOT / code) != audit['source_files'][code]:
                raise ValueError(f'projection code differs from audit: {code}')
        for name in ('unique_counts', 'first_indices', 'inverse'):
            relative = f'{grammar}/{name}.npy'
            if digest(args.projection_audit / relative) != audit['artifacts'][relative]:
                raise ValueError(f'audited mapping hash mismatch: {relative}')
        audited_counts = np.load(args.projection_audit / grammar / 'unique_counts.npy', mmap_mode='r')
        audited_first = np.load(args.projection_audit / grammar / 'first_indices.npy', mmap_mode='r')
        audited_inverse = np.load(args.projection_audit / grammar / 'inverse.npy', mmap_mode='r')
        audited_terms = np.load(args.projection_audit / 'sentence_terminal_counts.npy', mmap_mode='r')
    cfg.tokenizer.vocabulary = args.tokenizer
    model = OLMo.from_checkpoint(args.checkpoint, device='cuda')
    if args.attention == 'sdpa':
        model.config.flash_attention = False
        model.config.flex_attention = False
        for block in model.transformer.blocks:
            block.config.flash_attention = False
            block.config.flex_attention = False
            block.flash_attn_func = None
    bias = get_TG_generate_bias_func(cfg)
    kw = dict(tokenizer=None, dataset_path=args.data, vocab_path=args.tokenizer,
              model_ctx_len=cfg.model.max_sequence_length,
              transformer_grammar_type=cfg.model.transformer_grammar_type,
              pause_token_id=cfg.model.pause_token_id, generate_TG_attention_bias=bias)
    if args.format == 'terminal':
        ds = TerminalDocumentPerplexityDataset(**kw)
        counts = np.ones(len(ds), dtype=np.int64)
    else:
        ds = LazyTreeDataset(**kw, dataset_name=args.format, metric_type='doc')
        counts = np.load(Path(args.data) / 'valid_counts.npy')
        if len(counts) != len(ds) // 300 or np.any((counts < 1) | (counts > 300)):
            raise ValueError('invalid candidate counts')
    end = args.end_document if args.end_document is not None else len(ds.document_ends)
    docids = list(range(args.start_document, end))
    if args.document_ids:
        docids = [int(x) for x in args.document_ids.split(',')]
    if not docids or min(docids) < 0 or max(docids) >= len(ds.document_ends) or len(set(docids)) != len(docids):
        raise ValueError('invalid document selection')
    contract = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                    checkpoint_weights=input_identity(str(Path(args.checkpoint) / 'model.pt')),
                    checkpoint_config=digest(Path(args.checkpoint) / 'config.yaml'),
                    data=str(Path(args.data).resolve()), data_identity=input_identity(args.data), tokenizer=digest(args.tokenizer),
                    format=args.format, grammar=grammar, history='model_best',
                    aggregation='distinct_projected_candidate_joint_sum',
                    deduplication='exact_unpadded_token_ids_stable_first_occurrence',
                    projection_audit=digest(args.projection_audit / 'audit.json') if audit else None,
                    deduplication_code=digest(ROOT / 'scripts/projected_candidate_utils.py'),
                    padding='sentence_wide', precision=args.precision, attention=args.attention,
                    context=cfg.model.max_sequence_length,
                    scorer=digest(ROOT / 'olmo/train.py'), runner=digest(__file__),
                    model_code=digest(ROOT / 'olmo/model.py'), dataset_code=digest(ROOT / 'olmo/eval/downstream.py'),
                    mask_extension=digest(ROOT / 'olmo/data/tg_mask.cpython-310-x86_64-linux-gnu.so'))
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f'contract_{fingerprint}.json', contract)
    trainer = object.__new__(Trainer)
    trainer.device = torch.device('cuda')
    trainer.cfg = SimpleNamespace(model=cfg.model, autocast_precision=(torch.bfloat16 if args.precision == 'bf16' else torch.float32))
    trainer.dist_model = model
    trainer.cur_doc_id = None
    capture = Capture()
    capture.eval_loader = SimpleNamespace(dataset=SimpleNamespace(generate_TG_attention_bias=bias))
    physical = ds.SENT_SIZE
    started = time.time()
    for docid in docids:
        target = out / f'document_{docid:05d}.json'
        if target.exists():
            previous = json.loads(target.read_text())
            if previous['fingerprint'] != fingerprint or previous['document_id'] != docid:
                raise ValueError(f'incompatible saved document: {target}')
            continue
        # A resume may skip intervening docs, so reset explicitly at every doc.
        trainer.cur_doc_id = None
        rows = []
        for sent in range(int(ds.document_starts[docid]), int(ds.document_ends[docid])):
            valid = int(counts[sent])
            samples = [ds[sent * physical + k] for k in range(valid)]
            samples, first_indices, inverse = deduplicate_samples(samples)
            original_valid, valid = valid, len(samples)
            if audit and (valid != int(audited_counts[sent])
                          or not np.array_equal(first_indices, audited_first[sent, :valid])
                          or not np.array_equal(inverse, audited_inverse[sent, :original_valid])):
                raise ValueError(f'sentence {sent}: runtime projection differs from complete audit')
            lengths = [len(s['input_ids']) for s in samples]
            width = max(lengths)
            if width > cfg.model.max_sequence_length:
                raise ValueError(f'sentence {sent} exceeds context: {width}; refusing silent token loss')
            # Preserve projection / masking and each distinct real candidate.
            term_count = int(samples[0].get('term_count', sum(
                ds.vocab.is_terminal(t) or t == ds.vocab.eos for t in samples[0]['input_ids'])))
            if audit and term_count != int(audited_terms[sent]):
                raise ValueError(f'sentence {sent}: runtime denominator differs from audit')
            for s in samples:
                pad = width - len(s['input_ids'])
                s['input_ids'] = np.pad(s['input_ids'], (0, pad), constant_values=ds.vocab.pad)
                if 'label_mask' in s:
                    s['label_mask'] = np.pad(s['label_mask'], (0, pad), constant_values=False)
            capture.losses = []
            capture.eval_loader.dataset.SENT_SIZE = valid
            trainer.num_evaled = 0
            batch_size = min(args.batch_size, max(1, args.max_batch_tokens // width))
            for begin in range(0, valid, batch_size):
                batch = ds.collate_fn(samples[begin:begin + batch_size])
                batch['candidate_lengths'] = torch.tensor(lengths[begin:begin + batch_size])
                Trainer.TG_doc_eval_step(trainer, batch, capture)
            losses = np.asarray(capture.losses, dtype=np.float64)
            if len(losses) != valid or not np.isfinite(losses).all() or term_count <= 0:
                raise ValueError('incomplete/nonfinite sentence scoring')
            m = float(losses.min())
            nll = m - math.log(float(np.exp(m - losses).sum()))
            rows.append(dict(sentence_id=sent, nll=nll, terminal_count=term_count,
                             original_valid_candidates=original_valid, valid_candidates=valid,
                             removed_candidates=original_valid-valid,
                             selected_candidate=int(losses.argmin()),
                             selected_original_candidate=int(first_indices[losses.argmin()])))
        result = dict(document_id=docid, fingerprint=fingerprint, sentences=rows,
                      sentence_count=len(rows), terminal_count=sum(r['terminal_count'] for r in rows),
                      total_nll=math.fsum(r['nll'] for r in rows),
                      non_candidate0_count=sum(r['selected_candidate'] != 0 for r in rows),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                      job_id=os.environ.get('SLURM_JOB_ID'), elapsed_seconds=time.time()-started)
        atomic_json(target, result)
        print(json.dumps({k: result[k] for k in ('document_id','sentence_count','terminal_count','total_nll','elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
