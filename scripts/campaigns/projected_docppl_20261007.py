"""SIST campaign: audited sentence-projection deduplication for three Doc-PPLs.

All remote invocations, including preflight/finalization, run inside sbatch.
Workers claim disjoint complete-document shards; each GPU loads a full model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.merge_reserved_docppl import merge_documents
from scripts.native_document_results import input_identity


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def preflight(campaign, models):
    import torch
    import yaml
    audit = json.loads((campaign/'audit/audit.json').read_text())
    if audit['status'] != 'complete':
        raise ValueError('audit incomplete')
    data = ROOT/'dataset/bbc-news-reserved-clean-v1'
    for name, expected in audit['data_files'].items():
        if sha(data/name) != expected:
            raise ValueError(f'SIST data SHA256 differs: {name}')
    for name, expected in audit['artifacts'].items():
        if sha(campaign/'audit'/name) != expected:
            raise ValueError(f'SIST audit artifact SHA256 differs: {name}')
    tokenizer = ROOT/'dataset/bbc-news/TG_GPT2_tokenizer.json'
    if sha(tokenizer) != audit['tokenizer_sha256']:
        raise ValueError('SIST tokenizer SHA256 differs')
    for name, expected in json.loads((campaign/'provenance/source_manifest.json').read_text()).items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'SIST source SHA256 differs: {name}')
    checked = []
    for model in models:
        path = campaign/'checkpoints'/model['checkpoint']
        if sha(path/'model.pt') != model['weights_sha256']:
            raise ValueError(f'checkpoint weights differ: {model["id"]}')
        if sha(path/'config.original.yaml') != model['original_config_sha256']:
            raise ValueError('original configuration differs')
        cfg = yaml.safe_load((path/'config.yaml').read_text())
        original = yaml.safe_load((path/'config.original.yaml').read_text())
        original['model']['transformer_grammar_type'] = model['grammar']
        if cfg != original:
            raise ValueError('unexpected checkpoint config edit')
        checked.append(dict(model_id=model['id'], weights_sha256=model['weights_sha256'],
                            config_sha256=sha(path/'config.yaml'), weights_identity=input_identity(str(path/'model.pt'))))
    if not torch.cuda.is_available():
        raise ValueError('GPU unavailable inside allocated job')
    # Initialize CUDA and load the runtime extension on the actual batch node.
    from olmo.data.tg_mask import SentencepieceVocab
    from olmo.model import OLMo
    from olmo.train import Trainer
    from olmo.eval.downstream import TGPerplexityApproximationDataset
    SentencepieceVocab.from_vocab_file(str(tokenizer))
    torch.ones(4, device='cuda').sum().item()
    env = dict(host=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], python=sys.version,
        executable=sys.executable, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
        extension_ldd=subprocess.check_output(['ldd', str(ROOT/'olmo/data/tg_mask.cpython-310-x86_64-linux-gnu.so')], text=True),
        loaded_libraries=Path('/proc/self/maps').read_text())
    save(campaign/'provenance'/f'environment_{os.environ["SLURM_JOB_ID"]}.json', env)
    report = dict(status='complete', audit_sha256=sha(campaign/'audit/audit.json'),
                  source_manifest_sha256=sha(campaign/'provenance/source_manifest.json'),
                  data_identity=input_identity(str(data/'testppl/tree300')), checkpoints=checked,
                  job_id=os.environ['SLURM_JOB_ID'])
    save(campaign/'validation/input_validation.json', report)
    print(json.dumps(report), flush=True)


def validate_gate(campaign):
    gate = json.loads((campaign/'validation/input_validation.json').read_text())
    if gate['status'] != 'complete' or gate['audit_sha256'] != sha(campaign/'audit/audit.json'):
        raise ValueError('input validation is missing or stale')
    if gate['source_manifest_sha256'] != sha(campaign/'provenance/source_manifest.json'):
        raise ValueError('source identity changed after validation')
    if gate['data_identity'] != input_identity(str(ROOT/'dataset/bbc-news-reserved-clean-v1/testppl/tree300')):
        raise ValueError('input files changed after validation')
    for row in gate['checkpoints']:
        if row['weights_identity'] != input_identity(row['weights_identity']['path']):
            raise ValueError('checkpoint changed after validation')
        if sha(Path(row['weights_identity']['path']).with_name('config.yaml')) != row['config_sha256']:
            raise ValueError('checkpoint config changed after validation')
    for name, expected in json.loads((campaign/'provenance/source_manifest.json').read_text()).items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'source changed after validation: {name}')
    return gate


def command(campaign, model, output, batch_size):
    return [sys.executable, str(ROOT/'scripts/evaluate_reserved_document_ppl.py'),
        '--checkpoint', str(campaign/'checkpoints'/model['checkpoint']),
        '--data', str(ROOT/'dataset/bbc-news-reserved-clean-v1/testppl/tree300'),
        '--format', 'tree', '--tokenizer', str(ROOT/'dataset/bbc-news/TG_GPT2_tokenizer.json'),
        '--output', str(output), '--precision', 'fp32', '--attention', 'sdpa',
        '--projection-audit', str(campaign/'audit'), '--batch-size', str(batch_size),
        '--max-batch-tokens', '4096']


def pilot(campaign, model):
    validate_gate(campaign)
    differences = []
    for batch in (1, 16):
        folder = campaign/'pilot'/model['id']/f'batch{batch}'
        subprocess.run(command(campaign, model, folder, batch)+['--document-ids', '0,1'], check=True)
    for doc in (0, 1):
        rows = [json.loads((campaign/'pilot'/model['id']/f'batch{batch}'/f'document_{doc:05d}.json').read_text()) for batch in (1, 16)]
        if rows[0]['fingerprint'] != rows[1]['fingerprint']:
            raise ValueError('microbatch changed scientific contract')
        for a, b in zip(rows[0]['sentences'], rows[1]['sentences']):
            for key in ('sentence_id', 'terminal_count', 'original_valid_candidates', 'valid_candidates',
                        'removed_candidates', 'selected_original_candidate'):
                if a[key] != b[key]:
                    raise ValueError(f'microbatch changes {key}: {a} / {b}')
            delta = abs(a['nll']-b['nll'])
            differences.append(delta)
            if delta > 1e-3:
                raise ValueError(f'microbatch NLL differs by {delta}')
    # Exercise maximum sentence width, maximum unique K, and a long cache history.
    data = ROOT/'dataset/bbc-news-reserved-clean-v1/testppl/tree300'
    lengths = np.load(data/'tree_sent_index.npy', mmap_mode='r').reshape(-1, 300)
    sentence = int(np.argmax(lengths.max(axis=1)))
    docs = np.load(campaign/'audit/document_sentence_counts.npy')
    stress_doc = int(np.searchsorted(np.cumsum(docs), sentence, side='right'))
    subprocess.run(command(campaign, model, campaign/'pilot'/model['id']/'stress', 16)
                   +['--document-ids', str(stress_doc)], check=True)
    save(campaign/'validation'/f'pilot_{model["id"]}.json', dict(status='complete', model_id=model['id'],
         documents=[0, 1], stress_document=stress_doc, batches=[1, 16], sentence_count=len(differences),
         max_absolute_sentence_nll_difference=max(differences), selected_history_identical=True,
         input_validation_sha256=sha(campaign/'validation/input_validation.json'), job_id=os.environ['SLURM_JOB_ID']))


def finalize(campaign, models, require_complete=False):
    counts = np.load(campaign/'audit/document_sentence_counts.npy')
    tokens = np.load(campaign/'audit/document_terminal_counts.npy')
    original = np.load(campaign/'audit/original_counts.npy')
    terminal = np.load(campaign/'audit/sentence_terminal_counts.npy')
    report = []
    for model in models:
        folder = campaign/'results'/model['id']
        paths = sorted(folder.glob('document_*.json'))
        if len(paths) != len(counts):
            if require_complete:
                raise ValueError(f'{model["id"]}: only {len(paths)}/{len(counts)} documents')
            report.append(dict(model_id=model['id'], status='running', documents=len(paths), expected_documents=len(counts)))
            continue
        unique = np.load(campaign/'audit'/model['grammar']/'unique_counts.npy')
        first = np.load(campaign/'audit'/model['grammar']/'first_indices.npy', mmap_mode='r')
        row = merge_documents(folder, counts.tolist(), tokens.tolist(), unique)
        contract_path = folder/f'contract_{row["fingerprint"]}.json'
        contract = json.loads(contract_path.read_text())
        if hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() != row['fingerprint']:
            raise ValueError('result contract fingerprint mismatch')
        if (contract['aggregation'] != 'distinct_projected_candidate_joint_sum'
                or contract['grammar'] != model['grammar']
                or contract['projection_audit'] != sha(campaign/'audit/audit.json')):
            raise ValueError('incorrect post-projection result contract')
        for path in paths:
            for sentence in json.loads(path.read_text())['sentences']:
                sid = sentence['sentence_id']
                if (sentence['original_valid_candidates'] != int(original[sid])
                        or sentence['removed_candidates'] != int(original[sid]-unique[sid])
                        or sentence['terminal_count'] != int(terminal[sid])
                        or sentence['selected_original_candidate'] != int(first[sid, sentence['selected_candidate']])):
                    raise ValueError('sentence deduplication provenance mismatch')
        row.update(model_id=model['id'], weights_sha256=model['weights_sha256'],
                   original_candidates=int(original.sum()), unique_candidates=int(unique.sum()),
                   removed_candidates=int(original.sum(dtype=np.int64)-unique.sum(dtype=np.int64)))
        save(folder/'aggregate.json', row)
        report.append(row)
    complete = all(row['status'] == 'complete' for row in report)
    if complete:
        validate_gate(campaign)
        audit = json.loads((campaign/'audit/audit.json').read_text())
        for name, expected in audit['data_files'].items():
            if sha(ROOT/'dataset/bbc-news-reserved-clean-v1'/name) != expected:
                raise ValueError('final input content hash differs')
        for model in models:
            if sha(campaign/'checkpoints'/model['checkpoint']/'model.pt') != model['weights_sha256']:
                raise ValueError('final checkpoint content hash differs')
    save(campaign/('results.json' if complete else 'progress.json'), report)
    print(json.dumps(report), flush=True)


def worker(campaign, models, preferred):
    gate = validate_gate(campaign)
    for model in models:
        check = json.loads((campaign/'validation'/f'pilot_{model["id"]}.json').read_text())
        if check['status'] != 'complete' or check['input_validation_sha256'] != sha(campaign/'validation/input_validation.json'):
            raise ValueError('missing or stale model pilot')
    tasks = json.loads((campaign/'tasks.json').read_text())
    tasks.sort(key=lambda task: task['model_id'] != preferred)
    model_by_id = {model['id']: model for model in models}
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    def run_device(device):
        for task in tasks:
            claim = campaign/'claims'/task['id']
            try:
                claim.mkdir(parents=True)
            except FileExistsError:
                continue
            started = time.time()
            state = dict(task=task, job_id=os.environ['SLURM_JOB_ID'], host=socket.gethostname(),
                         device=device, status='running', started=started)
            save(claim/'status.json', state)
            model = model_by_id[task['model_id']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device,
                       TORCHINDUCTOR_CACHE_DIR=f'/tmp/projected_inductor_{os.environ["SLURM_JOB_ID"]}_{device}',
                       TRITON_CACHE_DIR=f'/tmp/projected_triton_{os.environ["SLURM_JOB_ID"]}_{device}')
            argv = command(campaign, model, campaign/'results'/model['id'], 16)
            argv += ['--start-document', str(task['start']), '--end-document', str(task['end'])]
            with open(campaign/'logs'/f'{task["id"]}_{os.environ["SLURM_JOB_ID"]}.log', 'w') as log:
                result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='complete' if result.returncode == 0 else 'failed',
                         returncode=result.returncode, elapsed_seconds=time.time()-started)
            save(claim/'status.json', state)
            if result.returncode:
                raise RuntimeError(f'shard {task["id"]} failed; see task log; claim retained for audited retry')
            print(json.dumps(state), flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        list(pool.map(run_device, devices))
    # Concurrent, idempotent merges validate complete coverage before publishing.
    finalize(campaign, models)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['preflight', 'pilot', 'worker', 'finalize'])
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--model')
    p.add_argument('--require-complete', action='store_true')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('SIST campaign must run inside sbatch')
    models = json.loads((args.campaign/'models.json').read_text())
    if args.stage == 'preflight':
        preflight(args.campaign, models)
    elif args.stage == 'pilot':
        pilot(args.campaign, next(model for model in models if model['id'] == args.model))
    elif args.stage == 'worker':
        worker(args.campaign, models, args.model)
    else:
        validate_gate(args.campaign)
        finalize(args.campaign, models, args.require_complete)


if __name__ == '__main__':
    main()
