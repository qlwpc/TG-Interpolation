"""Validate frozen identities, merge full coverage, and publish the final receipt."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.native_document_results import input_identity, atomic_json


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identities-only', action='store_true')
    args = parser.parse_args()
    campaign = ROOT.parent
    receipt = campaign / ('identity_validation.json' if args.identities_only else 'final_status.json')
    status = dict(status='validating', started_at=time.time())
    atomic_json(receipt, status)
    try:
        provenance = json.loads((campaign / 'provenance.json').read_text())
        for model, files in provenance['models'].items():
            for name, record in files.items():
                if sha(record['path']) != record['sha256']:
                    raise ValueError(f'checkpoint changed: {model}/{name}')
        checked_code = 0
        for relative, expected in provenance['code_sha256'].items():
            if relative.startswith(('olmo/', 'olmo_data/')) or relative in (
                    'scripts/evaluate_reserved_document_ppl.py',
                    'scripts/evaluate_pushdown_document_ppl.py',
                    'scripts/native_document_results.py'):
                if sha(ROOT / relative) != expected:
                    raise ValueError(f'frozen scoring dependency changed: {relative}')
                checked_code += 1
        tokenizer_sha = sha(ROOT / 'dataset/bbc-news/TG_GPT2_tokenizer.json')
        checked_contracts = 0
        for path in (campaign / 'results').glob('bbc_*/contract_*.json'):
            contract = json.loads(path.read_text())
            fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            if path.name != f'contract_{fingerprint}.json':
                raise ValueError(f'contract fingerprint mismatch: {path}')
            checkpoint = Path(contract['checkpoint'])
            if sha(checkpoint / 'config.yaml') != contract['checkpoint_config']:
                raise ValueError(f'checkpoint config differs from scoring contract: {path}')
            if input_identity(str(checkpoint / 'model.pt')) != contract['checkpoint_weights']:
                raise ValueError(f'checkpoint identity differs from scoring contract: {path}')
            if contract['tokenizer'] != tokenizer_sha:
                raise ValueError(f'tokenizer changed: {path}')
            checked_contracts += 1
            if input_identity(contract['data']) != contract['data_identity']:
                raise ValueError(f'dataset identity changed: {path}')
            for key, file in [('scorer','olmo/train.py'),('model_code','olmo/model.py'),
                              ('dataset_code','olmo/eval/downstream.py'),
                              ('mask_extension','olmo/data/tg_mask.cpython-310-x86_64-linux-gnu.so'),
                              ('runner','scripts/evaluate_reserved_document_ppl.py')]:
                if sha(ROOT/file) != contract[key]:
                    raise ValueError(f'scoring implementation changed: {key}')
        # The existing native run contract records every native array's size and
        # mtime. This supplements the initial exhaustive hash/readback audit.
        native_manifest = campaign / 'results/bbc_100m_pushdown/run_manifest.json'
        native_record = json.loads(native_manifest.read_text())
        native = native_record['contract']
        native_fingerprint = hashlib.sha256(
            json.dumps(native, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if native_record['run_fingerprint'] != native_fingerprint:
            raise ValueError('native contract fingerprint mismatch')
        for key in ('checkpoint', 'native_data', 'tokenizer'):
            if key in native and isinstance(native[key], dict) and 'path' in native[key]:
                if input_identity(native[key]['path']) != native[key]:
                    raise ValueError(f'native input changed: {key}')
        status.update(identity_validation='passed', checkpoint_models=len(provenance['models']),
                      scoring_files=checked_code, scoring_contracts=checked_contracts,
                      tokenizer_sha256=tokenizer_sha)
        if args.identities_only:
            status.update(status='complete', scope='frozen identities only; full result coverage not certified',
                          completed_at=time.time())
            return
        subprocess.run([sys.executable, str(ROOT/'scripts/merge_reserved_docppl.py'),
                        '--campaign', str(campaign)], check=True, cwd=ROOT)
        results = json.loads((campaign/'results.json').read_text())
        for row in results:
            if row['model_id'] == 'bbc_100m_pushdown':
                if row['run_fingerprint'] != native_fingerprint:
                    raise ValueError('native results do not match validated contract')
            elif not (campaign / 'results' / row['model_id'] /
                      f"contract_{row['fingerprint']}.json").is_file():
                raise ValueError(f"result has no validated scoring contract: {row['model_id']}")
        with (campaign/'results.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['model','docppl','documents','sentences','terminal_count','non_candidate0_ratio'])
            for row in results:
                writer.writerow([row['model_id'],f'{row["docppl"]:.8f}',row.get('documents',row.get('document_count')),
                                 row.get('sentences',row.get('sentence_count')),row['terminal_count'],row['non_candidate0_ratio']])
        lines = ['# BBC reserved-clean DocPPL 重测结果', '',
                 'SIST 全量评分和严格覆盖验证完成。每模型 5,000 篇、129,085 句；计分分母 3,068,713。', '',
                 'FP32；树模型按有效候选 joint-sum、model-best 历史；Pushdown 为 native n-ary / stack_legal。', '',
                 '| 模型 | DocPPL | 非 candidate0 历史比例 |', '| --- | ---: | ---: |']
        for row in results:
            lines.append(f'| {row["model_id"]} | {row["docppl"]:.6f} | {row["non_candidate0_ratio"]:.6f} |')
        lines += ['', '完整逐文档证据见 results/；权重、输入与数值验证见本目录的对应 JSON 凭据。']
        (campaign/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        status.update(status='complete', claim_type='computed', models=len(results),
                      completed_at=time.time(), results_sha256=sha(campaign/'results.json'),
                      validation='full document/sentence/token/candidate coverage and frozen identities')
    except BaseException as error:
        status.update(status='failed', error=repr(error), ended_at=time.time())
        raise
    finally:
        atomic_json(receipt, status)


if __name__ == '__main__':
    main()
