#!/usr/bin/env python3
"""Finalize, independently audit and publish the complete reserved-clean test top300."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from datatools.parse_test_docppl_data.generate_native_topk import _json_dump, _sha256_path


def main():
    root = REPO / 'dataset/bbc-news-reserved-clean-v1'
    output = root / 'testppl'
    output.mkdir(exist_ok=True)
    state_path = output / 'run_status.json'
    state = dict(status='finalizing', started_at=time.time(), hostname=socket.gethostname())
    _json_dump(state_path, state)
    canonical = root / 'canonical/test'
    tokenizer = REPO / 'dataset/bbc-news/TG_GPT2_tokenizer.json'
    native = root / 'native_model_topk_300_v2/test'
    tree, tg = output / 'tree300', output / 'tg300'
    base_hash = _sha256_path(root / 'manifest.json')
    commands = [
        ['datatools/parse_test_docppl_data/generate_native_topk.py', 'finalize',
         '--ppl-dir', canonical, '--test-tree', root / 'tree/test.npy', '--tokenizer', tokenizer,
         '--output', native, '--num-shards', '8'],
        ['datatools/reserved_clean/audit_native.py', '--dataset', root, '--split', 'test',
         '--canonical', canonical, '--native', native, '--tokenizer', tokenizer,
         '--output', output / 'native_validation.json'],
        ['datatools/parse_test_docppl_data/generate_labeled_topk.py', 'finalize'],
        ['datatools/parse_test_docppl_data/audit_labeled_topk.py', '--input', tree,
         '--output', output / 'tree300_validation.json'],
        ['datatools/parse_test_docppl_data/tree_to_tg.py', '--input-dir', tree, '--output-dir', tg,
         '--tokenizer', tokenizer, '--overwrite'],
        ['datatools/parse_test_docppl_data/tree_to_tg.py', '--input-dir', tree, '--output-dir', tg,
         '--tokenizer', tokenizer, '--validate-only'],
    ]
    try:
        for i, command in enumerate(commands):
            state.update(step=i, command=list(map(str, command)), updated_at=time.time())
            _json_dump(state_path, state)
            print('RUN', ' '.join(map(str, command)), flush=True)
            subprocess.run([sys.executable, *map(str, command)], cwd=REPO, check=True)
        for name in ('valid_counts.npy', 'proposal_scores.npy'):
            shutil.copyfile(tree / name, tg / name)
        tg_manifest = json.loads((tg / 'manifest.json').read_text())
        tg_manifest.update(valid_counts='valid_counts.npy', proposal_scores='proposal_scores.npy',
                           candidate_axis='identical to labeled Tree top300',
                           metadata_sha256={name: _sha256_path(tg / name)
                                            for name in ('valid_counts.npy', 'proposal_scores.npy')})
        _json_dump(tg / 'manifest.json', tg_manifest)
        native_audit = json.loads((output / 'native_validation.json').read_text())
        tree_audit = json.loads((output / 'tree300_validation.json').read_text())
        if (native_audit['documents'] != 5000 or native_audit['sentences'] != 129085 or
                tree_audit['sentences'] != 129085 or not native_audit['complete'] or
                not tree_audit['complete'] or not tg_manifest['exact']):
            raise ValueError('final output coverage/validation differs from frozen test')
        if _sha256_path(root / 'manifest.json') != base_hash:
            raise ValueError('base dataset manifest changed during finalization')
        result = dict(status='complete', document_count=5000, sentence_count=129085, physical_slots=300,
            base_manifest_sha256=base_hash, hostname=socket.gethostname(), completed_at=time.time(),
            tree='tree300', tg='tg300', native='../native_model_topk_300_v2/test',
            candidate_spaces={'tree_tg': 'exact labeled canonical CKY top300',
                              'gpst': 'native strict-binary top300 with fixed per-word BPE subtrees',
                              'pushdown': 'native unary-free n-ary top300'},
            shared_frozen_terminals=True, shared_document_sentence_boundaries=True,
            candidate_axes_independent_except_tree_tg=True, model_evaluation_run=False,
            validation_sha256={name: _sha256_path(output / name) for name in
                               ('native_validation.json', 'tree300_validation.json')})
        _json_dump(output / 'manifest.json', result)
        state.update(status='complete', completed_at=time.time())
    except BaseException as error:
        state.update(status='failed', error=repr(error), updated_at=time.time())
        raise
    finally:
        _json_dump(state_path, state)


if __name__ == '__main__':
    main()
