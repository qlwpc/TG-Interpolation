"""Independent identity audits must never publish a full-coverage receipt."""
import hashlib
import json
from scripts import finalize_reserved_docppl as finalizer


def test_identity_receipts_are_separate_even_on_failure(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    (tmp_path / 'provenance.json').write_text(json.dumps({'models': {}, 'code_sha256': {}}))
    native = tmp_path / 'results/bbc_100m_pushdown'
    native.mkdir(parents=True)
    fingerprint = hashlib.sha256(b'{}').hexdigest()
    manifest = native / 'run_manifest.json'
    manifest.write_text(json.dumps({'contract': {}, 'run_fingerprint': fingerprint}))
    monkeypatch.setattr(finalizer, 'ROOT', root)
    monkeypatch.setattr(finalizer, 'sha', lambda path: 'test')
    monkeypatch.setattr(finalizer.sys, 'argv', ['finalize', '--identities-only'])
    finalizer.main()
    receipt = tmp_path / 'identity_validation.json'
    assert json.loads(receipt.read_text())['status'] == 'complete'
    assert not (tmp_path / 'final_status.json').exists()
    manifest.write_text(json.dumps({'contract': {}, 'run_fingerprint': 'invalid'}))
    import pytest
    with pytest.raises(ValueError, match='native contract fingerprint'):
        finalizer.main()
    assert json.loads(receipt.read_text())['status'] == 'failed'
    assert not (tmp_path / 'final_status.json').exists()
