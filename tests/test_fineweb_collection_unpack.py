import hashlib
import json

import numpy as np
import pytest

from scripts.campaigns.fineweb_collection_unpack import unpack
from scripts.campaigns.fineweb_paired_collection import write_npz


def collection(tmp_path):
    root = tmp_path/'collection'
    (root/'train').mkdir(parents=True)
    (root/'export_receipts').mkdir()
    arrays = {'terminal': np.array([11, 12, 151643], dtype='<u4'),
              'tree': np.array([151900, 11, 12, 151901, 151643], dtype='<u4')}
    archive = root/'train'/'001-0000.npz'
    write_npz(archive, arrays)
    entry = {'file': 'train/001-0000.npz', 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    for key, a in arrays.items():
        entry[key+'_tokens'] = len(a)
        entry[key+'_token_bytes_sha256'] = hashlib.sha256(a.tobytes()).hexdigest()
    (root/'receipt.json').write_text(json.dumps({'status': 'complete', 'source_shards': [1]}))
    (root/'export_receipts'/'001.json').write_text(json.dumps({'files': [entry]}))
    return root, arrays, archive


def test_materialization_preserves_both_raw_byte_streams(tmp_path):
    root, arrays, _ = collection(tmp_path)
    output = tmp_path/'raw'
    result = unpack(root, output, reserve=0)
    assert result['status'] == 'complete'
    assert result['raw_bytes'] == 32
    for key, a in arrays.items():
        raw = output/key/'001-0000.npy'
        assert raw.read_bytes() == a.tobytes()
        assert np.array_equal(np.memmap(raw, dtype='<u4', mode='r'), a)


def test_corrupted_compressed_input_is_rejected(tmp_path):
    root, _, archive = collection(tmp_path)
    archive.write_bytes(archive.read_bytes()+b'corruption')
    with pytest.raises(ValueError, match='identity'):
        unpack(root, tmp_path/'raw', reserve=0)
