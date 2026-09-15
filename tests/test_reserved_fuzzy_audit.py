"""Compare the audit index against an exhaustive Python oracle."""
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer

from datatools.reserved_clean.common import normalize, sha_file
from diagnostics.audit_reserved_fuzzy import merge, run
from diagnostics.summarize_reserved_fuzzy import run as summarize


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    root = tmp_path_factory.mktemp("fuzzy")
    binary = root / "match"
    src = Path(__file__).resolve().parents[1] / "diagnostics/audit_reserved_fuzzy.cpp"
    subprocess.run(["g++", "-O2", "-std=c++17", str(src), "-o", str(binary)], check=True)
    return binary


def grams(words, n):
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def test_merge_accepts_long_reference_body(tmp_path):
    long_body = 'reference ' * 16000
    names = ['test_doc','exact','words','jaccard','jac_source','jac_body',
             'containment','cont_source','cont_body','local_pair','local_source','local_body','bitmap']
    values = ['0','0','1','0.1','0:0',long_body,'0','-','','0','-','','0']
    path = tmp_path / 'worker.tsv'
    path.write_text('\t'.join(names)+'\n'+'\t'.join(values)+'\n')
    rows = merge([path],[{'body':'target','candidate_id':0,'source_shard':'toy','source_row':0}])
    assert rows[0]['jac_body'] == long_body


def test_exhaustive_metrics_and_worker_union(binary, tmp_path):
    rng = random.Random(719)
    base = [f"w{i}" for i in range(110)]
    texts = ["a b", " ".join(base), " ".join(["same"] * 75)]
    refs = ["a b", " ".join(base[:25]), " ".join(base[45:75]),
            " ".join(["same"] * 60), " ".join(base[15:95])]
    for _ in range(12):
        w = base.copy()
        for i in rng.sample(range(len(w)), 18):
            w[i] = "changed" + str(i)
        texts.append(" ".join(w))
    refs += texts[4:9]
    refs += [" ".join(rng.choices(base, k=80)) for _ in range(8)]
    target_file = tmp_path / "targets.tsv"
    target_file.write_text("".join(f"{i}\t{s}\n" for i, s in enumerate(texts)))
    paths = []
    for worker in range(2):
        output = tmp_path / f"{worker}.tsv"; paths.append(output)
        subprocess.run([str(binary), str(target_file), str(output)],
                       input="".join(f"{i}:0\t{s}\n" for i, s in enumerate(refs) if i % 2 == worker),
                       text=True, check=True)
    targets = [{"body": s, "candidate_id": i, "source_shard": "toy", "source_row": i} for i, s in enumerate(texts)]
    result = merge(paths, targets)
    for text, row in zip(texts, result):
        w = text.split(); gs = grams(w, 5)
        jac = cont = local = 0; covered = set()
        for ref in refs:
            rw = ref.split(); rs = grams(rw, 5); shared = len(gs & rs)
            if shared:
                jac = max(jac, shared / len(gs | rs))
                if min(len(w), len(rw)) >= 50 and shared >= 20:
                    cont = max(cont, shared / min(len(gs), len(rs)))
            r13 = grams(rw, 13); pair = set()
            for i in range(len(w)-12):
                if tuple(w[i:i+13]) in r13:
                    pair.update(range(i, i+13))
            covered |= pair
            local = max(local, len(pair) / len(w))
        assert row["exact"] == (text in refs)
        assert row["jaccard"] == pytest.approx(jac)
        assert row["containment"] == pytest.approx(cont)
        assert row["local_pair"] == pytest.approx(local)
        assert row["coverage"] == pytest.approx(len(covered) / len(w))
        assert row["bitmap"] == "".join("1" if i in covered else "0" for i in range(len(w)))


def test_actual_array_pipeline_detects_injected_duplicates(tmp_path):
    """Positive controls: exact, case-only variant, and a clean document."""
    repo = Path(__file__).resolve().parents[1]
    tokenizer_path = repo / 'dataset/bbc-news/TG_GPT2_tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    texts = ['The river flooded several houses yesterday.',
             'A satellite measured distant stars during the night.',
             'Fresh bread tastes delicious with warm butter.']
    def encode(text):
        return np.array([50257,*tokenizer.encode(text).ids,50256],dtype='<u2')
    root=tmp_path/'dataset';(root/'terminal').mkdir(parents=True)
    (root/'docppl/test_1000').mkdir(parents=True)
    arrays=[encode(t) for t in texts]
    np.save(root/'terminal/test.npy',np.concatenate(arrays))
    import gzip
    with gzip.open(root/'test_selected.jsonl.gz','wt') as f:
        for i,t in enumerate(texts):
            f.write(json.dumps({'candidate_id':i,'source_shard':'toy','source_row':i,'body':normalize(t)})+'\n')
    (root/'docppl/test_1000/document_map.json').write_text(json.dumps([
        {'full_split_doc_id':i,'candidate_id':i} for i in range(len(texts))]))
    names=['terminal/test.npy','test_selected.jsonl.gz']
    (root/'manifest.json').write_text(json.dumps({'files':{n:sha_file(root/n) for n in names},
        'tokenizer_sha256':sha_file(tokenizer_path)}))
    train=tmp_path/'train.npy'
    np.save(train,np.concatenate([arrays[0],encode(texts[1].upper()),arrays[0]]))
    output=tmp_path/'audit'
    run(SimpleNamespace(output=output,dataset=root,tokenizer=tokenizer_path,train=train,
                        expected_train_sha256=sha_file(train),workers=2))
    result=json.loads((output/'summary.json').read_text())
    assert result['complete']
    assert result['train_documents']==3
    assert result['unique_strict_train']==2
    assert len(result['strict_matches'])==2
    assert {r['test_doc'] for r in result['strict_matches']}=={0}
    assert result['test']['normalized_exact']==2
    rows=[json.loads(s) for s in (output/'per_test.jsonl').read_text().splitlines()]
    assert not rows[2]['exact']
    report=tmp_path/'report.md'
    summarize(SimpleNamespace(audit=output,dataset=root,tokenizer=tokenizer_path,report=report))
    assert json.loads((output/'pair_validation.json').read_text())['complete']
    assert report.exists()
