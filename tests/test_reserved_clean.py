import csv
import io
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest

from datatools.reserved_clean.common import body_from_parse, ngrams, normalize
from datatools.reserved_clean.scan_arrays import documents


@pytest.fixture(scope="module")
def matcher(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path_factory.mktemp("matcher")/"match"
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(binary),
                    str(root/"datatools/reserved_clean/match.cpp")], check=True)
    return binary


def execute(binary, tmp_path, candidates, refs, mode="external"):
    p = tmp_path/"candidates.tsv"
    p.write_text("".join(f"{i}\t{b}\n" for i,b in enumerate(candidates)))
    subprocess.run([str(binary), str(p), str(tmp_path/"out.tsv"), str(tmp_path/"edges.tsv"), mode],
                   input="".join(f"{i}\t{b}\n" for i,b in enumerate(refs)), text=True, check=True)
    rows = list(csv.DictReader((tmp_path/"out.tsv").open(), delimiter="\t"))
    edges = list(csv.DictReader((tmp_path/"edges.tsv").open(), delimiter="\t"))
    return rows, edges


def test_leaf_normalization_is_independent_of_structure():
    a = '(S (ADJ (JJ GOOD)) (NP (NN -LRB-) (NN café) (NN -RRB-)) (VBD))'
    b = '(X (JJ good) (NN -LRB-) (NN café) (NN -RRB-))'
    assert body_from_parse(a) == body_from_parse(b) == 'good ( café )'
    assert normalize('Version 12!') != normalize('Version 13!')
    assert body_from_parse('(X bare words)') == 'bare words'


def test_chunked_array_boundaries_and_corruption():
    a = np.array([50257, 4, 5, 50256, 50257, 6, 50256, 50257, 9, 10, 50256], dtype='<u2')
    assert [(o, v.tolist()) for o,v in documents(a,chunk=3)] == [
        (0,a[:4].tolist()), (4,a[4:7].tolist()), (7,a[7:].tolist())]
    a[-1] = 7
    with pytest.raises(ValueError, match='EOS'):
        list(documents(a,chunk=3))


def test_exact_index_matches_exhaustive_oracle(matcher,tmp_path):
    rng = random.Random(8)
    base = [f'w{i}' for i in range(100)]
    candidates = [' '.join(base)]
    for j in range(12):
        v = base.copy()
        for i in rng.sample(range(100), j):
            v[i]=f'new{j}_{i}'
        candidates.append(' '.join(v))
    candidates += ['totally unrelated short document', ' '.join(base[10:70])]
    refs = [' '.join(base), ' '.join(f'z{i}' for i in range(110))]
    rows,_ = execute(matcher,tmp_path,candidates,refs)
    for c,r in zip(candidates,rows):
        expected = False
        for ref in refs:
            a,b=ngrams(c),ngrams(ref)
            k=len(a&b)
            if not k:
                continue
            expected |= k/len(a|b)>=.8 or (k/min(len(a),len(b))>=.8 and
                min(len(c.split()),len(ref.split()))>=50 and k>=20)
        assert bool(int(r['exact']) or int(r['near'])) == expected


def test_partial_coverage_unions_reference_positions(matcher,tmp_path):
    base = [f'w{i}' for i in range(100)]
    # Each ref is too short to veto via containment. Separate matching ranges
    # nevertheless cover 40% of candidate text, with no double counting.
    refs = [' '.join(base[:20]), ' '.join(base[40:60]), ' '.join(base[:15])]
    rows,_ = execute(matcher,tmp_path,[' '.join(base)],refs)
    assert rows[0]['near']=='0'
    assert int(rows[0]['covered_words'])==40
    assert rows[0]['covered_bitmap'].count('1')==40


def test_internal_excludes_self_and_links_versions(matcher,tmp_path):
    base = ' '.join(f'w{i}' for i in range(100))
    c=[base,base+' extra', ' '.join(f'z{i}' for i in range(100))]
    _,edges=execute(matcher,tmp_path,c,c,mode='internal')
    assert edges
    assert {(int(e['candidate']),int(e['source'])) for e in edges}=={(0,1)}


def test_short_containment_does_not_veto_whole_document(matcher,tmp_path):
    c=' '.join(f'w{i}' for i in range(100))
    rows,_=execute(matcher,tmp_path,[c],['w1 w2 w3 w4 w5'])
    assert rows[0]['near']=='0' and rows[0]['exact']=='0'


def test_common_templates_do_not_count_as_local_copy(matcher,tmp_path):
    template=' '.join(f't{i}' for i in range(13))
    c=[template+' '+' '.join(f'd{j}w{i}' for i in range(40)) for j in range(100)]
    rows,_=execute(matcher,tmp_path,c,[template])
    assert all(r['covered_words']=='0' for r in rows)


def test_reserved_shards_fail_before_train_outputs(tmp_path):
    from datatools.parse_pretrain_data.assemble_streams import assemble,read_order
    from datatools.reserved_clean.common import RESERVED
    with pytest.raises(ValueError,match='reserved evaluation shards'):
        assemble(tmp_path/'missing',tmp_path/'output',[RESERVED[0]],{}, {},50257)
    assert not (tmp_path/'output').exists()
    root=Path(__file__).resolve().parents[1]
    order=read_order(root/'datatools/parse_pretrain_data/bbc_train_shards.txt')
    assert len(order)==89 and not set(order)&set(RESERVED)


def test_group_partition_is_deterministic_disjoint_and_exact():
    from datatools.reserved_clean.build import stratified_partition
    from datatools.reserved_clean.common import RESERVED
    items=[{'group_id':f'{i:06d}','source_shard':RESERVED[i%5],'source_row':i,
            'terminal_tokens':100*(i%15+1)} for i in range(6400)]
    a=stratified_partition(items); b=stratified_partition(items[::-1])
    assert a==b
    assert {k:len(v) for k,v in a.items()}=={'dev':1000,'test':5000,'reserve':400}
    sets=[{r['group_id'] for r in rows} for rows in a.values()]
    assert len(set.union(*sets))==6400
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i))
    small=stratified_partition(items[:101])
    assert sum(map(len,small.values()))==101 and not small['reserve']


def test_canonical_encoding_and_native_adapter(tmp_path):
    import json
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from datatools.reserved_clean.build import encode_document
    from datatools.reserved_clean.common import sha_file
    from datatools.parse_test_docppl_data.reproduce_bbc_test import legacy_tokenizer
    from datatools.parse_test_docppl_data.generate_native_topk import CanonicalPPLCorpus,audit_alignment
    root=Path(__file__).resolve().parents[1]
    tokenizer_path=root/'dataset/bbc-news/TG_GPT2_tokenizer.json'
    tok=legacy_tokenizer(Tokenizer.from_file(str(tokenizer_path)))
    parsed='(S (NP (NNP Alice)) (VP (VBZ is) (ADJ (JJ happy))) (. .)) (Ċ Ċ) (S (NP (PRP She)) (VP (VBZ runs)) (. !))'
    arrays,bounds,repairs=encode_document(parsed,tok)
    assert repairs=={'ADJ_to_ADJP':1}
    nt=lambda a: a[(a<50268)|(a>50319)]
    np.testing.assert_array_equal(nt(arrays['tree']),arrays['terminal'])
    np.testing.assert_array_equal(nt(arrays['tg']),arrays['terminal'])
    np.save(tmp_path/'tree.npy',arrays['tree'])
    np.save(tmp_path/'sentence_offsets.npy',bounds)
    np.save(tmp_path/'document_sentence_counts.npy',np.array([len(bounds)-1],dtype=np.uint32))
    manifest={'format':'canonical-tree-sentences-v1','tree_path':'tree.npy',
        'document_count':1,'sentence_count':2,'tree_sha256':sha_file(tmp_path/'tree.npy'),
        'sentence_offsets_sha256':sha_file(tmp_path/'sentence_offsets.npy'),
        'document_sentence_counts_sha256':sha_file(tmp_path/'document_sentence_counts.npy')}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    corpus=CanonicalPPLCorpus(tmp_path,tokenizer_path)
    assert corpus.lengths.shape==(2,1)
    assert corpus.sentence_input(0).words
    audit_alignment(SimpleNamespace(ppl_dir=tmp_path,tokenizer=tokenizer_path,
        test_tree=tmp_path/'tree.npy',output=tmp_path/'audit'))
    audit=json.loads((tmp_path/'audit/alignment_audit.json').read_text())
    assert audit['document_offset']==0 and audit['exceptions']==[]
    manifest['tree_sha256']='bad'
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='fingerprint'):
        CanonicalPPLCorpus(tmp_path,tokenizer_path)


def test_build_propagates_contamination_through_content_groups(tmp_path):
    import base64,gzip,hashlib,json
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from datatools.reserved_clean.build import build,encode_document
    from datatools.reserved_clean.common import atomic_json,digest,emit,sha_file,RESERVED
    from datatools.parse_test_docppl_data.reproduce_bbc_test import legacy_tokenizer
    root=Path(__file__).resolve().parents[1]
    tokpath=root/'dataset/bbc-news/TG_GPT2_tokenizer.json'
    tok=legacy_tokenizer(Tokenizer.from_file(str(tokpath)))
    c=tmp_path/'candidates';c.mkdir();rows=[]
    with gzip.open(c/'candidates.jsonl.gz','wt') as f:
        for i in range(8):
            parsed='(S '+' '.join(f'(NN document{i}word{j})' for j in range(20))+')'
            arrays,_,_=encode_document(parsed,tok);blob=arrays['terminal'].tobytes()
            r={'candidate_id':i,'source_shard':RESERVED[i%5],'source_row':i,'parsed':parsed,
               'strict_sha256':hashlib.sha256(blob).hexdigest(),'terminal_b64':base64.b64encode(blob).decode(),
               'body':body_from_parse(parsed),'body_sha256':digest(body_from_parse(parsed))}
            rows.append(r);emit(f,r)
    (c/'raw_exclusions.jsonl').write_text('')
    atomic_json(c/'complete.json',{'complete':True,'candidates':8,
        'outputs':{p.name:sha_file(p) for p in c.iterdir() if p.is_file()}})
    (tmp_path/'arrays.jsonl').write_text('');(tmp_path/'local.jsonl').write_text('')
    atomic_json(tmp_path/'complete.json',{'complete':True,'outputs':{'arrays.jsonl':sha_file(tmp_path/'arrays.jsonl')}})
    folders=[]
    for j in range(2):
        p=tmp_path/f'match{j}';p.mkdir();folders.append(p)
        (p/'internal_edges.tsv').write_text('candidate\tsource\treason\n0\t1\tnear\n')
        with (p/'external.tsv').open('w') as f:
            f.write('candidate\texact\tnear\tcovered_bitmap\n')
            for i,r in enumerate(rows):f.write(f"{i}\t{int(j==0 and i==0)}\t0\t"+'0'*len(r['body'].split())+'\n')
        atomic_json(p/'complete.json',{'complete':True,'outputs':{x.name:sha_file(x) for x in p.iterdir()},
            'candidate_receipt_sha256':sha_file(c/'complete.json'),
            'prerequisite_receipts':{'arrays':sha_file(tmp_path/'complete.json')} if j==0 else {}})
    out=tmp_path/'built'
    build(SimpleNamespace(candidates=c,matches=folders,output=out,tokenizer=tokpath,seed=20260906,
        array_matches=tmp_path/'arrays.jsonl',local_references=tmp_path/'local.jsonl'))
    m=json.loads((out/'manifest.json').read_text())
    assert m['split_documents']=={'dev':1,'test':5,'reserve':0}
    final=[json.loads(s) for s in (out/'candidates.jsonl').read_text().splitlines()]
    assert final[0]['split'] is None and final[1]['split'] is None
    assert 'normalized_reference_exact' in final[1]['exclusion_reasons']
    for split in ('dev','test'):
        a=np.load(out/'terminal'/f'{split}.npy')
        assert np.count_nonzero(a==50257)==m['split_documents'][split]


def test_ppl_assembly_uses_manifest_counts_and_preflights(tmp_path,monkeypatch):
    import json,sys
    from datatools.parse_test_docppl_data.assemble_testppl import main
    native=tmp_path/'native';native.mkdir()
    (native/'manifest.json').write_text(json.dumps({'status':'complete','document_count':2,
        'sentence_count':3,'candidate_slots':300}))
    aligned=tmp_path/'aligned';aligned.mkdir()
    (aligned/'manifest.json').write_text(json.dumps({'contract':{'documents':2,'sentences':3}}))
    for prefix in ('tree','tg'):
        p=tmp_path/prefix;p.mkdir()
        np.save(p/f'{prefix}_doc_index.npy',np.array([1,2],dtype=np.uint32))
        np.save(p/f'{prefix}_sent_index.npy',np.ones(900,dtype=np.uint16))
        np.save(p/f'{prefix}_300.npy',np.zeros(900,dtype=np.uint16))
    output=tmp_path/'output'
    argv=['assemble','--native-source',str(native),'--tree-source',str(tmp_path/'tree'),
        '--tg-source',str(tmp_path/'tg'),'--aligned-source',str(aligned),'--output',str(output)]
    monkeypatch.setattr(sys,'argv',argv);main()
    m=json.loads((output/'manifest.json').read_text())
    assert (m['document_count'],m['sentence_count'])==(2,3)
    assert (output/'tree300/tree_300.npy').samefile(tmp_path/'tree/tree_300.npy')
    (aligned/'manifest.json').write_text(json.dumps({'contract':{'documents':2,'sentences':4}}))
    output2=tmp_path/'bad_output';argv[-1]=str(output2)
    with pytest.raises(ValueError,match='counts differ'):main()
    assert not output2.exists()


def test_parallel_reference_merge_unions_word_positions(matcher,tmp_path):
    import shutil
    from datatools.reserved_clean.filter import merge_external
    words=[f'w{i}' for i in range(100)];body=' '.join(words)
    for worker,ref in enumerate((' '.join(words[:20]),' '.join(words[40:60]))):
        p=tmp_path/f'part{worker}';p.mkdir()
        execute(matcher,p,[body],[ref])
        shutil.copyfile(p/'out.tsv',tmp_path/f'external.{worker}.tsv')
        shutil.copyfile(p/'edges.tsv',tmp_path/f'external_edges.{worker}.tsv')
    merge_external(tmp_path,2)
    r=list(csv.DictReader((tmp_path/'external.tsv').open(),delimiter='\t'))[0]
    assert int(r['covered_words'])==40 and r['covered_bitmap'].count('1')==40
    assert float(r['coverage'])==.4
    assert r['exact']=='0' and r['near']=='0'


def test_matching_orchestration_serial_parallel_parity(tmp_path):
    import json
    from types import SimpleNamespace
    from datatools.reserved_clean.common import atomic_json,digest,emit,sha_file
    from datatools.reserved_clean.filter import match
    candidates=tmp_path/'candidates';candidates.mkdir()
    words=[f'w{i}' for i in range(100)]
    (candidates/'candidates.tsv').write_text('0\t'+' '.join(words)+'\n')
    atomic_json(candidates/'complete.json',{'complete':True,'candidates':1,
        'outputs':{'candidates.tsv':sha_file(candidates/'candidates.tsv')}})
    refs=tmp_path/'refs.jsonl'
    with refs.open('w') as f:
        for i in range(256):
            body=' '.join((words[:20] if i<128 else words[40:60])+[f'new{i}'])
            emit(f,{'source':['synthetic',i],'body':body,'body_sha256':digest(body)})
    outcomes=[]
    for workers in (1,2):
        output=tmp_path/f'workers{workers}'
        match(SimpleNamespace(candidates=candidates,raw=None,arrays=None,references=[refs],
                              output=output,workers=workers))
        r=list(csv.DictReader((output/'external.tsv').open(),delimiter='\t'))[0]
        outcomes.append({k:r[k] for k in ('exact','near','covered_words','covered_bitmap')})
        complete=json.loads((output/'complete.json').read_text())
        assert complete['unique_references']==256 and complete['workers']==workers
    assert outcomes[0]==outcomes[1]
    assert outcomes[0]['covered_words']=='40'
