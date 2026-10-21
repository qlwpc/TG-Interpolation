"""Whole-pair commit, reversibility and interrupted-commit safety."""
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers

from scripts.campaigns import fineweb_bracket_repair as repair


@pytest.fixture
def tokenizer(tmp_path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tok = Tokenizer(models.BPE({s:i for i,s in enumerate(alphabet)}, []))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.normalizer = normalizers.NFC()
    tok.add_special_tokens(['<|endoftext|>', '<(S>', '<S)>', '<(NP>', '<NP)>'])
    path = tmp_path/'tokenizer.json'
    tok.save(str(path))
    return path, tok


def source_pair(tmp_path, tokenizer, leaves=None):
    path, tok = tokenizer
    if leaves is None:
        leaves = [' alpha-LRB-beta-RRB- ', ' [-LSB-a-RSB-] -LCB-b-RCB- ']
    root = tmp_path/'data'
    for fmt in ('tree', 'terminal'):
        (root/fmt).mkdir(parents=True)
    tree = []
    for leaf in leaves:
        tree.extend([tok.token_to_id('<(S>'), tok.token_to_id('<(NP>'),
                     *tok.encode(leaf, add_special_tokens=False).ids,
                     tok.token_to_id('<NP)>'), tok.token_to_id('<S)>'), tok.token_to_id('<|endoftext|>')])
    tree = np.array(tree, dtype='<u4')
    layout = repair.Layout(path)
    terminal = tree[~layout.nt[tree]]
    tree.tofile(root/'tree/a.npy')
    terminal.tofile(root/'terminal/a.npy')
    return root, {'tree': tree.tobytes(), 'terminal': terminal.tobytes()}


def run_prepare(root, campaign, tokenizer, commit=False, workers=1):
    return repair.prepare_shard(root, 'a.npy', 0, campaign, tokenizer[0], workers, 20, commit)


def test_candidate_detector_covers_every_ascii_segmentation():
    for marker in repair.BRACKET_MAPPING:
        for cuts in itertools.product([False, True], repeat=len(marker)-1):
            starts = [0] + [i+1 for i,c in enumerate(cuts) if c] + [len(marker)]
            pieces = [marker[a:b] for a,b in zip(starts, starts[1:])]
            if len(pieces) == 1:
                pieces[0] = 'prefix'+pieces[0]+'suffix'
            else:
                pieces[0], pieces[-1] = 'prefix'+pieces[0], pieces[-1]+'suffix'
            assert any(repair.signature_candidate(p) for p in pieces), pieces
    assert not repair.signature_candidate('Biology')
    assert not repair.signature_candidate('Book')


def test_typed_forest_validation_detects_crossed_labels(tokenizer):
    path, tok = tokenizer
    a = np.array([tok.token_to_id('<(S>'), tok.token_to_id('<(NP>'), tok.token_to_id('<S)>'),
                  tok.token_to_id('<NP)>'), tok.token_to_id('<|endoftext|>')], dtype='<u4')
    with pytest.raises(ValueError, match='labels'):
        repair.check_forest(a, repair.Layout(path))


def test_prepare_preserves_source_and_reverse_backup(tmp_path, tokenizer):
    root, old = source_pair(tmp_path, tokenizer)
    j = run_prepare(root, tmp_path/'campaign', tokenizer)
    assert j['status'] == 'verified' and j['verified']['residual_ptb_markers'] == 0
    assert j['verified']['documents'] == 2 and sum(j['counts'].values()) == 6
    for fmt in old:
        assert (root/fmt/'a.npy').read_bytes() == old[fmt]
        assert repair.replay(j['staged'][fmt], j['patch'], fmt, 'reverse') == j['original_sha256'][fmt]


def test_commit_and_exact_restore(tmp_path, tokenizer):
    root, old = source_pair(tmp_path, tokenizer)
    campaign = tmp_path/'campaign'
    j = run_prepare(root, campaign, tokenizer, commit=True, workers=2)
    assert j['status'] == 'committed'
    assert set(root.glob('*/a.npy')) == {root/'terminal/a.npy', root/'tree/a.npy'}
    for fmt in old:
        assert (root/fmt/'a.npy').read_bytes() != old[fmt]
        assert not Path(j['old_links'][fmt]).exists()
        assert not Path(j['staged'][fmt]).exists()
    layout = repair.Layout(tokenizer[0])
    tree = repair.raw_array(root/'tree/a.npy')
    terminal = repair.raw_array(root/'terminal/a.npy')
    np.testing.assert_array_equal(tree[~layout.nt[tree]], terminal)
    assert 'alpha(beta)' in layout.tok.decode(terminal.tolist(), skip_special_tokens=True)
    restored = tmp_path/'restored'
    repair.restore_original(campaign, 0, restored)
    for fmt in old:
        assert (restored/fmt/'a.npy').read_bytes() == old[fmt]
    assert run_prepare(root, campaign, tokenizer, commit=True)['status'] == 'committed'


def test_resume_half_committed_pair(tmp_path, tokenizer, monkeypatch):
    root, old = source_pair(tmp_path, tokenizer)
    campaign = tmp_path/'campaign'
    j = run_prepare(root, campaign, tokenizer)
    real_replace = repair.os.replace
    def crash_after_terminal(source, dest):
        if str(source) == j['staged']['tree']:
            raise RuntimeError('simulated crash between pair renames')
        return real_replace(source, dest)
    monkeypatch.setattr(repair.os, 'replace', crash_after_terminal)
    journal = campaign/'backups/000/journal.json'
    with pytest.raises(RuntimeError, match='simulated crash'):
        repair.finish_commit(journal)
    assert (root/'tree/a.npy').read_bytes() == old['tree']
    assert (root/'terminal/a.npy').read_bytes() != old['terminal']
    assert all(Path(p).read_bytes() == old[fmt] for fmt,p in j['old_links'].items())
    monkeypatch.setattr(repair.os, 'replace', real_replace)
    assert repair.finish_commit(journal)['status'] == 'committed'
    repair.restore_original(campaign, 0, tmp_path/'restored')
    for fmt in old:
        assert (tmp_path/'restored'/fmt/'a.npy').read_bytes() == old[fmt]


def test_corrupt_backup_prevents_any_source_replacement(tmp_path, tokenizer):
    root, old = source_pair(tmp_path, tokenizer)
    campaign = tmp_path/'campaign'
    j = run_prepare(root, campaign, tokenizer)
    with Path(j['patch']).open('ab') as f:
        f.write(b'corrupted')
    with pytest.raises(ValueError, match='backup fingerprint'):
        repair.finish_commit(campaign/'backups/000/journal.json')
    for fmt in old:
        assert (root/fmt/'a.npy').read_bytes() == old[fmt]


def test_interrupted_generation_restarts_without_touching_original(tmp_path, tokenizer, monkeypatch):
    root, old = source_pair(tmp_path, tokenizer)
    campaign = tmp_path/'campaign'
    original_verify = repair.verify_pair
    def fail_verify(*args):
        raise RuntimeError('simulated generation interruption')
    monkeypatch.setattr(repair, 'verify_pair', fail_verify)
    with pytest.raises(RuntimeError, match='simulated'):
        run_prepare(root, campaign, tokenizer, commit=True)
    for fmt in old:
        assert (root/fmt/'a.npy').read_bytes() == old[fmt]
    # A failed v1 generation must upgrade safely to the new EOS policy.
    path = campaign/'backups/000/journal.json'
    interrupted = json.loads(path.read_text())
    interrupted['protocol'] = repair.LEGACY_PROTOCOL
    path.write_text(json.dumps(interrupted))
    monkeypatch.setattr(repair, 'verify_pair', original_verify)
    assert run_prepare(root, campaign, tokenizer, commit=True)['status'] == 'committed'
    assert list((campaign/'backups/000').glob('journal.interrupted_*.json'))


def test_verified_journal_cannot_commit_a_different_dataset(tmp_path, tokenizer):
    root, old = source_pair(tmp_path, tokenizer)
    campaign = tmp_path/'campaign'
    run_prepare(root, campaign, tokenizer)
    other, _ = source_pair(tmp_path/'other', tokenizer)
    with pytest.raises(ValueError, match='incompatible'):
        run_prepare(other, campaign, tokenizer, commit=True)
    for fmt in old:
        assert (root/fmt/'a.npy').read_bytes() == old[fmt]
        assert (other/fmt/'a.npy').read_bytes() == old[fmt]


def test_input_mismatch_keeps_sources_untouched(tmp_path, tokenizer):
    root, _ = source_pair(tmp_path, tokenizer)
    path = root/'terminal/a.npy'
    a = repair.raw_array(path).copy()
    a[0] = 1
    a.tofile(path)
    before = {fmt: (root/fmt/'a.npy').read_bytes() for fmt in ('tree','terminal')}
    with pytest.raises(ValueError, match='mismatch'):
        run_prepare(root, tmp_path/'campaign', tokenizer, commit=True)
    for fmt, content in before.items():
        assert (root/fmt/'a.npy').read_bytes() == content


def test_no_change_commit_cleans_unused_staging(tmp_path, tokenizer):
    root, old = source_pair(tmp_path, tokenizer, [' Plain (native) [brackets] with B vitamins. '])
    j = run_prepare(root, tmp_path/'campaign', tokenizer, commit=True)
    assert not j['counts']
    for fmt in old:
        assert (root/fmt/'a.npy').read_bytes() == old[fmt]
        assert not Path(j['staged'][fmt]).exists()


def test_ptb_marker_split_by_tree_structure_blocks_commit(tmp_path, tokenizer):
    root, old = source_pair(tmp_path, tokenizer, [' -L', 'RB- '])
    # EOS separates documents, so no false positive across the EOS is allowed.
    j = run_prepare(root, tmp_path/'campaign', tokenizer, commit=True)
    assert not j['counts']
    # Within one document, however, such malformed lexical segmentation must not
    # silently retain the escape and claim success.
    tok = tokenizer[1]
    ids = [tok.token_to_id('<(S>'), tok.token_to_id('<(NP>'),
           *tok.encode('-L',add_special_tokens=False).ids, tok.token_to_id('<NP)>'),
           tok.token_to_id('<(NP>'), *tok.encode('RB-',add_special_tokens=False).ids,
           tok.token_to_id('<NP)>'), tok.token_to_id('<S)>'), tok.token_to_id('<|endoftext|>')]
    tree = np.array(ids,dtype='<u4'); layout = repair.Layout(tokenizer[0])
    tree.tofile(root/'tree/a.npy'); tree[~layout.nt[tree]].tofile(root/'terminal/a.npy')
    with pytest.raises(ValueError,match='remains'):
        run_prepare(root, tmp_path/'campaign2', tokenizer, commit=True)


def test_partial_utf8_lexical_run_cannot_damage_other_text(tmp_path, tokenizer):
    path, tok = tokenizer
    root, _ = source_pair(tmp_path, tokenizer)
    accent = tok.encode('é', add_special_tokens=False).ids
    assert len(accent) == 2
    ids = [tok.token_to_id('<(S>'), tok.token_to_id('<(NP>'), accent[0], tok.token_to_id('<NP)>'),
           tok.token_to_id('<(NP>'), accent[1], *tok.encode('-LRB-', add_special_tokens=False).ids,
           tok.token_to_id('<NP)>'), tok.token_to_id('<S)>'), tok.token_to_id('<|endoftext|>')]
    tree = np.array(ids, dtype='<u4'); layout = repair.Layout(path)
    terminal = tree[~layout.nt[tree]]
    tree.tofile(root/'tree/a.npy'); terminal.tofile(root/'terminal/a.npy')
    old = {fmt: (root/fmt/'a.npy').read_bytes() for fmt in ('tree','terminal')}
    assert 'é-LRB-' in tok.decode(terminal.tolist(),skip_special_tokens=True)
    with pytest.raises(ValueError,match='UTF-8 complete'):
        run_prepare(root, tmp_path/'campaign', tokenizer, commit=True)
    for fmt in old:
        assert (root/fmt/'a.npy').read_bytes() == old[fmt]


def test_unrelated_added_tokens_are_preserved_not_filtered(tmp_path, tokenizer):
    path, tok = tokenizer
    tok.add_special_tokens(['<|example-role|>'])
    tok.save(str(path))
    root, _ = source_pair(tmp_path, tokenizer, [' x-LRB-y-RRB-<|example-role|> z '])
    extra = tok.token_to_id('<|example-role|>')
    j = run_prepare(root, tmp_path/'campaign', tokenizer, commit=True)
    assert j['extra_added_tokens_preserved'] == {str(extra): 1}
    for fmt in ('tree','terminal'):
        assert np.count_nonzero(repair.raw_array(root/fmt/'a.npy') == extra) == 1


def test_old_document_boundaries_translate_to_same_documents(tmp_path, tokenizer):
    root, old = source_pair(tmp_path, tokenizer)
    layout = repair.Layout(tokenizer[0])
    old_terminal = np.frombuffer(old['terminal'],dtype='<u4')
    old_ends = (np.flatnonzero(old_terminal == layout.eos)+1).tolist()
    j = run_prepare(root, tmp_path/'campaign', tokenizer, commit=True)
    new_ends = (np.flatnonzero(repair.raw_array(root/'terminal/a.npy') == layout.eos)+1).tolist()
    mapping = repair.translate_boundaries(j['patch'],'terminal',[0,*old_ends])
    assert mapping == dict(zip([0,*old_ends],[0,*new_ends]))
    first = next(repair.read_patches(j['patch']))
    with pytest.raises(ValueError,match='ambiguous'):
        repair.translate_boundaries(j['patch'],'terminal',[first['terminal_old']+1])


def test_quota_fallback_requires_verified_headroom(tmp_path):
    p=tmp_path/'quota_snapshot.json'
    p.write_text(json.dumps({'fileset':'tukw-group','free_bytes':30<<30}))
    repair.quota_snapshot_fallback(tmp_path,20<<30)
    p.write_text(json.dumps({'fileset':'tukw-group','free_bytes':21<<30}))
    with pytest.raises(OSError,match='does not cover'):
        repair.quota_snapshot_fallback(tmp_path,20<<30)


@pytest.mark.parametrize('leaf', [
    ' CDCnews<|endoftext|>A number ',
    ' <|endoftext|>one<|endoftext|><|endoftext|>two<|endoftext|> ',
    ' -L<|endoftext|>RB-word-RRB- ',
])
def test_embedded_eos_delete_preserves_record_and_restores_exactly(tmp_path, tokenizer, leaf):
    root, old = source_pair(tmp_path, tokenizer, [leaf, ' Separate record. '])
    layout = repair.Layout(tokenizer[0])
    original_tree = np.frombuffer(old['tree'],dtype='<u4')
    original_term = np.frombuffer(old['terminal'],dtype='<u4')
    depth = np.cumsum(layout.delta[original_tree])
    inner = (original_tree==layout.eos) & (depth>0)
    campaign = tmp_path/'campaign'
    j = run_prepare(root,campaign,tokenizer,commit=True,workers=2)
    tree = repair.raw_array(root/'tree/a.npy')
    term = repair.raw_array(root/'terminal/a.npy')
    assert j['embedded_eos_removed']==int(inner.sum())==leaf.count('<|endoftext|>')
    assert j['documents']==j['record_end_eos_preserved']==2
    assert j['original_eos_markers']==2+int(inner.sum())
    assert j['verified']['residual_embedded_eos']==0
    np.testing.assert_array_equal(tree[layout.nt[tree]],original_tree[layout.nt[original_tree]])
    np.testing.assert_array_equal(tree[~layout.nt[tree]],term)
    assert repair.check_forest(tree,layout)==2
    assert np.count_nonzero(term==layout.eos)==2
    # Only the two actual record-ending EOS spellings are retained.
    expected = repair.normalize_ptb_brackets(leaf.replace('<|endoftext|>',''))+'<|endoftext|> Separate record. <|endoftext|>'
    assert layout.tok.decode(term.tolist(),skip_special_tokens=False)==expected
    if '-L' not in leaf:
        np.testing.assert_array_equal(tree,original_tree[~inner])
    restored=tmp_path/'restored'
    repair.restore_original(campaign,0,restored)
    for fmt,content in old.items():assert (restored/fmt/'a.npy').read_bytes()==content


def test_chunk_boundary_never_uses_inner_eos(tmp_path, tokenizer):
    root, old = source_pair(tmp_path,tokenizer,[' abcdefghijklmnop<|endoftext|>still inside this tree. '])
    layout=repair.Layout(tokenizer[0]);tree=np.frombuffer(old['tree'],dtype='<u4')
    first=int(np.flatnonzero(tree==layout.eos)[0])
    spans=list(repair.chunks(tree,layout.eos,first+1,layout=layout))
    assert spans==[(0,len(tree))]
    with pytest.raises(ValueError,match='crossing EOS'):
        repair.check_forest(tree,layout)
    assert repair.check_forest(tree,layout,allow_embedded_eos=True)==1


def test_inner_eos_policy_does_not_hide_mismatched_labels(tmp_path, tokenizer):
    root,old=source_pair(tmp_path,tokenizer,[' one<|endoftext|>two ']);layout=repair.Layout(tokenizer[0])
    tree=np.frombuffer(old['tree'],dtype='<u4').copy();tok=tokenizer[1]
    a,b=np.flatnonzero(tree==tok.token_to_id('<NP)>'))[0],np.flatnonzero(tree==tok.token_to_id('<S)>'))[0]
    tree[a],tree[b]=tree[b],tree[a];tree.tofile(root/'tree/a.npy')
    before={fmt:(root/fmt/'a.npy').read_bytes() for fmt in old}
    with pytest.raises(ValueError,match='labels'):
        run_prepare(root,tmp_path/'campaign',tokenizer,commit=True)
    for fmt,content in before.items():assert (root/fmt/'a.npy').read_bytes()==content


def test_depth_zero_eos_between_trees_is_preserved(tmp_path, tokenizer):
    root,old=source_pair(tmp_path,tokenizer,[' first ',' second '])
    j=run_prepare(root,tmp_path/'campaign',tokenizer,commit=True)
    assert j['embedded_eos_removed']==0 and j['documents']==2
    for fmt,content in old.items():assert (root/fmt/'a.npy').read_bytes()==content


def test_legacy_verified_journal_remains_resumable(tmp_path, tokenizer):
    root,old=source_pair(tmp_path,tokenizer)
    campaign=tmp_path/'campaign';run_prepare(root,campaign,tokenizer)
    path=campaign/'backups/000/journal.json';j=json.loads(path.read_text())
    j['protocol']=repair.LEGACY_PROTOCOL;path.write_text(json.dumps(j))
    assert run_prepare(root,campaign,tokenizer,commit=True)['status']=='committed'
    repair.restore_original(campaign,0,tmp_path/'restored')
    for fmt,content in old.items():assert (tmp_path/'restored'/fmt/'a.npy').read_bytes()==content


def test_parallel_locks_exclude_serial_and_duplicate_shard(tmp_path):
    with repair.dataset_lock(tmp_path,shared=True):
        with repair.dataset_lock(tmp_path,shared=True), repair.shard_lock(tmp_path,0):
            with repair.shard_lock(tmp_path,1):
                with pytest.raises(BlockingIOError):
                    with repair.shard_lock(tmp_path,0):pass
                with pytest.raises(BlockingIOError):
                    with repair.dataset_lock(tmp_path):pass
    with repair.dataset_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with repair.dataset_lock(tmp_path,shared=True):pass


@pytest.mark.parametrize('assignments,baseline', [
    ({'a':[0],'b':[0,1]},[]),
    ({'a':[0]},[]),
    ({'a':[0]},[0,1]),
    ({'a':[0],'b':[2]},[]),
])
def test_parallel_plan_rejects_overlap_gaps_and_invalid_indices(tmp_path,assignments,baseline):
    root=tmp_path/'data';campaign=tmp_path/'campaign';files=[root/'terminal/a.npy',root/'terminal/b.npy']
    plan={'data_root':str(root),'campaign':str(campaign),'file_names':[p.name for p in files],
          'assignments':assignments,'committed_baseline':baseline}
    with pytest.raises(ValueError,match='overlap|omit|invalid'):
        repair.validate_parallel_plan(plan,files,root,campaign,'a')


def test_parallel_progress_reports_peer_failure_and_never_false_complete(tmp_path):
    root=tmp_path/'data';campaign=tmp_path/'campaign';root.mkdir();campaign.mkdir()
    a={'runner_id':'a','job_id':'1','pid':1,'status':'completed','processing_shard':None}
    b={'runner_id':'b','job_id':'2','pid':2,'status':'failed','processing_shard':1,'error':'projection mismatch'}
    repair.publish_runner_progress(root,campaign,2,b)
    progress=repair.publish_runner_progress(root,campaign,2,a)
    assert progress['status']=='failed' and progress['completed_shards']==0
    assert progress['worker_failures']==[dict(b,updated_at=progress['worker_failures'][0]['updated_at'])]
    assert json.loads((root/'NATIVE_BRACKET_REPAIR_PROGRESS.json').read_text())==progress


def test_parallel_quota_includes_all_concurrent_staging(tmp_path):
    p=tmp_path/'quota_snapshot.json'
    p.write_text(json.dumps({'fileset':'tukw-group','free_bytes':30<<30,'parallel_staging_budget_bytes':12<<30}))
    with pytest.raises(OSError,match='budget'):
        repair.quota_snapshot_fallback(tmp_path,20<<30)
    p.write_text(json.dumps({'fileset':'tukw-group','free_bytes':40<<30,'parallel_staging_budget_bytes':12<<30}))
    repair.quota_snapshot_fallback(tmp_path,20<<30)


@pytest.mark.parametrize('error', [subprocess.TimeoutExpired('mmlsquota',30), FileNotFoundError('mmlsquota')])
def test_compute_quota_failure_requires_verified_snapshot(tmp_path,monkeypatch,error):
    campaign='/inspurfs/test-campaign'
    monkeypatch.setattr(repair,'campaign_bytes',lambda path: 1<<30)
    def unavailable(*args,**kwargs):raise error
    monkeypatch.setattr(repair.subprocess,'run',unavailable)
    fallback=repair.quota_snapshot_fallback
    monkeypatch.setattr(repair,'quota_snapshot_fallback',lambda path,cap: fallback(tmp_path,cap))
    snapshot=tmp_path/'quota_snapshot.json'
    snapshot.write_text(json.dumps({'fileset':'tukw-group','free_bytes':60<<30,'parallel_staging_budget_bytes':29<<30}))
    repair.check_backup_quota(campaign)
    snapshot.write_text(json.dumps({'fileset':'tukw-group','free_bytes':30<<30,'parallel_staging_budget_bytes':29<<30}))
    with pytest.raises(OSError,match='budget'):repair.check_backup_quota(campaign)
    snapshot.unlink()
    with pytest.raises(FileNotFoundError):repair.check_backup_quota(campaign)


def test_compute_quota_timeout_cannot_bypass_campaign_cap(monkeypatch):
    monkeypatch.setattr(repair,'campaign_bytes',lambda path: 18<<30)
    def query(*args,**kwargs):pytest.fail('over-budget campaign must fail before quota query')
    monkeypatch.setattr(repair.subprocess,'run',query)
    with pytest.raises(OSError,match='budget exhausted'):repair.check_backup_quota('/inspurfs/test-campaign')


def test_four_parallel_cli_owners_commit_and_restore_exact_pairs(tmp_path,tokenizer):
    root=tmp_path/'data';campaign=tmp_path/'campaign';originals={}
    for fmt in ('tree','terminal'):(root/fmt).mkdir(parents=True)
    for i in range(4):
        _,old=source_pair(tmp_path/f'input{i}',tokenizer,[' first-LRB-<|endoftext|>second-RRB- '])
        name=f'{i:02d}.npy';originals[name]=old
        for fmt,content in old.items():(root/fmt/name).write_bytes(content)
    campaign.mkdir();plan_path=tmp_path/'plan.json'
    plan={'data_root':str(root),'campaign':str(campaign),'file_names':sorted(originals),
          'assignments':{f'node{i}':[i] for i in range(4)},'committed_baseline':[]}
    plan_path.write_text(json.dumps(plan));processes=[]
    try:
        for i in range(4):
            log=(tmp_path/f'node{i}.log').open('w')
            args=[sys.executable,str(Path(repair.__file__).resolve()),'--data-root',str(root),
                  '--campaign',str(campaign),'--tokenizer',str(tokenizer[0]),'--workers','1',
                  '--chunk-tokens','20','--runner-id',f'node{i}','--assignment-plan',str(plan_path),'--commit']
            processes.append((subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT),log))
        for process,log in processes:
            assert process.wait(timeout=60)==0,Path(log.name).read_text()
    finally:
        for process,log in processes:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
            log.close()
    progress=json.loads((campaign/'progress.json').read_text())
    assert progress['status']=='complete' and progress['completed_shards']==4
    assert progress['embedded_eos_removed']==4 and progress['bracket_replacements']==8
    assert len(progress['workers'])==4 and not progress['worker_failures']
    assert json.loads((root/'NATIVE_BRACKET_REPAIR_PROGRESS.json').read_text())==progress
    for i,name in enumerate(sorted(originals)):
        restored=tmp_path/f'restored{i}';repair.restore_original(campaign,i,restored)
        for fmt,content in originals[name].items():assert (restored/fmt/name).read_bytes()==content


@pytest.mark.parametrize('leaf', [
    ' XY=#,#,<-LRB-><<-RRB->value>1 ',
    ' <-LSB->a<-RSB-> <-LCB->b<-RCB-> ',
    ' <-LRB->a<|endoftext|>b<-RRB-> ',
])
def test_dedicated_lexical_bracket_ids_preserve_angle_characters_and_restore(tmp_path,tokenizer,leaf):
    path,tok=tokenizer
    tokens=[f'<{s}>' for s in repair.BRACKET_MAPPING]
    tok.add_tokens(tokens);tok.save(str(path))
    root,old=source_pair(tmp_path,tokenizer,[leaf]);campaign=tmp_path/'campaign'
    j=run_prepare(root,campaign,tokenizer,commit=True);layout=repair.Layout(path)
    text=tok.decode(repair.raw_array(root/'terminal/a.npy').tolist(),skip_special_tokens=False)
    expected=repair.normalize_ptb_brackets(leaf.replace('<|endoftext|>',''))+'<|endoftext|>'
    assert text==expected
    assert '<(>' in text or '<[>' in text
    assert j['lexical_added_brackets_reencoded']
    tree=repair.raw_array(root/'tree/a.npy');term=repair.raw_array(root/'terminal/a.npy')
    assert not np.any(layout.lexical_bracket_added[tree])
    np.testing.assert_array_equal(tree[~layout.nt[tree]],term)
    repair.restore_original(campaign,0,tmp_path/'restored')
    for fmt,content in old.items():assert (tmp_path/'restored'/fmt/'a.npy').read_bytes()==content


def test_unrecognized_bracket_added_token_still_blocks_commit(tmp_path,tokenizer):
    path,tok=tokenizer;tok.add_tokens(['<prefix-LRB->']);tok.save(str(path))
    root,old=source_pair(tmp_path,tokenizer,[' <prefix-LRB-> '])
    with pytest.raises(ValueError,match='explicit policy'):
        run_prepare(root,tmp_path/'campaign',tokenizer,commit=True)
    for fmt,content in old.items():assert (root/fmt/'a.npy').read_bytes()==content


def test_atomic_json_concurrent_writers_never_share_temporary_file(tmp_path):
    path=tmp_path/'status.json';processes=[]
    code='''import sys
from pathlib import Path
from scripts.campaigns.fineweb_bracket_repair import atomic_json
writer=int(sys.argv[2])
for sequence in range(20):
 atomic_json(Path(sys.argv[1]),dict(writer=writer,sequence=sequence,payload=chr(97+writer)*16384))
'''
    try:
        for i in range(4):
            log=(tmp_path/f'writer{i}.log').open('w')
            processes.append((subprocess.Popen([sys.executable,'-c',code,str(path),str(i)],
                                               stdout=log,stderr=subprocess.STDOUT),log))
        for process,log in processes:assert process.wait(timeout=60)==0,Path(log.name).read_text()
    finally:
        for process,log in processes:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
            log.close()
    result=json.loads(path.read_text())
    assert result['payload']==chr(97+result['writer'])*16384
    assert not list(tmp_path.glob('status.json.tmp.*'))
