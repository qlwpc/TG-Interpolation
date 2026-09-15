"""Create an isolated grammar-corrected rerun without modifying historical checkpoints."""
import json
from pathlib import Path
import hashlib
import shutil

BASE = Path('/inspurfs/group/tukw/wangpch/docppl_reserved_20260907')

def main():
    campaign = BASE / 'grammar_fix_20260909'
    campaign.mkdir(exist_ok=True)
    repo = BASE / 'repo'
    if not (campaign/'repo').exists():
        (campaign/'repo').symlink_to(repo, target_is_directory=True)
    models = json.loads((BASE/'models.json').read_text())
    tasks = json.loads((BASE/'tasks.json').read_text())
    provenance = json.loads((BASE/'provenance.json').read_text())
    selected = []
    new_tasks = []
    evidence = {}
    for grammar in ('tree_noont', 'tree_compress', 'tree_triplecnt'):
        mid = 'bbc_100m_' + grammar
        model = next(m for m in models if m['id'] == mid)
        task = next(t for t in tasks if t['model'] == mid)
        original = Path(task['argv'][task['argv'].index('--checkpoint')+1])
        corrected = campaign/'checkpoints'/grammar
        corrected.mkdir(parents=True, exist_ok=True)
        text = (original/'config.yaml').read_text()
        assert text.count('transformer_grammar_type: tgtree') == 1
        fixed = text.replace('transformer_grammar_type: tgtree', 'transformer_grammar_type: '+grammar)
        (corrected/'config.yaml').write_text(fixed)
        if not (corrected/'model.pt').exists():
            (corrected/'model.pt').symlink_to(original/'model.pt')
        evidence[mid] = dict(grammar=grammar, original=str(original), corrected=str(corrected),
            original_config_sha256=hashlib.sha256(text.encode()).hexdigest(),
            corrected_config_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
            weights_sha256=provenance['models'][mid]['model.pt']['sha256'])
        selected.append(dict(model, checkpoint=str(corrected)))
        for old in tasks:
            if old['model'] != mid: continue
            t=json.loads(json.dumps(old));a=t['argv']
            a[a.index('--checkpoint')+1]=str(corrected)
            a[a.index('--output')+1]=str(campaign/'results'/mid)
            new_tasks.append(t)
    for name in ('input_validation.json','native_validation.json'):
        shutil.copy2(BASE/name, campaign/name)
    (campaign/'models.json').write_text(json.dumps(selected,indent=2)+'\n')
    (campaign/'tasks.json').write_text(json.dumps(new_tasks,indent=2)+'\n')
    (campaign/'grammar_correction.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (campaign/'logs').mkdir(exist_ok=True)
    print(campaign, len(new_tasks), 'tasks')

if __name__ == '__main__': main()
