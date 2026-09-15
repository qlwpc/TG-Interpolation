"""Certify only the three corrected models after full coverage and frozen-input checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.native_document_results import input_identity, atomic_json


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();c=a.campaign.resolve()
    status=dict(status='validating',started_at=time.time())
    atomic_json(c/'final_status.json',status)
    try:
        assert json.loads((c/'pilot_validation.json').read_text())['status']=='complete'
        records=json.loads((c/'grammar_correction.json').read_text())
        provenance=json.loads((c.parent/'provenance.json').read_text())
        root=(c/'repo').resolve()
        for file,expected in provenance['code_sha256'].items():
            if file.startswith(('olmo/','olmo_data/')) or file=='scripts/evaluate_reserved_document_ppl.py':
                assert sha(root/file)==expected, file
        for mid,r in records.items():
            cp=Path(r['corrected'])
            assert sha(cp/'model.pt')==r['weights_sha256']
            assert sha(cp/'config.yaml')==r['corrected_config_sha256']
            for path in (c/'results'/mid).glob('contract_*.json'):
                contract=json.loads(path.read_text())
                assert contract['checkpoint']==str(cp)
                assert contract['checkpoint_config']==r['corrected_config_sha256']
                assert input_identity(contract['data'])==contract['data_identity']
                assert input_identity(str(cp/'model.pt'))==contract['checkpoint_weights']
                assert sha(root/'dataset/bbc-news/TG_GPT2_tokenizer.json')==contract['tokenizer']
                assert path.name=='contract_'+hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()+'.json'
        subprocess.run([sys.executable,str(root/'scripts/merge_reserved_docppl.py'),'--campaign',str(c)],check=True)
        rows=json.loads((c/'results.json').read_text());assert len(rows)==3
        for row in rows:
            assert (c/'results'/row['model_id']/f"contract_{row['fingerprint']}.json").exists()
        status.update(status='complete',models=3,scope='three grammar-corrected models',results_sha256=sha(c/'results.json'),completed_at=time.time())
    except BaseException as e:
        status.update(status='failed',error=repr(e));raise
    finally:atomic_json(c/'final_status.json',status)

if __name__=='__main__':main()
