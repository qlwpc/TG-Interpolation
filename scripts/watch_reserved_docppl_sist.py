"""Low-frequency local observer: mirror progress and fetch validated final results.

This process never submits/cancels jobs and never runs model scoring locally.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT/'artifacts/evaluation/reserved_docppl_sist_20260907'
REMOTE = '/inspurfs/group/tukw/wangpch/docppl_reserved_20260907'
SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','SIST']


def write(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


def main():
    deadline = time.monotonic() + 60*3600
    write(LOCAL/'observer.json',dict(status='running',pid=os.getpid(),interval_seconds=300))
    while time.monotonic() < deadline:
        try:
            command = (f'cd {REMOTE}; PYTHONPATH=repo /public/home/wangpch/anaconda3/envs/LLM/bin/python '
                       'repo/scripts/merge_reserved_docppl.py --campaign . --allow-partial')
            proc = subprocess.run([*SSH,command],capture_output=True,text=True,timeout=120)
            if proc.returncode:
                print(f'progress aggregation: {proc.stderr[-1500:]}',flush=True)
            fetch = subprocess.run([*SSH,f'cat {REMOTE}/final_status.json'],capture_output=True,text=True,timeout=40)
            final = json.loads(fetch.stdout) if fetch.returncode == 0 else {'status':'running'}
            subprocess.run(['rsync','-a',f'SIST:{REMOTE}/progress.json',str(LOCAL)+'/'],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=60)
            write(LOCAL/'observer.json',dict(status=final['status'],pid=os.getpid(),checked_at=time.time(),
                                           interval_seconds=300,remote_final_status=final))
            if final['status'] == 'failed':
                write(LOCAL/'final_status.json',final)
                state = json.loads((LOCAL/'state.json').read_text())
                state.update(status='failed',final_status=final)
                write(LOCAL/'state.json',state)
                print(f'final validation failed: {final}',flush=True)
                return 1
            if final['status'] == 'complete':
                subprocess.run(['rsync','-a',f'SIST:{REMOTE}/results/',str(LOCAL/'results')+'/'],check=True,timeout=1800)
                for name in ('results.json','results.csv','summary.md','final_status.json','provenance.json','full_job_ids.txt'):
                    subprocess.run(['rsync','-a',f'SIST:{REMOTE}/{name}',str(LOCAL)+'/'],check=True,timeout=60)
                digest = hashlib.sha256((LOCAL/'results.json').read_bytes()).hexdigest()
                if digest != final['results_sha256']:
                    raise ValueError('downloaded results differ from final receipt')
                state = json.loads((LOCAL/'state.json').read_text())
                state.update(status='complete',completed_at=time.time(),final_status=final,
                             results=str(LOCAL/'results.json'),summary=str(LOCAL/'summary.md'))
                write(LOCAL/'state.json',state)
                print('All 22 full-test results validated and mirrored locally.',flush=True)
                return 0
        except Exception as error:
            print(f'observer retry after error: {error!r}',flush=True)
        for _ in range(5):
            time.sleep(60)
    write(LOCAL/'observer.json',dict(status='observer_timeout',pid=os.getpid(),ended_at=time.time()))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
