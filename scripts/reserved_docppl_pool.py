"""Run independent complete-document tasks on Slurm-assigned GPUs.

Atomic directory claims work across nodes without a database or scheduler calls.
Failed tasks remain visible and must be explicitly repaired/released.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def atomic(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--class', dest='pool_class', choices=['shanghai', 'critical', 'terminal'], required=True)
    p.add_argument('--stage', choices=['pilot', 'full'], default='full')
    args = p.parse_args()
    campaign = args.campaign.resolve()
    tasks = json.loads((campaign / 'tasks.json').read_text())
    for receipt, key in [('input_validation.json', 'status'), ('native_validation.json', 'complete')]:
        value = json.loads((campaign / receipt).read_text())[key]
        if value not in (True, 'complete'):
            raise RuntimeError('input validation is not complete')
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if not devices or not all(devices):
        raise RuntimeError('Slurm must assign visible GPUs')
    threads = str(max(1, min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '4')) // len(devices))))
    (campaign / 'claims').mkdir(exist_ok=True)
    (campaign / 'task_logs').mkdir(exist_ok=True)
    print(f'host={socket.gethostname()} assigned_devices={devices} class={args.pool_class}', flush=True)

    def worker(device):
        failures = []
        for task in tasks:
            if task['stage'] != args.stage:
                continue
            if args.pool_class == 'critical' and not task['critical_allowed']:
                continue
            if args.pool_class == 'terminal':
                argv = task['argv']
                if '--format' not in argv or argv[argv.index('--format') + 1] != 'terminal':
                    continue
            claim = campaign / 'claims' / task['id']
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            status = dict(task_id=task['id'], job_id=os.environ['SLURM_JOB_ID'],
                          host=socket.gethostname(), device=device, started=time.time(),
                          command=[sys.executable, *task['argv']], status='running')
            atomic(claim / 'status.json', status)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS=threads,
                       MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS='1', SLURM_CPUS_PER_TASK=threads)
            # Keep compilation caches unique per assigned device/process worker.
            env['TORCHINDUCTOR_CACHE_DIR'] = f'/tmp/rc_inductor_{os.environ["SLURM_JOB_ID"]}_{device}'
            env['TRITON_CACHE_DIR'] = f'/tmp/rc_triton_{os.environ["SLURM_JOB_ID"]}_{device}'
            logfile = campaign / 'task_logs' / (task['id'] + '.log')
            print(f'start {task["id"]} device={device}', flush=True)
            with logfile.open('a') as stream:
                code = subprocess.call(status['command'], env=env, stdout=stream, stderr=subprocess.STDOUT)
            status.update(status='complete' if code == 0 else 'failed', returncode=code, ended=time.time())
            atomic(claim / 'status.json', status)
            print(f'end {task["id"]} returncode={code}', flush=True)
            if code:
                failures.append(task['id'])
                break
        return failures

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        failures = [t for group in pool.map(worker, devices) for t in group]
    if failures:
        raise SystemExit(f'failed tasks: {failures}')


if __name__ == '__main__':
    main()
