#!/usr/bin/env python3
"""Check filesystem Unix sockets and spawned DataLoader tensor IPC before training.

Run under the same Python and TMPDIR as the training allocation. This does not
validate GPUs, NCCL, model kernels, real data collation, or microbatch capacity.
"""
import argparse
import datetime
import json
from multiprocessing.connection import Listener
from multiprocessing.util import get_temp_dir
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new preflight receipt: {args.output}')
    actual_tmpdir = tempfile.gettempdir()
    requested_tmpdir = os.environ.get('TMPDIR')
    if requested_tmpdir and Path(requested_tmpdir).resolve() != Path(actual_tmpdir).resolve():
        raise RuntimeError(f'TMPDIR unavailable: requested {requested_tmpdir}, Python selected {actual_tmpdir}')
    # Explicit pathname also tests Python versions that default to abstract sockets.
    address = tempfile.mktemp(prefix='listener-', dir=get_temp_dir())
    with Listener(address=address, family='AF_UNIX'):
        address_bytes = len(os.fsencode(address))
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(1)
    expected = torch.arange(16, dtype=torch.int64)
    loader = DataLoader(TensorDataset(expected), batch_size=4, num_workers=2,
                        multiprocessing_context='spawn', timeout=60)
    actual = torch.cat([batch[0] for batch in loader])
    if not torch.equal(actual, expected):
        raise RuntimeError('DataLoader tensor transfer differs from the input')
    result = dict(status='passed', checked_at=datetime.datetime.now().astimezone().isoformat(),
                  job_id=os.environ.get('SLURM_JOB_ID'), tmpdir=actual_tmpdir,
                  socket_address_bytes=address_bytes, spawn_workers=2, batches_received=4,
                  tensor_ipc='exact equality', torch=torch.__version__)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print('IPC_PREFLIGHT_OK', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
