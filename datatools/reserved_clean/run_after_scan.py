"""Resume the exact/near duplicate pipeline after two independently logged scans."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from datatools.reserved_clean.common import atomic_json


def run(args):
    state = args.root/"driver_status.json"
    atomic_json(state, {"status": "waiting_for_scans", "raw_pid": args.raw_pid, "array_pid": args.array_pid})
    for folder, pid in (("raw", args.raw_pid), ("arrays", args.array_pid)):
        while not (args.root/folder/"complete.json").exists():
            try:
                os.kill(pid, 0)
            except ProcessLookupError as exc:
                raise RuntimeError(f"{folder} process ended without completion receipt; inspect {folder}.log") from exc
            time.sleep(15)
    atomic_json(state, {"status": "preparing_candidates"})
    subprocess.run([sys.executable, "-m", "datatools.reserved_clean.filter", "prepare", "--raw", str(args.root/"raw"),
                    "--output", str(args.root/"candidates")], check=True)
    atomic_json(state, {"status": "matching_all_historical_and_actual_train_bodies"})
    subprocess.run([sys.executable, "-m", "datatools.reserved_clean.filter", "match", "--raw", str(args.root/"raw"),
                    "--arrays", str(args.root/"arrays"), "--candidates", str(args.root/"candidates"),
                    "--output", str(args.root/"matching")], check=True)
    atomic_json(state, {"status": "remote_matching_complete", "local_dev_test_audit_pending": True})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--raw-pid", type=int, required=True)
    p.add_argument("--array-pid", type=int, required=True)
    args = p.parse_args()
    try:
        run(args)
    except BaseException as exc:
        atomic_json(args.root/"driver_status.json", {"status": "failed", "error": repr(exc)})
        raise
