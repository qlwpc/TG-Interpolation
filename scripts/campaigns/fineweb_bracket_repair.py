"""Repair paired historical FineWeb raw-u32 streams with reversible backups.

Lexical PTB brackets are normalized and EOS IDs inside open trees are deleted
from both streams. Tree markers, outer EOS, record order and storage are preserved. Before
commit, verify the full pair and reconstruct the ORIGINAL SHA-256 from the new
files plus a compressed delta. A journal and temporary original hardlinks protect
the two-file commit; --commit is explicit and resumable. No corpus filtering.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datatools.ptb_brackets import BRACKET_MAPPING, normalize_ptb_brackets

LEGACY_PROTOCOL = 'fineweb-paired-native-brackets-delta-v1'
PROTOCOL = 'fineweb-paired-native-brackets-inner-eos-delta-v2'
PATCH_HEADER = struct.Struct('<QQQQII')
STOP = False
WORKER = None


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f'{path.name}.tmp.{os.uname().nodename}.{os.getpid()}.{uuid.uuid4().hex}')
    try:
        with tmp.open('x') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def identity(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'inode': s.st_ino,
            'device': s.st_dev, 'mode': stat.S_IMODE(s.st_mode), 'uid': s.st_uid, 'gid': s.st_gid}


def stable_identity(info):
    # NFS mount device numbers can differ between compute nodes.
    return {k: v for k, v in info.items() if k != 'device'}


def raw_array(path):
    path = Path(path)
    with path.open('rb') as f:
        if f.read(6) == b'\x93NUMPY':
            raise ValueError(f'expected headerless raw-u32, not standard NPY: {path}')
    if not path.stat().st_size or path.stat().st_size % 4:
        raise ValueError(f'invalid raw-u32 size: {path}')
    return np.memmap(path, dtype='<u4', mode='r')


def signature_candidate(piece):
    """Any ASCII B in a literal -[LR][RCS]B- must overlap such a piece.

    ByteLevel preserves ASCII letters in vocabulary spellings. Compare the
    available left/right context; this includes single-character B and markers
    split arbitrarily across BPE IDs, while excluding ordinary Biology/Book IDs.
    """
    for i, ch in enumerate(piece):
        if ch != 'B':
            continue
        left, right = min(i, 3), min(len(piece) - i - 1, 1)
        if any(piece[i-left:i+right+1] == marker[3-left:4+right] for marker in BRACKET_MAPPING):
            return True
    return False


class Layout:
    def __init__(self, tokenizer_path):
        spec = json.loads(Path(tokenizer_path).read_text())
        if spec['model']['type'] != 'BPE' or spec['decoder']['type'] != 'ByteLevel':
            raise ValueError('requires an explicitly registered ByteLevel BPE tokenizer')
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        vocab = self.tok.get_vocab()
        self.size = self.tok.get_vocab_size()
        self.eos = vocab['<|endoftext|>']
        self.nt = np.zeros(self.size, dtype=bool)
        self.delta = np.zeros(self.size, dtype=np.int8)
        self.label = np.zeros(self.size, dtype=np.int16)
        labels = sorted({s[2:-1] for s in vocab if re.fullmatch(r'<\([A-Za-z0-9]+>', s)})
        if not labels:
            raise ValueError('missing tree labels')
        for i, label in enumerate(labels, 1):
            op, cl = vocab.get(f'<({label}>'), vocab.get(f'<{label})>')
            if op is None or cl is None:
                raise ValueError('unpaired structural vocabulary')
            self.nt[[op, cl]] = True
            self.delta[op], self.delta[cl] = 1, -1
            self.label[[op, cl]] = i
        self.boundary = self.nt.copy()
        self.boundary[self.eos] = True
        self.forbidden = np.zeros(self.size, dtype=bool)
        self.extra = np.zeros(self.size, dtype=bool)
        self.lexical_bracket_added = np.zeros(self.size,dtype=bool)
        bracket_added_pieces = {}
        for token in spec.get('added_tokens', []):
            tid = token['id']
            if tid != self.eos and not self.nt[tid]:
                if (re.fullmatch(r'<-(?:LRB|RRB|LCB|RCB|LSB|RSB)->',token['content'])
                        and not token.get('lstrip') and not token.get('rstrip')):
                    # Apply the SAME textual rule to this dedicated lexical ID:
                    # <-LRB-> becomes <(>, retaining its literal angle brackets.
                    self.lexical_bracket_added[tid] = True
                    bracket_added_pieces[tid] = token['content']
                    continue
                # Unrelated added tokens are copied unchanged, never re-encoded.
                # Dedicated angle-wrapped PTB IDs need a separate explicit policy.
                self.extra[tid] = True
                self.boundary[tid] = True
                if any(s in token['content'] for s in BRACKET_MAPPING):
                    self.forbidden[tid] = True
        self.candidate = np.zeros(self.size, dtype=bool)
        self.pieces = [''] * self.size
        for piece, tid in spec['model']['vocab'].items():
            self.pieces[tid] = piece
            if not self.boundary[tid] and not self.forbidden[tid] and signature_candidate(piece):
                self.candidate[tid] = True
        for tid,piece in bracket_added_pieces.items():
            self.pieces[tid] = piece
            self.candidate[tid] = True
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        chars = list(byte_values)
        next_char = 256
        for byte in range(256):
            if byte not in byte_values:
                byte_values.append(byte)
                chars.append(next_char)
                next_char += 1
        self.byte_translate = dict(zip(chars, byte_values))


def check_ids(a, layout):
    if len(a) and int(a.max()) >= layout.size:
        raise ValueError('out-of-vocabulary tokens')
    if np.any(layout.forbidden[a]):
        bad = np.unique(a[layout.forbidden[a]]).tolist()
        raise ValueError(f'dedicated angle-wrapped PTB tokens require explicit policy: {bad}')


def forest_state(a, layout, allow_embedded_eos=False):
    """Validate typed trees; count only EOS at empty-stack record boundaries."""
    check_ids(a, layout)
    depth = np.cumsum(layout.delta[a], dtype=np.int64)
    if len(depth) and (depth.min() < 0 or depth[-1] != 0
            or (not allow_embedded_eos and np.any(depth[a == layout.eos] != 0))):
        raise ValueError('unbalanced structure or a tree crossing EOS')
    positions = np.flatnonzero(layout.nt[a])
    events = a[positions]
    if not len(events):
        raise ValueError('tree chunk has no structural markers')
    kinds = layout.delta[events]
    event_depth = depth[positions] + (kinds == -1)
    order = np.argsort(event_depth, kind='stable')
    d, k, labels = event_depth[order], kinds[order], layout.label[events[order]]
    starts = np.r_[0, np.flatnonzero(np.diff(d)) + 1]
    sizes = np.diff(np.r_[starts, len(d)])
    if np.any(sizes % 2):
        raise ValueError('unfinished structure at chunk boundary')
    if np.any(k[::2] != 1) or np.any(k[1::2] != -1) or np.any(labels[::2] != labels[1::2]):
        raise ValueError('mismatched tree labels')
    return int(np.count_nonzero((a == layout.eos) & (depth == 0))), depth


def check_forest(a, layout, allow_embedded_eos=False):
    return forest_state(a, layout, allow_embedded_eos)[0]


def chunks(a, eos, chunk_tokens, max_chunk_tokens=64 << 20, layout=None):
    start = 0
    while start < len(a):
        target = min(len(a), start + chunk_tokens)
        while True:
            window = a[start:target]
            boundary = window == eos
            if layout is not None:
                check_ids(window, layout)
                depth = np.cumsum(layout.delta[window], dtype=np.int64)
                if len(depth) and depth.min() < 0:
                    raise ValueError('unbalanced structure before chunk boundary')
                boundary &= depth == 0
            ends = np.flatnonzero(boundary)
            if len(ends):
                end = start + int(ends[-1]) + 1
                break
            if target == len(a) or target - start >= max_chunk_tokens:
                raise ValueError('missing EOS or document exceeds bounded chunk capacity')
            target = min(len(a), start + min(2 * (target - start), max_chunk_tokens))
        yield start, end
        start = end


def worker_init(tree_path, terminal_path, tokenizer):
    global WORKER
    WORKER = (raw_array(tree_path), raw_array(terminal_path), Layout(tokenizer))


def analyze_chunk(task):
    tree, terminal, layout = WORKER
    start, end, term_start, term_end = task
    a = tree[start:end]
    documents, depth = forest_state(a, layout, allow_embedded_eos=True)
    leaf_mask = ~layout.nt[a]
    projected = a[leaf_mask]
    if not np.array_equal(projected, terminal[term_start:term_end]):
        raise ValueError(f'input tree/terminal mismatch at tree token {start}')
    embedded = np.flatnonzero((a == layout.eos) & (depth > 0))
    boundary_mask = layout.boundary[a].copy()
    boundary_mask[embedded] = False
    boundaries = np.flatnonzero(boundary_mask)
    candidates = np.union1d(np.flatnonzero(layout.candidate[a]), embedded)
    end_indices = np.unique(np.searchsorted(boundaries, candidates))
    patches, counts, examples = [], Counter(), []
    prefix = np.r_[0, np.cumsum(leaf_mask, dtype=np.int64)]
    for idx in end_indices:
        left = int(boundaries[idx - 1]) + 1 if idx else 0
        right = int(boundaries[idx])
        old = a[left:right]
        cleaned = old[old != layout.eos]
        text = layout.tok.decode(cleaned.tolist(), skip_special_tokens=False)
        normalized = normalize_ptb_brackets(text)
        if normalized == text:
            # No re-tokenization for EOS-only changes: every other ID survives.
            for pos in np.flatnonzero(old == layout.eos):
                at = left + int(pos)
                patches.append((at, int(prefix[at]), a[at:at+1].tobytes(), b''))
            continue
        # Re-encoding a partial UTF-8 character would replace original bytes by
        # U+FFFD, even though the full terminal stream might decode correctly.
        spelling = ''.join(layout.pieces[int(i)] for i in cleaned)
        try:
            strict_text = spelling.translate(layout.byte_translate).encode('latin1').decode('utf-8', errors='strict')
        except (UnicodeError, KeyError) as error:
            raise ValueError('affected lexical run is not UTF-8 complete; refusing collateral repair') from error
        if strict_text != text:
            raise ValueError('ByteLevel decode differs from original lexical bytes')
        new = np.array(layout.tok.encode(normalized, add_special_tokens=False).ids, dtype='<u4')
        if not len(new) or np.any(layout.boundary[new]) or np.any(layout.forbidden[new]):
            raise ValueError('lexical encoding introduced boundary/added tokens')
        if layout.tok.decode(new.tolist(), skip_special_tokens=False) != normalized:
            raise ValueError('encoding changed text beyond bracket normalization')
        counts.update({s: text.count(s) for s in BRACKET_MAPPING if s in text})
        patches.append((left, int(prefix[left]), old.tobytes(), new.tobytes()))
        if len(examples) < 3:
            match = re.search(r'-(?:LRB|RRB|LCB|RCB|LSB|RSB)-', text)
            lo, hi = max(0, match.start()-70), min(len(text), match.end()+70)
            examples.append({'tree_old_start': start+left, 'terminal_old_start': term_start+int(prefix[left]),
                             'before': text[lo:hi], 'after': normalize_ptb_brackets(text[lo:hi])})
    patches.sort(key=lambda patch:patch[0])
    eos_examples = [{'tree_old_offset':start+int(pos),
                     'terminal_old_offset':term_start+int(prefix[pos]),
                     'open_depth':int(depth[pos]), 'deleted_id':layout.eos,
                     'context':layout.tok.decode(projected[max(0,int(prefix[pos])-6):int(prefix[pos])+7].tolist(),
                                               skip_special_tokens=False)} for pos in embedded[:12]]
    return {'patches': patches, 'counts': dict(counts), 'documents': documents,
            'embedded_eos_removed':len(embedded), 'embedded_eos_examples':eos_examples,
            'lexical_added_brackets_reencoded':dict(zip(*[x.tolist() for x in
                np.unique(a[layout.lexical_bracket_added[a]],return_counts=True)])),
            'candidate_runs': len(end_indices), 'examples': examples,
            'extra_added_tokens': dict(zip(*[a.tolist() for a in np.unique(a[layout.extra[a]], return_counts=True)]))}


def residual_markers(a, layout):
    candidates = np.flatnonzero(layout.candidate[a])
    for pos in candidates:
        # An ASCII 5-character marker spans at most 5 BPE tokens. Keeping EOS
        # visible prevents falsely joining text across document boundaries.
        window = a[max(0, int(pos)-4):min(len(a), int(pos)+5)]
        text = layout.tok.decode(window.tolist(), skip_special_tokens=False)
        if normalize_ptb_brackets(text) != text:
            raise ValueError(f'PTB marker remains in output near terminal token {pos}')


def write_array(handle, array, digest):
    data = memoryview(np.ascontiguousarray(array)).cast('B')
    handle.write(data)
    digest.update(data)


def read_patches(path):
    with gzip.open(path, 'rb') as f:
        if f.read(8) != b'FWPTB001':
            raise ValueError('invalid delta backup magic')
        while True:
            header = f.read(PATCH_HEADER.size)
            if not header:
                break
            if len(header) != PATCH_HEADER.size:
                raise ValueError('truncated delta backup')
            to, tn, po, pn, old_len, new_len = PATCH_HEADER.unpack(header)
            old, new = f.read(4*old_len), f.read(4*new_len)
            if len(old) != 4*old_len or len(new) != 4*new_len:
                raise ValueError('truncated delta payload')
            yield {'tree_old': to, 'tree_new': tn, 'terminal_old': po, 'terminal_new': pn,
                   'old': old, 'new': new, 'old_len': old_len, 'new_len': new_len}


def replay(source, patch_path, fmt, direction, output=None):
    """Hash (or write) an exact forward/reverse reconstruction, without decoding."""
    a = raw_array(source)
    digest, cursor = hashlib.sha256(), 0
    out = Path(output).open('wb') if output is not None else None
    def emit(data):
        view = memoryview(data).cast('B')
        digest.update(view)
        if out:
            out.write(view)
    try:
        for p in read_patches(patch_path):
            reverse = direction == 'reverse'
            offset = p[f'{fmt}_new' if reverse else f'{fmt}_old']
            expected, replacement = (p['new'], p['old']) if reverse else (p['old'], p['new'])
            length = len(expected)//4
            if offset < cursor or offset+length > len(a) or a[offset:offset+length].tobytes() != expected:
                raise ValueError('delta does not match input bytes / ordered ranges')
            for lo in range(cursor, offset, 4 << 20):
                emit(a[lo:min(offset, lo+(4 << 20))])
            emit(replacement)
            cursor = offset+length
        for lo in range(cursor, len(a), 4 << 20):
            emit(a[lo:lo+(4 << 20)])
        if out:
            out.flush()
            os.fsync(out.fileno())
    finally:
        if out:
            out.close()
    return digest.hexdigest()


def translate_boundaries(patch_path, fmt, offsets):
    """Translate old document boundaries; reject ambiguous interior token offsets."""
    delta, result = 0, {}
    wanted = iter(sorted(set(offsets)))
    offset = next(wanted, None)
    for p in read_patches(patch_path):
        old = p[f'{fmt}_old']
        while offset is not None and offset <= old:
            result[offset] = offset + delta
            offset = next(wanted, None)
        if offset is not None and offset < old+p['old_len']:
            raise ValueError('offset is inside a re-encoded lexical run; token mapping is ambiguous')
        delta += p['new_len']-p['old_len']
    while offset is not None:
        result[offset] = offset+delta
        offset = next(wanted, None)
    return result


def verify_pair(tree_path, terminal_path, layout, chunk_tokens):
    tree, term = raw_array(tree_path), raw_array(terminal_path)
    cursor, documents = 0, 0
    digests = {k: hashlib.sha256() for k in ('tree', 'terminal', 'structural', 'extra_added')}
    for start, end in chunks(tree, layout.eos, chunk_tokens, layout=layout):
        a = tree[start:end]
        documents += check_forest(a, layout)
        projected = a[~layout.nt[a]]
        if not np.array_equal(projected, term[cursor:cursor+len(projected)]):
            raise ValueError('staged output projection differs from terminal')
        residual_markers(projected, layout)
        digests['tree'].update(memoryview(a).cast('B'))
        digests['terminal'].update(memoryview(term[cursor:cursor+len(projected)]).cast('B'))
        digests['structural'].update(memoryview(a[layout.nt[a]]).cast('B'))
        digests['extra_added'].update(memoryview(a[layout.extra[a]]).cast('B'))
        cursor += len(projected)
    if cursor != len(term):
        raise ValueError('staged terminal has unmatched tail')
    return {'documents': documents, 'tokens': {'tree': len(tree), 'terminal': len(term)},
            'sha256': {k: d.hexdigest() for k, d in digests.items()},
            'projection_equal': True, 'typed_tree_labels_balanced': True,
            'residual_embedded_eos':0, 'residual_ptb_markers': 0}


@contextmanager
def file_lock(path, shared=False, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(lock.fileno(), mode | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def dataset_lock(root, shared=False):
    return file_lock(Path(root)/'.native_bracket_repair.lock', shared=shared)


def shard_lock(root, index):
    return file_lock(Path(root)/'.native_bracket_shard_locks'/f'{index:03d}.lock')


def preserve_attributes(source, target):
    src = Path(source).stat()
    shutil.copystat(source, target)
    if Path(target).stat().st_gid != src.st_gid:
        os.chown(target, -1, src.st_gid)
    os.utime(target, None)  # cache invalidation: repaired content has a new mtime


def finish_commit(journal_path):
    journal_path = Path(journal_path)
    j = json.loads(journal_path.read_text())
    if j['status'] == 'committed':
        if sha_file(j['patch']) != j['patch_sha256']:
            raise ValueError('permanent backup fingerprint changed')
        for fmt in ('terminal', 'tree'):
            if stable_identity(identity(j['source'][fmt])) != stable_identity(j['committed_identity'][fmt]):
                raise ValueError('completed source identity changed; inspect before resuming')
        for p in list(j['old_links'].values()) + list(j['staged'].values()):
            Path(p).unlink(missing_ok=True)
        stage_dir = Path(j['staged']['tree']).parent
        if stage_dir.exists() and not any(stage_dir.iterdir()):
            stage_dir.rmdir()
        return j
    if j['status'] not in ('verified', 'committing'):
        raise ValueError('only verified output can be committed')
    if sha_file(j['patch']) != j['patch_sha256']:
        raise ValueError('backup fingerprint changed before commit')
    for fmt in ('terminal', 'tree'):
        src, old = Path(j['source'][fmt]), Path(j['old_links'][fmt])
        current = sha_file(src)
        if current not in (j['original_sha256'][fmt], j['new_sha256'][fmt]):
            raise ValueError('source changed externally; refusing commit')
        if not old.exists() and current == j['original_sha256'][fmt]:
            os.link(src, old)
            sync_dir(old.parent)
        elif old.exists() and sha_file(old) != j['original_sha256'][fmt]:
            raise ValueError('temporary original hardlink no longer matches original')
        if current != j['new_sha256'][fmt]:
            stage = Path(j['staged'][fmt])
            if sha_file(stage) != j['new_sha256'][fmt]:
                raise ValueError('staged output changed before commit')
    j['status'] = 'committing'
    atomic_json(journal_path, j)
    for fmt in ('terminal', 'tree'):
        src = Path(j['source'][fmt])
        if sha_file(src) != j['new_sha256'][fmt]:
            os.replace(j['staged'][fmt], src)
            sync_dir(src.parent)
    for fmt in ('terminal', 'tree'):
        if sha_file(j['source'][fmt]) != j['new_sha256'][fmt]:
            raise ValueError('post-commit hash verification failed; original links retained')
    j['status'] = 'committed'
    j['committed_at'] = time.time()
    j['committed_identity'] = {fmt: identity(p) for fmt, p in j['source'].items()}
    atomic_json(journal_path, j)
    # Original bytes are now recoverable from the verified delta. Free only the
    # temporary hardlinks; the permanent compressed backup and receipt remain.
    for p in list(j['old_links'].values()) + list(j['staged'].values()):
        Path(p).unlink(missing_ok=True)
    stage_dir = Path(j['staged']['tree']).parent
    if stage_dir.exists() and not any(stage_dir.iterdir()):
        stage_dir.rmdir()
    return j


def campaign_bytes(campaign):
    used = 0
    for path in Path(campaign).rglob('*'):
        try:
            info = path.stat()
        except FileNotFoundError:  # Another runner atomically replaced its JSON.
            continue
        if stat.S_ISREG(info.st_mode):
            used += info.st_size
    return used


def quota_snapshot_fallback(campaign, cap_bytes):
    snapshot = json.loads((Path(campaign)/'quota_snapshot.json').read_text())
    required = cap_bytes + (4 << 30) + snapshot.get('parallel_staging_budget_bytes', 0)
    if snapshot['fileset'] != 'tukw-group' or snapshot['free_bytes'] < required:
        raise OSError('login-side quota snapshot does not cover campaign budget')


def check_backup_quota(campaign, cap_bytes=20 << 30):
    # GPFS df reports the whole filesystem, not the 15-TiB tukw fileset limit.
    if not str(campaign).startswith('/inspurfs/'):
        return
    used = campaign_bytes(campaign)
    if used + (4 << 30) > cap_bytes:
        raise OSError(f'campaign backup budget exhausted: {used}/{cap_bytes} bytes')
    try:
        result = subprocess.run(['/usr/lpp/mmfs/bin/mmlsquota', '-Y', '-j', 'tukw-group', 'inspurfs'],
                                capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as error:
        quota_snapshot_fallback(campaign, cap_bytes)
        print(f'Compute quota query unavailable: {error}; enforcing {cap_bytes} byte campaign cap', flush=True)
        return
    if result.returncode:
        quota_snapshot_fallback(campaign, cap_bytes)
        print(f'Compute quota query unavailable: {result.stderr.strip()}; enforcing {cap_bytes} byte campaign cap', flush=True)
        return
    lines = [line.split(':') for line in result.stdout.splitlines() if line.startswith('mmlsquota:fileset:')]
    header = next(row for row in lines if row[2] == 'HEADER')
    row = next(row for row in lines if row[2] != 'HEADER')
    fields = dict(zip(header[6:], row[6:]))
    free = (int(fields['blockLimit']) - int(fields['blockUsage']) - int(fields['blockInDoubt'])) * 1024
    if free < 4 << 30:
        raise OSError(f'fileset backup headroom below 4 GiB: {free} bytes')


def prepare_shard(root, name, index, campaign, tokenizer, workers, chunk_tokens, commit=False, backup_cap_bytes=20 << 30):
    root, campaign = Path(root), Path(campaign)
    backup = campaign / 'backups' / f'{index:03d}'
    journal_path = backup / 'journal.json'
    if journal_path.exists():
        j = json.loads(journal_path.read_text())
        expected_source = {fmt: str(root/fmt/name) for fmt in ('terminal','tree')}
        if (j['name'] != name or j['protocol'] not in (PROTOCOL, LEGACY_PROTOCOL)
                or j['tokenizer_sha256'] != sha_file(tokenizer)
                or j['source'] != expected_source):
            raise ValueError('incompatible previous shard journal')
        if j['status'] in ('verified', 'committing', 'committed'):
            if commit:
                j = finish_commit(journal_path)
            return j
        if j['status'] != 'generating' or any(Path(p).exists() for p in j['old_links'].values()):
            raise ValueError('unsafe interrupted state; original hardlinks retained')
        if any(stable_identity(identity(p)) != stable_identity(j['original_identity'][fmt]) for fmt, p in j['source'].items()):
            raise ValueError('interrupted source changed; refusing automatic restart')
        stamp = time.time_ns()
        os.replace(journal_path, backup/f'journal.interrupted_{stamp}.json')
        if Path(j['patch']).exists():
            os.replace(j['patch'], backup/f'patch.interrupted_{stamp}.bin.gz')
        for p in j['staged'].values():
            Path(p).unlink(missing_ok=True)  # generated outputs only; source unchanged
        Path(next(iter(j['staged'].values()))).parent.rmdir()
    if STOP:
        raise InterruptedError('stop requested before shard generation')
    source = {fmt: root / fmt / name for fmt in ('terminal', 'tree')}
    originals = {fmt: identity(p) for fmt, p in source.items()}
    need = sum(s['bytes'] for s in originals.values()) + (2 << 30)
    if shutil.disk_usage(root).free < need:
        raise OSError(f'not enough same-filesystem staging space: need {need} bytes')
    check_backup_quota(campaign, backup_cap_bytes)
    initial_backup_usage = campaign_bytes(campaign) if str(campaign).startswith('/inspurfs/') else 0
    backup.mkdir(parents=True, exist_ok=True)
    stage = root / '.native_bracket_staging' / f'{index:03d}'
    if stage.exists():
        raise FileExistsError('staging already exists without compatible receipt')
    stage.mkdir(parents=True)
    staged = {fmt: stage / f'{fmt}.raw' for fmt in source}
    patch = backup / 'patch.bin.gz'
    layout = Layout(tokenizer)
    tree = raw_array(source['tree'])
    terminal = raw_array(source['terminal'])
    old_h = {fmt: hashlib.sha256() for fmt in source}
    new_h = {fmt: hashlib.sha256() for fmt in source}
    structural_h = hashlib.sha256()
    extra_h = hashlib.sha256()
    extra_counts = Counter()
    lexical_added_counts = Counter()
    counts, examples, documents, patch_count, candidate_runs = Counter(), [], 0, 0, 0
    embedded_eos_removed, eos_examples = 0, []
    old_terminal_cursor, new_offsets = 0, {'tree': 0, 'terminal': 0}
    started = time.monotonic()
    j = {'protocol': PROTOCOL, 'status': 'generating', 'shard_index': index, 'name': name,
         'source': {k: str(p) for k, p in source.items()}, 'staged': {k: str(p) for k, p in staged.items()},
         'old_links': {k: str(stage / f'{k}.original') for k in source}, 'original_identity': originals,
         'tokenizer_sha256': sha_file(tokenizer), 'script_sha256': sha_file(__file__),
         'normalizer_sha256': sha_file(Path(__file__).resolve().parents[2] / 'datatools/ptb_brackets.py'),
         'environment': {'python': sys.version, 'numpy': np.__version__,
                         'tokenizers': importlib.metadata.version('tokenizers')},
         'patch': str(patch), 'job_id': os.getenv('SLURM_JOB_ID'), 'host': os.uname().nodename}
    j['backup_cap_bytes'] = backup_cap_bytes
    j['embedded_eos_policy'] = 'delete EOS only at positive tree depth from both streams; preserve empty-stack EOS'
    atomic_json(journal_path, j)
    with staged['tree'].open('wb') as tree_out, staged['terminal'].open('wb') as term_out, gzip.open(patch, 'wb', compresslevel=1) as patch_out:
        outputs = {'tree': tree_out, 'terminal': term_out}
        patch_out.write(b'FWPTB001')
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=worker_init,
                initargs=(str(source['tree']), str(source['terminal']), str(tokenizer))) as pool:
            pending = deque()
            def consume(item):
                nonlocal documents, patch_count, candidate_runs, embedded_eos_removed
                task, future = item
                start, end, ts, te = task
                result = future.result()
                a, b = tree[start:end], terminal[ts:te]
                old_h['tree'].update(memoryview(a).cast('B'))
                old_h['terminal'].update(memoryview(b).cast('B'))
                structural_h.update(memoryview(a[layout.nt[a]]).cast('B'))
                extra_h.update(memoryview(a[layout.extra[a]]).cast('B'))
                tree_pos, term_pos = 0, 0
                for tr, po, old, new in result['patches']:
                    old_ids, new_ids = np.frombuffer(old, dtype='<u4'), np.frombuffer(new, dtype='<u4')
                    if not np.array_equal(a[tr:tr+len(old_ids)], old_ids) or not np.array_equal(b[po:po+len(old_ids)], old_ids):
                        raise ValueError('source changed during generation')
                    write_array(tree_out, a[tree_pos:tr], new_h['tree'])
                    write_array(term_out, b[term_pos:po], new_h['terminal'])
                    tn = new_offsets['tree'] + tr - tree_pos
                    pn = new_offsets['terminal'] + po - term_pos
                    patch_out.write(PATCH_HEADER.pack(start+tr, tn, ts+po, pn, len(old_ids), len(new_ids)))
                    patch_out.write(old)
                    patch_out.write(new)
                    new_offsets['tree'] = tn + len(new_ids)
                    new_offsets['terminal'] = pn + len(new_ids)
                    write_array(tree_out, new_ids, new_h['tree'])
                    write_array(term_out, new_ids, new_h['terminal'])
                    tree_pos, term_pos = tr+len(old_ids), po+len(old_ids)
                    patch_count += 1
                write_array(tree_out, a[tree_pos:], new_h['tree'])
                write_array(term_out, b[term_pos:], new_h['terminal'])
                new_offsets['tree'] += len(a)-tree_pos
                new_offsets['terminal'] += len(b)-term_pos
                counts.update(result['counts'])
                extra_counts.update(result['extra_added_tokens'])
                lexical_added_counts.update(result['lexical_added_brackets_reencoded'])
                documents += result['documents']
                embedded_eos_removed += result['embedded_eos_removed']
                eos_examples.extend(result['embedded_eos_examples'][:max(0,12-len(eos_examples))])
                candidate_runs += result['candidate_runs']
                if initial_backup_usage and initial_backup_usage + patch.stat().st_size > backup_cap_bytes:
                    raise OSError('compressed backup exceeds campaign cap; source untouched')
                if len(examples) < 12:
                    examples.extend(result['examples'][:12-len(examples)])
                print(f'{index:03d}: {end}/{len(tree)} tree tokens; {sum(counts.values())} bracket replacements; '
                      f'{embedded_eos_removed} embedded EOS removed', flush=True)
            for start, end in chunks(tree, layout.eos, chunk_tokens, layout=layout):
                if STOP:
                    raise InterruptedError('stop requested during generation; source untouched')
                a = tree[start:end]
                check_ids(a, layout)
                length = int(np.count_nonzero(~layout.nt[a]))
                task = (start, end, old_terminal_cursor, old_terminal_cursor+length)
                old_terminal_cursor += length
                pending.append((task, pool.submit(analyze_chunk, task)))
                if len(pending) >= 2*workers:
                    consume(pending.popleft())
            while pending:
                consume(pending.popleft())
        if old_terminal_cursor != len(terminal):
            raise ValueError('source terminal unmatched tail')
        for handle in outputs.values():
            handle.flush()
            os.fsync(handle.fileno())
    with patch.open('rb') as f:
        os.fsync(f.fileno())
    sync_dir(backup)
    # Release original NFS mappings before unlinking temporary original links.
    del a
    tree._mmap.close()
    terminal._mmap.close()
    for fmt in source:
        if identity(source[fmt]) != originals[fmt]:
            raise ValueError('source identity changed while processing')
        preserve_attributes(source[fmt], staged[fmt])
    print(f'{index:03d}: generation complete; validating complete pair and reverse backup', flush=True)
    verified = verify_pair(staged['tree'], staged['terminal'], layout, chunk_tokens)
    old_sha = {k: h.hexdigest() for k, h in old_h.items()}
    new_sha = {k: h.hexdigest() for k, h in new_h.items()}
    if (verified['sha256']['structural'] != structural_h.hexdigest()
            or verified['sha256']['extra_added'] != extra_h.hexdigest() or verified['documents'] != documents):
        raise ValueError('structure sequence or document count changed')
    for fmt in source:
        if verified['sha256'][fmt] != new_sha[fmt]:
            raise ValueError('staged output differs from generated hash')
        if replay(staged[fmt], patch, fmt, 'reverse') != old_sha[fmt]:
            raise ValueError('delta cannot reconstruct exact original SHA-256')
    if STOP:
        raise InterruptedError('stop requested before commit; source untouched')
    j.update(status='verified', original_sha256=old_sha, new_sha256=new_sha,
             patch_sha256=sha_file(patch), patch_bytes=patch.stat().st_size,
             counts=dict(counts), changed_lexical_runs=patch_count,
             candidate_runs=candidate_runs, documents=documents, original_tokens={k:s['bytes']//4 for k,s in originals.items()},
             extra_added_tokens_preserved=dict(extra_counts),
             lexical_added_brackets_reencoded=dict(lexical_added_counts),
             dedicated_bracket_policy='apply six textual PTB replacements; preserve literal angle characters',
             new_tokens=new_offsets, verified=verified, reverse_backup_original_sha256_verified=True,
             embedded_eos_removed=embedded_eos_removed, embedded_eos_examples=eos_examples,
             original_eos_markers=documents+embedded_eos_removed, record_end_eos_preserved=documents,
             only_bracket_text_changes=embedded_eos_removed==0,
             only_brackets_and_embedded_eos_changes=True,
             tree_structure_unchanged=True, corpus_filtering_applied=False,
             examples=examples, generation_and_validation_seconds=time.monotonic()-started,
             storage_format='headerless little-endian uint32; EOS-only; same filenames')
    atomic_json(journal_path, j)
    if commit:
        j = finish_commit(journal_path)
    return j


def restore_original(campaign, index, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('restore writes only to a fresh output directory')
    j = json.loads((Path(campaign)/'backups'/f'{index:03d}'/'journal.json').read_text())
    if j['status'] != 'committed' or sha_file(j['patch']) != j['patch_sha256']:
        raise ValueError('requires a committed pair and intact backup')
    for fmt in ('terminal', 'tree'):
        if sha_file(j['source'][fmt]) != j['new_sha256'][fmt]:
            raise ValueError('current source no longer matches repaired version')
    output.mkdir(parents=True)
    for fmt in ('terminal', 'tree'):
        (output/fmt).mkdir()
        restored = output/fmt/j['name']
        if replay(j['source'][fmt], j['patch'], fmt, 'reverse', restored) != j['original_sha256'][fmt]:
            raise ValueError('restoration fingerprint mismatch')
        os.chmod(restored, j['original_identity'][fmt]['mode'])
    atomic_json(output/'restore_receipt.json', {'status': 'complete', 'protocol': PROTOCOL,
        'shard_index': index, 'original_sha256': j['original_sha256'], 'source_modified': False})


def request_stop(signum, frame):
    global STOP
    STOP = True


def validate_parallel_plan(plan, files, root, campaign, runner_id):
    if (plan['data_root'] != str(Path(root).absolute())
            or plan['campaign'] != str(Path(campaign).absolute())
            or plan['file_names'] != [p.name for p in files]):
        raise ValueError('parallel plan does not match dataset / campaign / filenames')
    assignments = plan['assignments']
    if runner_id not in assignments:
        raise ValueError('runner missing from parallel plan')
    selected = [i for indices in assignments.values() for i in indices]
    baseline = plan['committed_baseline']
    if (any(type(i) is not int or not 0 <= i < len(files) for i in selected+baseline)
            or len(set(selected+baseline)) != len(selected+baseline)
            or set(selected+baseline) != set(range(len(files)))):
        raise ValueError('parallel assignments overlap, omit shards, or have invalid indices')
    if any(not indices or indices != sorted(indices) for indices in assignments.values()):
        raise ValueError('parallel assignments must be nonempty sorted lists')
    return assignments[runner_id]


def publish_runner_progress(root, campaign, total_shards, runner):
    """Serialize shared progress writes; per-shard journals remain authoritative."""
    root, campaign = Path(root), Path(campaign)
    runners = campaign/'runners'
    runners.mkdir(exist_ok=True)
    # Coordinate on the same cold-data filesystem whose cross-node flock
    # behavior is validated, rather than the separate campaign mount.
    with file_lock(root/'.native_bracket_progress.lock', blocking=True):
        status_path = runners/f'{runner["runner_id"]}.json'
        if status_path.exists():
            old = json.loads(status_path.read_text())
            if (old.get('job_id'),old.get('pid')) != (runner.get('job_id'),runner.get('pid')):
                history = runners/'history'/runner['runner_id']
                history.mkdir(parents=True,exist_ok=True)
                atomic_json(history/f'{time.time_ns()}.json',old)
        runner = dict(runner, updated_at=time.time())
        atomic_json(status_path,runner)
        committed = []
        for receipt in sorted((campaign/'backups').glob('*/journal.json')):
            row = json.loads(receipt.read_text())
            if row['status']=='committed':
                committed.append(row)
        states = [json.loads(p.read_text()) for p in sorted(runners.glob('*.json'))]
        failures = [r for r in states if r['status'] in ('failed','interrupted')]
        complete = (len(committed)==total_shards
                    and all(r['status']=='completed' for r in states))
        progress = {'protocol':PROTOCOL,'execution_mode':'disjoint_parallel_shards',
                    'total_shards':total_shards,'completed_shards':len(committed),
                    'completed_indices':[r['shard_index'] for r in committed],
                    'status':'complete' if complete else ('failed' if failures else 'partial'),
                    'bracket_replacements':sum(sum(r['counts'].values()) for r in committed),
                    'embedded_eos_removed':sum(r.get('embedded_eos_removed',0) for r in committed),
                    'delta_backup_bytes':sum(r.get('patch_bytes',0) for r in committed),
                    'workers':states,'worker_failures':failures,
                    'processing_shards':[r['processing_shard'] for r in states
                                         if r['status']=='running' and r.get('processing_shard') is not None],
                    'campaign':str(campaign.absolute()),'updated_at':time.time(),
                    'legacy_coordinates':'refer to original streams; use reversible delta'}
        atomic_json(campaign/'progress.json',progress)
        atomic_json(root/'NATIVE_BRACKET_REPAIR_PROGRESS.json',progress)
        return progress


def run_parallel(args, files, indices):
    root, campaign = args.data_root.absolute(), args.campaign.absolute()
    runner = {'runner_id':args.runner_id,'status':'running','processing_shard':None,
              'assigned_indices':indices,'completed_assigned':[],
              'job_id':os.getenv('SLURM_JOB_ID'),'pid':os.getpid(),'host':os.uname().nodename,
              'cpus':os.getenv('SLURM_CPUS_PER_TASK'),'workers':args.workers,
              'plan_sha256':sha_file(args.assignment_plan),'script_sha256':sha_file(__file__)}
    # Shared dataset lock excludes serial repair / restore; a shard lock excludes
    # a duplicate owner. A runner lease prevents competing status writers.
    with dataset_lock(root,shared=True), file_lock(root/'.native_bracket_runner_locks'/f'{args.runner_id}.lock'):
        publish_runner_progress(root,campaign,len(files),runner)
        try:
            for index in indices:
                if STOP:
                    raise InterruptedError('stop requested between shards')
                name = files[index].name
                runner.update(processing_shard=index,name=name)
                publish_runner_progress(root,campaign,len(files),runner)
                print(f'START {args.runner_id} {index:03d} {name}',flush=True)
                with shard_lock(root,index):
                    j = prepare_shard(root,name,index,campaign,args.tokenizer,
                                      args.workers,args.chunk_tokens,args.commit,args.backup_cap_gib << 30)
                runner['completed_assigned'].append(index)
                runner.update(last_status=j['status'],processing_shard=None)
                publish_runner_progress(root,campaign,len(files),runner)
                print(f'DONE {args.runner_id} {index:03d} {j["status"]}; '
                      f'{sum(j["counts"].values())} replacements',flush=True)
            runner.update(status='completed',processing_shard=None)
            publish_runner_progress(root,campaign,len(files),runner)
        except BaseException as error:
            runner.update(status='interrupted' if STOP else 'failed',error=str(error))
            publish_runner_progress(root,campaign,len(files),runner)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--chunk-tokens', type=int, default=1 << 20)
    p.add_argument('--start-shard', type=int, default=0)
    p.add_argument('--stop-shard', type=int, help='exclusive')
    p.add_argument('--runner-id', help='Unique owner of one disjoint parallel assignment')
    p.add_argument('--assignment-plan', type=Path)
    p.add_argument('--commit', action='store_true')
    p.add_argument('--backup-cap-gib', type=int, default=20)
    p.add_argument('--restore-shard', type=int)
    p.add_argument('--restore-output', type=Path)
    p.add_argument('--translate-shard', type=int)
    p.add_argument('--translate-format', choices=('terminal','tree'), default='terminal')
    p.add_argument('--old-boundary', type=int, action='append')
    args = p.parse_args()
    if args.workers < 1 or args.chunk_tokens < 16:
        p.error('workers must be positive and chunks at least 16 tokens')
    if bool(args.runner_id) != bool(args.assignment_plan):
        p.error('--runner-id and --assignment-plan must be supplied together')
    if args.runner_id and (not re.fullmatch(r'[A-Za-z0-9_-]+',args.runner_id)
                           or args.start_shard != 0 or args.stop_shard is not None
                           or args.restore_shard is not None or args.translate_shard is not None):
        p.error('parallel mode requires a safe runner ID and plan-only shard selection')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    args.campaign.mkdir(parents=True, exist_ok=True)
    if args.translate_shard is not None:
        j = json.loads((args.campaign/'backups'/f'{args.translate_shard:03d}'/'journal.json').read_text())
        if j['status'] != 'committed' or not args.old_boundary:
            p.error('translation requires a committed shard and --old-boundary values')
        print(json.dumps({'coordinates': translate_boundaries(j['patch'], args.translate_format, args.old_boundary),
                          'target_sha256': j['new_sha256'][args.translate_format]}))
        return
    if args.restore_shard is not None:
        if not args.restore_output:
            p.error('--restore-output is required with --restore-shard')
        with dataset_lock(args.data_root):
            restore_original(args.campaign, args.restore_shard, args.restore_output)
        return
    files = sorted((args.data_root / 'terminal').glob('*.npy'))
    if not files or [p.name for p in files] != sorted(p.name for p in (args.data_root / 'tree').glob('*.npy')):
        raise ValueError('missing/mismatched paired shard filenames')
    if args.runner_id:
        plan = json.loads(args.assignment_plan.read_text())
        indices = validate_parallel_plan(plan,files,args.data_root,args.campaign,args.runner_id)
        run_parallel(args,files,indices)
        return
    selected = files[args.start_shard:args.stop_shard]
    if not selected:
        raise ValueError('empty shard selection')
    with dataset_lock(args.data_root):
        for index in range(args.start_shard, args.start_shard+len(selected)):
            if STOP:
                raise InterruptedError('stop requested between shards')
            name = files[index].name
            atomic_json(args.data_root/'NATIVE_BRACKET_REPAIR_PROGRESS.json', {
                'protocol': PROTOCOL, 'processing_shard': index, 'name': name, 'status': 'in_progress',
                'campaign': str(args.campaign.absolute()), 'job_id': os.getenv('SLURM_JOB_ID'),
                'legacy_coordinates': 'refer to original streams; reversible delta is retained'})
            print(f'START {index:03d} {name}', flush=True)
            try:
                j = prepare_shard(args.data_root.absolute(), name, index, args.campaign.absolute(),
                                  args.tokenizer, args.workers, args.chunk_tokens, args.commit,
                                  args.backup_cap_gib << 30)
            except BaseException as error:
                failed = {'protocol': PROTOCOL, 'status': 'interrupted' if STOP else 'failed',
                          'processing_shard': index, 'name': name, 'error': str(error),
                          'campaign': str(args.campaign.absolute()), 'job_id': os.getenv('SLURM_JOB_ID'),
                          'journal': str(args.campaign.absolute()/'backups'/f'{index:03d}'/'journal.json')}
                atomic_json(args.campaign/'progress.json', failed)
                atomic_json(args.data_root/'NATIVE_BRACKET_REPAIR_PROGRESS.json', failed)
                raise
            completed = []
            for receipt in sorted((args.campaign/'backups').glob('*/journal.json')):
                row = json.loads(receipt.read_text())
                if row['status'] == 'committed':
                    completed.append(row)
            progress = {'protocol': PROTOCOL, 'last_processed_shard': index, 'last_status': j['status'],
                        'total_shards': len(files), 'job_id': os.getenv('SLURM_JOB_ID'),
                        'completed_shards': len(completed),
                        'status': 'complete' if len(completed) == len(files) else 'partial',
                        'bracket_replacements': sum(sum(r['counts'].values()) for r in completed),
                        'embedded_eos_removed':sum(r.get('embedded_eos_removed',0) for r in completed),
                        'delta_backup_bytes': sum(r.get('patch_bytes',0) for r in completed),
                        'campaign': str(args.campaign.absolute()), 'legacy_coordinates': 'refer to original streams; use reversible delta',
                        'updated_at': time.time()}
            atomic_json(args.campaign/'progress.json', progress)
            atomic_json(args.data_root/'NATIVE_BRACKET_REPAIR_PROGRESS.json', progress)
            print(f'DONE {index:03d} {j["status"]}; {sum(j["counts"].values())} replacements', flush=True)


if __name__ == '__main__':
    main()
