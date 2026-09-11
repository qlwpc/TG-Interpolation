from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

RESERVED = tuple(f"CC-MAIN-2023-{s}" for s in ("06", "14", "23", "40", "50"))
PTB = dict(zip(("-LRB-", "-RRB-", "-LCB-", "-RCB-", "-LSB-", "-RSB-", "Ċ"),
               ("(", ")", "{", "}", "[", "]", "\n")))
LEAF = re.compile(r"\([^()\s]+\s+([^()\s]+)\)")
WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def records(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def normalize(text):
    # Token spacing makes parser leaves and decoded BPE text comparable. Numbers
    # and punctuation are preserved. This string is NEVER used as scoring text.
    text = unicodedata.normalize("NFKC", text)
    for a, b in PTB.items():
        text = text.replace(a, b)
    return " ".join(WORD.findall(text.casefold()))


def body_from_parse(parsed):
    # Fast common path, with an independent NLTK leaf parser for unusual forms.
    from diagnostics.audit_bbc_raw_duplicates import TREE_EVENTS
    leaves, cursor, depth = [], 0, 0
    for m in TREE_EVENTS.finditer(parsed):
        if parsed[cursor:m.start()].strip():
            break
        cursor = m.end()
        preterminal, leaf, label = m.groups()
        if preterminal is not None:
            leaves.append(PTB.get(leaf, leaf))
        elif label is not None:
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                break
    else:
        if not parsed[cursor:].strip() and depth == 0:
            return normalize(" ".join(leaves))
    from nltk import Tree
    root = Tree.fromstring("(ROOT " + parsed.strip() + ")")
    return normalize(" ".join(PTB.get(x, x) for x in root.leaves()))


def ngrams(text, n=5):
    words = text.split()
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}
