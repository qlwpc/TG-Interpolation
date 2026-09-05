"""One-shot dependency bootstrap for the parse/tokenize pipeline (datatools).

Makes the parse scripts self-sufficient: running this before (or from inside)
``benepar_parse.py`` / ``parse_input.py`` / ``worker.py`` guarantees the
models and tokenizers they need are present.  Every step is idempotent and
offline-safe when the artifact already exists locally.

Ensured artifacts:
  1. benepar models  ``benepar_en3``, ``benepar_en3_large``
     (unzipped under the nltk_data search path, e.g. ``~/nltk_data/models``)
  2. spaCy model     ``en_core_web_md`` (sentence segmentation)
  3. HF tokenizer    ``t5-small`` (word->subword budgeting in parse_input)
     Downloads fall back to ``hf-mirror.com`` when huggingface.co is
     unreachable (the mirror is reachable from CN clusters).

Usage:
    python -m datatools.parse_pretrain_data.setup_parse_deps
    python -m datatools.parse_pretrain_data.setup_parse_deps --check

Exit code 0 = all present (or installed), 1 = something missing (``--check``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENEPAR_MODELS = ("benepar_en3", "benepar_en3_large")
SPACY_MODEL = "en_core_web_md"
HF_TOKENIZER = "t5-small"
HF_MIRROR = "https://hf-mirror.com"
HUB_DATASET = "permutans/fineweb-bbc-news"

# spaCy 3.8.x is version-locked to en_core_web_md-3.8.x
SPACY_WHEEL_SOURCES = [
    # GitHub releases (canonical)
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{SPACY_MODEL}-3.8.0/{SPACY_MODEL}-3.8.0-py3-none-any.whl",
    # HuggingFace mirror fallback (spacy/en_core_web_md repo on the mirror)
    f"{HF_MIRROR}/spacy/{SPACY_MODEL}/resolve/main/{SPACY_MODEL}-3.8.0-py3-none-any.whl",
]


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def nltk_data_dirs() -> list[Path]:
    """Candidate nltk_data roots, matching nltk's own search order plus the
    hardcoded legacy path used by parse_input.py."""
    dirs = [Path.home() / "nltk_data"]
    env = os.environ.get("NLTK_DATA")
    if env:
        dirs = [Path(p) for p in env.split(os.pathsep)] + dirs
    dirs.append(Path("/2024233198/nltk_data"))  # legacy path on the H800 cluster
    return dirs


def benepar_model_dir(model: str) -> Path | None:
    """Existing unzipped benepar model dir under any nltk_data root, if any."""
    for root in nltk_data_dirs():
        candidate = root / "models" / model
        if candidate.is_dir():
            return candidate
    return None


def spacy_model_ok(model: str = SPACY_MODEL) -> bool:
    try:
        import spacy.util

        return bool(spacy.util.is_package(model))
    except Exception:
        return False


def hf_snapshot_ok(repo_id: str) -> bool:
    """True if the HF cache already holds a usable snapshot (no network)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# installers
# ---------------------------------------------------------------------------

def install_benepar_model(model: str) -> bool:
    import benepar

    print(f"[setup] downloading benepar model {model!r} ...")
    try:
        benepar.download(model)
    except Exception as e:  # network down and not cached
        print(f"[setup] ERROR: benepar.download({model}) failed: {e}")
        return False
    return benepar_model_dir(model) is not None


def install_spacy_model(model: str = SPACY_MODEL) -> bool:
    import requests

    for url in SPACY_WHEEL_SOURCES:
        try:
            probe = requests.head(url, timeout=8, allow_redirects=True)
            if probe.status_code >= 400:
                print(f"[setup] source unreachable ({probe.status_code}): {url}")
                continue
        except Exception as e:
            print(f"[setup] source unreachable: {url} ({e.__class__.__name__})")
            continue
        print(f"[setup] pip installing {model} from {url} ...")
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "--no-deps", url])
        if rc == 0 and spacy_model_ok(model):
            return True
    print(f"[setup] ERROR: could not install spacy model {model} from any source")
    return False


def install_hf_tokenizer(repo_id: str) -> bool:
    """Prefetch via subprocess so HF_ENDPOINT is honoured before transformers
    gets imported in this process."""
    for endpoint in (None, HF_MIRROR):  # None = default hub; then mirror
        env = dict(os.environ)
        if endpoint:
            env["HF_ENDPOINT"] = endpoint
        src = f"{endpoint or 'https://huggingface.co'}"
        print(f"[setup] prefetching {repo_id!r} from {src} ...")
        rc = subprocess.call(
            [
                sys.executable, "-c",
                "from huggingface_hub import snapshot_download;"
                f"snapshot_download({repo_id!r})",
            ],
            env=env,
        )
        if rc == 0 and hf_snapshot_ok(repo_id):
            return True
        print(f"[setup] prefetch from {src} failed; trying next source")
    print(f"[setup] ERROR: could not prefetch {repo_id}")
    return False


# ---------------------------------------------------------------------------
# direct file fetch (bypasses hf_hub_download's HEAD-metadata handshake,
# which some proxies — including hf-mirror.com — answer 403)
# ---------------------------------------------------------------------------

def fetch_hf_file(
    repo_id: str,
    filename: str,
    out_path: str | Path,
    repo_type: str = "dataset",
    revision: str = "main",
    expected_size: int | None = None,
) -> Path:
    """Streaming-GET one file from the Hub (or the mirror) to ``out_path``.

    Uses plain GET instead of hf_hub_download so that proxies which 403 on
    HEAD still work.  Tries huggingface.co first, then hf-mirror.com.
    """
    import requests
    from tqdm import tqdm

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "datasets" if repo_type == "dataset" else "models"
    last_err: Exception | None = None
    for endpoint in ("https://huggingface.co", HF_MIRROR):
        url = f"{endpoint}/{prefix}/{repo_id}/resolve/{revision}/{filename}"
        fd, name = tempfile.mkstemp(prefix=f".{out_path.name}.", suffix=".part", dir=out_path.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            with requests.get(url, timeout=30, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                with open(temporary, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=out_path.name
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
                expected = expected_size if expected_size is not None else (None if r.headers.get("content-encoding") else total)
                if expected is not None and temporary.stat().st_size != expected:
                    raise IOError(f"incomplete download: {temporary.stat().st_size} bytes, expected {expected}")
            os.replace(temporary, out_path)
            return out_path
        except Exception as e:  # try next endpoint
            last_err = e
            print(f"[setup] GET failed from {endpoint}: {e.__class__.__name__}")
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"could not fetch {repo_id}/{filename}: {last_err}")


def list_dataset_files(repo_id: str, path: str = "") -> list[dict]:
    """List a dataset repo's files via the (mirror-friendly) tree API."""
    import requests

    for endpoint in ("https://huggingface.co", HF_MIRROR):
        try:
            r = requests.get(
                f"{endpoint}/api/datasets/{repo_id}/tree/main/{path}".rstrip("/"),
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    raise RuntimeError(f"could not list {repo_id}/{path}")


def fetch_bbc_shard(config: str, out_root: str | Path = "dataset/bbc-news-raw") -> list[Path]:
    """Download every parquet of one BBC config into ``<out_root>/<config>/``."""
    files = sorted((e for e in list_dataset_files(HUB_DATASET, path=config)
                    if e["type"] == "file" and e["path"].endswith(".parquet")), key=lambda e: e["path"])
    if not files:
        raise RuntimeError(f"no files under {HUB_DATASET}/{config}")
    out_paths = []
    for entry in files:
        out = Path(out_root) / config / Path(entry["path"]).name
        if out.exists() and out.stat().st_size == entry.get("size", -1):
            print(f"[setup] already downloaded: {out}")
        else:
            fetch_hf_file(HUB_DATASET, entry["path"], out, repo_type="dataset", expected_size=entry.get("size"))
        out_paths.append(out)
    return out_paths


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def ensure_all(check_only: bool = False) -> bool:
    """Ensure every artifact; returns True when everything is present."""
    ok = True
    rows = []

    for model in BENEPAR_MODELS:
        present = benepar_model_dir(model) is not None
        if not present and not check_only:
            present = install_benepar_model(model)
        rows.append((f"benepar:{model}", present))
        ok &= present

    present = spacy_model_ok()
    if not present and not check_only:
        present = install_spacy_model()
    rows.append((f"spacy:{SPACY_MODEL}", present))
    ok &= present

    present = hf_snapshot_ok(HF_TOKENIZER)
    if not present and not check_only:
        present = install_hf_tokenizer(HF_TOKENIZER)
    rows.append((f"hf:{HF_TOKENIZER}", present))
    ok &= present

    width = max(len(name) for name, _ in rows)
    for name, present in rows:
        print(f"  {name:<{width}}  {'OK' if present else 'MISSING'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only, no installs")
    parser.add_argument(
        "--fetch-bbc-shard", metavar="CONFIG",
        help="download all parquets of one BBC config (e.g. CC-MAIN-2013-20) "
        "into dataset/bbc-news-raw/<CONFIG>/ for offline --data_files parsing",
    )
    args = parser.parse_args(argv)
    if args.fetch_bbc_shard:
        outs = fetch_bbc_shard(args.fetch_bbc_shard)
        for o in outs:
            print(f"  fetched {o}")
        return 0
    ok = ensure_all(check_only=args.check)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
