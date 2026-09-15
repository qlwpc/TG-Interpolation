"""Require exact full-test document/sentence/token coverage before publishing."""
import argparse
import json
import math
from pathlib import Path


def merge_documents(folder, document_counts, terminal_counts, expected_candidates=None):
    paths = sorted(Path(folder).glob('document_*.json'))
    if len(paths) != len(document_counts):
        raise ValueError(f'incomplete documents: {len(paths)}/{len(document_counts)}')
    fingerprint = None
    sentence_id = 0
    rows = []
    for docid, (path, expected_sentences, expected_tokens) in enumerate(zip(paths, document_counts, terminal_counts)):
        row = json.loads(path.read_text())
        if path.name != f'document_{docid:05d}.json' or row['document_id'] != docid:
            raise ValueError('document order/identity mismatch')
        if fingerprint is None:
            fingerprint = row['fingerprint']
        if row['fingerprint'] != fingerprint:
            raise ValueError('mixed scoring fingerprints')
        sentences = row['sentences']
        if len(sentences) != expected_sentences or row['sentence_count'] != expected_sentences:
            raise ValueError('sentence coverage mismatch')
        if [r['sentence_id'] for r in sentences] != list(range(sentence_id, sentence_id + expected_sentences)):
            raise ValueError('sentence identity mismatch')
        sentence_id += expected_sentences
        if sum(r['terminal_count'] for r in sentences) != expected_tokens or row['terminal_count'] != expected_tokens:
            raise ValueError('terminal denominator mismatch')
        if not all(math.isfinite(r['nll']) and 1 <= r['valid_candidates'] <= 300 and
                   0 <= r['selected_candidate'] < r['valid_candidates'] for r in sentences):
            raise ValueError('invalid sentence result')
        if expected_candidates is not None and any(
                r['valid_candidates'] != int(expected_candidates[r['sentence_id']]) for r in sentences):
            raise ValueError('scored candidate counts differ from frozen logical counts')
        if not math.isclose(math.fsum(r['nll'] for r in sentences), row['total_nll'], rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError('document NLL mismatch')
        if sum(r['selected_candidate'] != 0 for r in sentences) != row['non_candidate0_count']:
            raise ValueError('selection count mismatch')
        rows.append(row)
    nll = math.fsum(r['total_nll'] for r in rows)
    tokens = sum(terminal_counts)
    selected = sum(r['non_candidate0_count'] for r in rows)
    return dict(status='complete', documents=len(rows), sentences=sentence_id, terminal_count=tokens,
                total_nll=nll, docppl=math.exp(nll/tokens), non_candidate0_count=selected,
                non_candidate0_ratio=selected/sentence_id, fingerprint=fingerprint,
                job_ids=sorted(set(r['job_id'] for r in rows)))


def main():
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true', help='publish progress only until all models complete')
    args = p.parse_args()
    campaign = args.campaign
    data = campaign / 'repo/dataset/bbc-news-reserved-clean-v1'
    counts = np.load(data / 'canonical/test/document_sentence_counts.npy').astype(int).tolist()
    offsets = np.load(data / 'terminal/test_doc_offsets.npy')
    tokens = (np.diff(offsets)-1).astype(int).tolist()
    models = json.loads((campaign / 'models.json').read_text())
    report = []
    for model in models:
        folder = campaign / 'results' / model['id']
        paths = sorted(folder.glob('document_*.json'))
        if args.allow_partial and len(paths) < 5000:
            report.append(dict(model_id=model['id'], status='running', documents=len(paths), expected_documents=5000))
            continue
        if model['format'] == 'pushdown':
            from scripts.native_document_results import aggregate_rows
            paths = sorted(folder.glob('document_*.json'))
            rows = [json.loads(p.read_text()) for p in paths]
            row = aggregate_rows('pushdown', rows, expected_ids=set(range(5000)), expected_samples=300)
            for docid, (path, item) in enumerate(zip(paths, rows)):
                if (path.name != f'document_{docid:05d}.json' or item['document_id'] != docid
                        or item['terminal_count'] != tokens[docid] or item['sentence_count'] != counts[docid]):
                    raise ValueError('native document identity/coverage mismatch')
            if row['terminal_count'] != sum(tokens) or row['sentence_count'] != sum(counts):
                raise ValueError('native coverage differs from frozen test')
            row['docppl'] = row['legacy_perplexity']
            row['status'] = 'complete'
        else:
            expected = (np.ones(sum(counts), dtype=np.int64) if model['format'] == 'terminal'
                        else np.load(data / f'testppl/{model["format"]}300/valid_counts.npy'))
            row = merge_documents(folder, counts, tokens, expected)
        row['model_id'] = model['id']
        (folder / 'aggregate.json').write_text(json.dumps(row, indent=2)+'\n')
        report.append(row)
    name = 'results.json' if all(r['status'] == 'complete' for r in report) else 'progress.json'
    tmp = campaign / (name + '.tmp')
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    tmp.replace(campaign / name)


if __name__ == '__main__':
    main()
