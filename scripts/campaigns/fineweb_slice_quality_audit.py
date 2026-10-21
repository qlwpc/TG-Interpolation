"""Inventory visible representation and content signals in FineWeb JSONL spans.

This is a read-only audit, not a training-data filter. Signals such as tables,
code, quotes and navigation phrases need contextual review before any removal.
All character evidence names its text field; source token coordinates stay raw.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


VERSION = 'fineweb-visible-slice-audit-v3'
PATTERNS = {
    'uncorrected_machine_read_notice': re.compile(r'\b(?:uncorrected machine\s*-\s*read text|uncorrected material)\b', re.I),
    'replacement_character': re.compile('\ufffd'),
    'control_character': re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'),
    'mojibake_candidate': re.compile(r'Ã[\u0080-\u00bf]|â€[™œ\u009d“”¦]|Â[\u00a0£©®°]|ðŸ'),
    'literal_tree_marker': re.compile(r'<\((?:ADJP|ADVP|CONJP|FRAG|INTJ|LST|NAC|NML|NP|PP|PRN|PRT|QP|RRC|S|SBAR|SBARQ|SINV|SQ|UCP|VP|WHADJP|WHADVP|WHNP|WHPP|X)>|<(?:ADJP|ADVP|CONJP|FRAG|INTJ|LST|NAC|NML|NP|PP|PRN|PRT|QP|RRC|S|SBAR|SBARQ|SINV|SQ|UCP|VP|WHADJP|WHADVP|WHNP|WHPP|X)\)>'),
    'ptb_quote': re.compile(r'(?<!\w)(?:``|\'\')(?!\w)'),
    'split_contraction': re.compile(r'\b[A-Za-z]+\s+(?:n[\'’]t|[\'’](?:s|re|ve|ll|d|m))\b'),
    'space_before_punctuation': re.compile(r'(?<=[A-Za-z0-9]) +[,.!?;:](?=\s|$)'),
    'html_entity_candidate': re.compile(r'&\s*(?:amp|nbsp|lt|gt|quot|apos|#\s*\d{2,7}|#x\s*[0-9A-Fa-f]{2,6})\s*;', re.I),
    'html_tag_candidate': re.compile(r'<\s*/?\s*(?:div|p|span|a|br|ul|li|table|td|tr|html|body|script|style|h[1-6])\b[^<>\n]{0,200}>', re.I),
    'wiki_citation_markup_candidate': re.compile(r'\{\{|\}\}|\[\[[^\]\n]{1,100}\]\]|\|\s*(?:url|title|date|publisher|accessdate|archiveurl|author|doi|isbn)\s*=', re.I),
    'web_ui_phrase_candidate': re.compile(r'\b(?:skip to (?:main )?content|accept all cookies|cookie (?:settings|preferences)|enable javascript|javascript is disabled|subscribe to our newsletter|leave a reply|your email address will not be published|log in to (?:comment|continue))\b', re.I),
    'code_syntax_candidate': re.compile(r'\b(?:def\s+[A-Za-z_]\w*\s*\(|function\s+[A-Za-z_]\w*\s*\(|public\s+static\s+|console\s*\.\s*log\s*\()|#\s*include\s*<'),
    'function_equation_candidate': re.compile(r'\b[A-Za-zΑ-Ωα-ω]\w{0,3}\s*\([^()\n]{0,30}\)\s*=\s*(?:ln|log|exp|sin|cos|tan|sqrt)\s*\([^()\n]{1,100}\)', re.I),
}
PTB_BRACKET = re.compile(r'-(?:LRB|RRB|LCB|RCB|LSB|RSB)-')


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def evidence(text, start, end, field='text'):
    return {'text_field': field, 'start_char': start, 'end_char': end,
            'matched': text[start:end], 'context': text[max(0, start - 130):min(len(text), end + 180)]}


def inspect_text(text):
    signals = {}
    for name, pattern in PATTERNS.items():
        count, examples = 0, []
        for match in pattern.finditer(text):
            count += 1
            if len(examples) < 2:
                examples.append(evidence(text, match.start(), match.end()))
        if count:
            signals[name] = {'occurrences': count, 'examples': examples}
    if text.count('|') >= 10:
        match = re.search(r'[^\n]{0,90}\|[^\n]{0,90}', text)
        signals['pipe_table_candidate'] = {'occurrences': text.count('|'),
                                            'examples': [evidence(text, match.start(), match.end())]}
    lines = defaultdict(list)
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if len(stripped) >= 80 and len(stripped.split()) >= 8:
            lines[stripped].append(pos + len(line) - len(line.lstrip()))
        pos += len(line)
    repeats = [(line, starts) for line, starts in lines.items() if len(starts) >= 3]
    if repeats:
        repeated_chars = sum((len(starts) - 1) * len(line) for line, starts in repeats)
        line, starts = max(repeats, key=lambda item: len(item[1]) * len(item[0]))
        signals['repeated_long_line_candidate'] = {
            'occurrences': sum(len(starts) - 1 for _, starts in repeats),
            'extra_repeated_character_fraction': repeated_chars / max(1, len(text)),
            'most_repeated_line_copies': len(starts),
            'examples': [evidence(text, start, min(start + len(line), start + 250)) for start in starts[:2]],
        }
    return signals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True, help='fresh audit directory')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('use a fresh audit directory')
    args.output.mkdir(parents=True)
    script_sha256 = sha256_file(Path(__file__))
    counts, by_decision, occurrences = Counter(), defaultdict(Counter), Counter()
    selections, tokens = Counter(), Counter()
    candidates, content_groups, rows = defaultdict(list), defaultdict(list), []
    seen = set()
    fields = ['sample_id', 'decision', 'source_file', 'start_token', 'end_token',
              'source_qwen_tokens', 'token_bytes_sha256', 'signals']
    for source in args.input:
        with source.open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                sid = record['sample_id']
                if sid in seen:
                    raise ValueError(f'duplicate sample identity: {sid}')
                seen.add(sid)
                text = record['text']
                raw = record.get('source_text', text)
                signals = inspect_text(text)
                raw_matches = list(PTB_BRACKET.finditer(raw))
                if raw_matches:
                    signals['source_ptb_brackets'] = {'occurrences': len(raw_matches),
                        'examples': [evidence(raw, m.start(), m.end(), 'source_text' if 'source_text' in record else 'text') for m in raw_matches[:2]]}
                text_matches = list(PTB_BRACKET.finditer(text))
                if text_matches:
                    signals['output_ptb_brackets'] = {'occurrences': len(text_matches),
                        'examples': [evidence(text, m.start(), m.end()) for m in text_matches[:2]]}
                decision = record.get('formula_filter', {}).get('decision', 'not_filtered')
                selections[decision] += 1
                tokens[decision] += record['tokens']
                ref = {'sample_id': sid, 'decision': decision, 'source_file': record['file'],
                       'start_token': record['start_token'], 'end_token': record['end_token'],
                       'source_qwen_tokens': record['tokens'], 'token_bytes_sha256': record['token_bytes_sha256']}
                content_groups[hashlib.sha256(text.encode('utf-8')).hexdigest()].append(ref)
                rows.append({**ref, 'signals': json.dumps({key: value['occurrences'] for key, value in signals.items()}, ensure_ascii=False)})
                for name, signal in signals.items():
                    counts[name] += 1
                    by_decision[decision][name] += 1
                    occurrences[name] += signal['occurrences']
                    group = candidates[name]
                    group.append({**ref, **signal})
                    group.sort(key=lambda r: (-r['occurrences'], r['source_qwen_tokens'], r['sample_id']))
                    del group[15:]
                if len(seen) % 1000 == 0:
                    print(f'{len(seen)} records audited', flush=True)
    duplicates = [{'text_sha256': digest, 'records': refs} for digest, refs in content_groups.items() if len(refs) > 1]
    with (args.output/'index.tsv').open('w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    (args.output/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2)+'\n')
    (args.output/'exact_duplicate_groups.json').write_text(json.dumps(duplicates, ensure_ascii=False, indent=2)+'\n')
    summary = {'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'audit_version': VERSION, 'executed_script_sha256': script_sha256,
               'inputs': {str(path): sha256_file(path) for path in args.input},
               'records': len(seen), 'formula_decisions': dict(selections), 'source_qwen_tokens': dict(tokens),
               'signal_records': dict(counts), 'signal_occurrences': dict(occurrences),
               'signal_records_by_formula_decision': {key: dict(value) for key, value in by_decision.items()},
               'exact_duplicate_groups': len(duplicates), 'exact_duplicate_records': sum(len(group['records']) for group in duplicates),
               'sampling_scope': '7872 token-stratified EOS spans, not uniform-document prevalence or a full-corpus audit',
               'text_field': 'text in native-bracket output; source_ptb_brackets uses original decoded text',
               'signals_are_automatic_filter_rules': False, 'training_data_or_filter_decisions_modified': False,
               'not_checked': ['original webpage completeness or upstream revision identity',
                               'all raw token IDs or special tokens hidden by decode(skip_special_tokens=True)',
                               'terminal/tree document alignment', 'near-duplicates across all 100BT',
                               'factual accuracy, language identification or model training effects']}
    summary['output_sha256'] = {name: sha256_file(args.output/name) for name in ['index.tsv', 'candidates.json', 'exact_duplicate_groups.json']}
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: summary[key] for key in ['records', 'signal_records', 'exact_duplicate_groups', 'exact_duplicate_records']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
