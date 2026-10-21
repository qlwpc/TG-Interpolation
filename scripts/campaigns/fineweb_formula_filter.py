"""Audit and filter explicit mathematical formulas in FineWeb document JSONL.

No topic or model-quality filtering is performed. Output text restores native
brackets by default; source text, coordinates and hashes remain traceable.
Whole documents with confirmed formulas are separated from keep/review records;
the source files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from datatools import ptb_brackets


VERSION = 'formula-syntax-v5'
MATH_CONTEXT = re.compile(
    r'\b(?:equation|equations|formula|formulas|formulae|algebra|algebraic|'
    r'calculus|polynomial|theorem|derivative|integral|numerator|denominator|'
    r'coefficient|coefficients|quadratic|proportional|logarithm|exponent|'
    r'trigonometric|matrix|matrices|vector|vectors)\b', re.I)
CODE_CONTEXT = re.compile(
    r'\b(?:def\s+\w+\s*(?:\(|-LRB-)|function\s+\w+\s*(?:\(|-LRB-)|'
    r'public\s+static\s+|console\s*\.|printf\s*\(|'
    r'const\s+\w+\s*=|var\s+\w+\s*=|import\s+\w+)')
URL = re.compile(r'https?://\S+', re.I)
LATEX_ENV = re.compile(r'\\\s*begin\s*\{\s*(?:equation|align|eqnarray|gather|multline|math|displaymath)\*?\s*\}', re.I)
LATEX_OPERATION = re.compile(r'\\\s*(?:[dt]?frac|sqrt|binom)\s*(?:\[[^\]\n]{0,30}\]\s*)?\{', re.I)
LATEX_CALCULUS = re.compile(r'\\\s*(?:sum|prod|i{0,2}int|lim)\s*(?:[_^]|\\\s*(?:limits|infty)|\{)', re.I)
MATHML = re.compile(r'<\s*(?:\w+:)?math\b', re.I)
LATEX_DELIMITED = re.compile(r'\\\[(?:.|\n){1,2500}?\\\]|\$\$(?:.|\n){1,2500}?\$\$')
INLINE_TEX = re.compile(r'\$[^$\n]{1,180}\$|\\\([^\n]{1,500}?\\\)')
EQUALITY = re.compile(
    r'(?<![\w])(?P<lhs>(?:[A-Za-zΑ-Ωα-ω]{1,6}|\d+(?:\.\d+)?)'
    r'(?:\s*[_^]\s*[\w{}]+)?(?:\s*\([^=\n]{0,45}\))?'
    r'(?:\s*[+*/^×÷−-]\s*(?:[A-Za-zΑ-Ωα-ω]{1,6}|\d+(?:\.\d+)?)){0,3})'
    r'\s*(?<![<>!:])=(?!=)\s*(?P<rhs>[^\n]{1,160})')
NUMERIC_OPERATION = re.compile(r'(?<![\w.])\d+(?:\.\d+)?\s*[+*/^×÷−-]\s*\d+(?:\.\d+)?(?![\w.])')
SYMBOLIC_OPERATION = re.compile(r'(?<!\w)[A-Za-zΑ-Ωα-ω]{1,3}\s*[+*/^×÷−-]\s*(?:[A-Za-zΑ-Ωα-ω]{1,3}|\d+)(?!\w)')
POWER = re.compile(r'(?:[A-Za-zΑ-Ωα-ω]|\d)\s*\^\s*(?:[\w{]|[-+]?\d)|[A-Za-zΑ-Ωα-ω][²³⁴⁵⁶⁷⁸⁹]')
CALCULUS_GLYPH = re.compile(r'[∑∏∫]\s*(?:[a-zΑ-Ωα-ω0-9(_^]|\\)')
SIMPLE_STATISTIC = re.compile(r'^(?:[npN]|[pP]|OR|HR|RR|CI|SD|SE|M|df|R2|r|score)\s*$', re.I)
SIMPLE_VALUE = re.compile(r'^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?\s*%?(?=\s|$|[),.;|]|-RRB-)')
VALUE_WITH_INTERVAL = re.compile(r'^[-+]?\d+(?:\.\d+)?\s*(?:\(|-LRB-)\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*(?:\)|-RRB-)')
UNIT_NAME = re.compile(r'^(?:fs|ps|ns|us|μs|ms|s|mm|cm|m|km|mg|g|kg|ml|mL|L|mV|V|Hz|kHz|MHz)$')
UNIT_VALUE = re.compile(r'^\d+(?:\.\d+)?(?:\s*\^?\s*-\s*\d+)?\s+(?:s|seconds?|m|meters?|g|grams?|kg|mg|V|Hz|L)\b')
ARITHMETIC_START = re.compile(r'^[+*/^×÷−-]')
META_NAME = re.compile(r'^(?:url|date|title|symbol|publisher|retrieved|accessdate|link_id|id|iid|isbn|lang|referer)$', re.I)
CURRENCY_NAME = re.compile(r'\b(?:USD|EUR|GBP|dollars?|pesos?|zloty|yen|yuan|rupees?|rand|ringgit|won|fee|price|cost)\b', re.I)
EXPR_TOKEN = re.compile(r'\s*(\d+(?:\.\d+)?|\.\d+|[A-Za-zΑ-Ωα-ω][A-Za-zΑ-Ωα-ω0-9²³⁴⁵⁶⁷⁸⁹]*|[_^{}()[\]+*/×÷√−=-]|\.)')
EXPR_STOP = {'is', 'are', 'was', 'were', 'and', 'or', 'the', 'of', 'in', 'to', 'for', 'but', 'can', 'as', 'at', 'it', 'by', 'not', 'has', 'from', 'then', 'we'}


def expression_prefix(rhs):
    """Bound equation evidence to the mathematical prefix, not following prose."""
    pos = end = 0
    while pos < len(rhs):
        m = EXPR_TOKEN.match(rhs, pos)
        if not m:
            break
        word = m.group(1)
        if word[0].isalpha():
            if word.lower() in EXPR_STOP:
                break
            allowed = (len(word) <= 3 or (word.isupper() and len(word) <= 6)
                       or word in {'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'pi', 'ln'}
                       or bool(re.search(r'[Α-Ωα-ω²³⁴⁵⁶⁷⁸⁹]', word)))
            if not allowed:
                break
        end = m.end()
        pos = end
    return rhs[:end].strip().rstrip('.')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def classify(text):
    """Return keep/drop/review and bounded evidence for each distinct signal."""
    raw_text = text
    # Equal-width replacements preserve all original evidence coordinates and
    # prevent PTB bracket names such as LRB-n from looking like subtraction.
    for before, after in [('-LRB-', '(    '), ('-RRB-', '    )'), ('-LSB-', '[    '),
                          ('-RSB-', '    ]'), ('-LCB-', '{    '), ('-RCB-', '    }')]:
        text = text.replace(before, after)
    signals = []
    urls = list(URL.finditer(text)) if 'http' in text.lower() else []

    def add(kind, match, confidence, start=None, end=None):
        a = match.start() if start is None else start
        b = match.end() if end is None else end
        if any(u.start() <= a < u.end() for u in urls):
            return
        signals.append({'kind': kind, 'confidence': confidence, 'start_char': a,
                        'end_char': b, 'matched': raw_text[a:b][:180],
                        'context': raw_text[max(0, a - 100):min(len(text), b + 120)]})

    if '\\' in text:
        for kind, pattern in [('latex_environment', LATEX_ENV), ('latex_fraction_root', LATEX_OPERATION),
                              ('latex_calculus', LATEX_CALCULUS)]:
            for m in pattern.finditer(text):
                add(kind, m, 'confirmed')
    if '<' in text:
        for m in MATHML.finditer(text):
            add('mathml', m, 'confirmed')
    if '\\[' in text or '$$' in text:
        for m in LATEX_DELIMITED.finditer(text):
            if re.search(r'[=^_∑∫√]|\\\s*(?:frac|sqrt|sum|int)', m.group()):
                add('display_math', m, 'confirmed')
    if '$' in text or '\\(' in text:
        for m in INLINE_TEX.finditer(text):
            # Dollar prices and PHP variables can resemble paired TeX dollars.
            content = m.group()
            if CURRENCY_NAME.search(content) or 'http' in content.lower():
                continue
            if re.fullmatch(r'\$\s*\d[\d,.\s]*=\s*(?:us|hk|si|tt|ec|r)\s*\$', content, re.I):
                continue
            if re.search(r'\$\s*(?:[A-Za-z]{2,}|\[)', content) and not re.search(r'\\\s*(?:frac|sqrt|sum|int)', content):
                # Shell/PHP variables, currency-code pairs and BBCode links.
                continue
            if re.search(r'[=^∑∫√]|\\\s*(?:frac|sqrt|sum|int)', content):
                add('inline_math', m, 'confirmed')
            elif '_' in content and MATH_CONTEXT.search(text[max(0, m.start() - 300):m.end() + 300]):
                add('inline_variable_notation', m, 'uncertain')
    if any(c in text for c in '∑∏∫'):
        for m in CALCULUS_GLYPH.finditer(text):
            add('calculus_glyph', m, 'confirmed')
    if '=' in text:
        for m in EQUALITY.finditer(text):
            a, b = m.start(), m.end()
            if any(u.start() <= a < u.end() for u in urls):
                continue
            lhs = m.group('lhs').strip()
            if META_NAME.fullmatch(lhs) or CURRENCY_NAME.fullmatch(lhs):
                continue
            rhs = re.split(r'[,;]|\s+(?:where|which|with|and|is|are|means|denotes|represents|because)\b', m.group('rhs'), maxsplit=1, flags=re.I)[0].strip()
            vicinity = text[max(0, a - 350):min(len(text), b + 350)]
            math_cues = len(set(x.group().lower() for x in MATH_CONTEXT.finditer(vicinity)))
            code = bool(CODE_CONTEXT.search(vicinity))
            # Study measurements/assignments are not a mathematical derivation.
            value = SIMPLE_VALUE.match(rhs)
            if SIMPLE_STATISTIC.fullmatch(lhs) and value:
                if not ARITHMETIC_START.match(rhs[value.end():].lstrip()):
                    continue
            fraction = re.match(r'^\d+\s*/\s*\d+(?!\w)', rhs)
            if SIMPLE_STATISTIC.fullmatch(lhs) and fraction and not math_cues:
                if not re.match(r'^[+*/^×÷(−-]', rhs[fraction.end():].lstrip()):
                    continue
            if VALUE_WITH_INTERVAL.match(rhs) or (UNIT_NAME.fullmatch(lhs) and UNIT_VALUE.match(rhs)):
                continue
            if rhs.startswith(('"', "'", '’', '‘')) and not math_cues:
                continue
            if not math_cues and re.fullmatch(r'[A-Za-z]{2,6}\s*-\s*[A-Za-z]{2,6}', lhs):
                continue
            if not math_cues and re.match(r'^(?:CH\d*|COOH|OH|NH\d*|CO\d*)\b', lhs) and re.match(r'^(?:CH\d*|COOH|OH|NH\d*|CO\d*)\b', rhs):
                continue
            if not math_cues and re.search(r'\b(?:UHD|pixels?|resolution)\b', vicinity, re.I) and re.match(r'^\d+\s*[×x]\s*\d+', rhs):
                continue
            rhs = expression_prefix(rhs)
            if not rhs:
                continue
            if not math_cues and re.fullmatch(r'[A-Za-zΑ-Ωα-ω]{1,6}\s*/\s*[A-Za-zΑ-Ωα-ω]{1,6}', lhs):
                if re.fullmatch(r'[-+]?(?:\d+(?:\.\d+)?|\.\d+)', rhs.strip('()[] ')):
                    # A stated ratio value, rather than a derivation/calculation.
                    continue
            lhs_op = bool(SYMBOLIC_OPERATION.search(lhs) or NUMERIC_OPERATION.search(lhs) or POWER.search(lhs))
            numeric_op = bool(NUMERIC_OPERATION.search(rhs))
            power_rhs = bool(POWER.search(rhs))
            symbolic_op = bool(SYMBOLIC_OPERATION.search(rhs) or power_rhs)
            greek_product = bool(re.match(r'^[A-Za-zΑ-Ωα-ω]+', rhs) and re.search(r'[a-zA-Z][Α-Ωα-ω]|[Α-Ωα-ω][a-zA-Z]', rhs.split()[0]))
            if lhs_op or numeric_op or symbolic_op or greek_product:
                rhs_math_start = bool(re.match(r'^(?:[-+(]|\d|[A-Za-zΑ-Ωα-ω]{1,3}\b)', rhs))
                strong_numeric_op = numeric_op and bool(re.search(r'[+*/^×÷]', rhs))
                strong_lhs_op = lhs_op and (bool(re.search(r'[+*/^×÷]', lhs)) or math_cues >= 1)
                confidence = 'confirmed' if (not code and ((strong_lhs_op and rhs_math_start) or strong_numeric_op or power_rhs or math_cues >= 1 or greek_product)) else 'uncertain'
                add('equation_expression', m, confidence,
                    end=m.start('rhs') + len(rhs))
            elif math_cues >= 2 and re.fullmatch(r'[A-Za-zΑ-Ωα-ω0-9 _^{}().+*/×÷−-]{1,30}', rhs) and len(rhs.split()) <= 5:
                add('symbolic_equality', m, 'uncertain', end=m.start('rhs') + len(rhs))
    # A large opaque symbol region can contain formulas lost in text extraction.
    if not signals and re.search(r'[=^_∑∏∫√]', text):
        for m in re.finditer(r'[^\n]{0,100}[∑∏∫√][^\n]{0,100}', text):
            if MATH_CONTEXT.search(text[max(0, m.start() - 300):m.end() + 300]):
                add('ambiguous_math_symbol', m, 'uncertain')
    confirmed = sum(s['confidence'] == 'confirmed' for s in signals)
    decision = 'drop' if confirmed else ('review' if signals else 'keep')
    counts = Counter(s['kind'] for s in signals)
    return {'decision': decision, 'confirmed_signals': confirmed,
            'uncertain_signals': len(signals) - confirmed, 'signal_counts': dict(counts),
            'signals': signals[:20], 'signals_truncated': len(signals) > 20}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='document JSONL with text and original source coordinates')
    p.add_argument('--output', type=Path, required=True, help='fresh directory')
    p.add_argument('--text-policy', choices=('native-brackets', 'source'), default='native-brackets',
                   help='restore all six PTB bracket escapes, including those inside words')
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    executed_source_sha256 = sha256_file(Path(__file__))
    normalizer_source_sha256 = sha256_file(Path(ptb_brackets.__file__))
    args.output.mkdir(parents=True)
    files = {k: (args.output / f'{k}.jsonl').open('w') for k in ('keep', 'drop', 'review')}
    counts, tokens, kinds = Counter(), Counter(), Counter()
    replacements = Counter()
    changed_records = 0
    references = {}
    with (args.output / 'index.tsv').open('w') as index:
        w = csv.DictWriter(index, fieldnames=['sample_id', 'decision', 'source_file', 'start_token', 'end_token',
                                            'source_tokens', 'token_bytes_sha256', 'confirmed_signals', 'uncertain_signals'], delimiter='\t')
        w.writeheader()
        try:
            for number, line in enumerate(args.input.open(), 1):
                record = json.loads(line)
                raw_text = record.get('source_text', record['text'])
                text = ptb_brackets.normalize_ptb_brackets(raw_text) if args.text_policy == 'native-brackets' else raw_text
                changes = {key: raw_text.count(key) for key in ptb_brackets.BRACKET_MAPPING
                           if args.text_policy == 'native-brackets' and key in raw_text}
                if text != raw_text:
                    record['source_text'] = raw_text
                    changed_records += 1
                replacements.update(changes)
                record['text'] = text
                record['text_normalization'] = {
                    'policy': ptb_brackets.NATIVE_BRACKET_POLICY if args.text_policy == 'native-brackets' else 'source',
                    'replacements': changes,
                    'source_text_sha256': hashlib.sha256(raw_text.encode('utf-8')).hexdigest(),
                    'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    'token_coordinates': 'unchanged source Qwen stream; not normalized or GPST token positions',
                }
                result = classify(record['text'])
                result['evidence_text_field'] = 'text'
                decision = result['decision']
                counts[decision] += 1
                tokens[decision] += record['tokens']
                kinds.update(result['signal_counts'])
                record['formula_filter'] = result
                original = record['file']
                record.setdefault('resolved_file_on_sample_node', original)
                record['file'] = str(Path('/public/home/wangpch/SIST-assets/datasets/fineweb-edu-v2/terminal') / Path(original).name)
                files[decision].write(json.dumps(record, ensure_ascii=False) + '\n')
                w.writerow({'sample_id': record['sample_id'], 'decision': decision, 'source_file': record['file'],
                            'start_token': record['start_token'], 'end_token': record['end_token'],
                            'source_tokens': record['tokens'], 'token_bytes_sha256': record['token_bytes_sha256'],
                            'confirmed_signals': result['confirmed_signals'], 'uncertain_signals': result['uncertain_signals']})
                # Smallest examples are easier to inspect; decisions use full text.
                examples = references.setdefault(decision, [])
                examples.append({'sample_id': record['sample_id'], 'source_tokens': record['tokens'],
                                 'signals': result['signals'], 'text_head': record['text'][:1200]})
                examples.sort(key=lambda x: (x['source_tokens'], x['sample_id']))
                del examples[20:]
                if number % 1000 == 0:
                    print(f'{number} records: {dict(counts)}', flush=True)
        finally:
            for f in files.values():
                f.close()
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'rule_version': VERSION,
               'input': str(args.input.resolve()), 'input_sha256': sha256_file(args.input),
               'script_sha256': executed_source_sha256, 'records': sum(counts.values()),
               'counts': {k: counts[k] for k in files}, 'source_qwen_tokens': {k: tokens[k] for k in files},
               'signal_counts': dict(kinds), 'selection_unit': 'whole EOS-delimited source document span',
               'topic_filter': False, 'model_inference_or_training': False,
               'sampling_scope': 'existing token-stratified 7872-span audit sample; not uniform document sample or full 100BT scan',
               'review_policy': 'separate uncertain records; no automatic exclusion as confirmed formulas',
               'text_policy': args.text_policy, 'changed_text_records': changed_records,
               'bracket_replacements': dict(replacements),
               'normalization_module_sha256': normalizer_source_sha256,
               'normalization_policy': ptb_brackets.NATIVE_BRACKET_POLICY if args.text_policy == 'native-brackets' else 'source',
               'source_text_policy': 'source_text retained for changed records; text itself is original otherwise',
               'formula_evidence_coordinate_space': 'characters in output text field',
               'output_sha256': {f'{k}.jsonl': sha256_file(args.output / f'{k}.jsonl') for k in files}}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'examples.json').write_text(json.dumps(references, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'counts': summary['counts'], 'source_qwen_tokens': summary['source_qwen_tokens']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
