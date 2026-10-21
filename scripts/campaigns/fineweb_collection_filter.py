"""Conservative whole-document exclusions for the paired 20B collection.

Text is used only to decide inclusion. No accepted terminal or tree is rewritten.
"""
from __future__ import annotations

import re
from collections import Counter

from scripts.campaigns.fineweb_formula_filter import classify as formula_classify

VERSION = 'paired-collection-quality-v2'
FUNCTION_EQUATION = re.compile(
    r'(?<!\w)[A-Za-zΑ-Ωα-ω]{1,6}\s*\([^\n=]{1,45}\)\s*=(?!=)\s*'
    r'(?:[a-zΑ-Ωα-ω]{1,3}\s*\(|(?:ln|log|exp|sin|cos|tan|sqrt)\b|[-+]?\d|[√∑∫])')
WEB_UI = re.compile(
    r'\b(?:accept\s+(?:all\s+)?cookies|reject\s+(?:all\s+)?cookies|'
    r'cookie\s+(?:settings|preferences|consent|policy)|'
    r'(?:this\s+(?:website|site)|we)\s+uses?\s+cookies|'
    r'by\s+(?:using|continuing|browsing)[^\n.]{0,100}\bcookies|'
    r'manage\s+(?:your\s+)?(?:cookie\s+preferences|consent)|'
    r'skip\s+to\s+(?:main\s+)?content|your\s+email\s+address\s+will\s+not\s+be\s+published|'
    r'leave\s+a\s+reply|log\s*in\s+to\s+(?:comment|continue)|'
    r'(?:please\s+)?enable\s+javascript|javascript\s+is\s+disabled|'
    r'subscribe\s+to\s+(?:our|the)\s+newsletter)\b', re.I)
EXTRACTION_WARNING = re.compile(
    r'\b(?:uncorrected\s+(?:OCR|machine)|(?:machine|automatically)[-\s]+extracted\s+text|'
    r'OCR\s+(?:errors|artefacts|artifacts)|'
    r'(?:this|the)\s+(?:text|document|transcription)[^\n.]{0,60}'
    r'not\s+(?:been\s+)?(?:proofread|corrected))\b', re.I)
MOJIBAKE = re.compile(r'(?:â\s*[€€™œž]|Ã\s*[¼¶¤©±¯]|Â\s*[£©®°])')
RAW_HTML = re.compile(r'<\s*/?\s*(?:html|body|script|iframe|div|table|tr|td)\b', re.I)


def classify(text):
    reasons = []
    formula = formula_classify(text)
    if formula['decision'] != 'keep':
        reasons.append('formula_' + formula['decision'])
    if FUNCTION_EQUATION.search(text):
        reasons.append('function_equation')
    for name, pattern in [('web_boilerplate', WEB_UI), ('extraction_warning', EXTRACTION_WARNING),
                          ('mojibake', MOJIBAKE), ('raw_html', RAW_HTML)]:
        if pattern.search(text):
            reasons.append(name)
    # Only dominant repeated long lines; citations, lists and numeric tables
    # alone are not hard exclusions.
    lines = [' '.join(line.split()) for line in text.splitlines()]
    counts = Counter(line for line in lines if len(line) >= 80)
    if any(n >= 4 and n * len(line) >= .3 * len(text) for line, n in counts.items()):
        reasons.append('dominant_repeated_line')
    return {'keep': not reasons, 'reasons': reasons, 'formula': formula['decision']}
