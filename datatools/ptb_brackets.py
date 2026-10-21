"""Restore PTB bracket escapes in terminal text, without changing spacing.

Apply to plain text or parse leaves, never to serialized tree labels. Benepar
also escapes brackets inside a leaf (e.g. ``alpha-LRB-beta-RRB-``).
"""

BRACKET_MAPPING = {
    "-LRB-": "(", "-RRB-": ")", "-LCB-": "{", "-RCB-": "}",
    "-LSB-": "[", "-RSB-": "]",
}
NATIVE_BRACKET_POLICY = "native-ptb-brackets-v2"
PRESERVED_BRACKET_POLICY = "ptb-brackets-preserved-v1"


def normalize_ptb_brackets(text: str) -> str:
    for escaped, literal in BRACKET_MAPPING.items():
        text = text.replace(escaped, literal)
    return text
