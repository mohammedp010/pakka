"""Find and mask personal identifiers common in Indian support chats.

Regex plus checksums (Verhoeff for Aadhaar, Luhn for cards), so random 12- or 16-digit numbers
don't trip it. shortcut: no detection of personal names; add an NER model (e.g. Presidio) if a
deployment's answers can contain them.
"""

import re

_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
      [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 2, 4, 1],
      [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 2, 1, 8, 7, 6, 0, 3], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def verhoeff_digit(digits):
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[(i + 1) % 8][int(ch)]]
    return _INV[c]


def _verhoeff_ok(digits):
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def luhn_ok(digits):
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def _digits(s):
    return re.sub(r"\D", "", s)


# (kind, pattern, extra check on the match). Order matters only for overlapping matches.
_RULES = [
    ("email", re.compile(r"[\w.\-]+@[A-Za-z][\w\-]*(?:\.[\w\-]+)+"), None),
    ("upi", re.compile(r"[\w.\-]+@[A-Za-z][\w\-]*"), None),
    ("aadhaar", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"), lambda m: _verhoeff_ok(_digits(m))),
    ("card", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), lambda m: luhn_ok(_digits(m))),
    ("phone", re.compile(r"(?<!\d)(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)"), None),
    ("pan", re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b"), None),
    ("ifsc", re.compile(r"\b[A-Za-z]{4}0[A-Za-z0-9]{6}\b"), None),
    ("order_id", re.compile(r"(?i)\b(?:order|ord|od)\s*(?:no\.?|number|id)?\s*#?\s*\d{4,}\b|#\d{4,}\b"), None),
]


def _spans(text):
    found = []
    for kind, pattern, ok in _RULES:
        for m in pattern.finditer(text):
            if ok is None or ok(m.group()):
                found.append((m.start(), m.end(), kind))
    kept, end = [], -1
    for start, stop, kind in sorted(found, key=lambda s: (s[0], -s[1])):  # earliest, then longest
        if start >= end:
            kept.append((start, stop, kind))
            end = stop
    return kept


def find_pii(text):
    """[(kind, matched text)] for every identifier found."""
    return [(kind, text[start:stop]) for start, stop, kind in _spans(text)]


def mask(text):
    for start, stop, kind in reversed(_spans(text)):
        text = f"{text[:start]}<{kind.upper()}>{text[stop:]}"
    return text
