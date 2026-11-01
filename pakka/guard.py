"""Rule guards: cheap checks that veto a cache hit when the two questions can't want the same answer.

`check(query, cached_question, cached_answer)` returns None when the hit may be served, or the reason
it may not: "negation", "slot", "action" or "entity". Embeddings score these pairs as near-identical,
so the score alone can't catch them.
"""

import re

_EN_NEG = (r"not|no|non|never|none|nothing|without|cannot|nor|\w+n't|dont|didnt|doesnt|isnt|wasnt|werent|cant|wont"
           r"|havent|hasnt|hadnt|arent|shouldnt|couldnt|wouldnt|aint")
# Romanized Hindi spellings. shortcut: "nai" can also mean "new"; drop it if that costs real hits.
_HINGLISH_NEG = r"nahi|nahin|nahee|nahii|nhi|nhin|nai|mat|bina"
_NEGATION = re.compile(rf"\b(?:{_EN_NEG}|{_HINGLISH_NEG})\b")
# Devanagari has no \b for whole words, so require a non-Devanagari character (or edge) on both sides.
_NEGATION_HI = re.compile(r"(?<![ऀ-ॿ])(?:नहीं|नही|मत|बिना)(?![ऀ-ॿ])")

# Yes/no tails ("or not", "ya nahi", "nhi h kya") ask a question; they don't flip its meaning.
_TAGS = re.compile(r"\bor not\b|\bya (?:nahi|nahin|nhi|nai)\b|\b(?:nahi|nahin|nhi) (?:hai|h|he) (?:kya|kyaa)\b|या नहीं")

# Numbers, order IDs, amounts and UPI IDs / emails: if they differ, so does the answer.
_SLOTS = re.compile(r"[\w.\-]+@[\w.\-]+|\d+")

# Actions that read alike but need different answers.
_CONFUSABLE = [("block", "unblock"), ("lock", "unlock"), ("subscribe", "unsubscribe"), ("link", "unlink"),
               ("activate", "deactivate"), ("enable", "disable"), ("delete", "deactivate"),
               ("add", "remove"), ("start", "stop"), ("upgrade", "downgrade"), ("increase", "decrease")]


# Scope words that move a question outside what a domestic FAQ answer covers.
_SCOPE = {"country", "countries", "international", "abroad", "overseas", "outside", "foreign", "videsh"}


def _negated(text):
    text = _TAGS.sub(" ", text.lower().replace("’", "'"))
    return bool(_NEGATION.search(text) or _NEGATION_HI.search(text))


def _words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def _names(text):
    """Capitalised words after the first (likely names: Nepal, Flipkart) plus scope words."""
    tokens = re.findall(r"[A-Za-z][A-Za-z']*", text)
    return {t.lower() for i, t in enumerate(tokens) if (i and t[0].isupper() and t != "I") or t.lower() in _SCOPE}


def check(query, cached, answer=""):
    known = f"{cached} {answer}"
    if _negated(query) != _negated(cached):
        return "negation"
    if set(_SLOTS.findall(query.lower())) - set(_SLOTS.findall(known.lower())):
        return "slot"
    q, c = _words(query), _words(cached)
    for a, b in _CONFUSABLE:
        if (a in q and b in c and b not in q and a not in c) or (b in q and a in c and a not in q and b not in c):
            return "action"
    if _names(query) - _words(known):
        return "entity"
    return None
