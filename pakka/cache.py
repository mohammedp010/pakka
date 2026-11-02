"""Cache backends.

Every backend answers one question: which stored question is nearest, and how similar is it
(cosine, higher is closer)? Deciding hit or miss is the caller's job, so a threshold sweep
needs only one lookup per question.
"""

import re
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Match:
    question: str
    answer: str
    score: float  # cosine similarity
    meta: dict = field(default_factory=dict)


class NumpyCache:
    """In-memory brute-force search over unit vectors, so a dot product is the cosine.

    shortcut: O(n) scan per lookup, fine to ~100k entries; use RedisVLCache beyond that.
    """

    def __init__(self, embed, clock=time.time):
        self.embed, self.clock = embed, clock
        self._vectors = None
        self._entries = []  # (question, answer, meta, expires_at or None)
        self._expiring = False  # any entry with a ttl; until then every entry is live

    def store(self, question, answer, meta=None, ttl=None):
        vector = self.embed([question])
        self._vectors = vector if self._vectors is None else np.vstack([self._vectors, vector])
        self._entries.append((question, answer, meta or {}, None if ttl is None else self.clock() + ttl))
        self._expiring = self._expiring or ttl is not None

    def nearest(self, question, allow=None):
        """Nearest live entry. `allow` maps meta fields to the values a match may have."""
        if not self._entries:
            return None
        scores = self._vectors @ self.embed([question])[0]
        if allow or self._expiring:  # the per-entry check is a Python loop, so skip it when nothing can be filtered
            now = self.clock()
            live = np.array([(expires is None or expires > now) and all(meta.get(k) in v for k, v in (allow or {}).items())
                             for _, _, meta, expires in self._entries], dtype=bool)
            if not live.any():
                return None
            scores = np.where(live, scores, -np.inf)
        i = int(np.argmax(scores))
        stored_question, answer, meta, _ = self._entries[i]
        return Match(stored_question, answer, float(scores[i]), meta)

    def forget(self, field, value):
        """Delete every entry whose meta[field] == value (e.g. one user's data). Returns how many."""
        keep = [i for i, (_, _, meta, _) in enumerate(self._entries) if meta.get(field) != value]
        removed = len(self._entries) - len(keep)
        self._entries = [self._entries[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else None
        return removed


def normalize(text):
    return " ".join(re.findall(r"\w+", text.lower()))


class ExactCache:
    """Exact match after lowercasing and dropping punctuation: score 1.0 on a match, else 0.0.
    The baseline every semantic cache has to beat."""

    def __init__(self):
        self._entries = {}

    def store(self, question, answer, meta=None):
        self._entries[normalize(question)] = Match(question, answer, 1.0, meta or {})

    def nearest(self, question):
        if not self._entries:
            return None
        return self._entries.get(normalize(question)) or Match("", "", 0.0, {})
