"""Cache backends.

Every backend answers one question: which stored question is nearest, and how similar is it
(cosine, higher is closer)? Deciding hit or miss is the caller's job, so a threshold sweep
needs only one lookup per question.
"""

import re
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

    def __init__(self, embed):
        self.embed = embed
        self._vectors = None
        self._entries = []

    def store(self, question, answer, meta=None):
        vector = self.embed([question])
        self._vectors = vector if self._vectors is None else np.vstack([self._vectors, vector])
        self._entries.append((question, answer, meta or {}))

    def nearest(self, question):
        if not self._entries:
            return None
        scores = self._vectors @ self.embed([question])[0]
        i = int(np.argmax(scores))
        stored_question, answer, meta = self._entries[i]
        return Match(stored_question, answer, float(scores[i]), meta)


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
