import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAQS = json.loads((ROOT / "data" / "faqs.json").read_text())
PILOT = [json.loads(line) for line in (ROOT / "data" / "pilot.jsonl").read_text().splitlines() if line.strip()]

DIM = 256


def fake_embed(texts):
    """Hashed bag-of-words, unit-normalised: shared words raise cosine. Deterministic and needs no model."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in re.findall(r"\w+", text.lower()):
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


class FakeLLM:
    """Stands in for the real LLM: instant, free, and records every call."""

    def __init__(self, answer="LLM answer", tokens=42):
        self.answer, self.tokens, self.calls = answer, tokens, []

    def __call__(self, question):
        self.calls.append(question)
        return f"{self.answer}: {question}", self.tokens
