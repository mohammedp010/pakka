"""The request flow: look up the cache, serve a hit, or call the LLM on a miss."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class Result:
    answer: str
    hit: bool
    reason: str  # "hit", "below_threshold" or "empty_cache"
    score: float | None
    matched_id: str | None
    latency_ms: float
    tokens: int


class CachedAnswerer:
    """`llm(question) -> (answer, tokens)`. Hit when similarity is at or above `threshold`."""

    def __init__(self, cache, llm, threshold, log_path=None, store_misses=True):
        self.cache, self.llm, self.threshold = cache, llm, threshold
        self.log_path, self.store_misses = log_path, store_misses

    def ask(self, question):
        start = time.perf_counter()
        match = self.cache.nearest(question)
        if match and match.score >= self.threshold:
            answer, hit, reason, tokens = match.answer, True, "hit", 0
        else:
            answer, tokens = self.llm(question)
            hit, reason = False, "empty_cache" if match is None else "below_threshold"
            if self.store_misses:
                self.cache.store(question, answer)
        result = Result(
            answer, hit, reason,
            score=None if match is None else match.score,
            matched_id=match.meta.get("id") if match and hit else None,
            latency_ms=(time.perf_counter() - start) * 1000,
            tokens=tokens,
        )
        self._log(question, result)
        return result

    def _log(self, question, result):
        if not self.log_path:
            return
        # shortcut: logs the raw question; mask PII before logging in phase 3 (policy).
        row = {"ts": datetime.now(timezone.utc).isoformat(), "question": question, **asdict(result)}
        del row["answer"]
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
