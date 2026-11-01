"""The request flow: look up the cache, serve a hit, or call the LLM on a miss."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class Result:
    answer: str
    hit: bool
    reason: str  # "hit", "below_threshold", "empty_cache" or "blocked:<guard reason>"
    score: float | None
    matched_id: str | None
    latency_ms: float
    tokens: int


class CachedAnswerer:
    """`llm(question) -> (answer, tokens)`. Hit when similarity is at or above `threshold` and the
    optional `guard(question, cached_question, cached_answer)` returns no veto reason."""

    def __init__(self, cache, llm, threshold, log_path=None, store_misses=True, guard=None):
        self.cache, self.llm, self.threshold = cache, llm, threshold
        self.log_path, self.store_misses, self.guard = log_path, store_misses, guard

    def ask(self, question):
        start = time.perf_counter()
        match = self.cache.nearest(question)
        veto = None
        if match and match.score >= self.threshold and self.guard:
            veto = self.guard(question, match.question, match.answer)
        if match and match.score >= self.threshold and not veto:
            answer, hit, reason, tokens = match.answer, True, "hit", 0
        else:
            answer, tokens = self.llm(question)
            hit = False
            reason = f"blocked:{veto}" if veto else "empty_cache" if match is None else "below_threshold"
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
