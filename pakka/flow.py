"""The request flow: look up the cache, serve a hit, or call the LLM on a miss."""

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from pakka.pii import mask


@dataclass
class Result:
    answer: str
    hit: bool
    reason: str  # "hit", "below_threshold", "empty_cache", "blocked:<guard reason>", "blocked:verify" or "blocked:verify_error"
    score: float | None
    matched_id: str | None
    latency_ms: float
    tokens: int
    stored: str | None = None  # on a miss: "stored", "queued", "skipped:<why>", or None if not storing


class CachedAnswerer:
    """`llm(question) -> (answer, tokens)`. Hit when similarity is at or above `threshold` and the
    optional `guard(question, cached_question, cached_answer)` returns no veto reason. With a
    `policy` (pakka.policy.Policy), lookups and stores respect partitions, versions, expiry and PII. With
    `verify(question, cached_question, cached_answer) -> bool` (e.g. pakka.verify.load_verifier("jev")), a hit the
    guards let through is served only if it says yes; it sees PII-masked text, and an error counts as a miss."""

    def __init__(self, cache, llm, threshold, log_path=None, store_misses=True, guard=None, policy=None, verify=None):
        self.cache, self.llm, self.threshold = cache, llm, threshold
        self.log_path, self.store_misses, self.guard, self.policy = log_path, store_misses, guard, policy
        self.verify = verify

    def ask(self, question, partition=None):
        start = time.perf_counter()
        if self.policy:
            match = self.cache.nearest(question, allow=self.policy.allow(partition))
        else:
            match = self.cache.nearest(question)
        veto = None
        if match and match.score >= self.threshold and self.guard:
            veto = self.guard(question, match.question, match.answer)
        if match and match.score >= self.threshold and not veto and self.verify:
            veto = self._verify(question, match)
        stored = None
        if match and match.score >= self.threshold and not veto:
            answer, hit, reason, tokens = match.answer, True, "hit", 0
        else:
            answer, tokens = self.llm(question)
            hit = False
            reason = f"blocked:{veto}" if veto else "empty_cache" if match is None else "below_threshold"
            if self.store_misses and self.policy:
                stored = self.policy.admit(self.cache, question, answer, partition)
            elif self.store_misses:
                self.cache.store(question, answer)
                stored = "stored"
        result = Result(
            answer, hit, reason,
            score=None if match is None else match.score,
            matched_id=match.meta.get("id") if match and hit else None,
            latency_ms=(time.perf_counter() - start) * 1000,
            tokens=tokens,
            stored=stored,
        )
        self._log(question, result)
        return result

    def _verify(self, question, match):
        try:  # masked, since a verifier is often a remote API; any failure, a timeout too, must not serve the hit
            ok = self.verify(mask(question), mask(match.question), mask(match.answer))
        except Exception:
            return "verify_error"
        return None if ok else "verify"

    def _log(self, question, result):
        if not self.log_path:
            return
        # Identifiers are masked so logs can be kept for audits without holding customers' data.
        row = {"ts": datetime.now(timezone.utc).isoformat(), "question": mask(question), **asdict(result)}
        del row["answer"]
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
