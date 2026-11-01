"""What a cache may serve and store.

- Partitions: an answer written for one customer or tenant is only served back to that partition.
  Curated answers live in the GLOBAL partition, which everyone can read.
- Versions: entries are tagged with the knowledge-base version; editing the FAQ retires them all.
- Expiry: answers stored after an LLM call expire after `ttl` seconds.
- Admission: nothing containing personal identifiers is stored, and with `pending_path` set, new
  answers for the shared GLOBAL partition wait in a queue for approval instead of being served.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pakka.pii import find_pii

GLOBAL = "global"


@dataclass
class Policy:
    kb_version: str | None = None
    ttl: float | None = 7 * 86400
    pending_path: Path | str | None = None

    def allow(self, partition):
        allow = {"partition": {GLOBAL, partition or GLOBAL}}
        if self.kb_version:
            allow["kb_version"] = {self.kb_version}
        return allow

    def admit(self, cache, question, answer, partition):
        """Store, queue or skip one new question/answer pair. Returns what happened."""
        if not answer.strip():
            return "skipped:empty"
        if find_pii(question) or find_pii(answer):
            return "skipped:pii"
        target = partition or GLOBAL
        meta = {"partition": target, "kb_version": self.kb_version}
        if target == GLOBAL and self.pending_path:
            with open(self.pending_path, "a") as f:
                f.write(json.dumps({"question": question, "answer": answer, **meta}, ensure_ascii=False) + "\n")
            return "queued"
        cache.store(question, answer, meta, ttl=self.ttl)
        return "stored"
