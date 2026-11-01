"""How often does a cache serve an answer from before an FAQ edit? With and without version tags.

    python scripts/stale_demo.py

Simulates a persistent cache (as Redis would be) that outlives an FAQ edit: the bot answers every
benchmark question under FAQ v1 (LLM answers are stored), the FAQ is edited, the bot restarts and
re-seeds the FAQ as v2, and every question is asked again. No LLM calls: a stub answers.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pakka.cache import NumpyCache  # noqa: E402
from pakka.embed import load_embedder  # noqa: E402
from pakka.flow import CachedAnswerer  # noqa: E402
from pakka.guard import check  # noqa: E402
from pakka.policy import GLOBAL, Policy  # noqa: E402

THRESHOLD = 0.74  # the guarded setting from the phase 2 bench


def main():
    faqs = json.loads((ROOT / "data/faqs.json").read_text())
    questions = [json.loads(line)["q"] for line in (ROOT / "data/bench_v1.jsonl").read_text().splitlines()]
    cache = NumpyCache(load_embedder("paraphrase-multilingual-MiniLM-L12-v2"))

    def seed(version):
        for faq in faqs:
            cache.store(faq["question"], faq["answer"], {"id": faq["id"], "partition": GLOBAL, "kb_version": version})

    seed("v1")
    v1 = CachedAnswerer(cache, lambda q: (f"(answer written under FAQ v1) {q}", 1), THRESHOLD, guard=check,
                        policy=Policy(kb_version="v1", ttl=None))
    stored = sum(v1.ask(q).stored == "stored" for q in questions)
    print(f"FAQ v1: {stored} LLM answers stored alongside the 30 FAQs. Now the FAQ is edited and re-seeded as v2.\n")
    seed("v2")

    for label, version in [("without version tags", None), ("with version tags", "v2")]:
        allow = {"partition": {GLOBAL}} if version is None else {"partition": {GLOBAL}, "kb_version": {version}}
        hits = stale = 0
        for q in questions:
            m = cache.nearest(q, allow=allow)
            if m and m.score >= THRESHOLD and not check(q, m.question, m.answer):
                hits += 1
                stale += m.meta.get("kb_version") != "v2"
        print(f"{label}: {hits} hits, {stale} from before the edit ({stale / hits:.0%} stale)")


if __name__ == "__main__":
    main()
