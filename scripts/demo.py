"""The 90-second demo: one conversation through Dukaan Bot that shows every part of pakka.

    python scripts/demo.py              # real LLM via OpenRouter (about 4 calls, well under $0.01)
    python scripts/demo.py --fake-llm   # rehearse for free

Each step says what should happen and prints a check mark or a cross, so the demo checks itself.
"""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.dukaan_bot.app import build  # noqa: E402

STEPS = [  # (user, question, expected outcome, what to say)
    ("ravi", "Can I get my order gift wrapped?", "miss", "a new question goes to the LLM"),
    ("ravi", "Do you gift wrap orders?", "hit", "same meaning, new words: answered from cache"),
    ("ravi", "order gift wrap ho sakta hai?", "hit", "and in Hinglish"),
    ("ravi", "order gift wrap mat karna, plain packing chahiye", "blocked:negation", "'mat' flips the meaning: guard vetoes"),
    ("ravi", "Order 4521 ka gift wrap status?", "blocked:slot+skipped:pii",
     "an order number: the generic answer is vetoed, and the exchange is never stored"),
    ("priya", "Do you gift wrap orders?", "miss", "another customer never sees Ravi's cached answer"),
    ("priya", "How long does a refund take?", "hit", "the shared FAQ answers everyone"),
]


def outcome(r):
    base = "hit" if r.hit else r.reason if r.reason.startswith("blocked:") else "miss"
    return base + ("+skipped:pii" if r.stored == "skipped:pii" else "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fake-llm", action="store_true")
    args = p.parse_args()
    tmp = Path(tempfile.mkdtemp())
    llm = (lambda q: ("(fake LLM answer)", 0)) if args.fake_llm else None
    bot = build(llm=llm, log_path=tmp / "demo.jsonl", pending_path=tmp / "pending.jsonl",
                approved_path=tmp / "approved.jsonl")
    failures = 0
    for user, question, expected, say in STEPS:
        r = bot.ask(question, partition=user)
        got = outcome(r)
        failures += got != expected
        mark = "✓" if got == expected else f"✗ expected {expected}"
        score = "-" if r.score is None else f"{r.score:.2f}"
        print(f"\n[{user}] {question}\n  {r.answer[:110]}\n  {got.upper()} | similarity {score} | "
              f"{r.latency_ms:.0f} ms | {r.tokens} tokens  {mark}  ({say})")
    print(f"\n{len(STEPS) - failures}/{len(STEPS)} steps behaved as expected")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
