"""Replay the labelled pilot set through the real bot and print before/after numbers.

    python -m examples.dukaan_bot.replay --no-cache   # baseline: every question goes to the LLM
    python -m examples.dukaan_bot.replay              # with the cache

The cache keeps only the 30 seeded FAQs during a replay (no storing misses), so each hit can be
checked against its label. After both runs it prints the README headline line.
"""

import argparse
import json
import os
import statistics

from examples.dukaan_bot.app import ROOT, build


def select(rows, split):
    """Only the rows of one benchmark split (e.g. held-out "test"), or all rows."""
    return rows if split is None else [r for r in rows if r.get("split") == split]


def summarize(results, data):
    n = len(results)
    hits = [r for r in results if r.hit]
    return {
        "n": n,
        "hits": len(hits),
        "false_hits": sum(r.hit and r.matched_id != item["label"] for r, item in zip(results, data)),
        "llm_calls": n - len(hits),
        "tokens": sum(r.tokens for r in results),
        "avg_ms": statistics.mean(r.latency_ms for r in results),
        "p95_ms": statistics.quantiles([r.latency_ms for r in results], n=20)[-1],
    }


def headline(before, after, threshold):
    return (f"Cache hit rate {after['hits'] / after['n']:.0%} on {after['n']} test questions at threshold "
            f"{threshold} with {after['false_hits']} false hits; LLM calls down "
            f"{1 - after['llm_calls'] / before['llm_calls']:.0%}; average latency down from "
            f"{before['avg_ms']:.0f} ms to {after['avg_ms']:.0f} ms.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "pilot.jsonl"))
    p.add_argument("--threshold", type=float, default=float(os.environ.get("PAKKA_THRESHOLD", "0.70")))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--split", help="replay only this split of a benchmark file, e.g. test")
    p.add_argument("--approved", help="approved answers to load as extra keys (default data/approved.jsonl)")
    args = p.parse_args()

    mode = "nocache" if args.no_cache else "cache"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    # Similarity never exceeds 1, so a threshold of 2 turns the cache off.
    extra = {"approved_path": args.approved} if args.approved else {}
    answerer = build(threshold=2.0 if args.no_cache else args.threshold,
                     log_path=logs / f"replay_{mode}.jsonl", store_misses=False, **extra)
    data = select([json.loads(line) for line in open(args.data) if line.strip()], args.split)
    results = [answerer.ask(item["q"]) for item in data]
    summary = {**summarize(results, data), "threshold": args.threshold}
    (logs / f"replay_{mode}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    before, after = logs / "replay_nocache.json", logs / "replay_cache.json"
    if before.exists() and after.exists():
        after_summary = json.loads(after.read_text())
        print("\n" + headline(json.loads(before.read_text()), after_summary, after_summary["threshold"]))


if __name__ == "__main__":
    main()
