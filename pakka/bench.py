"""Threshold sweep: hit rate, false hits and latency for each threshold on a labelled set.

    python -m pakka.bench --embedder all-MiniLM-L6-v2

Each labelled row has the FAQ id it should match (`label`, or None for should-miss), and what
the cache actually returned (`matched_id`, `score`, `lookup_ms`).
"""

import argparse
import json
import time
from pathlib import Path


def sweep(rows, thresholds, llm_ms):
    results = []
    for t in thresholds:
        hits = correct = false_hits = missed = 0
        total_ms = 0.0
        for r in rows:
            hit = r["score"] >= t
            hits += hit
            correct += hit and r["matched_id"] == r["label"]
            false_hits += hit and r["matched_id"] != r["label"]
            missed += not hit and r["label"] is not None
            total_ms += r["lookup_ms"] + (0 if hit else llm_ms)
        results.append({
            "threshold": t, "hits": hits, "correct": correct, "false_hits": false_hits,
            "missed": missed, "hit_rate": hits / len(rows), "avg_latency_ms": total_ms / len(rows),
        })
    return results


def format_table(results):
    lines = ["| Threshold | Hit rate | Correct hits | False hits | Missed paraphrases | Avg latency |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['threshold']:.2f} | {r['hit_rate']:.0%} | {r['correct']} | {r['false_hits']} "
                     f"| {r['missed']} | {r['avg_latency_ms']:.0f} ms |")
    return "\n".join(lines)


def score_rows(cache, data):
    rows = []
    for item in data:
        start = time.perf_counter()
        match = cache.nearest(item["q"])
        rows.append({**item, "matched_id": match.meta.get("id"), "score": match.score,
                     "lookup_ms": (time.perf_counter() - start) * 1000})
    return rows


def main():
    from pakka.cache import NumpyCache
    from pakka.embed import load_embedder

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--faqs", default="data/faqs.json")
    p.add_argument("--data", default="data/pilot.jsonl")
    p.add_argument("--embedder", default="all-MiniLM-L6-v2")
    p.add_argument("--llm-ms", type=float, default=1500, help="LLM latency charged to each miss")
    p.add_argument("--thresholds", default="0.75,0.80,0.85,0.90,0.95")
    args = p.parse_args()

    cache = NumpyCache(load_embedder(args.embedder))
    for faq in json.loads(Path(args.faqs).read_text()):
        cache.store(faq["question"], faq["answer"], {"id": faq["id"]})
    data = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    cache.nearest("warm up")  # first call pays one-time model setup
    rows = score_rows(cache, data)
    print(f"{args.embedder} on {len(rows)} questions (miss latency assumed {args.llm_ms:.0f} ms)\n")
    print(format_table(sweep(rows, [float(t) for t in args.thresholds.split(",")], args.llm_ms)))


if __name__ == "__main__":
    main()
