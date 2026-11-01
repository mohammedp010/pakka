"""Measure a semantic cache on a labelled question set.

    python -m pakka.bench --data data/bench_v1.jsonl --embedder exact --embedder all-MiniLM-L6-v2

Each labelled row has the FAQ id it should match (`label`, or None for should-miss) and, after
lookup, what the cache returned (`matched_id`, `score`, `lookup_ms`). Rows tagged `split: dev`
choose the threshold; rows tagged `split: test` report the result, so the number is not tuned
on the data it is reported on.
"""

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


def is_correct(row):
    return row["matched_id"] == row["label"]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n), (centre + margin) / (1 + z * z / n)


def is_served(row, threshold):
    """Answered from cache: similar enough, and no guard vetoed it."""
    return row["score"] >= threshold and not row.get("blocked")


def evaluate(rows, threshold, llm_ms=0):
    served = [r for r in rows if is_served(r, threshold)]
    wrong = [r for r in served if not is_correct(r)]
    vetoed = [r for r in rows if r["score"] >= threshold and r.get("blocked")]
    return {
        "threshold": threshold,
        "hits": len(served),
        "correct": len(served) - len(wrong),
        "false_hits": len(wrong),
        "missed": sum(r["label"] is not None and not is_served(r, threshold) for r in rows),
        "blocked_by": dict(Counter(r["blocked"] for r in vetoed)),
        "blocked_correct": sum(map(is_correct, vetoed)),  # good hits a guard threw away
        "hit_rate": len(served) / len(rows),
        "false_hit_rate": len(wrong) / len(served) if served else 0.0,
        "false_hit_upper": wilson(len(wrong), len(served))[1],
        "false_by_kind": dict(Counter(r.get("kind") for r in wrong)),
        "avg_latency_ms": sum(r["lookup_ms"] + (0 if is_served(r, threshold) else llm_ms)
                              for r in rows) / len(rows) if "lookup_ms" in rows[0] else None,
    }


def sweep(rows, thresholds, llm_ms):
    return [evaluate(rows, t, llm_ms) for t in thresholds]


def pchr_auc(rows):
    """Area under precision vs cache-hit ratio (Redis, EMNLP 2026): serve the k most similar
    rows for every k and average the precision. Independent of any threshold."""
    ranked = sorted(rows, key=lambda r: -1 if r.get("blocked") else r["score"], reverse=True)
    correct, total = 0, 0.0
    for k, row in enumerate(ranked, 1):
        correct += is_correct(row)
        total += correct / k
    return total / len(ranked)


def pick_threshold(rows, budget):
    """Lowest threshold (most hits) whose false-hit rate upper bound stays within `budget`."""
    for t in sorted({r["score"] for r in rows}):  # ascending, so the first fit serves the most
        if evaluate(rows, t)["false_hit_upper"] <= budget:
            return t
    return None


def score_rows(cache, data, guard=None):
    """Look up every question once; with `guard(query, cached_question, cached_answer)`, also record its veto."""
    rows = []
    for item in data:
        start = time.perf_counter()
        match = cache.nearest(item["q"])
        row = {**item, "matched_id": match.meta.get("id"), "matched_q": match.question, "score": match.score,
               "lookup_ms": (time.perf_counter() - start) * 1000}
        if guard:
            start = time.perf_counter()
            row["blocked"] = guard(item["q"], match.question, match.answer)
            row["guard_ms"] = (time.perf_counter() - start) * 1000
        rows.append(row)
    return rows


def verify_rows(rows, verify, floor):
    """Re-score grey-band candidates (at or above `floor`, not vetoed) with `verify(pairs) -> scores`,
    e.g. a cross-encoder. Everything else can no longer hit."""
    todo = [i for i, r in enumerate(rows) if r["score"] >= floor and not r.get("blocked")]
    scores = dict(zip(todo, verify([(rows[i]["q"], rows[i]["matched_q"]) for i in todo]) if todo else []))
    return [{**r, "score": float(scores[i]) if i in scores else -1.0} for i, r in enumerate(rows)]


def caught(rows, threshold, lang):
    pool = [r for r in rows if r["label"] is not None and r.get("lang") == lang]
    if not pool:
        return "-"
    return f"{sum(is_served(r, threshold) and is_correct(r) for r in pool)}/{len(pool)}"


def report_encoder(name, rows, budget):
    dev = [r for r in rows if r.get("split") != "test"]
    test = [r for r in rows if r.get("split") == "test"] or dev
    t = pick_threshold(dev, budget)
    line = {"encoder": name, "auc": pchr_auc(test), "threshold": t,
            "lookup_p50_ms": statistics.median(r["lookup_ms"] for r in test),
            "guard_p50_us": statistics.median(r["guard_ms"] * 1000 for r in test) if "guard_ms" in test[0] else None}
    if t is not None:
        line.update(evaluate(test, t))
        line.update({lang: caught(test, t, lang) for lang in ("en", "hinglish", "hi")})
    return line


def format_report(lines, budget, n_test):
    out = [f"Threshold chosen on dev so the false-hit rate's 95% upper bound stays under {budget:.0%}; "
           f"results on {n_test} held-out questions.\n",
           "| Encoder | P-CHR AUC | Threshold | Hit rate | Correct | False hits | False-hit rate (95% upper) "
           "| English caught | Hinglish caught | Hindi caught | Lookup p50 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in lines:
        if x["threshold"] is None:
            out.append(f"| {x['encoder']} | {x['auc']:.3f} | none fits budget | | | | | | | | "
                       f"{x['lookup_p50_ms']:.0f} ms |")
            continue
        out.append(f"| {x['encoder']} | {x['auc']:.3f} | {x['threshold']:.3f} | {x['hit_rate']:.0%} "
                   f"| {x['correct']} | {x['false_hits']} | {x['false_hit_rate']:.1%} ({x['false_hit_upper']:.1%}) "
                   f"| {x['en']} | {x['hinglish']} | {x['hi']} | {x['lookup_p50_ms']:.0f} ms |")
    kinds = sorted({k for x in lines for k in x.get("false_by_kind", {})}, key=str)
    if kinds:
        out += ["\nFalse hits by question type:\n", "| Encoder | " + " | ".join(map(str, kinds)) + " |",
                "|---|" + "---|" * len(kinds)]
        for x in lines:
            if x["threshold"] is not None:
                out.append(f"| {x['encoder']} | " + " | ".join(str(x["false_by_kind"].get(k, 0)) for k in kinds) + " |")
    guarded = [x for x in lines if x.get("guard_p50_us") is not None and x["threshold"] is not None]
    if guarded:
        out += ["\nGuard vetoes on held-out at the chosen threshold:\n",
                "| Encoder | negation | slot | action | Good hits lost | Guard p50 |", "|---|---|---|---|---|---|"]
        for x in guarded:
            b = x["blocked_by"]
            out.append(f"| {x['encoder']} | {b.get('negation', 0)} | {b.get('slot', 0)} | {b.get('action', 0)} "
                       f"| {x['blocked_correct']} | {x['guard_p50_us']:.0f} µs |")
    return "\n".join(out)


def load_cache(name, faqs):
    from pakka.cache import ExactCache, NumpyCache

    if name == "exact":
        cache = ExactCache()
    else:
        from pakka.embed import load_embedder

        cache = NumpyCache(load_embedder(name))
    for faq in faqs:
        cache.store(faq["question"], faq["answer"], {"id": faq["id"]})
    cache.nearest("warm up")  # first call pays one-time model setup
    return cache


def load_verifier(name):
    """A cross-encoder as `verify(pairs) -> scores`, plus its median latency for one pair in ms."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name)
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict([("refund kab milega?", "How long does a refund take?")], show_progress_bar=False)
        timings.append((time.perf_counter() - start) * 1000)
    return (lambda pairs: model.predict(pairs, show_progress_bar=False).tolist()), statistics.median(timings[5:])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--faqs", default="data/faqs.json")
    p.add_argument("--data", default="data/bench_v1.jsonl")
    p.add_argument("--embedder", action="append", help="sentence-transformers model, or 'exact' (repeatable)")
    p.add_argument("--budget", type=float, default=0.05, help="max false-hit rate (95%% upper bound)")
    p.add_argument("--thresholds", help="also print a sweep over these thresholds, e.g. 0.6,0.7,0.8")
    p.add_argument("--llm-ms", type=float, default=1260, help="LLM latency charged to each miss in a sweep")
    p.add_argument("--fail-above", type=float, help="exit 1 if any encoder's false-hit upper bound exceeds this")
    p.add_argument("--guard", action="store_true", help="also report each encoder with pakka.guard rules on")
    p.add_argument("--verify", action="append", default=[], help="cross-encoder to re-score grey-band hits (repeatable)")
    p.add_argument("--floor", type=float, default=0.6, help="lowest similarity a verifier looks at")
    args = p.parse_args(argv)

    faqs = json.loads(Path(args.faqs).read_text())
    data = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    lines = []
    for name in args.embedder or ["paraphrase-multilingual-MiniLM-L12-v2"]:
        from pakka.guard import check

        rows = score_rows(load_cache(name, faqs), data, guard=check if args.guard else None)
        lines.append(report_encoder(name, [{k: v for k, v in r.items() if k not in ("blocked", "guard_ms")}
                                           for r in rows], args.budget))
        if args.guard:
            lines.append(report_encoder(f"{name} + guards", rows, args.budget))
        for model in args.verify:
            verify, single_ms = load_verifier(model)
            line = report_encoder(f"{name}{' + guards' if args.guard else ''} + {model}",
                                  verify_rows(rows, verify, args.floor), args.budget)
            line["lookup_p50_ms"] += single_ms
            lines.append(line)
        if args.thresholds:
            print(f"\n{name} sweep (all rows):\n")
            print(format_table(sweep(rows, [float(t) for t in args.thresholds.split(",")], args.llm_ms)))
    print(format_report(lines, args.budget, sum(r.get("split") == "test" for r in data) or len(data)))

    if args.fail_above is not None:
        failing = [x["encoder"] for x in lines if x.get("false_hit_upper", 1.0) > args.fail_above]
        if failing:
            print(f"\nFAIL: false-hit upper bound above {args.fail_above:.1%} for {', '.join(failing)}")
            sys.exit(1)


def format_table(results):
    lines = ["| Threshold | Hit rate | Correct hits | False hits | Missed paraphrases | Avg latency |",
             "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['threshold']:.2f} | {r['hit_rate']:.0%} | {r['correct']} | {r['false_hits']} "
                     f"| {r['missed']} | {r['avg_latency_ms']:.0f} ms |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
