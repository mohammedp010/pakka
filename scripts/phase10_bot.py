"""Phase 10: Dukaan Bot end to end on Bitext, through the real LLM.

    .venv/bin/python scripts/phase10_bot.py            # 4,000 held-out questions, the three CPU setups
    .venv/bin/python scripts/phase10_bot.py --n 100    # dry run: the first 100 of the same questions
    .venv/bin/python scripts/phase10_bot.py --setups free,qwen+jev

The bot's knowledge base is Bitext's 27 seed FAQs, each with Bitext's own answer (data/public/bitext.faqs.json).
The questions come from Bitext's test split, which no threshold was picked on and Phase 9 never trained on,
shuffled with a fixed seed, so a smaller --n is the start of a larger one and reuses its calls.

1. Every question goes once to the bot's real LLM, as a bot without a cache sends it (the baseline): answer,
   tokens, cost and time.
2. Each setup replays the questions through the bot's own `build()`: real embedder, guards and verifier, nothing
   stored. A miss gets the baseline's answer and adds its time; a verifier check adds the time it took.
3. Every served hit goes, without its intent label, to the label audit's judges (two, a third on their splits):
   is the reply the customer got a correct answer to the message? The LLM's own reply to the same questions is
   judged the same way, as a control. (The audit's own prompt, "is B the same request as A, with the same
   specifics?", called many fitting Bitext answers wrong, so this asks about the reply itself.)
Every call is kept in data/public/phase10/, so a rerun costs nothing.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from examples.dukaan_bot.app import VERIFY_THRESHOLD, build  # noqa: E402
from examples.dukaan_bot.llm import DEFAULT_MODEL, ENV_FILE, URL, make_chat, system_prompt  # noqa: E402
from pakka.bench import wilson  # noqa: E402
from pakka.testset import parse_json  # noqa: E402
from pakka.embed import DEFAULT_EMBEDDER, GPU_EMBEDDER, THRESHOLDS, TUNED_EMBEDDER, load_embedder  # noqa: E402
from pakka.verify import JEV_MIN_SCORE, _post, jev_verifier, openrouter_verifier  # noqa: E402
from scripts import label_audit as la  # noqa: E402
from scripts.public_bench import log, pct, table  # noqa: E402

OUT = ROOT / "data/public/phase10"
CHEAP = "openrouter:qwen/qwen3.7-flash"
SETUPS = {  # name: (embedder, threshold, verifier)
    "free": (DEFAULT_EMBEDDER, THRESHOLDS[DEFAULT_EMBEDDER], None),
    "tuned+llm": (TUNED_EMBEDDER, VERIFY_THRESHOLD, CHEAP),
    "tuned+jev": (TUNED_EMBEDDER, VERIFY_THRESHOLD, "jev"),
    "qwen+jev": (GPU_EMBEDDER, VERIFY_THRESHOLD, "jev"),
}
VERIFY_COST = {"jev": 0.0000175, CHEAP: 0.000006}  # $ per check, measured in Phase 8.5
JUDGE_SYSTEM = "You check the replies of an online store's customer support bot."
JUDGE_PROMPT = """For each numbered customer message, is the bot's reply a correct answer to the message as asked? True
when it addresses this customer's request and nothing in it is wrong for them. False when it answers a different or
only related request (another action, product, account type, fee or policy), or misses what was asked. Replies are
templates: text in {{{{double braces}}}} gets filled in, and asking for details or pointing to a page or to support
is fine when it is the right next step for this request. Don't judge tone or length.

{items}

Reply with JSON only: {{"1": true or false, ... "{n}": ...}}."""


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def fill(units, fn, store, path, workers=8):
    """`store.update(fn(unit))` for every unit, in parallel, saving `store` to `path` every 200 units."""
    with ThreadPoolExecutor(workers) as pool:
        for i in range(0, len(units), 200):
            for got in pool.map(fn, units[i:i + 200]):
                store.update(got)
            path.write_text(json.dumps(store))
            log(f"  {path.name}: {min(i + 200, len(units))} of {len(units)}")


def key(*parts):
    return hashlib.md5(json.dumps(parts).encode()).hexdigest()


def questions(n):
    faqs = json.loads((ROOT / "data/public/bitext.faqs.json").read_text())
    rows = [json.loads(line) for line in open(ROOT / "data/public/bitext.jsonl")]
    test = list({r["q"]: r for r in rows if r["split"] == "test"}.values())
    random.Random(10).shuffle(test)
    return faqs, test[:n]


def baseline(faqs, rows):
    """The no-cache bot: every question to the LLM once. {question: {answer, tokens, cost, ms}}."""
    path, system = OUT / "llm.json", system_prompt(faqs)
    store = read(path)
    client = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})

    def ask(q):
        body = {"model": DEFAULT_MODEL, "max_tokens": 300, "reasoning": {"enabled": False}, "usage": {"include": True},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}]}
        try:
            start = time.perf_counter()
            data = _post(client, URL, body, attempts=4)
            ms = (time.perf_counter() - start) * 1000
            text, usage = data["choices"][0]["message"]["content"].strip(), data["usage"]
        except Exception as e:  # left out, so a rerun asks again
            log(f"  LLM failed on {q[:40]!r}: {e}")
            return {}
        return {q: {"answer": text, "tokens": usage["total_tokens"], "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"], "cost": usage.get("cost"), "ms": ms}}

    fill([r["q"] for r in rows if r["q"] not in store], ask, store, path)
    if any(r["q"] not in store for r in rows):
        raise SystemExit("Some LLM calls failed; run again to retry just those")
    return store


def check_pairs(verifier, pairs, store):
    """Ask `verifier` about new (question, cached question) pairs as the bot would (one try, 5 s); keep the reply
    (None on an error, which the bot counts as a miss) and the time it took."""
    if verifier == "jev":
        ask = jev_verifier(timeout=5, attempts=1).ask
    else:
        ask = openrouter_verifier(verifier.removeprefix("openrouter:"), timeout=5, attempts=1).ask

    def timed(pair):
        start = time.perf_counter()
        try:
            value = ask(pair)
        except Exception:
            value = None
        return {key(verifier, *pair): {"value": value, "ms": (time.perf_counter() - start) * 1000}}

    fill(pairs, timed, store, OUT / "verify.json", workers=4)


def replay(name, embed, faqs, rows, base, checks):
    """One setup through the bot's own build(). Rows: hit, reason, matched_id, checked, ms (as a user would wait)."""
    _, threshold, verifier = SETUPS[name]
    state = {}

    def verify(question, cached_question, cached_answer):
        state["checked"] = True
        got = checks.get(key(verifier, question, cached_question))
        if got is None:
            state["missing"].append((question, cached_question))
            return False
        state["ms"] += got["ms"]
        if got["value"] is None:
            raise RuntimeError("verifier error")
        return got["value"] >= JEV_MIN_SCORE if verifier == "jev" else got["value"]

    log_path = OUT / f"requests_{name}.jsonl"
    for attempt in (1, 2):  # the first pass finds the pairs a verifier hasn't seen yet
        log_path.unlink(missing_ok=True)
        answerer = build(embed=embed, llm=lambda q: (base[q]["answer"], base[q]["tokens"]), threshold=threshold,
                         log_path=log_path, store_misses=False, faqs=faqs, verify=verify if verifier else None,
                         pending_path=OUT / "pending.jsonl", approved_path=OUT / "approved.jsonl")
        state["missing"], out = [], []
        for r in rows:
            state.update(checked=False, ms=0.0)
            res = answerer.ask(r["q"])
            out.append({"q": r["q"], "label": r["label"], "hit": res.hit, "reason": res.reason, "score": res.score,
                        "matched_id": res.matched_id, "checked": state["checked"],
                        "ms": res.latency_ms + state["ms"] + (0 if res.hit else base[r["q"]]["ms"])})
        if not state["missing"]:
            return out
        log(f"{name}: checking {len(state['missing'])} new pairs with {verifier}")
        check_pairs(verifier, list(dict.fromkeys(state["missing"])), checks)
    raise RuntimeError("verifier pairs still missing after a fill")


def judge_batch(items, chat):
    text = "\n".join(f"{i}. Customer: {la.cut(it['q'])}\n   Reply: {it['answer'][:1500]}" for i, it in enumerate(items, 1))
    reply = parse_json(chat(JUDGE_SYSTEM, JUDGE_PROMPT.format(items=text, n=len(items)))[0])
    verdicts = [reply.get(str(i)) for i in range(1, len(items) + 1)]
    if not all(isinstance(v, bool) for v in verdicts):
        raise ValueError(f"expected true/false for 1-{len(items)}, got {reply!r}")
    return verdicts


def judge(items):
    """{key: True if the reply correctly answers the question}, by majority of the label audit's three judges, and
    the items the first two split on, with all three votes."""
    path = OUT / "judges.json"
    store = read(path)

    def ask(model, picked, reasoning=None):
        chat, done = make_chat(model, max_tokens=8000, reasoning=reasoning), store.setdefault(model, {})
        todo = list({it["key"]: it for it in picked if it["key"] not in done}.values())
        batches = [todo[i:i + la.BATCH] for i in range(0, len(todo), la.BATCH)]

        def one(batch):
            for attempt in range(3):
                try:
                    return dict(zip([it["key"] for it in batch], judge_batch(batch, chat)))
                except Exception as e:  # bad JSON or a flaky call
                    log(f"  {model}: retry {attempt + 1}/3: {e}")
            # DeepSeek sometimes thinks past max_tokens on one batch; alone, each reply usually goes through
            return {k: v for it in batch for k, v in one([it]).items()} if len(batch) > 1 else {}

        with ThreadPoolExecutor(6) as pool:
            for i in range(0, len(batches), 40):
                for got in pool.map(one, batches[i:i + 40]):
                    done.update(got)
                path.write_text(json.dumps(store))
        if any(it["key"] not in done for it in picked):
            raise SystemExit(f"{model} failed on some replies; run again to retry just those")
        return [done[it["key"]] for it in picked]

    a, b = (ask(m, items) for m in la.JUDGES)
    split = [it for it, x, y in zip(items, a, b) if x != y]
    third = dict(zip([it["key"] for it in split], ask(la.ADJUDICATOR, split, {"effort": "low"})))
    verdicts = {it["key"]: x if x == y else third[it["key"]] for it, x, y in zip(items, a, b)}
    log(f"judged {len(items)} replies; the first two split on {len(split)}")
    return verdicts, [{**it, "votes": [x, y, third.get(it["key"])]} for it, x, y in zip(items, a, b) if x != y]


def judged_items(results, faqs, base):
    """Every served hit's (question, stored answer), and the LLM's own answer to each of those questions."""
    answer = {f["id"]: f["answer"] for f in faqs}
    hits = dict.fromkeys((r["q"], r["matched_id"]) for rows in results.values() for r in rows if r["hit"])
    served = [(q, answer[i]) for q, i in hits] + [(q, base[q]["answer"]) for q in dict.fromkeys(q for q, _ in hits)]
    return [{"q": q, "answer": a, "key": key(q, a)} for q, a in served]


def p50_p90(xs):
    if len(xs) < 2:
        return None
    q = statistics.quantiles(xs, n=10)
    return [q[4], q[8]]


def summarize(rows, base, verdicts, verifier, answer):
    hits = [r for r in rows if r["hit"]]
    wrong = sum(not verdicts[key(r["q"], answer[r["matched_id"]])] for r in hits)
    checks = sum(r["checked"] for r in rows)
    verify_cost = checks * VERIFY_COST.get(verifier, 0.0)
    saved = sum(base[r["q"]]["cost"] for r in hits)
    return {"n": len(rows), "hits": len(hits), "wrong": wrong, "wrong_upper": wilson(wrong, len(hits))[1],
            "llm_wrong": sum(not verdicts[key(r["q"], base[r["q"]]["answer"])] for r in hits),
            "label_wrong": sum(r["matched_id"] != r["label"] for r in hits), "checks": checks,
            "verify_errors": sum(r["reason"] == "blocked:verify_error" for r in rows),
            "ms_hits": p50_p90([r["ms"] for r in hits]), "ms_misses": p50_p90([r["ms"] for r in rows if not r["hit"]]),
            "tokens_saved": sum(base[r["q"]]["tokens"] for r in hits), "dollars_saved": saved,
            "verify_dollars": verify_cost, "net_dollars": saved - verify_cost}


def ms(pair):
    return "-" if pair is None else f"{pair[0]:,.0f} / {pair[1]:,.0f}"


def report(summary, base_summary):
    rows = [["No cache (every question to the LLM)", "0", "-", "-", "-", "-", ms(base_summary["ms"]),
             f"{base_summary['tokens']:,} used", f"${base_summary['dollars']:.2f} spent"]]
    for name, s in summary.items():
        embedder, threshold, verifier = SETUPS[name]
        rows.append([f"{embedder.split('/')[-1]} @ {threshold}" + (f" + {verifier}" if verifier else ", free"),
                     f"{s['hits']:,} ({pct(s['hits'] / s['n'], 1)})",
                     f"{s['wrong']} ({pct(s['wrong'] / s['hits'] if s['hits'] else 0)}, {pct(s['wrong_upper'])})",
                     str(s["llm_wrong"]), str(s["label_wrong"]), ms(s["ms_hits"]), ms(s["ms_misses"]), f"{s['tokens_saved']:,}",
                     f"${s['net_dollars']:.3f} ({pct(s['net_dollars'] / base_summary['dollars'], 1)})"])
    return table(["Setup", "Answered from cache", "Wrong answers served, judges (rate, 95% upper)",
                  "LLM's own answer wrong, same questions", "Intent label differs", "Hits p50 / p90 ms", "Misses p50 / p90 ms", "LLM tokens saved",
                  "Saved, verifier paid"], rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--setups", default="free,tuned+llm,tuned+jev")
    args = p.parse_args()
    if ENV_FILE.exists():  # keys for the LLM, judges and verifiers; never printed
        for line in ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() and value.strip():
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
    for name in ("PAKKA_VERIFY", "PAKKA_GUARD", "PAKKA_THRESHOLD", "PAKKA_FAQS"):
        os.environ.pop(name, None)  # every setup here is spelled out below
    OUT.mkdir(parents=True, exist_ok=True)

    faqs, rows = questions(args.n)
    log(f"{len(rows)} Bitext test questions, {len(faqs)} FAQs")
    base = baseline(faqs, rows)
    base_summary = {"n": len(rows), "tokens": sum(base[r["q"]]["tokens"] for r in rows),
                    "dollars": sum(base[r["q"]]["cost"] for r in rows), "ms": p50_p90([base[r["q"]]["ms"] for r in rows])}
    log(f"baseline: {base_summary['tokens']:,} tokens, ${base_summary['dollars']:.3f}")

    checks, results, embedders = read(OUT / "verify.json"), {}, {}
    for name in args.setups.split(","):
        embedder = SETUPS[name][0]
        if embedder not in embedders:
            embedders[embedder] = load_embedder(embedder)
        results[name] = replay(name, embedders[embedder], faqs, rows, base, checks)
        log(f"{name}: {sum(r['hit'] for r in results[name])} hits")

    verdicts, splits = judge(judged_items(results, faqs, base))
    answer = {f["id"]: f["answer"] for f in faqs}
    summary = {name: summarize(rs, base, verdicts, SETUPS[name][2], answer) for name, rs in results.items()}
    (OUT / f"summary_{args.n}.json").write_text(json.dumps({"baseline": base_summary, "setups": summary}, indent=1))
    (OUT / f"splits_{args.n}.json").write_text(json.dumps(splits, indent=1, ensure_ascii=False))
    print(f"\n{len(rows)} questions, knowledge base of {len(faqs)} Bitext FAQs, LLM {DEFAULT_MODEL}.\n")
    print(report(summary, base_summary))


if __name__ == "__main__":
    main()
