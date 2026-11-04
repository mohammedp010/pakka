"""Phase 10b: bring the real bot's wrong cached answers down to about the LLM's own rate, measured honestly.

    .venv/bin/python scripts/phase10b.py --dev 100 --test 0       # dry run: 100 dev questions, project the cost
    .venv/bin/python scripts/phase10b.py                          # 3,000 dev questions, Phase 10's 4,000 test ones

Every choice is made on Bitext's dev split, never on the 4,000 test questions Phase 10 used:
1. Knowledge base: the FAQs whose answers caused the most wrong hits on dev were rewritten into complete answers
   (data/public/phase10b/faqs_fixed.json). Both the original and the fixed KB are reported.
2. Thresholds: for each setup, the similarity threshold (and the verifier's own bar) is picked on dev so the 95%
   Wilson upper bound of wrong answers stays within the budget (3%, and 5%); only then is it run on test.
3. Verifiers: Jev and the cheap LLM from Phase 10 (they see the two questions), stronger OpenRouter models that also
   see the stored reply and give P(yes) from logprobs, and "agree" (Jev and an LLM verifier must both pass).
4. Judges: the same three as Phase 10, on the reply the customer would get (data/public/phase10/judges.json, shared).

Nothing here re-asks the bot's LLM: test misses use Phase 10's recorded answers, cost and time.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from examples.dukaan_bot.app import build  # noqa: E402
from examples.dukaan_bot.llm import ENV_FILE, api_key  # noqa: E402
from pakka.bench import wilson  # noqa: E402
from pakka.embed import DEFAULT_EMBEDDER, TUNED_EMBEDDER, load_embedder  # noqa: E402
from pakka.verify import OPENROUTER_URL, _post  # noqa: E402
from scripts import phase10_bot as p10  # noqa: E402
from scripts.public_bench import log, pct, table  # noqa: E402

OUT = ROOT / "data/public/phase10b"
FLOOR = 0.6  # every lookup at this similarity or more that the guards pass is a candidate; thresholds go up from it
TIMEOUT_MS = 5000  # the bot gives a verifier one try of 5 s; slower counts as a miss
# Different families from the judges (DeepSeek, Haiku 5.5, GPT-5.4). Qwen3.5 and Gemma 4 are open weights and return
# logprobs on OpenRouter; Gemini doesn't, so it is a plain yes / no. (No logprob endpoint, 4 Nov: Qwen3-235B-2507,
# Llama 3.3 70B, GPT-4o-mini.)
MODELS = ["qwen/qwen3.5-397b-a17b", "google/gemma-4-31b-it", "google/gemini-3.1-flash-lite"]
NO_LOGPROBS = {"google/gemini-3.1-flash-lite"}
REPLY_CHECK = (
    "You check a support bot's answer cache before it sends a stored reply. The reply was written for "
    "cached_question. Would it also correctly answer new_question? Answer yes only if the reply deals with what "
    "new_question actually asks for: the same action, item, account type, fee or policy, and it is a useful "
    "answer, not just a request to ask the question. Text in {{double braces}} is filled in later. Answer no if it "
    "answers a different or only related request, the opposite action, or leaves out what was asked.\n"
    "Reply with one word: yes or no.")
VERIFY_GRID = [0.5, 0.7, 0.8, 0.84, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
SPEND_CAP = 10.0  # $ for this phase, from the key's usage when it started


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def key_usage():
    """Dollars used on the OpenRouter key so far (the key itself is never printed)."""
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {api_key()}"}, timeout=30)
    return r.json()["data"]["usage"]


def spend_check(stage):
    path = OUT / "spend.json"
    spend = read(path)
    now = key_usage()
    spend.setdefault("start", 4.106427829)  # the key's usage before this phase's first call
    spend[stage] = now
    path.write_text(json.dumps(spend, indent=1))
    log(f"OpenRouter spend this phase: ${now - spend['start']:.3f} ({stage})")
    if now - spend["start"] > SPEND_CAP:
        raise SystemExit(f"Hard stop: over ${SPEND_CAP} this phase")


def dev_questions(n):
    rows = [json.loads(line) for line in open(ROOT / "data/public/bitext.jsonl")]
    dev = list({r["q"]: r for r in rows if r["split"] == "dev"}.values())
    random.Random(10).shuffle(dev)
    return dev[:n]


def candidates(embed, faqs, rows):
    """Each question through the bot's own build() at the floor, guards on, nothing stored. A row is a candidate
    when it reaches the verifier; `pair` is what the verifier sees (masked, as the bot sends it)."""
    seen = {}

    def record(question, cached_question, cached_answer):
        seen["pair"] = [question, cached_question, cached_answer]
        return True

    answerer = build(embed=embed, llm=lambda q: ("", 0), threshold=FLOOR, log_path=OUT / "requests.jsonl",
                     store_misses=False, faqs=faqs, verify=record,
                     pending_path=OUT / "pending.jsonl", approved_path=OUT / "approved.jsonl")
    out = []
    for r in rows:
        seen.clear()
        res = answerer.ask(r["q"])
        out.append({"q": r["q"], "label": r["label"], "cand": res.hit, "reason": res.reason, "sim": res.score,
                    "id": res.matched_id, "lookup_ms": res.latency_ms, "pair": seen.get("pair")})
    (OUT / "requests.jsonl").unlink(missing_ok=True)
    return out


def p_yes(top):
    """P(yes) from the first answer token's top logprobs, normalised over yes and no."""
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("yes"))
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("no"))
    return yes / (yes + no) if yes + no else None


def llm_check(model, pairs, store, workers=8):
    """P(yes) from `model` for each (question, cached question, cached reply), kept in `store`."""
    client = httpx.Client(timeout=30, headers={"Authorization": f"Bearer {api_key()}"})

    def ask(pair):
        body = {"model": model, "temperature": 0, "max_tokens": 5, "reasoning": {"enabled": False},
                "usage": {"include": True}, "messages": [{"role": "system", "content": REPLY_CHECK},
                             {"role": "user", "content": f"cached_question: {pair[1]}\nreply: {pair[2]}\n"
                                                         f"new_question: {pair[0]}"}]}
        if model not in NO_LOGPROBS:
            body.update(logprobs=True, top_logprobs=10, provider={"require_parameters": True})
        try:
            start = time.perf_counter()
            data = _post(client, OPENROUTER_URL, body, attempts=3)
            ms = (time.perf_counter() - start) * 1000
            choice = data["choices"][0]
            tokens = (choice.get("logprobs") or {}).get("content") or []
            first = next((t for t in tokens if t["token"].strip()), None)
            p = p_yes(first["top_logprobs"]) if first else None
            if p is None:  # no logprobs: fall back to the word itself
                p = 1.0 if (choice["message"].get("content") or "").strip().lower().startswith("yes") else 0.0
        except Exception as e:  # left out, so a rerun asks again
            log(f"  {model} failed: {e}")
            return {}
        return {p10.key(model, REPLY_CHECK, *pair): {"p": p, "ms": ms, "cost": data.get("usage", {}).get("cost") or 0.0,
                                                     "logprobs": bool(first)}}

    todo = list({p10.key(model, REPLY_CHECK, *p): p for p in pairs
                 if p10.key(model, REPLY_CHECK, *p) not in store}.values())
    p10.fill(todo, ask, store, OUT / "verify_llm.json", workers=workers)


def verifier_values(cands, jev, llm):
    """Attach every verifier's reply to each candidate: (value, ms), value None on an error or a timeout."""
    for c in cands:
        if not c["cand"]:
            continue
        q, cq, a = c["pair"]
        v = {}
        for name in ("jev", p10.CHEAP):
            got = jev.get(p10.key(name, q, cq))
            v[name] = (None, TIMEOUT_MS) if got is None or got["value"] is None else (got["value"], got["ms"])
        for m in MODELS:
            got = llm.get(p10.key(m, REPLY_CHECK, q, cq, a))
            v[m] = (None, TIMEOUT_MS) if got is None else (got["p"] if got["ms"] <= TIMEOUT_MS else None, got["ms"])
            v[m + ":cost"] = got["cost"] if got else 0.0
        c["v"] = v


def setups():
    """name: (embedder, verifiers). Each verifier gets its own bar from VERIFY_GRID, except the cheap LLM's yes/no."""
    out = {"free": (DEFAULT_EMBEDDER, []), "tuned": (TUNED_EMBEDDER, []), "tuned+cheap LLM": (TUNED_EMBEDDER, [p10.CHEAP]),
           "tuned+Jev": (TUNED_EMBEDDER, ["jev"])}
    out.update({f"tuned+{m.split('/')[-1]}": (TUNED_EMBEDDER, [m]) for m in MODELS})
    out.update({f"tuned+agree(Jev, {m.split('/')[-1]})": (TUNED_EMBEDDER, ["jev", m]) for m in MODELS})
    return out


def passes(c, verifiers, bars):
    for name, bar in zip(verifiers, bars):
        value = c["v"][name][0]
        if value is None or (value < bar if bar is not None else not value):
            return False
    return True


def served(cands, verifiers, sim, bars):
    return [c for c in cands if c["cand"] and c["sim"] >= sim and passes(c, verifiers, bars)]


def grids(verifiers):
    import itertools

    return list(itertools.product(*[[None] if v == p10.CHEAP else VERIFY_GRID for v in verifiers]))


def pick(cands, verifiers, wrong, budget):
    """(sim, bars) serving the most dev questions while the wrong rate's 95% upper bound stays within `budget`."""
    best = None
    for bars in grids(verifiers):
        ok = sorted((c["sim"], wrong(c)) for c in cands if c["cand"] and passes(c, verifiers, bars))
        bad = sum(w for _, w in ok)
        for i, (sim, w) in enumerate(ok):  # ascending: the first that fits serves the most at this bar
            # a threshold can't split equal similarities, so only try where the similarity changes
            if (i == 0 or ok[i - 1][0] != sim) and wilson(bad, len(ok) - i)[1] <= budget:
                if best is None or len(ok) - i > best[0]:
                    best = (len(ok) - i, sim, bars)
                break
            bad -= w
    return None if best is None else best[1:]


def p50_p90(xs):
    return p10.p50_p90(xs)


def measure(cands, verifiers, sim, bars, wrong, base=None, llm_wrong=None):
    """Test metrics at a fixed (sim, bars). Verifiers in an agree setup run in parallel, so a check waits for the
    slowest one."""
    hits, ms_hit, ms_miss, checks, verify_cost = [], [], [], 0, 0.0
    for c in cands:
        checked = c["cand"] and c["sim"] >= sim and verifiers
        wait = c["lookup_ms"] + (max(c["v"][v][1] for v in verifiers) if checked else 0)
        if checked:
            checks += 1
            verify_cost += sum(c["v"].get(v + ":cost", p10.VERIFY_COST.get(v, 0.0)) for v in verifiers)
        if c["cand"] and c["sim"] >= sim and (not verifiers or passes(c, verifiers, bars)):
            hits.append(c)
            ms_hit.append(wait)
        else:
            ms_miss.append(wait + (base[c["q"]]["ms"] if base else 0))
    bad = sum(wrong(c) for c in hits)
    out = {"n": len(cands), "hits": len(hits), "wrong": bad, "wrong_upper": wilson(bad, len(hits))[1],
           "ms_hits": p50_p90(ms_hit), "ms_misses": p50_p90(ms_miss), "checks": checks, "verify_dollars": verify_cost,
           "label_wrong": sum(c["id"] != c["label"] for c in hits)}
    if base:
        out["dollars_saved"] = sum(base[c["q"]]["cost"] for c in hits)
        out["net_dollars"] = out["dollars_saved"] - verify_cost
        out["llm_wrong"] = sum(llm_wrong(c) for c in hits)
    return out


def unsettled(items, fallback):
    """p10.judge from the stored votes only: a split with no third vote counts as `fallback`."""
    store = read(p10.OUT / "judges.json")
    a, b, third = (store[m] for m in [*p10.la.JUDGES, p10.la.ADJUDICATOR])
    missing = [it for it in items if it["key"] not in a or it["key"] not in b]
    if missing:
        raise SystemExit(f"{len(missing)} replies have no votes from the first two judges")
    verdicts, disputed, left = {}, [], 0
    for it in items:
        x, y = a[it["key"]], b[it["key"]]
        if x != y:
            left += it["key"] not in third
            disputed.append({**it, "votes": [x, y, third.get(it["key"])]})
        verdicts[it["key"]] = x if x == y else third.get(it["key"], fallback == "right")
    log(f"{len(disputed)} splits, {left} unsettled, counted {fallback}")
    return verdicts, disputed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dev", type=int, default=3000)
    p.add_argument("--test", type=int, default=4000)
    p.add_argument("--kb", default="orig,fixed")
    p.add_argument("--no-llm-verify", action="store_true", help="skip the new OpenRouter verifiers (and their cost)")
    p.add_argument("--unsettled", choices=["wrong", "right"],
                   help="no new judge calls: count a split the third judge hasn't settled as wrong (or right)")
    args = p.parse_args()
    if ENV_FILE.exists():  # keys, never printed
        for line in ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() and value.strip():
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
    for name in ("PAKKA_VERIFY", "PAKKA_GUARD", "PAKKA_THRESHOLD", "PAKKA_FAQS"):
        os.environ.pop(name, None)
    OUT.mkdir(parents=True, exist_ok=True)
    spend_check("run start")

    kbs = {"orig": json.loads((ROOT / "data/public/bitext.faqs.json").read_text())}
    if "fixed" in args.kb and (OUT / "faqs_fixed.json").exists():
        kbs["fixed"] = json.loads((OUT / "faqs_fixed.json").read_text())
    splits = {"dev": dev_questions(args.dev), "test": p10.questions(args.test)[1]}
    base = p10.read(p10.OUT / "llm.json")
    log(f"{len(splits['dev'])} dev, {len(splits['test'])} test questions; KBs {list(kbs)}")

    embedders = {e: load_embedder(e) for e in (DEFAULT_EMBEDDER, TUNED_EMBEDDER)}
    cands = {(e, kb, s): candidates(embedders[e], faqs, rows)
             for e in embedders for kb, faqs in kbs.items() for s, rows in splits.items() if rows}
    for k, cs in cands.items():
        log(f"{k}: {sum(c['cand'] for c in cs)} candidates of {len(cs)}")

    # Verifiers. Jev and the cheap LLM see only the two questions, so one set of calls serves both KBs.
    checks = p10.read(p10.OUT / "verify.json")
    pairs = list(dict.fromkeys(tuple(c["pair"][:2]) for (e, _, _), cs in cands.items() if e == TUNED_EMBEDDER
                               for c in cs if c["cand"]))
    for name in ("jev", p10.CHEAP):
        todo = [pr for pr in pairs if p10.key(name, *pr) not in checks]
        log(f"{name}: {len(todo)} new pairs")
        p10.check_pairs(name, todo, checks)
    spend_check("cheap verifiers")
    llm = read(OUT / "verify_llm.json")
    if not args.no_llm_verify:
        triples = list(dict.fromkeys(tuple(c["pair"]) for (e, _, _), cs in cands.items() if e == TUNED_EMBEDDER
                                     for c in cs if c["cand"]))
        for m in MODELS:
            llm_check(m, triples, llm)
            spend_check(m)
    for cs in cands.values():
        verifier_values(cs, checks, llm)

    # Judges: every candidate's reply, and on test the LLM's own reply to every question, as a control.
    answer = {kb: {f["id"]: f["answer"] for f in faqs} for kb, faqs in kbs.items()}
    items = {}
    for (e, kb, s), cs in cands.items():
        for c in cs:
            if c["cand"]:
                a = answer[kb][c["id"]]
                items[p10.key(c["q"], a)] = {"q": c["q"], "answer": a, "key": p10.key(c["q"], a)}
    for r in splits["test"]:
        a = base[r["q"]]["answer"]
        items[p10.key(r["q"], a)] = {"q": r["q"], "answer": a, "key": p10.key(r["q"], a)}
    log(f"judging {len(items)} replies (most already judged in Phase 10)")
    verdicts, disputed = unsettled(list(items.values()), args.unsettled) if args.unsettled else p10.judge(list(items.values()))
    spend_check("judges")
    (OUT / f"splits_{args.dev}_{args.test}.json").write_text(json.dumps(disputed, indent=1, ensure_ascii=False))

    def wrong_fn(kb):
        return lambda c: not verdicts[p10.key(c["q"], answer[kb][c["id"]])]

    def llm_wrong(c):
        return not verdicts[p10.key(c["q"], base[c["q"]]["answer"])]

    results = {"llm_wrong_all": sum(not verdicts[p10.key(r["q"], base[r["q"]]["answer"])] for r in splits["test"]),
               "n_test": len(splits["test"]), "n_dev": len(splits["dev"]), "setups": {}}
    fixed = {"free": (0.79, ()), "tuned+cheap LLM": (0.6, (None,)), "tuned+Jev": (0.6, (0.84,))}  # Phase 10's settings
    for kb in kbs:
        for name, (e, verifiers) in setups().items():
            if args.no_llm_verify and any(v in MODELS for v in verifiers):
                continue
            dev, test = cands[(e, kb, "dev")], cands.get((e, kb, "test"))
            row = {}
            for budget in (0.03, 0.05):
                chosen = pick(dev, verifiers, wrong_fn(kb), budget)
                row[f"{budget:.0%}"] = r = {"sim": None if chosen is None else chosen[0],
                                            "bars": None if chosen is None else chosen[1]}
                if chosen:
                    r["dev"] = measure(dev, verifiers, *chosen, wrong_fn(kb))
                    if test:
                        r["test"] = measure(test, verifiers, *chosen, wrong_fn(kb), base, llm_wrong)
            if name in fixed and test:
                row["phase10"] = {"sim": fixed[name][0], "bars": fixed[name][1],
                                  "test": measure(test, verifiers, *fixed[name], wrong_fn(kb), base, llm_wrong)}
            results["setups"][f"{kb}|{name}"] = row
            log(f"{kb} {name}: " + "; ".join(f"{b}: {r.get('sim')}/{r.get('bars')} dev "
                                             f"{r.get('dev', {}).get('hits')}/{r.get('dev', {}).get('wrong')} test "
                                             f"{r.get('test', {}).get('hits')}/{r.get('test', {}).get('wrong')}"
                                             for b, r in row.items()))
    # Per FAQ wrong counts on dev at the floor, the list the KB fix was chosen from.
    for kb in kbs:
        per = {}
        for c in cands[(TUNED_EMBEDDER, kb, "dev")]:
            if c["cand"] and wrong_fn(kb)(c):
                per[c["id"]] = per.get(c["id"], 0) + 1
        results[f"dev_wrong_by_faq|{kb}"] = dict(sorted(per.items(), key=lambda kv: -kv[1]))
    (OUT / f"results_{args.dev}_{args.test}.json").write_text(json.dumps(results, indent=1))
    if splits["test"]:
        print(report(results))


def report(results, budget="3%"):
    base = read(p10.OUT / "summary_4000.json")["baseline"]
    rows = [["No cache (every question to the LLM)", "0", f"{results['llm_wrong_all']} of {results['n_test']:,} "
             f"({pct(results['llm_wrong_all'] / results['n_test'])}, {pct(wilson(results['llm_wrong_all'], results['n_test'])[1])})",
             "-", "-", p10.ms(base["ms"]), f"${base['dollars']:.2f} spent"]]
    for name, row in results["setups"].items():
        r = row.get(budget, {})
        t = r.get("test")
        if not t:
            rows.append([name, "none fits on dev", "", "", "", "", ""])
            continue
        bars = ", ".join("yes" if b is None else str(b) for b in r["bars"])
        rows.append([f"{name} @ {r['sim']:.3f}" + (f" / {bars}" if bars else ""),
                     f"{t['hits']:,} ({pct(t['hits'] / t['n'])})",
                     f"{t['wrong']} ({pct(t['wrong'] / t['hits'] if t['hits'] else 0)}, {pct(t['wrong_upper'])})",
                     str(t["llm_wrong"]), p10.ms(t["ms_hits"]), p10.ms(t["ms_misses"]),
                     f"${t['net_dollars']:.3f} ({pct(t['net_dollars'] / base['dollars'])})"])
    return table(["Setup @ similarity / verifier bar (picked on dev)", "Answered from cache", "Wrong (rate, 95% upper)",
                  "LLM wrong, same questions", "Hits p50 / p90 ms", "Misses p50 / p90 ms", "Saved, verifier paid"], rows)


if __name__ == "__main__":
    main()
