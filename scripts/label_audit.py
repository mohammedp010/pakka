"""Audit public-dataset labels with two LLMs: are the hits a label calls wrong (or right) really so?

    python scripts/label_audit.py

Samples held-out hits of MiniLM-L12 (pakka's default in Phase 7; plain cache) on every public dataset: labelled false hits,
labelled true hits, and labelled-true hits the guards veto. Each pair goes, without its label, to two judges from
different model families, 5 numbered pairs per call (in batches of 20 the judges grew stricter and agreed less).
A third judge from a third family, with some reasoning, settles the pairs the two split on, so every verdict is a
majority of three; it also re-asks a sample of pairs the two agreed on, to show how far their agreement holds.
The sample and every reply are kept in data/public/audit/, so a rerun costs nothing.
"""

import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.dukaan_bot.llm import make_chat  # noqa: E402
from pakka.bench import is_correct, pair_rows, replay_stream, score_with_method  # noqa: E402
from pakka.guard import check  # noqa: E402
from pakka.testset import parse_json  # noqa: E402
from scripts import public_bench as pb  # noqa: E402

JUDGES = ["deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]  # two families, so errors don't line up
ADJUDICATOR, SPOT_CHECKS = "openai/gpt-5.4", 200  # settles their splits; re-asks this many of their agreements
AUDIT = ROOT / "data/public/audit"
DATASETS = pb.FAQ + ["langcache"] + pb.STREAM
PER_POOL, VETOED = 75, 20  # labelled false and labelled true hits per dataset, plus vetoed true hits
BATCH, CUT = 5, 500  # pairs per call; characters kept of each text
ANSWERS = {"ours", "bitext"}  # datasets whose stored answer is real text, not an intent name

SYSTEM = "You check a semantic cache for a chatbot. It stored an answer to question A and wants to serve it for question B."
INSTRUCTIONS = """For each numbered pair, would the answer to A also be a correct and complete answer to B as asked?
Say true only for the same request with the same specifics: names, numbers, dates, places, direction and negation
must match. When A's stored answer is shown, judge that answer. For classification prompts ("Which category does
the text belong to? ..."), true means both texts get the same answer.

{pairs}

Reply with JSON only: {{"1": true or false, ... "{n}": ...}}."""


def cut(text):
    return text if len(text) <= CUT else text[:CUT] + "…"


def prompt(items):
    pairs = [f"{i}. A: {cut(it['cached'])}\n   B: {cut(it['q'])}"
             + (f"\n   Stored answer for A: {cut(it['answer'])}" if it["answer"] else "") for i, it in enumerate(items, 1)]
    return INSTRUCTIONS.format(pairs="\n".join(pairs), n=len(items))


def judge(items, chat):
    """One verdict per pair: True when the cached answer serves the new question."""
    reply = parse_json(chat(SYSTEM, prompt(items))[0])
    verdicts = [reply.get(str(i)) for i in range(1, len(items) + 1)]
    if not all(isinstance(v, bool) for v in verdicts):
        raise ValueError(f"expected true/false for pairs 1-{len(items)}, got {reply!r}")
    return verdicts


def cached(model, items, chat, attempts=3):
    """Ask one judge about one batch once; the reply is kept under a hash of the model and prompt."""
    path = AUDIT / f"{pb.slug(model)}_{hashlib.md5((model + prompt(items)).encode()).hexdigest()[:16]}.json"
    if path.exists():
        return json.loads(path.read_text())
    for i in range(attempts):
        try:
            verdicts = judge(items, chat)
            path.write_text(json.dumps(verdicts))
            return verdicts
        except Exception as e:  # bad JSON or a flaky API call: try again, then give up loudly
            print(f"  {model}: retry {i + 1}/{attempts}: {e}")
    raise RuntimeError(f"{model} failed {attempts} times")


def held_out_hits(name):
    """Held-out rows the default embedder's plain cache serves: at the dev-picked threshold for streams (0.90 when
    none fits), otherwise the 10% it scores highest, the band every operating threshold falls in."""
    model = pb.DEFAULT
    if name in pb.STREAM:
        _, rows = pb.load(name)
        vectors = pb.embedder(model, name, [r["q"] for r in rows])([r["q"] for r in rows])
        t = pb.result(model, name)["plain"]["threshold"] or pb.REDIS_T
        return [{**r, "answer": ""} for r in replay_stream(vectors, rows, t)
                if r["split"] == "test" and r["matched_q"] is not None and r["score"] >= t]
    if name in pb.PAIRS:
        _, pairs = pb.load(name)
        embed = pb.embedder(model, name, [p["q"] for p in pairs] + [p["q2"] for p in pairs])
        rows = [{**r, "q": r["q2"], "matched_q": r["q"], "answer": ""}  # q2 asks, q is cached, as in pair_rows
                for r in pair_rows(embed([p["q"] for p in pairs]), embed([p["q2"] for p in pairs]), pairs)]
    else:
        faqs, data = pb.load(name)
        embed = pb.embedder(model, name, [f["question"] for f in faqs] + [r["q"] for r in data])
        answer = {f["id"]: f["answer"] for f in faqs}
        rows = [{**r, "answer": answer[r["matched_id"]] if name in ANSWERS else ""}
                for r in score_with_method(model, embed, faqs, data)]
    test = [r for r in rows if r["split"] == "test"]
    return sorted(test, key=lambda r: -r["score"])[:len(test) // 10]


def sample(name, rng):
    """(items, band): the audited hits of one dataset, and the band's size and labelled false-hit rate."""
    hits = held_out_hits(name)
    items = [{"dataset": name, "q": r["q"], "cached": r["matched_q"], "answer": r["answer"], "label_wrong": not is_correct(r),
              "vetoed": check(r["q"], r["matched_q"], r["answer"])} for r in hits]
    return draw(items, rng), {"hits": len(items), "false_hit_rate": sum(i["label_wrong"] for i in items) / len(items)}


def draw(items, rng):
    """Labelled false and true hits, PER_POOL each, then up to VETOED vetoed true hits not drawn already."""
    wrong, right = [i for i in items if i["label_wrong"]], [i for i in items if not i["label_wrong"]]
    false, true = rng.sample(wrong, min(PER_POOL, len(wrong))), rng.sample(right, min(PER_POOL, len(right)))
    vetoed = [i for i in right if i["vetoed"] and not any(i is t for t in true)]
    return ([{**i, "pool": "false"} for i in false] + [{**i, "pool": "true"} for i in true]
            + [{**i, "pool": "veto"} for i in rng.sample(vetoed, min(VETOED, len(vetoed)))])


def summarize(items, a, b, third=None):
    """Per dataset: how many labels the judges overturn, how often the two agree, and how many vetoed labelled-true
    hits they call wrong (the guard was right, the label too coarse). A verdict is the two judges' when they agree,
    otherwise `third[i]`'s; a split nobody settled leaves the label standing."""
    out = {}
    for i, (it, x, y) in enumerate(zip(items, a, b)):
        verdict = x if x == y else (third or {}).get(i)
        overturned = verdict is not None and verdict == it["label_wrong"]  # True means the cached answer serves
        s = out.setdefault(it["dataset"], dict(false_hits=0, false_hits_overturned=0, true_hits=0, true_hits_overturned=0,
                                               vetoed=0, vetoed_right=0, agree=0, n=0))
        s["n"] += 1
        s["agree"] += x == y
        if it["pool"] == "false":
            s["false_hits"] += 1
            s["false_hits_overturned"] += overturned
        if it["pool"] == "true":
            s["true_hits"] += 1
            s["true_hits_overturned"] += overturned
        if it["vetoed"] and not it["label_wrong"]:
            s["vetoed"] += 1
            s["vetoed_right"] += overturned
    for s in out.values():
        s["agreement"] = s.pop("agree") / s["n"]
    return out


def report(summary, bands):
    rows = []
    for name, s in summary.items():
        fo = s["false_hits_overturned"] / s["false_hits"] if s["false_hits"] else 0.0
        to = s["true_hits_overturned"] / s["true_hits"] if s["true_hits"] else 0.0
        fhr = bands[name]["false_hit_rate"]
        noise, judged = fhr * fo + (1 - fhr) * to, fhr * (1 - fo) + (1 - fhr) * to
        rows.append([pb.NAMES[name], f"{s['false_hits']} / {s['true_hits']}", pb.pct(s["agreement"], 0),
                     f"{s['false_hits_overturned']} of {s['false_hits']} ({pb.pct(fo, 0)})",
                     f"{s['true_hits_overturned']} of {s['true_hits']} ({pb.pct(to, 0)})", pb.pct(noise, 0),
                     f"{pb.pct(fhr, 0)} → {pb.pct(judged, 0)}",
                     f"{s['vetoed_right']} of {s['vetoed']}" if s["vetoed"] else "-"])
    return pb.table(["Dataset", "Audited (false / true hits)", "First two judges agree",
                     "Labelled false hits the judges call fine", "Labelled true hits the judges call wrong",
                     "Hits whose label the judges overturn (est.)", "False-hit rate: labels → judges",
                     "Vetoed true hits the judges call wrong"], rows)


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)

    def everything():
        rng, items, bands = random.Random(7), [], {}
        for name in DATASETS:
            got, bands[name] = sample(name, rng)
            items += got
            pb.log(f"{name}: sampled {len(got)} of {bands[name]['hits']} hits")
        rng.shuffle(items)  # batches mix datasets, so a judge can't settle into one dataset's habits
        return {"items": items, "bands": bands}

    drawn = pb.memo(AUDIT / "sample.json", everything)
    items, bands = drawn["items"], drawn["bands"]
    def ask(model, picked, reasoning=None):
        chat = make_chat(model, max_tokens=8000, reasoning=reasoning)  # DeepSeek sometimes thinks with reasoning off
        batches = [picked[i:i + BATCH] for i in range(0, len(picked), BATCH)]
        with ThreadPoolExecutor(6) as pool:
            out = [v for part in pool.map(lambda b: cached(model, b, chat), batches) for v in part]
        pb.log(f"{model}: {len(out)} verdicts")
        return out

    a, b = (ask(model, items) for model in JUDGES)
    split = [i for i in range(len(items)) if a[i] != b[i]]
    spot = random.Random(11).sample([i for i in range(len(items)) if a[i] == b[i]], SPOT_CHECKS)
    asked = sorted(split + spot)
    third = dict(zip(asked, ask(ADJUDICATOR, [items[i] for i in asked], {"effort": "low"})))
    summary = summarize(items, a, b, third)
    (AUDIT / "summary.json").write_text(json.dumps({"summary": summary, "bands": bands}, indent=1))
    print(f"\n{len(items)} pairs. The first two judges agree on {1 - len(split) / len(items):.0%}. Where they split, "
          f"{ADJUDICATOR} sides with {JUDGES[0]} on {sum(third[i] == a[i] for i in split) / len(split):.0%}. On "
          f"{len(spot)} pairs they agreed on, it agrees with them on {sum(third[i] == a[i] for i in spot) / len(spot):.0%}.\n")
    print(report(summary, bands))


if __name__ == "__main__":
    main()
