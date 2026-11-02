"""Measure semantic caches on public datasets and write docs/public_results.md.

    python scripts/public_bench.py

Downloads and converts the datasets (scripts/fetch_public.py), embeds every text once per model
(data/public/emb/), runs the FAQ, stream and pairs protocols plain and with pakka guards, and writes the doc.
Every measurement is kept in data/public/results/, so a rerun resumes where it stopped; delete that folder
to measure again after changing pakka. After changing only the guards, drop the "guards" key from each
result file (and the `+guards` files) and just those get measured again.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pakka.bench import (evaluate, is_correct, pair_rows, pchr_auc, pick_stream_threshold, pick_threshold,  # noqa: E402
                         replay_stream, report_encoder, score_with_method)
from pakka.embed import disk_cached, load_embedder  # noqa: E402
from pakka.guard import check  # noqa: E402
from scripts import fetch_public  # noqa: E402

PUBLIC, DOC = ROOT / "data/public", ROOT / "docs/public_results.md"
EMB, RESULTS = PUBLIC / "emb", PUBLIC / "results"
MODELS = {  # model: short name; local models slowest last, then API models (they cost money, about $0.70 together)
    "paraphrase-multilingual-MiniLM-L12-v2": "MiniLM-L12 (pakka default)",
    "redis/langcache-embed-v1": "langcache-embed-v1",
    "intfloat/multilingual-e5-small": "e5-small",
    "intfloat/multilingual-e5-base": "e5-base",
    "l3cube-pune/indic-sentence-similarity-sbert": "IndicSBERT",
    "sentence-transformers/LaBSE": "LaBSE",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "BAAI/bge-m3": "bge-m3",
    "openrouter:qwen/qwen3-embedding-8b": "Qwen3-Embedding-8B (API)",
    "openrouter:openai/text-embedding-3-small": "OpenAI 3-small (API)",
}
DEFAULT, REDIS, REDIS_T = "paraphrase-multilingual-MiniLM-L12-v2", "redis/langcache-embed-v1", 0.90  # RedisVL 0.29 defaults
BUDGET = 0.05  # false-hit budget (95% upper bound) the dev split picks thresholds for
MAX_TOKENS = 128  # MiniLM's own limit: every model sees about the same text, and long chat prompts embed faster
HALF = {"BAAI/bge-m3", "Qwen/Qwen3-Embedding-0.6B"}  # float16 weights: big models fit next to everything else in 8 GB
FAQ = ["ours", "banking77", "clinc150", "bitext", "hinglish_top", "massive"]
STREAM = ["vcache_lmarena", "vcache_classification", "vcache_searchqueries"]
PAIRS = ["langcache", "langcache_strat"]
SETUPS = (("plain", None), ("guards", check))
CURVE_BINS = [0, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 150000]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def slug(model):
    return model.replace("/", "__").replace(":", "_")


def memo(path, fn):
    """Run `fn` once and keep its JSON result at `path`."""
    if path.exists():
        return json.loads(path.read_text())
    out = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def load(name):
    if name == "langcache_strat":
        return None, fetch_public.stratified(load("langcache")[1])
    faqs, data = ((ROOT / "data/faqs.json", ROOT / "data/bench_v1.jsonl") if name == "ours"
                  else (PUBLIC / f"{name}.faqs.json", PUBLIC / f"{name}.jsonl"))
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    return (json.loads(faqs.read_text()) if faqs.exists() else None), rows


_loaded = {}


def embedder(model, name, texts):
    """`embed` for one dataset, backed by data/public/emb/, with every text of the dataset embedded up front."""

    def embed(batch):
        if model not in _loaded:
            _loaded.clear()  # one model in memory at a time: the Mac has 8 GB
            if model.startswith("openrouter:") and not os.environ.get("OPENROUTER_API_KEY"):
                from examples.dukaan_bot.llm import api_key  # reads pakka/.env
                os.environ["OPENROUTER_API_KEY"] = api_key() or ""
            _loaded[model] = load_embedder(model, max_length=MAX_TOKENS, half=model in HALF)
        start = time.time()
        out = _loaded[model](batch)
        log(f"  {MODELS[model]}: embedded {len(batch)} {name} texts in {time.time() - start:.0f}s")
        return out

    cached = disk_cached(embed, EMB / slug(model) / f"{name.removesuffix('_strat')}.npz")  # same texts, same file
    cached(texts)
    cached.save()
    return cached


def compact(rows):
    """Held-out rows as [score, wrong, blocked, lang, kind, should_hit], enough to evaluate any threshold later."""
    return [[r["score"], int(not is_correct(r)), int(bool(r.get("blocked"))), r.get("lang"), r.get("kind"),
             int(r["label"] is not None)] for r in rows if r["split"] == "test"]


def at(rows, t):
    """False-hit rate, its 95% upper bound and hit rate of compact rows at threshold t."""
    from pakka.bench import wilson

    served = [w for s, w, b, *_ in rows if s >= t and not b]
    return {"false_hit_rate": sum(served) / len(served) if served else 0.0, "false_hit_upper": wilson(sum(served), len(served))[1],
            "hit_rate": len(served) / len(rows), "hits": len(served), "false_hits": sum(served)}


def test_eval(rows, t):
    out = evaluate([r for r in rows if r["split"] == "test"], t)
    out.pop("avg_latency_ms")
    return out


def faq_result(model, name, setups):
    faqs, data = load(name)
    embed = embedder(model, name, [f["question"] for f in faqs] + [r["q"] for r in data])
    out = {}
    for setup, guard in setups:
        rows = score_with_method(model, embed, faqs, data, guard=guard)
        out[setup] = {**report_encoder(model, rows, BUDGET), "fixed": test_eval(rows, REDIS_T), "test": compact(rows)}
    return out


def stream_result(model, name, setups):
    _, rows = load(name)
    vectors = embedder(model, name, [r["q"] for r in rows])([r["q"] for r in rows])
    out = {}
    for setup, guard in setups:
        t, replayed = pick_stream_threshold(vectors, rows, BUDGET, guard)
        out[setup] = {"threshold": t, **(test_eval(replayed, t) if t else {})}
        if model == REDIS:
            out[setup]["fixed"] = test_eval(replay_stream(vectors, rows, REDIS_T, guard), REDIS_T)
        log(f"  {MODELS[model]} {name} {setup}: threshold {t}")
    return out


def pairs_result(model, name, setups):
    _, pairs = load(name)
    embed = embedder(model, name, [p["q"] for p in pairs] + [p["q2"] for p in pairs])
    v1, v2 = embed([p["q"] for p in pairs]), embed([p["q2"] for p in pairs])
    out = {}
    for setup, guard in setups:
        rows = pair_rows(v1, v2, pairs, guard)
        dev, test = [r for r in rows if r["split"] == "dev"], [r for r in rows if r["split"] == "test"]
        t = pick_threshold(dev, BUDGET)
        by_source = {}
        for source in sorted({r["source"] for r in test}):
            sub = [r for r in test if r["source"] == source]
            by_source[source] = {"n": len(sub), "positives": sum(r["label"] is not None for r in sub), "auc": pchr_auc(sub),
                                 "at_t": evaluate(sub, t) if t else None, "fixed": evaluate(sub, REDIS_T)}
        out[setup] = {"threshold": t, "auc": pchr_auc(test), **(evaluate(test, t) if t else {}),
                      "fixed": evaluate(test, REDIS_T), "by_source": by_source, "test": compact(rows)}
    return out


def result(model, name):
    """Plain and guarded results of one model on one dataset, measuring only the setups the file lacks."""
    path = RESULTS / slug(model) / f"{name}.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    todo = [s for s in SETUPS if s[0] not in out]
    if todo:
        fn = faq_result if name in FAQ else stream_result if name in STREAM else pairs_result
        out.update(fn(model, name, todo))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return out


def stream_at(model, name, t, guard=False):
    """A stream's held-out result when replayed at threshold t (for thresholds picked on other datasets)."""

    def run():
        _, rows = load(name)
        vectors = embedder(model, name, [r["q"] for r in rows])([r["q"] for r in rows])
        return test_eval(replay_stream(vectors, rows, t, check if guard else None), t)

    return memo(RESULTS / slug(model) / f"{name}@{t:.4f}{'+guards' if guard else ''}.json", run)


def transfer(model):
    """Threshold picked on each dataset's dev split (plain cache), applied to every dataset's test split."""
    names = FAQ + PAIRS + STREAM
    picked = {a: result(model, a)["plain"]["threshold"] for a in names}
    matrix = {}
    for a, t in picked.items():
        if t is None:
            continue
        for b in names:
            r = stream_at(model, b, t) if b in STREAM else at(result(model, b)["plain"]["test"], t)
            matrix[f"{a}>{b}"] = {k: r[k] for k in ("false_hit_rate", "false_hit_upper", "hit_rate")}
    return {"picked": picked, "matrix": matrix}


def curve(model, t, guard=False):
    """False-hit rate on vCache SearchQueries by how many entries the cache held when each question arrived."""

    def run():
        _, rows = load("vcache_searchqueries")
        vectors = embedder(model, "vcache_searchqueries", [r["q"] for r in rows])([r["q"] for r in rows])
        replayed = replay_stream(vectors, rows, t, check if guard else None)
        bins = []
        for lo, hi in zip(CURVE_BINS, CURVE_BINS[1:]):
            part = [r for r in replayed if lo <= r["cache_size"] < hi]
            if part:
                served = [r for r in part if r["score"] >= t and not r["blocked"]]
                wrong = sum(not is_correct(r) for r in served)
                bins.append({"lo": lo, "hi": hi, "questions": len(part), "hits": len(served), "false_hits": wrong})
        return bins

    return memo(RESULTS / slug(model) / f"curve@{t:.4f}{'+guards' if guard else ''}.json", run)


def curves():
    """(label, bins) for the default embedder at its dev-picked thresholds (0.90 when none fits) and RedisVL defaults."""
    picked, lines = result(DEFAULT, "vcache_searchqueries"), []
    for setup, guard in SETUPS:
        t = picked[setup]["threshold"]
        label = f"{MODELS[DEFAULT]}{' + guards' if guard else ''} at " + (f"{t} (dev-picked)" if t else f"{REDIS_T} (none fits dev)")
        lines.append((label, curve(DEFAULT, t or REDIS_T, guard=bool(guard))))
    return lines + [(f"RedisVL defaults ({REDIS_T})", curve(REDIS, REDIS_T))]


# --- the doc: generated tables go between these markers; prose outside them survives a rerun

START, END = "<!-- generated by scripts/public_bench.py: start -->", "<!-- generated by scripts/public_bench.py: end -->"
NAMES = {
    "ours": "HinglishCacheBench v1 (ours)", "banking77": "Banking77", "clinc150": "CLINC150", "bitext": "Bitext",
    "hinglish_top": "Hinglish-TOP", "massive": "MASSIVE en+hi", "langcache": "LangCache pairs", "langcache_strat": "LangCache pairs, stratified split",
    "vcache_lmarena": "vCache LmArena", "vcache_classification": "vCache Classification",
    "vcache_searchqueries": "vCache SearchQueries",
}


def pct(x, digits=1):
    return "-" if x is None else f"{x:.{digits}%}"


def cell(r):
    """'false-hit rate · hit rate' of one result."""
    return "no threshold fits" if r is None or r.get("hit_rate") is None else f"{pct(r['false_hit_rate'])} · {pct(r['hit_rate'], 0)}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def exact_floor(name):
    """An exact-match cache replayed over a stream: hits on identical text that the labels call wrong are label
    noise, a floor no embedder can get under."""
    from pakka.cache import normalize

    seen, hits, wrong = {}, 0, 0
    for r in load(name)[1]:
        key = normalize(r["q"])
        if key in seen:
            hits, wrong = hits + 1, wrong + (seen[key] != r["label"])
        else:
            seen[key] = r["label"]
    return hits, wrong


def plain_fhr_at(rows, hits):
    """Plain cache's false-hit rate if it served as many held-out questions as the guarded one did."""
    top = sorted(rows, key=lambda r: -r[0])[:hits]
    return sum(r[1] for r in top) / hits if hits else None


SHARE = 0.10


def top_share(rows):
    """The SHARE of held-out questions an embedder scores highest, guard vetoes never served: every embedder at
    the same hit rate, so they compare where no threshold fits the budget."""
    return sorted((r for r in rows if not r[2]), key=lambda r: -r[0])[:int(len(rows) * SHARE)]


def language_cells(rows, langs):
    """False-hit rate, share of each language's should-hit questions answered right, and slot-trap false hits."""
    served = top_share(rows)
    caught = [pct(sum(r[3] == lang and r[5] and not r[1] for r in served) / max(1, sum(r[3] == lang and r[5] for r in rows)), 0)
              for lang in langs]
    return [pct(sum(r[1] for r in served) / len(served))] + caught + [sum(r[1] and r[4] == "slot" for r in served)]


def measured():
    """Models with every dataset measured. The rest show as not run yet (a slow model can be finished later)."""
    return [m for m in MODELS if all((RESULTS / slug(m) / f"{n}.json").exists() for n in FAQ + PAIRS + STREAM)]


PENDING = "not run yet"


def render():
    names, models = FAQ + PAIRS + STREAM, measured()
    pending = [m for m in MODELS if m not in models]
    res = {(m, n): result(m, n) for m in models for n in names}
    held_out = {n: sum(r["split"] == "test" for r in load(n)[1]) for n in names}
    protocol = {n: "faq" if n in FAQ else "pairs" if n in PAIRS else "stream" for n in names}
    out = []

    def best(n, setup="guards"):
        fits = [m for m in models if res[m, n][setup].get("hit_rate") is not None]
        return max(fits, key=lambda m: res[m, n][setup]["hit_rate"], default=None)

    out += ["## Main table\n",
            f"Held-out (test) split of each dataset. Each cell is **false-hit rate · hit rate**. \"RedisVL defaults\" is "
            f"`{REDIS}` at similarity {REDIS_T} (RedisVL 0.29's out-of-the-box semantic cache), no tuning. \"Tuned\" picks "
            f"the threshold on the dev split so the false-hit rate's 95% upper bound stays under {BUDGET:.0%}, then "
            "reports test. LangCache is pairs: hit rate there is the share of pairs scored as the same.\n"]
    rows = []
    for n in names:
        b = best(n)
        rows.append([NAMES[n], protocol[n], f"{held_out[n]:,}", cell(res[REDIS, n]["plain"]["fixed"]),
                     cell(res[REDIS, n]["guards"]["fixed"]), cell(res[DEFAULT, n]["plain"]), cell(res[DEFAULT, n]["guards"]),
                     f"{MODELS[b]}: {cell(res[b, n]['guards'])}" if b else "no threshold fits"])
    out += [table(["Dataset", "Protocol", "Held-out", "RedisVL defaults", "RedisVL defaults + pakka guards",
                   f"{MODELS[DEFAULT]}, tuned", "… + guards", "Best measured embedder, tuned + guards"], rows)]
    if pending:
        out += [f"\nNot measured on every dataset yet: {', '.join(MODELS[m] for m in pending)}, marked \"{PENDING}\" below "
                "(results for what has run are under \"Better embedders\").\n"]
    floors = {n: exact_floor(n) for n in STREAM}
    out += ["\nLabel-noise floor of the streams: an exact-match cache (same text after lowercasing) replayed over the whole "
            "stream is already \"wrong\" on " + "; ".join(f"{NAMES[n]} {w:,} of {h:,} hits ({pct(w / h)})" for n, (h, w) in floors.items())
            + ", because identical prompts sit in different equivalence classes. No embedder can beat that floor on "
            "these labels.\n"]

    out += ["\n## Embedders, threshold-free: P-CHR AUC\n",
            "Area under precision vs cache-hit ratio (higher is better), plain cache, held-out split. It doesn't depend on "
            "any threshold, so it compares embedders fairly. Streams have none: what a stream stores depends on the threshold.\n"]
    rows = []
    for n in FAQ + PAIRS:
        aucs = {m: res[m, n]["plain"]["auc"] for m in models}
        top = max(aucs.values())
        rows.append([NAMES[n]] + [f"**{aucs[m]:.3f}**" if aucs[m] == top else f"{aucs[m]:.3f}" for m in models] + [PENDING] * len(pending))
    rows.append(["Mean"] + [f"{sum(res[m, n]['plain']['auc'] for n in FAQ + PAIRS) / len(FAQ + PAIRS):.3f}" for m in models]
                + [PENDING] * len(pending))
    out += [table(["Dataset"] + [MODELS[m] for m in models + pending], rows)]

    out += [f"\n## Embedders at a {BUDGET:.0%} false-hit budget, with guards\n",
            "Hit rate on held-out at the dev-picked threshold, with the held-out false-hit rate in brackets: above "
            f"{BUDGET:.0%} means the threshold picked on dev didn't hold.\n"]
    rows = []
    for n in names:
        lines = {m: res[m, n]["guards"] for m in models}
        top = max((x["hit_rate"] for x in lines.values() if x.get("hit_rate") is not None), default=None)
        rows.append([NAMES[n]] + ["none fits" if x.get("hit_rate") is None else
                                  ("**{}** ({})" if x["hit_rate"] == top else "{} ({})").format(pct(x["hit_rate"], 0), pct(x["false_hit_rate"]))
                                  for x in lines.values()] + [PENDING] * len(pending))
    out += [table(["Dataset"] + [MODELS[m] for m in models + pending], rows)]

    out += [f"\n## Do guards help on data they weren't designed for? ({MODELS[DEFAULT]})\n",
            "Same budget, threshold picked on dev with and without guards. \"Plain at the same hit rate\" is the plain "
            "cache's held-out false-hit rate if it served as many questions as the guarded one did.\n"]
    rows = []
    for n in names:
        p, g = res[DEFAULT, n]["plain"], res[DEFAULT, n]["guards"]
        same = plain_fhr_at(p["test"], g["hits"]) if "test" in p and g.get("hits") else None
        vetoes = ", ".join(f"{k} {v}" for k, v in sorted(g.get("blocked_by", {}).items())) or "-"
        rows.append([NAMES[n], cell(p), cell(g), pct(same), g.get("blocked_correct", "-"), vetoes])
    out += [table(["Dataset", "Plain, tuned", "Guards, tuned", "Plain at the same hit rate", "Good hits vetoed", "Vetoes by rule"], rows)]
    up = down = 0
    for m in models:
        for n in names:
            p, g = res[m, n]["plain"].get("hit_rate"), res[m, n]["guards"].get("hit_rate")
            up += (g or 0) > (p or 0)
            down += (g or 0) < (p or 0)
    out += [f"\nAcross the {len(models)} measured embedders and {len(names)} datasets, guards raised the hit rate at the same budget "
            f"in {up} cases and lowered it in {down}.\n"]

    tr = transfer(DEFAULT)
    out += [f"## Threshold transfer ({MODELS[DEFAULT]}, plain)\n",
            f"Each row picks its threshold on one dataset's dev split ({BUDGET:.0%} budget); each column applies it to "
            f"another dataset's held-out split. Cells are **false-hit rate · hit rate**; bold means the measured false-hit "
            f"rate itself is over the {BUDGET:.0%} budget. The diagonal is the honest in-domain number.\n"]
    rows = []
    for a in names:
        t = tr["picked"][a]
        if t is None:
            rows.append([f"{NAMES[a]} (none fits)"] + ["-"] * len(names))
            continue
        cells = []
        for b in names:
            x = tr["matrix"][f"{a}>{b}"]
            c = cell(x)
            cells.append(f"**{c}**" if x["false_hit_rate"] > BUDGET else c)
        rows.append([f"{NAMES[a]} ({t:.3f})"] + cells)
    out += [table(["Picked on ↓ / applied to →"] + [NAMES[n] for n in names], rows)]

    fits = [m for m in models if res[m, "langcache_strat"]["plain"]["threshold"]]
    out += ["\n## LangCache pairs by source\n",
            "Redis's own sentence-pair benchmark, test split. Each cell is **false-hit rate · duplicates caught** "
            "(share of same-meaning pairs scored as hits). The tuned threshold comes from the validation split, which has "
            "no QQP, PAWS or APT pairs. On the stratified split, whose dev holds a fifth of every source, a threshold "
            + (f"fits for {', '.join(MODELS[m] for m in fits)} only.\n" if fits else f"fits for none of the {len(models)} embedders.\n")]
    rows = []
    lc = {m: res[m, "langcache"]["plain"] for m in (DEFAULT, REDIS)}
    for source, s in lc[DEFAULT]["by_source"].items():
        def dup(r, pos=s["positives"]):
            return "-" if r is None else f"{pct(r['false_hit_rate'])} · {pct(r['correct'] / pos, 0) if pos else '-'}"
        auc = (lambda a: f"{a:.3f}") if s["positives"] else (lambda a: "n/a, no same pairs")
        rows.append([source, f"{s['n']:,} ({s['positives']:,})", auc(s["auc"]), auc(lc[REDIS]["by_source"][source]["auc"]),
                     dup(s["at_t"]), dup(lc[REDIS]["by_source"][source]["fixed"])])
    out += [table(["Source", "Pairs (same)", f"AUC {MODELS[DEFAULT]}", f"AUC {MODELS[REDIS]}",
                   f"{MODELS[DEFAULT]} at {lc[DEFAULT]['threshold']:.3f}", f"RedisVL defaults ({REDIS_T})"], rows)]

    lines = curves()
    out += ["\n## False hits as the cache grows (vCache SearchQueries)\n",
            "The 150,000 search queries replayed in order, every miss stored. Questions are grouped by how many entries the "
            "cache held when they arrived; the threshold stays where dev (the first 30,000) put it. Each cell is "
            "**false-hit rate · hit rate · wrong answers per 100 questions**.\n"]
    bins = [f"{b['lo'] // 1000}k–{b['hi'] // 1000}k" for b in lines[0][1]]
    rows = [[label] + [f"{pct(b['false_hits'] / b['hits'] if b['hits'] else 0)} · {pct(b['hits'] / b['questions'], 0)} · "
                       f"{100 * b['false_hits'] / b['questions']:.1f}" for b in bins_] for label, bins_ in lines]
    out += [table(["Cache size when asked →"] + bins, rows)]
    out += ["\n```mermaid", "xychart-beta", '    title "False-hit rate (%) by cache size, one line per row above"',
            "    x-axis [" + ", ".join(f'"{b}"' for b in bins) + "]", '    y-axis "False-hit rate (%)"']
    out += ["    line [" + ", ".join(f"{100 * b['false_hits'] / b['hits'] if b['hits'] else 0:.1f}" for b in bins_) + "]"
            for _, bins_ in lines]
    out += ["```\n"]

    out += ["## English, Hinglish and Hindi written by people\n",
            f"No embedder has a threshold that meets the budget on Hinglish-TOP or MASSIVE, so here each one serves the "
            f"{SHARE:.0%} of held-out questions it scores highest, with guards (the same hit rate for all). Columns: "
            "false-hit rate of those hits, share of each language's should-hit questions answered right, and how many "
            "hits were slot traps (same intent, different slot value).\n"]
    rows = []
    for m in models:
        rows.append([MODELS[m]] + language_cells(res[m, "hinglish_top"]["guards"]["test"], ("en", "hinglish"))
                    + language_cells(res[m, "massive"]["guards"]["test"], ("en", "hi")))
    rows += [[MODELS[m]] + [PENDING] * 8 for m in pending]
    out += [table(["Embedder", "Hinglish-TOP false-hit rate", "English caught", "Hinglish caught", "Slot-trap false hits",
                   "MASSIVE false-hit rate", "English caught", "Hindi caught", "Slot-trap false hits"], rows)]
    return "\n".join(out) + "\n"


def write_doc(tables):
    """Put the generated tables between the markers, keeping any prose around them."""
    old = DOC.read_text() if DOC.exists() else f"# Public benchmark results\n\n{START}\n{END}\n"
    head, rest = old.split(START)
    DOC.write_text(f"{head}{START}\n{tables}{END}{rest.split(END, 1)[1]}")


def main(argv=None):
    """`python scripts/public_bench.py [model ...]` measures these models (default: all), then writes the doc
    from every model measured so far."""
    models = (sys.argv[1:] if argv is None else argv) or list(MODELS)
    unknown = set(models) - set(MODELS)
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(sorted(unknown))}; choose from {', '.join(MODELS)}")
    fetch_public.main([])
    for model in models:
        for name in FAQ + PAIRS + STREAM:
            start = time.time()
            result(model, name)
            log(f"{MODELS[model]} {name} done in {time.time() - start:.0f}s")
    transfer(DEFAULT)
    curves()
    write_doc(render())
    log(f"wrote {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
