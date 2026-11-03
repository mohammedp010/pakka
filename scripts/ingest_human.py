"""Turn the Phase 8 Google Forms answers into data/bench_human.jsonl, labelled blind by two LLMs.

    python scripts/ingest_human.py          # reads every *.csv in ../pakka-private/

People's raw answers stay outside the repo. Only rows with the consent box ticked are read; each line of an
answer is one message. Messages with personal identifiers are dropped, repeats removed, and each one keeps the
FAQ and kind of the form question it answered (scripts/make_form.gs). Two models from different families label
every message without seeing that; a message goes to `pakka review` when they split or when they disagree with
what the form asked for. Every judge reply is cached next to the CSVs, so a rerun costs nothing. One writer in
five goes to dev and the rest to test, by person, so nobody's phrasing lands in both and nobody tunes on test.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.dukaan_bot.llm import make_chat  # noqa: E402
from pakka.pii import find_pii  # noqa: E402
from pakka.testset import dedupe, judge, reconcile  # noqa: E402

JUDGES = ["deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]  # two families, so errors don't line up
BATCH = 10  # messages per judge call; smaller batches made the Phase 7 judges more consistent
PLACEHOLDERS = {"", "-", "na", "n/a", "nil", "none", "nothing", "no"}  # what people type to get past a required box
FORM_GS, OUT, PRIVATE =ROOT / "scripts/make_form.gs", ROOT / "data/bench_human.jsonl", ROOT.parent / "pakka-private"

# shortcut: a word list, not a model; Hinglish with none of these words is tagged en. Each is a common romanized
# Hindi word that isn't an English one (no "main", "do", "to", "me", "par", "bata"), so one is enough.
HINDI_WORDS = set("""
hai hain h kya kyaa nahi nhi nahin nai kaise kese kaisa kab kaha kahan kitna kitne kitni mera meri mere mujhe muje
mujhko hum humko hame hamara aap aapka aapko apna apne ka ki ke ko se mein mai karna karni karu karun karo kar kare
karein chahiye chaiye chahie aur bhi toh yaar bhai paisa paise wapas vapas kyu kyun kyon abhi raha rahi rahe gaya
gayi gaye diya diye liya wala wali wale hoon hu tha thi sakta sakti sakte milega milegi milta hoga hogi ho kuch koi
agar lekin ye yeh woh wo batao dena lena ji kyuki kaun konsa kon din ghar bina baad pehle jaldi accha acha theek thik
""".split())


def load_form(path=FORM_GS):
    """The FORM block of make_form.gs: the exact questions people saw."""
    text = path.read_text()
    return json.loads(text[text.index("// form:start") + len("// form:start"):text.index("// form:end")])


def questions(form):
    """Every answer question across the variants; "anything else" appears once."""
    return [q for v in form["variants"].values() for q in v["situations"] + v["twists"]] + [form["other"]]


def split_cell(cell):
    """Each non-empty line of an answer is one message, without list markers, wrapping quotes or "NA"-style fillers."""
    lines = (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip().strip("\"'“”‘’").strip() for line in cell.splitlines())
    return [m for m in lines if m.lower().rstrip(".") not in PLACEHOLDERS]


def lang_of(q):
    if re.search(r"[ऀ-ॿ]", q):
        return "hi"
    return "hinglish" if HINDI_WORDS & set(re.findall(r"[a-z]+", q.lower())) else "en"


def read_responses(paths, form):
    """One item per message from consented rows, with the FAQ (`near`) and kind its form question asked for.
    Writers are numbered 1, 2, ... in file and row order."""
    columns, consent = {q["title"]: q for q in questions(form)}, form["consent"]
    items, writer = [], 0
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            unknown = set(reader.fieldnames or []) - set(columns) - {"Timestamp", consent["title"]}
            if unknown:
                raise ValueError(f"{path.name}: columns {sorted(unknown)} aren't in scripts/make_form.gs; "
                                 "the form's question titles must match it")
            for answers in reader:
                if (answers.get(consent["title"]) or "").strip() != consent["choice"]:
                    continue
                writer += 1
                for title, cell in answers.items():
                    if title in columns:
                        q = columns[title]
                        items += [{"q": m, "kind": q["kind"], "near": q["faq"],
                                   "intended": q["faq"] if q["kind"] == "paraphrase" else None, "writer": writer}
                                  for m in split_cell(cell or "")]
    return items


def clean(items):
    """(kept, how many were dropped for personal data): repeats removed, each message tagged with its language."""
    safe = [i for i in items if not find_pii(i["q"])]
    return [{**i, "lang": lang_of(i["q"])} for i in dedupe(safe)], len(items) - len(safe)


def reconcile2(item, a, b):
    """reconcile() against the first judge, plus a review whenever the second judge disagrees with it."""
    out = reconcile(item, a)
    return {**out, "judged2": b, "needs_review": out["needs_review"] or a != b}


def row(item):
    keys = ("q", "label", "kind", "near", "lang", "split", "source", "writer", "intended", "judged", "judged2",
            "needs_review")
    return {k: item.get(k) for k in keys} | {"split": "dev" if item["writer"] % 5 == 0 else "test", "source": "human"}


def cached(path, fn, attempts=3):
    """Run one LLM call once and keep its result at `path`, so a rerun or a failure elsewhere never redoes it."""
    if path.exists():
        return json.loads(path.read_text())
    for i in range(attempts):
        try:
            result = fn()
            path.write_text(json.dumps(result))
            return result
        except Exception as e:  # bad JSON or a flaky API call: try again, then give up loudly
            print(f"  {path.name}: retry {i + 1}/{attempts}: {e}")
    raise RuntimeError(f"{path.name} failed {attempts} times")


def label_all(items, faqs, chats, cache):
    """Both judges label every item blind, BATCH at a time; each call is cached under a hash of what it asked."""
    votes = []
    for model, chat in chats.items():
        def ask(batch, model=model, chat=chat):
            key = hashlib.md5(json.dumps([model, faqs, [i["q"] for i in batch]]).encode()).hexdigest()[:16]
            return cached(cache / f"{model.replace('/', '__')}_{key}.json", lambda: judge(batch, faqs, chat, batch=BATCH))
        with ThreadPoolExecutor(6) as pool:
            votes.append([x for part in pool.map(ask, [items[i:i + BATCH] for i in range(0, len(items), BATCH)]) for x in part])
    return [reconcile2(item, a, b) for item, a, b in zip(items, *votes)]


def counts(rows, field):
    out = {}
    for r in rows:
        out[r[field]] = out.get(r[field], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: str(kv[0])))


def main(argv=None, chats=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--private", type=Path, default=PRIVATE, help="folder with the forms' CSV exports, outside the repo")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--force", action="store_true", help="rebuild even if --out already has human-reviewed labels")
    args = p.parse_args(argv)

    private = args.private.resolve()
    if private.is_relative_to(ROOT):
        sys.exit(f"{private} is inside the repo. Keep people's raw answers outside the repo, e.g. in {PRIVATE}.")
    if args.out.exists() and '"reviewed": true' in args.out.read_text() and not args.force:
        sys.exit(f"{args.out} has human-reviewed labels; rebuilding would erase them. Use --force to do it anyway.")
    paths = sorted(private.glob("*.csv"))
    if not paths:
        sys.exit(f"No .csv files in {private}. Download each form's responses there first.")

    faqs = json.loads((ROOT / "data/faqs.json").read_text())
    raw = read_responses(paths, load_form())
    items, pii = clean(raw)
    cache = private / "llm"
    cache.mkdir(exist_ok=True)
    chats = chats or {m: make_chat(m, max_tokens=8000) for m in JUDGES}  # DeepSeek sometimes thinks with reasoning off
    rows = [row(i) for i in label_all(items, faqs, chats, cache)]
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    agree = sum(r["judged"] == r["judged2"] for r in rows)
    print(f"{len({i['writer'] for i in raw})} people wrote {len(raw)} messages; dropped {pii} with personal data and "
          f"{len(raw) - pii - len(items)} repeats.\nwritten: {len(rows)} to {args.out} | judges agree on {agree}/{len(rows)} "
          f"| needs review: {sum(r['needs_review'] for r in rows)} | should-miss: {sum(r['label'] is None for r in rows)}")
    for field in ("lang", "kind", "split"):
        print(f"  by {field}: {counts(rows, field)}")
    print(f"Next: pakka review {args.out}")


if __name__ == "__main__":
    main()
