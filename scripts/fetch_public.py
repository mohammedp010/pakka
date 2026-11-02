"""Download public datasets into data/public/ (gitignored) and convert them to pakka's formats.

    python scripts/fetch_public.py                    # every dataset, skipping ones already converted
    python scripts/fetch_public.py banking77 --force  # just these, converted again

Each dataset is read straight from a pinned Hugging Face revision (only the columns we need, so the
6 GB vCache files stay on the server) and written as one of three protocols:

- faq:    `<name>.faqs.json` (id, question, answer) + `<name>.jsonl` (q, label, kind, lang, split, source),
          the format `pakka bench` already reads
- stream: `<name>.jsonl` in arrival order; `label` is the equivalence class, the first fifth is dev
- pairs:  `<name>.jsonl` with `q`, `q2` and `label` 1 (same meaning) or 0
"""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pakka.cache import normalize  # noqa: E402

OUT = ROOT / "data/public"


def split_of(text):
    """Stable 50/50 dev/test split for datasets that ship without one."""
    return "dev" if int(hashlib.md5(text.encode()).hexdigest(), 16) % 2 == 0 else "test"


def row(q, label, split, source, kind="paraphrase", lang="en"):
    return {"q": q, "label": label, "kind": kind, "lang": lang, "split": split, "source": source}


def finish(faqs, rows):
    """Keep each question once, and drop rows that are just an FAQ question again (a trivial exact hit,
    or label noise when the parse disagrees)."""
    seen = {normalize(f["question"]) for f in faqs}
    kept = []
    for r in rows:
        if normalize(r["q"]) not in seen:
            seen.add(normalize(r["q"]))
            kept.append(r)
    return faqs, kept


# --- FAQ protocol from intent datasets: the first training question per intent becomes its FAQ


def seed_intents(examples, answer=lambda intent, ex: intent):
    """(faqs, rest): the first example per intent is its FAQ; `examples` are (text, intent, extra) tuples."""
    faqs, rest, seen = [], [], set()
    for text, intent, ex in examples:
        if intent in seen:
            rest.append((text, intent))
        else:
            seen.add(intent)
            faqs.append({"id": intent, "question": text, "answer": answer(intent, ex)})
    return faqs, rest


def from_banking77(splits, names):
    faqs, rest = seed_intents((r["text"], names[r["label"]], r) for r in splits["train"])
    rows = [row(t, i, "dev", "banking77") for t, i in rest]
    rows += [row(r["text"], names[r["label"]], "test", "banking77") for r in splits["test"]]
    return finish(faqs, rows)


def from_clinc150(splits, names):
    faqs, _ = seed_intents((r["text"], names[r["intent"]], r) for r in splits["train"] if names[r["intent"]] != "oos")
    rows = []
    for name, split in (("validation", "dev"), ("test", "test")):
        for r in splits[name]:
            intent = names[r["intent"]]
            rows.append(row(r["text"], None, split, "clinc150", kind="out_of_scope") if intent == "oos"
                        else row(r["text"], intent, split, "clinc150"))
    return finish(faqs, rows)


def from_bitext(data):
    faqs, rest = seed_intents(((r["instruction"], r["intent"], r) for r in data), lambda _, r: r["response"])
    return finish(faqs, [row(t, i, split_of(t), "bitext") for t, i in rest])


# --- FAQ protocol from semantic parses: same parse should hit, same intent with other slot values must miss


def _value(text):
    return "".join(re.findall(r"\w+", text.lower()))  # "6 am" and "6am" are the same slot value


def top_meaning(parse, values=True):
    """What a TOP parse asks for: intent and slot values, nested, without filler words. With
    `values=False`, the skeleton (intent and slot names only)."""
    tokens = parse.split()

    def node():
        label, words, kids = tokens.pop(0)[1:], [], []
        while tokens[0] != "]":
            if tokens[0].startswith("["):
                kids.append(node())
            else:
                words.append(tokens.pop(0))
        tokens.pop(0)
        value = _value(" ".join(words)) if values and label.startswith("SL:") and not kids else ""
        return label, value, tuple(sorted(kids))

    return node()


def massive_meaning(intent, annot, values=True):
    slots = re.findall(r"\[(\w+) : ([^\]]+)\]", annot)
    return intent, tuple(sorted((name, _value(v) if values else "") for name, v in slots))


def from_parses(records, source):
    """FAQ protocol from `records` (split, lang, text, key, skeleton, intent). Every distinct training parse
    becomes an FAQ (its first English question). A question with that exact parse should hit it, the seed's
    own translations included; same skeleton with other slot values is a `slot` trap; anything else is
    `uncovered`. Both must miss."""
    faqs, ids, rows = [], {}, []
    train = [r for r in records if r["split"] == "train"]
    for r in train:
        if r["lang"] == "en" and r["key"] not in ids:
            ids[r["key"]] = f"{r['intent']}_{hashlib.md5(repr(r['key']).encode()).hexdigest()[:8]}"
            faqs.append({"id": ids[r["key"]], "question": r["text"], "answer": r["intent"]})
            r["seed"] = True
    skeletons = {r["skeleton"] for r in train if r["lang"] == "en"}
    for r in records:
        if r.get("seed"):
            continue
        label = ids.get(r["key"])
        kind = "paraphrase" if label else "slot" if r["skeleton"] in skeletons else "uncovered"
        split = split_of(r["text"]) if r["split"] == "train" else r["split"]
        rows.append(row(r["text"], label, split, source, kind=kind, lang=r["lang"]))
    return finish(faqs, rows)


def own_question(example_id):
    """Key and skeleton for a catch-all intent, whose slots don't say what is asked ("check my car is ready"
    and "check my laptop is working" are both general_quirky): only the utterance's own translations match."""
    return {"key": ("open", example_id), "skeleton": ("open", example_id)}


def from_hinglish_top(splits):
    records = []
    for name, split in (("train", "train"), ("validation", "dev"), ("test", "test")):
        for ex in splits[name]:
            intent = ex["en_parse"].split()[0][4:]  # "[IN:CREATE_ALARM" -> "CREATE_ALARM"
            meaning = dict(key=top_meaning(ex["en_parse"]), skeleton=top_meaning(ex["en_parse"], values=False),
                           intent=intent)
            if intent.startswith("UNSUPPORTED"):
                meaning.update(own_question(ex["en_query"]))
            records += [{"split": split, "lang": "en", "text": ex["en_query"], **meaning},
                        {"split": split, "lang": "hinglish", "text": ex["cs_query"], **meaning}]
    return from_parses(records, "hinglish_top")


_LOCALIZED = {"localization", "unchanged_localization"}  # MASSIVE: the slot value was swapped for a local one
_OPEN_ENDED = {"general_quirky", "qa_factoid", "qa_maths", "social_post"}  # MASSIVE intents whose slots miss the question


def from_massive(locales, names=None):
    """`locales`: {"en": en-US rows, "hi": hi-IN rows}. Keys come from the English annotation; a Hindi row
    whose slots were localized (Boston became Mumbai) asks something else, so it gets its own key."""
    english = {r["id"]: r for r in locales["en"]}
    records = []
    for lang in ("en", "hi"):
        for r in locales[lang]:
            en = english[r["id"]]
            intent = names[en["intent"]] if names else str(en["intent"])
            meaning = {"key": massive_meaning(intent, en["annot_utt"]),
                       "skeleton": massive_meaning(intent, en["annot_utt"], values=False)}
            if intent in _OPEN_ENDED:
                meaning = own_question(r["id"])
            elif lang == "hi" and _LOCALIZED & set(r["slot_method"]["method"]):
                meaning["key"] = ("localized", r["id"])
            records.append({"split": r["partition"], "lang": lang, "text": r["utt"], "intent": intent, **meaning})
    return from_parses(records, "massive")


# --- stream and pairs protocols


def from_stream(data, text, label, source, dev_share=0.2):
    """Arrival order is kept (the vCache files are already shuffled). The first fifth tunes the threshold;
    the rest is reported, with the cache it built still in place."""
    cut = int(len(data) * dev_share)
    return [{"q": r[text], "label": r[label], "kind": "stream", "lang": "en", "split": "dev" if i < cut else "test",
             "source": source} for i, r in enumerate(data)]


def from_langcache(splits):
    return [{"q": r["sentence1"], "q2": r["sentence2"], "label": int(r["label"]), "kind": "pair", "lang": "en",
             "split": "dev" if name == "validation" else "test", "source": r["source"]}
            for name in ("validation", "test") for r in splits[name]]


def jsonl(rows):
    """One JSON object per line, with Unicode line separators escaped: str.splitlines() would cut there."""
    lines = (json.dumps(r, ensure_ascii=False) for r in rows)
    return "".join(line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029").replace("\x85", "\\u0085") + "\n"
                   for line in lines)


# --- downloads, pinned to the revisions these results were measured on


def hf(repo, revision, path, columns=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{repo}@{revision}/{path}") as f:
        return pq.read_table(f, columns=columns).to_pylist()


def class_names(repo, revision, path, column):
    """Label names a Hugging Face parquet file keeps in its metadata."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    with HfFileSystem().open(f"datasets/{repo}@{revision}/{path}") as f:
        info = json.loads(pq.read_schema(f).metadata[b"huggingface"])["info"]
    return info["features"][column]["names"]


def fetch_banking77():
    repo, rev = "PolyAI/banking77", "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"
    splits = {s: hf(repo, rev, f"data/{s}-00000-of-00001.parquet") for s in ("train", "test")}
    return from_banking77(splits, class_names(repo, rev, "data/test-00000-of-00001.parquet", "label"))


def fetch_clinc150():
    repo, rev = "clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec"
    splits = {s: hf(repo, rev, f"plus/{s}-00000-of-00001.parquet") for s in ("train", "validation", "test")}
    return from_clinc150(splits, class_names(repo, rev, "plus/test-00000-of-00001.parquet", "intent"))


def fetch_bitext():
    return from_bitext(hf("bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                          "f93dfd8473c480417bd50123f605f11d638f014c", "default/train/0000.parquet"))


def fetch_hinglish_top():
    repo, rev = "WillHeld/hinglish_top", "53404b8688a8bb2504a3717a345f8fc85c29ee61"
    files = {"train": "train-00000-of-00001-edd2764b201a5c1f", "validation": "validation-00000-of-00001-5abcf62ed020ba5a",
             "test": "test-00000-of-00001-cf691a6b0d345ef6"}
    return from_hinglish_top({s: hf(repo, rev, f"data/{f}.parquet") for s, f in files.items()})


def fetch_massive():
    repo, rev = "AmazonScience/massive", "ed58ac423a2f4121720918bf5301577edce4ffd3"  # refs/convert/parquet
    cols = ["id", "partition", "intent", "utt", "annot_utt", "slot_method"]
    locales = {lang: [r for s in ("train", "validation", "test") for r in hf(repo, rev, f"{loc}/{s}/0000.parquet", cols)]
               for lang, loc in (("en", "en-US"), ("hi", "hi-IN"))}
    return from_massive(locales, class_names(repo, rev, "en-US/test/0000.parquet", "intent"))


def fetch_langcache():
    repo, rev = "redis/langcache-sentencepairs-v1", "fcf7322a21d28643896a6e808fe069bf1417fe77"
    return from_langcache({s: hf(repo, rev, f"all/{s}-00000-of-00001.parquet") for s in ("validation", "test")})


def vcache(name, revision, label):
    return lambda: from_stream(hf(f"vCache/SemBenchmark{name}", revision, "train.parquet", ["prompt", label]),
                               "prompt", label, f"vcache_{name.lower()}")


DATASETS = {  # name: (protocol, fetch)
    "banking77": ("faq", fetch_banking77),
    "clinc150": ("faq", fetch_clinc150),
    "bitext": ("faq", fetch_bitext),
    "hinglish_top": ("faq", fetch_hinglish_top),
    "massive": ("faq", fetch_massive),
    "langcache": ("pairs", fetch_langcache),
    "vcache_lmarena": ("stream", vcache("LmArena", "c6f36229bef6a5f7936e5b9d982bd5766ef2f862", "ID_Set")),
    "vcache_classification": ("stream", vcache("Classification", "373719c21898ba0914b1934e88e0486de2420c02",
                                               "response_llama_3_8b")),  # class = the one-word answer
    "vcache_searchqueries": ("stream", vcache("SearchQueries", "5159bff8ccf5d7c57dd82cbb24a76e2e5d24f1c0", "id_set")),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    force = "--force" in argv
    names = [a for a in argv if a != "--force"] or list(DATASETS)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        protocol, fetch = DATASETS[name]
        path = OUT / f"{name}.jsonl"
        if path.exists() and not force:
            continue
        out = fetch()
        faqs, rows = out if protocol == "faq" else (None, out)
        if faqs is not None:
            (OUT / f"{name}.faqs.json").write_text(json.dumps(faqs, ensure_ascii=False, indent=1))
        path.write_text(jsonl(rows))
        counts = {}
        for r in rows:
            k = f"{r['split']}/{r['kind']}/{r['lang']}"
            counts[k] = counts.get(k, 0) + 1
        print(f"{name}: {len(rows)} rows" + (f", {len(faqs)} FAQs" if faqs else "") + f" | {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
