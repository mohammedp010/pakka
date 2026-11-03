"""Phase 9: training data for the Kaggle fine-tune, and the eval files it is measured on.

    python scripts/phase9_data.py

Writes data/public/phase9/ (gitignored):
- upload/triplets.jsonl  (anchor FAQ question, paraphrase, trap) from the dev splits of Hinglish-TOP, MASSIVE
  and half of v1 dev, built with pakka.train.training_triplets
- upload/pairs.jsonl     LangCache pairs from its Hugging Face *train* split, up to 3,000 per source
- upload/phase9_kaggle.py the trainer; zip upload/ and add it to a Kaggle notebook
- ours_calib.jsonl, human_calib.jsonl: v1 test or human test, with the *other* half of v1 dev as dev, so a model
  trained on v1 dev never picks its threshold on rows it was trained on

Only dev or train rows are used. Hinglish-TOP's and MASSIVE's HF train rows are not safe: pakka's test split
reuses half of them (fetch_public.from_parses), so only pakka's dev rows go in. Any training text that is also a
test text of any dataset (normalized), or any human-written message, is dropped.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pakka.cache import normalize  # noqa: E402
from pakka.train import training_triplets  # noqa: E402
from scripts.fetch_public import jsonl, split_of  # noqa: E402

PUBLIC = ROOT / "data/public"
OUT = PUBLIC / "phase9"
LANGCACHE = ("redis/langcache-sentencepairs-v1", "fcf7322a21d28643896a6e808fe069bf1417fe77")
PER_SOURCE = 3000  # LangCache pairs per source: enough signal, and the upload stays a few MB


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def v1_halves():
    """v1 dev split in two by a hash of the text: (train half, calibration half)."""
    dev = [r for r in read(ROOT / "data/bench_v1.jsonl") if r["split"] == "dev"]
    return [r for r in dev if split_of(r["q"]) == "dev"], [r for r in dev if split_of(r["q"]) == "test"]


def held_out_texts():
    """Normalized text of every test row of every dataset, and of every human-written message."""
    texts = set()
    for path in [*PUBLIC.glob("*.jsonl"), ROOT / "data/bench_v1.jsonl"]:
        if path.name == "human_eval.jsonl":
            continue
        for r in read(path):
            if r["split"] == "test":
                texts.update(normalize(r[k]) for k in ("q", "q2") if k in r)
    texts.update(normalize(r["q"]) for r in read(ROOT / "data/bench_human.jsonl"))
    return texts


def faq_triplets():
    """{source: triplets} from each FAQ-protocol dataset's dev rows."""
    out = {}
    for name in ("hinglish_top", "massive"):
        faqs = json.loads((PUBLIC / f"{name}.faqs.json").read_text())
        out[name] = training_triplets([r for r in read(PUBLIC / f"{name}.jsonl") if r["split"] == "dev"], faqs)
    out["ours"] = training_triplets(v1_halves()[0], json.loads((ROOT / "data/faqs.json").read_text()))
    return out


def sample_pairs(rows, per_source=PER_SOURCE):
    """Up to `per_source` pairs per source, picked by a hash of the text so the sample never changes."""
    def key(r):
        return hashlib.md5((r["sentence1"] + "\t" + r["sentence2"]).encode()).hexdigest()

    out = []
    for source in sorted({r["source"] for r in rows}):
        out += sorted((r for r in rows if r["source"] == source), key=key)[:per_source]
    return out


def langcache_pairs():
    cached = OUT / "langcache_train_sample.jsonl"
    if not cached.exists():  # the train split is 150 MB: one resumable download (not hf()'s ranged reads), keep the sample
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(LANGCACHE[0], "all/train-00000-of-00001.parquet", repo_type="dataset", revision=LANGCACHE[1])
        rows = pq.read_table(path, columns=["sentence1", "sentence2", "label", "source"]).to_pylist()
        OUT.mkdir(parents=True, exist_ok=True)
        cached.write_text(jsonl(sample_pairs(rows, PER_SOURCE * 2)))  # spare rows for the leakage filter
    return read(cached)


def main():
    blocked = held_out_texts()
    clean = lambda *texts: not any(normalize(t) in blocked for t in texts)  # noqa: E731
    upload = OUT / "upload"
    upload.mkdir(parents=True, exist_ok=True)

    triplets = [{"source": s, "anchor": a, "positive": p, "negative": n}
                for s, ts in faq_triplets().items() for a, p, n in ts if clean(p, n)]
    pairs = [{"source": "langcache", "sub": r["source"], "s1": r["sentence1"], "s2": r["sentence2"],
              "label": float(r["label"])} for r in sample_pairs([r for r in langcache_pairs()
                                                                  if clean(r["sentence1"], r["sentence2"])])]
    (upload / "triplets.jsonl").write_text(jsonl(triplets))
    (upload / "pairs.jsonl").write_text(jsonl(pairs))
    shutil.copy(ROOT / "scripts/phase9_kaggle.py", upload)

    calib, test_v1 = v1_halves()[1], [r for r in read(ROOT / "data/bench_v1.jsonl") if r["split"] == "test"]
    test_human = [r for r in read(ROOT / "data/bench_human.jsonl") if r["split"] == "test"]
    (OUT / "ours_calib.jsonl").write_text(jsonl(calib + test_v1))
    (OUT / "human_calib.jsonl").write_text(jsonl(calib + test_human))

    counts = {}
    for t in triplets:
        counts[t["source"]] = counts.get(t["source"], 0) + 1
    print(f"triplets {counts}, langcache pairs {len(pairs)} ({sum(p['label'] for p in pairs):.0f} same), "
          f"v1 dev calibration half {len(calib)}")
    print(f"upload folder: {upload} ({sum(f.stat().st_size for f in upload.iterdir()) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
