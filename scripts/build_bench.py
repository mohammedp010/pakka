"""Build data/bench_v1.jsonl: the 100 hand-written pilot questions plus LLM-written paraphrases and
trap questions for every FAQ, labelled again by a second model family.

    python scripts/build_bench.py

Raw generations and labels are cached in data/raw/, so a rerun only redoes what is missing.
Items the two models disagree on get "needs_review": true; check them with `python -m pakka.review`.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.dukaan_bot.llm import make_chat  # noqa: E402
from pakka.testset import assign_splits, dedupe, generate_for_faq, generate_other, judge, reconcile  # noqa: E402

GEN_MODEL = "anthropic/claude-haiku-5.5"  # writes the questions
JUDGE_MODEL = "deepseek/deepseek-v4.1-flash"  # labels them; a different family, so errors don't line up
RAW, OUT = ROOT / "data/raw", ROOT / "data/bench_v1.jsonl"


def cached(name, fn, *args, attempts=3):
    """Run one LLM step once and keep its result in data/raw/, so a failure elsewhere never redoes it."""
    path = RAW / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    for i in range(attempts):
        try:
            result = fn(*args)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
            return result
        except Exception as e:  # bad JSON or a flaky API call: try again, then give up loudly
            print(f"  {name}: retry {i + 1}/{attempts}: {e}")
    raise RuntimeError(f"{name} failed {attempts} times")


def main():
    if OUT.exists() and '"reviewed": true' in OUT.read_text() and "--force" not in sys.argv:
        sys.exit(f"{OUT.relative_to(ROOT)} has human-reviewed labels; rebuilding would erase them. Use --force to do it anyway.")
    faqs = json.loads((ROOT / "data/faqs.json").read_text())
    RAW.mkdir(exist_ok=True)

    gen = make_chat(GEN_MODEL, max_tokens=4000)
    with ThreadPoolExecutor(6) as pool:
        per_faq = list(pool.map(lambda f: cached(f"gen_{f['id']}", generate_for_faq, f, faqs, gen), faqs))
    generated = [item for batch in per_faq for item in batch] + cached("gen_other", generate_other, faqs, gen)
    print(f"generated: {len(generated)}")

    labeller = make_chat(JUDGE_MODEL, max_tokens=1000)
    chunks = [generated[i:i + 25] for i in range(0, len(generated), 25)]
    with ThreadPoolExecutor(6) as pool:
        parts = pool.map(lambda ic: cached(f"judge_{ic[0]:03d}", judge, ic[1], faqs, labeller), enumerate(chunks))
        labels = [x for part in parts for x in part]

    pilot = [{**json.loads(line), "source": "hand", "needs_review": False}
             for line in (ROOT / "data/pilot.jsonl").read_text().splitlines()]
    pilot = [{**p, "near": p["label"], "intended": p["label"], "judged": None} for p in pilot]
    machine = [{**reconcile(item, label), "source": "generated"} for item, label in zip(generated, labels)]
    items = assign_splits(dedupe(pilot + machine))
    OUT.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))

    agree = sum(not m["needs_review"] for m in machine)
    print(f"written: {len(items)} to {OUT.relative_to(ROOT)} | models agreed on {agree}/{len(machine)} "
          f"| needs review: {sum(i['needs_review'] for i in items)}")
    for field in ("kind", "lang", "split"):
        counts = {}
        for i in items:
            counts[i[field]] = counts.get(i[field], 0) + 1
        print(f"  by {field}: {dict(sorted(counts.items()))}")
    print(f"  should-miss: {sum(i['label'] is None for i in items)}")


if __name__ == "__main__":
    main()
