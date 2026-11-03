"""Check the labels two LLMs disagreed on, one question at a time.

    python -m pakka.review data/bench_v1.jsonl

The file is saved after every answer, so you can quit any time and carry on later.
"""

import argparse
import json
from pathlib import Path

DROP = object()
HELP = "w = writer's label, j = judge's label, n = should miss, 1-30 = that FAQ, d = drop, s = skip, q = quit"


def decide(item, answer, faq_ids):
    """Apply one review answer. Returns the updated item, DROP, or raises ValueError for bad input."""
    answer = answer.strip().lower()
    if answer == "d":
        return DROP
    if answer == "w":
        label = item["intended"]
    elif answer == "j":
        label = item["judged"]
    elif answer == "n":
        label = None
    elif answer.isdigit() and f"faq_{int(answer):02d}" in faq_ids:
        label = f"faq_{int(answer):02d}"
    else:
        raise ValueError(answer)
    return {**item, "label": label, "needs_review": False, "reviewed": True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data", type=Path)
    p.add_argument("--faqs", type=Path, default=Path("data/faqs.json"))
    args = p.parse_args(argv)

    faqs = {f["id"]: f for f in json.loads(args.faqs.read_text())}
    items = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    todo = [i for i, item in enumerate(items) if item.get("needs_review")]
    print(f"{len(todo)} questions to review. {HELP}\n")

    def name(label):
        return "should miss" if label is None else f"{label}: {faqs[label]['question']}"

    for n, index in enumerate(todo, 1):
        item = items[index]
        print(f"[{n}/{len(todo)}] {item['kind']} ({item['lang']}), written for {item['near'] or 'no FAQ'}")
        print(f"  {item['q']}")
        print(f"  w) writer: {name(item['intended'])}\n  j) judge:  {name(item['judged'])}")
        if "judged2" in item:  # human-written sets have two judges; type the FAQ number or n to take this one
            print(f"     judge 2: {name(item['judged2'])}")
        while True:
            answer = input("> ").strip().lower()
            if answer in ("s", "q"):
                break
            try:
                items[index] = decide(item, answer, faqs)
                break
            except ValueError:
                print(f"  {HELP}")
        if answer == "q":
            break
        kept = [x for x in items if x is not DROP]
        args.data.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept))
        print()
    left = sum(1 for x in items if x is not DROP and x.get("needs_review"))
    print(f"Saved. {left} still need review.")


if __name__ == "__main__":
    main()
