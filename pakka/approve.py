"""Approve answers waiting to enter the shared cache.

    python -m pakka.approve data/pending.jsonl --to data/approved.jsonl

y = approve, n = discard, s = keep for later, q = quit. Both files are saved after every answer.
"""

import argparse
import json
from pathlib import Path


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pending", type=Path)
    p.add_argument("--to", type=Path, default=Path("data/approved.jsonl"))
    args = p.parse_args(argv)

    rows = [json.loads(line) for line in args.pending.read_text().splitlines() if line.strip()]
    kept = []
    for i, row in enumerate(rows):
        print(f"\n[{i + 1}/{len(rows)}] Q: {row['question']}\n  A: {row['answer']}")
        answer = ""
        while answer not in ("y", "n", "s", "q"):
            answer = input("approve? [y/n/s/q] ").strip().lower()
        if answer == "q":
            kept += rows[i:]
            break
        if answer == "y":
            with open(args.to, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if answer == "s":
            kept.append(row)
        _write(args.pending, kept + rows[i + 1:])
    _write(args.pending, kept)
    print(f"\n{len(kept)} left in {args.pending}")


if __name__ == "__main__":
    main()
