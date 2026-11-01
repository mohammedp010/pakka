"""Fine-tune the Dukaan Bot embedder on the benchmark's dev split (never the held-out split).

    python scripts/finetune.py [--base paraphrase-multilingual-MiniLM-L12-v2] [--out models/pakka-minilm-hinglish]

About a minute on a laptop CPU. Then measure it: python -m pakka.bench --guard --embedder models/pakka-minilm-hinglish
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pakka.train import fine_tune, training_triplets  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--out", default=str(ROOT / "models" / "pakka-minilm-hinglish"))
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    faqs = json.loads((ROOT / "data/faqs.json").read_text())
    rows = [json.loads(line) for line in (ROOT / "data/bench_v1.jsonl").read_text().splitlines()]
    dev = [r for r in rows if r["split"] == "dev"]
    triplets = training_triplets(dev, faqs)
    print(f"training on {len(triplets)} triplets from {len(dev)} dev questions")
    print(f"saved to {fine_tune(args.base, triplets, args.out, epochs=args.epochs)}")


if __name__ == "__main__":
    main()
