"""Write the benchmark's dev paraphrases as approved answers, standing in for past customer
questions a team has approved. Used to measure the extra-keys setup with the real bot.

    python scripts/approved_from_dev.py > logs/approved_dev.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.dukaan_bot.app import FAQS, kb_version  # noqa: E402

answer = {f["id"]: f["answer"] for f in FAQS}
for line in (ROOT / "data/bench_v1.jsonl").read_text().splitlines():
    row = json.loads(line)
    if row["split"] == "dev" and row["label"]:
        print(json.dumps({"question": row["q"], "answer": answer[row["label"]], "kb_version": kb_version(FAQS),
                          "id": row["label"]}, ensure_ascii=False))
