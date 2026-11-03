"""Phase 9: measure the fine-tuned models from Kaggle against their base model, on data each never saw.

    python scripts/phase9_eval.py                      # every model unzipped into models/phase9/
    python scripts/phase9_eval.py --base Qwen/Qwen3-Embedding-0.6B

Each `...-pakka-no_X` model is measured on dataset X (leave-one-dataset-out); `...-pakka-all` on the datasets no
run trained on (Banking77, CLINC150, Bitext, the vCache streams) and on the human test split. Thresholds are
picked on dev and reported on test, exactly as scripts/public_bench.py does, guards on and off; results are
kept in data/public/results/<model>/, so a rerun only measures what is new. The `-all` model was trained on
half of v1 dev, so on v1 and human test its threshold is picked on the other half (phase9/*_calib.jsonl), and so
is the base model's in that row. Prints markdown tables.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # models are named by their path relative to the repo, as the embedding cache keys them

from pakka.bench import report_encoder, score_with_method  # noqa: E402
from pakka.guard import check  # noqa: E402
from scripts import public_bench as pb  # noqa: E402

FAQS = ROOT / "data/faqs.json"
OWN = {  # v1-FAQ sets measured at the budgets the README uses: (data file, dev rows the threshold is picked on)
    "ours": "data/bench_v1.jsonl",
    "human": "data/public/human_eval.jsonl",
    "ours_calib": "data/public/phase9/ours_calib.jsonl",
    "human_calib": "data/public/phase9/human_calib.jsonl",
}
UNSEEN = ["banking77", "clinc150", "bitext", "vcache_classification", "vcache_lmarena", "vcache_searchqueries"]
HELD_OUT = {"langcache": ["langcache", "langcache_strat"], "hinglish_top": ["hinglish_top"], "massive": ["massive"],
            "ours": ["ours@", "human@"]}  # "@": the v1-FAQ sets at 10% and 5% budgets


def own_result(model, name):
    """Guarded result on a v1-FAQ set at a 10% and a 5% budget."""
    def run():
        faqs = json.loads(FAQS.read_text())
        data = [json.loads(line) for line in Path(OWN[name]).read_text().splitlines() if line.strip()]
        embed = pb.embedder(model, name, [f["question"] for f in faqs] + [r["q"] for r in data])
        rows = score_with_method(model, embed, faqs, data, guard=check)
        return {str(b): {k: v for k, v in report_encoder(model, rows, b).items() if k != "en"} for b in (0.10, 0.05)}

    return pb.memo(pb.RESULTS / pb.slug(model) / f"phase9_{name}.json", run)


def served(r):
    if r.get("hits") is None:
        return "none fits"
    return f"{r['correct']} / {r['false_hits']} ({pb.pct(r['false_hit_rate'])}, {pb.pct(r['false_hit_upper'])})"


def public_cells(model, name):
    g = pb.result(model, name)["guards"]
    top = pb.top_share(g["test"]) if "test" in g else None
    return [f"{g['auc']:.3f}" if g.get("auc") is not None else "-", served(g),
            pb.pct(sum(r[1] for r in top) / len(top)) if top else "-"]


def compare(base, tuned, names):
    """Rows of base vs fine-tuned on `names`."""
    rows = []
    for name in names:
        if name.endswith("@"):
            own = name[:-1]
            for b in ("0.1", "0.05"):
                x, y = own_result(base, own)[b], own_result(tuned, own)[b]
                rows.append([f"{own} ({float(b):.0%} budget)", f"{x['auc']:.3f} → {y['auc']:.3f}", served(x), served(y), "", ""])
        else:
            x, y = public_cells(base, name), public_cells(tuned, name)
            rows.append([name, f"{x[0]} → {y[0]}", x[1], y[1], x[2], y[2]])
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=pb.DEFAULT)
    p.add_argument("--models", default="models/phase9")
    p.add_argument("--skip", default="", help="datasets to leave out, comma-separated (the vCache streams take hours with Qwen3)")
    args = p.parse_args()
    stem = args.base.split("/")[-1].lower()
    tuned = sorted(str(d.relative_to(ROOT)) for d in (ROOT / args.models).glob(f"{stem}-pakka-*") if d.is_dir())
    if not tuned:
        sys.exit(f"no {stem}-pakka-* folders in {args.models}: unzip the Kaggle outputs there first")
    for m in [args.base, *tuned]:
        pb.MODELS.setdefault(m, Path(m).name)
        if "qwen3" in m.lower():
            pb.HALF.add(m)

    header = ["Held-out set", "AUC, guards: base → tuned", "Base: right / wrong (FHR, upper)", "Tuned", "Base: FHR top 10%", "Tuned"]
    for m in tuned:
        run = m.rsplit("-pakka-", 1)[1]
        names = UNSEEN + ["ours_calib@", "human_calib@"] if run == "all" else HELD_OUT[run.removeprefix("no_")]
        names = [n for n in names if n not in args.skip.split(",")]
        print(f"\n### {Path(m).name} vs {args.base} (5% budget unless noted)\n")
        print(pb.table(header, compare(args.base, m, names)))


if __name__ == "__main__":
    main()
