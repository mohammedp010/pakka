"""Draw the README charts into docs/img/ from results already measured.

    pip install matplotlib
    python scripts/charts.py

Reads data/public/results/ (scripts/public_bench.py) and data/public/phase10/summary_4000.json
(scripts/phase10_bot.py); it measures nothing itself, so run those first.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.public_bench import FAQ, NAMES, PAIRS, REDIS, STREAM, curves, result  # noqa: E402

IMG = ROOT / "docs/img"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # dataviz reference palette, slots 1-3 and 7
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BUDGET = 0.05

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160})


def redis_defaults():
    """False-hit rate of RedisVL's out-of-the-box cache on each dataset's held-out split."""
    names = [n for n in FAQ + PAIRS + STREAM if n != "langcache_strat"]  # same pairs as langcache, other split
    rates = sorted(((result(REDIS, n)["plain"]["fixed"]["false_hit_rate"], NAMES[n]) for n in names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([n for _, n in rates], [100 * r for r, _ in rates], height=0.6,
            color=[SERIES[0] if r > BUDGET else "#a9c8ee" for r, _ in rates])
    for i, (r, _) in enumerate(rates):
        ax.text(100 * r + 0.8, i, f"{100 * r:.1f}%", va="center", color=INK)
    ax.axvline(100 * BUDGET, color=MUTED, ls="--", lw=1)
    ax.text(100 * BUDGET + 0.8, len(rates) - 0.45, "5% budget", color=MUTED, fontsize=9)
    ax.set_xlabel("Wrong answers among cache hits (%)")
    fig.suptitle("RedisVL defaults (langcache-embed-v1 at similarity 0.90), held-out splits", x=0.01, ha="left", color=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig, sum(r > BUDGET for r, _ in rates), len(rates)


def bot_run():
    """Phase 10: share answered from cache against share of those answers the judges call wrong."""
    s = json.loads((ROOT / "data/public/phase10/summary_4000.json").read_text())["setups"]
    labels = {"free": "MiniLM (default), free", "tuned+llm": "Fine-tuned MiniLM + cheap LLM",
              "tuned+jev": "Fine-tuned MiniLM + Jev", "qwen+jev": "Qwen3-0.6B (GPU) + Jev"}
    fig, ax = plt.subplots(figsize=(7, 4))
    llm = [100 * r["llm_wrong"] / r["hits"] for r in s.values()]
    ax.axhspan(min(llm), max(llm), color=GRID)
    ax.text(1, max(llm) + 0.6, "the LLM's own answers to the same questions", color=MUTED, fontsize=9)
    for (key, r), color in zip(s.items(), SERIES):
        x, y = 100 * r["hits"] / r["n"], 100 * r["wrong"] / r["hits"]
        ax.plot(x, y, "o", ms=9, color=color, mec="white", mew=2)
        ax.annotate(labels[key], (x, y), xytext=(8, -4 if key != "tuned+llm" else -16), textcoords="offset points", color=INK, fontsize=9)
    ax.set_xlim(0, 75)
    ax.set_ylim(0, 32)
    ax.set_xlabel("Questions answered from cache (%)")
    ax.set_ylabel("Cached answers judged wrong (%)")
    ax.set_title("Dukaan Bot on 4,000 Bitext test questions", loc="left", color=INK)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def cache_size():
    """vCache SearchQueries: the wrong share of hits stays flat as the cache grows, wrong answers per 100 rise."""
    lines = curves()
    bins = [f"{b['lo'] // 1000}k–{b['hi'] // 1000}k" for b in lines[0][1]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
    for (label, bs), color in zip(lines, SERIES):
        label = label.replace(" (pakka default)", "").replace(" (none fits dev)", "")
        left.plot(bins, [100 * b["false_hits"] / b["hits"] for b in bs], "-o", lw=2, ms=5, color=color, label=label)
        right.plot(bins, [100 * b["false_hits"] / b["questions"] for b in bs], "-o", lw=2, ms=5, color=color)
    left.set_title("Wrong share of hits (%)", loc="left", color=INK)
    right.set_title("Wrong answers per 100 questions", loc="left", color=INK)
    left.set_ylim(0, 60)
    right.set_ylim(0, 25)
    for ax in (left, right):
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("Cache size when the question arrived")
        ax.tick_params(axis="x", rotation=35)
    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("vCache SearchQueries, 150,000 queries replayed in order, every miss stored", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    fig, over, total = redis_defaults()
    assert (over, total) == (6, 10), (over, total)  # the README's "6 of 10 datasets"
    for name, f in (("redisvl_defaults.png", fig), ("bot_run.png", bot_run()), ("cache_size.png", cache_size())):
        f.savefig(IMG / name)
        print("wrote", (IMG / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
