"""Phase 9 fine-tune, self-contained for a Kaggle T4 notebook (no pakka install needed).

    python phase9_kaggle.py --base minilm              # 5 runs: all + leave-one-dataset-out, a few minutes each
    python phase9_kaggle.py --base qwen --runs all,no_ours
    python phase9_kaggle.py --base minilm --smoke      # 64 examples, CPU is fine: checks the script runs

Reads triplets.jsonl and pairs.jsonl from the folder this file is in (built by scripts/phase9_data.py) and
writes one model folder per run, zipped, into --out. Run `no_X` trains without any part of dataset X, so X
can be measured on data the model never saw. The recipe is pakka/train.py's: one epoch, lr 2e-5,
MultipleNegativesRankingLoss on (FAQ question, paraphrase, trap) with no duplicate texts in a batch; plus
CoSENT on LangCache's labelled pairs, whose "different" pairs are word-order and near-miss traps.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # Kaggle's "T4 x2" would split each batch, and its in-batch negatives
BASES = {  # short name: (model, lr, batch, float32 weights)
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 2e-5, 64, True),
    "qwen": ("Qwen/Qwen3-Embedding-0.6B", 1e-5, 32, False),
}
SOURCES = ["langcache", "hinglish_top", "massive", "ours"]
# pakka.embed.PREFIXES: what pakka puts before every text for this model, so training must see it too
QWEN_PREFIX = "Instruct: Retrieve semantically similar text.\nQuery: "


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def datasets_for(run, triplets, pairs, prefix=""):
    """{name: {column: texts}} for one run, without `datasets` (so pakka's CI can test it)."""
    left_out = run.removeprefix("no_") if run != "all" else None
    t = [r for r in triplets if r["source"] != left_out]
    p = [r for r in pairs if r["source"] != left_out]
    out = {}
    if t:
        out["faq"] = {k: [prefix + r[k] for r in t] for k in ("anchor", "positive", "negative")}
    if p:
        out["pairs"] = {"sentence1": [prefix + r["s1"] for r in p], "sentence2": [prefix + r["s2"] for r in p],
                        "label": [r["label"] for r in p]}
    return out


def train(base, run, triplets, pairs, out_dir, smoke=False):
    import torch
    from datasets import Dataset, DatasetDict
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CoSENTLoss, MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    name, lr, batch, fp32 = BASES[base]
    gpu = torch.cuda.is_available()
    # float32 master weights: Qwen3 loads in bf16 by default, and fp16 mixed precision (a T4 has no bf16) can't unscale bf16 grads
    model = SentenceTransformer(name, device="cuda" if gpu else "cpu", model_kwargs={"dtype": torch.float32})
    model.max_seq_length = 128 if fp32 else 64  # pakka embeds at most 128 tokens; support questions are short
    if not fp32:  # Qwen3-0.6B in a T4's 15 GB. Set on the inner HF model: the Trainer's own switch breaks on
        # some sentence-transformers/transformers pairs (Kaggle's 5.7 + 5.20: unexpected `every_n_layers`)
        inner = getattr(model[0], "auto_model", None) or model[0].model
        inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    data = {k: Dataset.from_dict(v) for k, v in datasets_for(run, triplets, pairs, QWEN_PREFIX if base == "qwen" else "").items()}
    losses = {"faq": MultipleNegativesRankingLoss(model), "pairs": CoSENTLoss(model)}
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "checkpoints"), num_train_epochs=1, per_device_train_batch_size=8 if smoke else batch,
        learning_rate=lr, warmup_ratio=0.1, seed=0, save_strategy="no", report_to=[], logging_steps=50,
        fp16=gpu,
        # In-batch negatives: two paraphrases of one FAQ in a batch would be pushed apart, so forbid duplicates.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    start = time.time()
    SentenceTransformerTrainer(model=model, args=args, train_dataset=DatasetDict(data),
                               loss={k: losses[k] for k in data}).train()
    if not fp32 or run != "all":
        # Half the download. Only MiniLM's `all` model stays float32: it is the one that could ship, and float16
        # runs slowly on a CPU. Eval-only models move scores by about 0.0002; base Qwen3 is measured in float16 too.
        model.half()
    folder = out_dir / f"{name.split('/')[-1].lower()}-pakka-{run}"  # keeps "qwen3-embedding" in the name: pakka adds the prefix
    model.save(str(folder))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    (folder / "phase9_run.json").write_text(json.dumps({
        "base": name, "run": run, "minutes": round((time.time() - start) / 60, 1),
        "examples": {k: len(v) for k, v in data.items()}, "lr": lr, "batch": batch, "epochs": 1}))
    shutil.make_archive(str(folder), "zip", folder)
    print(f"{run}: {sum(len(v) for v in data.values())} examples, {(time.time() - start) / 60:.1f} min -> {folder}.zip",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=list(BASES), default="minilm")
    p.add_argument("--runs", default=",".join(["all"] + [f"no_{s}" for s in SOURCES]))
    p.add_argument("--data", default=str(HERE))
    p.add_argument("--out", default="/kaggle/working" if Path("/kaggle/working").exists() else "phase9_models")
    p.add_argument("--smoke", action="store_true", help="64 examples of each kind, to check the script runs")
    args = p.parse_args()

    import sentence_transformers
    import torch
    import transformers
    print(f"sentence-transformers {sentence_transformers.__version__}, transformers {transformers.__version__}, "
          f"torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    triplets, pairs = read(Path(args.data) / "triplets.jsonl"), read(Path(args.data) / "pairs.jsonl")
    if args.smoke:
        triplets = [r for s in SOURCES for r in [t for t in triplets if t["source"] == s][:16]]
        pairs = pairs[:64]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for run in args.runs.split(","):
        assert run == "all" or run.removeprefix("no_") in SOURCES, f"unknown run {run}"
        train(args.base, run, triplets, pairs, out, args.smoke)


if __name__ == "__main__":
    main()
