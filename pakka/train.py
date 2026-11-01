"""Fine-tune an embedder on your own labelled questions, with trap questions as hard negatives.

Train only on the dev split and measure on held-out, or the bench numbers mean nothing.
"""


def training_triplets(rows, faqs):
    """(FAQ question, a paraphrase of it, a question that must not get its answer) per paraphrase.
    The negative is a trap written against the same FAQ when there is one, else any should-miss question."""
    question = {f["id"]: f["question"] for f in faqs}
    traps = {}
    for r in rows:
        if r["label"] is None and r.get("near"):
            traps.setdefault(r["near"], []).append(r["q"])
    should_miss = [r["q"] for r in rows if r["label"] is None]
    used, out = {}, []
    for i, r in enumerate(r for r in rows if r["label"] is not None):
        own = traps.get(r["label"])
        if own:
            negative = own[used.get(r["label"], 0) % len(own)]
            used[r["label"]] = used.get(r["label"], 0) + 1
        else:
            negative = should_miss[i % len(should_miss)]
        out.append((question[r["label"]], r["q"], negative))
    return out


def fine_tune(model_name, triplets, out_dir, epochs=1, batch_size=32, seed=0):
    """One epoch is the Redis recipe: more raised in-domain precision but caused forgetting elsewhere."""
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(model_name)
    anchors, positives, negatives = zip(*triplets)
    data = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=2e-5, warmup_steps=0.1, seed=seed, save_strategy="no", report_to=[], logging_steps=5,
        # In-batch negatives: two paraphrases of one FAQ in a batch would be pushed apart, so forbid duplicates.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=data,
                               loss=MultipleNegativesRankingLoss(model)).train()
    model.save(str(out_dir))
    return out_dir
