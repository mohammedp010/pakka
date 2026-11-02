"""Sentence-transformers embedders that return unit vectors, so dot product equals cosine."""

import hashlib
from pathlib import Path

import numpy as np


def load_embedder(model_name="all-MiniLM-L6-v2", max_length=None, half=False):
    """`max_length` caps tokens per text (long chat prompts are cut there), never raising a model's own limit.
    `half` loads float16 weights, half the memory, for a big model on a small machine; vectors stay float32."""
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, model_kwargs={"dtype": torch.float16} if half else None)
    if max_length:
        model.max_seq_length = min(model.max_seq_length, max_length)
    # E5 models are trained with a "query: " prefix for symmetric similarity.
    prefix = "query: " if "e5" in model_name.lower() else ""

    def embed(texts):
        vectors = model.encode([prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    return embed


def disk_cached(embed, path):
    """`embed` backed by every vector it has made, kept at `path` (.npz keyed by a hash of the text), so a
    model embeds each text once across runs. Texts it hasn't seen are embedded in one batch; `.save()` writes."""
    path, store, new = Path(path), {}, []
    if path.exists():
        with np.load(path) as z:
            store = dict(zip(z["keys"].tolist(), z["vectors"]))

    def cached(texts):
        keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        missing = {k: t for k, t in zip(keys, texts) if k not in store}
        if missing:
            store.update(zip(missing, embed(list(missing.values()))))
            new.append(len(missing))
        return np.stack([store[k] for k in keys])

    def save():
        if new:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, keys=np.array(list(store)), vectors=np.stack(list(store.values())))
            new.clear()

    cached.save = save
    return cached
