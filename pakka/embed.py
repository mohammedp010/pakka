"""Sentence-transformers or API embedders that return unit vectors, so dot product equals cosine."""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

API_URL = "https://openrouter.ai/api/v1/embeddings"
API_DIMENSIONS = 1024  # API models return Matryoshka vectors: 1024 keeps 150k texts in memory on 8 GB
# What a model card asks to put before each text (matched on the model name); a cache pair gets it on both sides.
PREFIXES = {"e5": "query: ", "qwen3-embedding": "Instruct: Retrieve semantically similar text.\nQuery: "}
DEFAULT_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
FAST_EMBEDDER = "paraphrase-multilingual-MiniLM-L12-v2"  # ~9 ms a question on CPU, vs ~235 ms: for small servers
# Picked on HinglishCacheBench v1 dev with pakka.guard on, for a 10% false-hit budget (95% upper bound).
THRESHOLDS = {DEFAULT_EMBEDDER: 0.925, FAST_EMBEDDER: 0.79}


def _prefix(model_name):
    return next((p for key, p in PREFIXES.items() if key in model_name.lower()), "")


def load_embedder(model_name=DEFAULT_EMBEDDER, max_length=None, half=False, transport=None, batch=256):
    """`max_length` caps tokens per text (long chat prompts are cut there), never raising a model's own limit.
    `half` loads float16 weights, half the memory, for a big model on a small machine; vectors stay float32.
    `openrouter:<model>` (e.g. `openrouter:qwen/qwen3-embedding-8b`) embeds through OpenRouter's API with
    OPENROUTER_API_KEY, `batch` texts per call."""
    if model_name.startswith("openrouter:"):
        return _api_embedder(model_name.removeprefix("openrouter:"), max_length, transport, batch)
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, model_kwargs={"dtype": torch.float16} if half else None)
    if max_length:
        model.max_seq_length = min(model.max_seq_length, max_length)
    prefix = _prefix(model_name)

    def embed(texts):
        vectors = np.asarray(model.encode([prefix + t for t in texts], convert_to_numpy=True), dtype=np.float32)
        # Normalized here, in float32: a bf16 model (Qwen3's default) normalizes to within 0.4% of unit length
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    return embed


def _api_embedder(model, max_length, transport, batch):
    import httpx

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(f"Set OPENROUTER_API_KEY to embed with {model} through OpenRouter")
    client = httpx.Client(timeout=120, headers={"Authorization": f"Bearer {key}"}, transport=transport)
    prefix = _prefix(model)
    cut = max_length * 4 if max_length else None  # shortcut: ~4 characters per token; use a tokenizer if it matters

    def post(texts):
        for attempt in range(5):  # a long run meets the odd rate limit or dropped connection
            try:
                response = client.post(API_URL, json={"model": model, "input": texts, "dimensions": API_DIMENSIONS})
                response.raise_for_status()
                return [d["embedding"] for d in sorted(response.json()["data"], key=lambda d: d["index"])]
            except (httpx.HTTPError, KeyError):
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    def embed(texts):
        chunks = [[prefix + (t[:cut] or " ") for t in texts[i:i + batch]] for i in range(0, len(texts), batch)]
        with ThreadPoolExecutor(16) as pool:  # a call takes seconds on some providers; keep many in flight
            vectors = np.array([v for part in pool.map(post, chunks) for v in part], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

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
