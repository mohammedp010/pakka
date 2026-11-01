"""Sentence-transformers embedders that return unit vectors, so dot product equals cosine."""


def load_embedder(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    # E5 models are trained with a "query: " prefix for symmetric similarity.
    prefix = "query: " if "e5" in model_name.lower() else ""

    def embed(texts):
        return model.encode([prefix + t for t in texts], normalize_embeddings=True, convert_to_numpy=True)

    return embed
