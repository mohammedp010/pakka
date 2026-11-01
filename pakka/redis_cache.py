"""RedisVL SemanticCache behind the same `store` / `nearest` interface as NumpyCache."""

from pakka.cache import Match


class RedisVLCache:
    """Redis reports cosine distance; we convert it back to similarity so a threshold
    means the same number on every backend."""

    def __init__(self, embed, name="pakka", redis_url="redis://localhost:6379", ttl=None):
        from redisvl.extensions.cache.llm import SemanticCache
        from redisvl.utils.vectorize import CustomVectorizer

        vectorizer = CustomVectorizer(
            embed=lambda text: embed([text])[0].tolist(),
            embed_many=lambda texts: embed(texts).tolist(),
        )
        # Widest distance so Redis always returns the nearest entry; the threshold is applied by the caller.
        self._cache = SemanticCache(name=name, redis_url=redis_url, vectorizer=vectorizer, ttl=ttl,
                                    distance_threshold=2.0, overwrite=True)

    def store(self, question, answer, meta=None):
        self._cache.store(question, answer, metadata=meta or {})

    def nearest(self, question):
        found = self._cache.check(prompt=question, num_results=1)
        if not found:
            return None
        top = found[0]
        return Match(top["prompt"], top["response"], 1 - float(top["vector_distance"]), top.get("metadata") or {})

    def clear(self):
        self._cache.clear()
