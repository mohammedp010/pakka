"""RedisVL SemanticCache behind the same `store` / `nearest` / `forget` interface as NumpyCache."""

from pakka.cache import Match

FILTER_FIELDS = ("partition", "kb_version")  # meta fields Redis can filter on, as tags


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
                                    distance_threshold=2.0, overwrite=True,
                                    filterable_fields=[{"name": f, "type": "tag"} for f in FILTER_FIELDS])

    def store(self, question, answer, meta=None, ttl=None):
        meta = meta or {}
        filters = {f: str(meta[f]) for f in FILTER_FIELDS if meta.get(f) is not None}
        self._cache.store(question, answer, metadata=meta, filters=filters or None,
                          ttl=None if ttl is None else int(ttl))

    def nearest(self, question, allow=None):
        """`allow` maps FILTER_FIELDS to the values a match may have."""
        found = self._cache.check(prompt=question, num_results=1, filter_expression=_filter(allow))
        if not found:
            return None
        top = found[0]
        return Match(top["prompt"], top["response"], 1 - float(top["vector_distance"]), top.get("metadata") or {})

    def forget(self, field, value):
        from redisvl.query import FilterQuery
        from redisvl.query.filter import Tag

        docs = self._cache.index.query(FilterQuery(Tag(field) == str(value), return_fields=["id"], num_results=10_000))
        keys = [d["id"] for d in docs]
        if keys:
            self._cache.drop(keys=keys)
        return len(keys)

    def clear(self):
        self._cache.clear()


def _filter(allow):
    from redisvl.query.filter import Tag

    expression = None
    for field, values in (allow or {}).items():
        if field not in FILTER_FIELDS:
            raise ValueError(f"Redis can only filter on {FILTER_FIELDS}, not {field!r}")
        clause = Tag(field) == [str(v) for v in values if v is not None]
        expression = clause if expression is None else expression & clause
    return expression
