"""Real embedding model and real Redis. Run with: pytest -m integration"""

import os

import numpy as np
import pytest
from conftest import FAQS

from pakka.cache import NumpyCache

pytestmark = pytest.mark.integration
REDIS_URL = os.environ.get("PAKKA_REDIS_URL", "redis://localhost:6380")


@pytest.fixture(scope="module")
def embed():
    from pakka.embed import load_embedder

    return load_embedder()  # the default, Qwen3-Embedding-0.6B


def seeded(cache):
    for faq in FAQS:
        cache.store(faq["question"], faq["answer"], {"id": faq["id"]})
    return cache


def test_vectors_are_unit_length(embed):
    norms = np.linalg.norm(embed(["refund kab milega?", "How long does a refund take?"]), axis=1)
    assert norms == pytest.approx([1.0, 1.0], abs=1e-5)


def test_english_paraphrase_finds_its_faq(embed):
    match = seeded(NumpyCache(embed)).nearest("How many days does it take to get my refund?")
    assert match.meta["id"] == "faq_01" and match.score > 0.6


@pytest.fixture
def redis_cache(embed):
    import redis

    try:
        redis.Redis.from_url(REDIS_URL).execute_command("FT._LIST")
    except Exception as e:
        pytest.skip(f"no Redis with search at {REDIS_URL}: {e}")
    from pakka.redis_cache import RedisVLCache

    cache = RedisVLCache(embed, name="pakka_test", redis_url=REDIS_URL)
    yield cache
    cache.clear()


def test_redisvl_empty_cache_has_no_nearest(redis_cache):
    assert redis_cache.nearest("refund kab milega?") is None


def test_redisvl_returns_the_same_match_and_score_as_numpy(embed, redis_cache):
    in_memory, in_redis = seeded(NumpyCache(embed)), seeded(redis_cache)
    for q in ["refund kab milega?", "Do you deliver to Dubai?", "How do I cancel my order?"]:
        a, b = in_memory.nearest(q), in_redis.nearest(q)
        assert b.meta["id"] == a.meta["id"]
        assert b.score == pytest.approx(a.score, abs=1e-3)


# --- phase 3: the policy features behave the same on Redis

def test_redisvl_filters_by_partition_and_version(redis_cache):
    redis_cache.store("refund kab milega?", "ravi's answer", {"partition": "ravi", "kb_version": "v1"})
    redis_cache.store("refund kab milega", "old global", {"partition": "global", "kb_version": "v0"})
    redis_cache.store("How long does a refund take?", "global v1", {"partition": "global", "kb_version": "v1"})
    allow = {"partition": {"global", "priya"}, "kb_version": {"v1"}}
    assert redis_cache.nearest("refund kab milega?", allow=allow).answer == "global v1"
    assert redis_cache.nearest("refund kab milega?", allow={"partition": {"ravi"}}).answer == "ravi's answer"


def test_redisvl_forget_deletes_one_partition(redis_cache):
    redis_cache.store("refund kab milega?", "for ravi", {"partition": "ravi"})
    redis_cache.store("order kab aayega?", "for ravi too", {"partition": "ravi"})
    redis_cache.store("COD hai kya?", "global", {"partition": "global"})
    assert redis_cache.forget("partition", "ravi") == 2
    assert redis_cache.nearest("refund kab milega?").answer == "global"


def test_redisvl_entries_expire_after_their_ttl(redis_cache):
    import time

    redis_cache.store("COD charges kitne hain?", "Rs 20", {"partition": "global"}, ttl=1)
    assert redis_cache.nearest("COD charges kitne hain?").answer == "Rs 20"
    time.sleep(2)
    assert redis_cache.nearest("COD charges kitne hain?") is None


def test_redisvl_canary_one_customers_answer_never_reaches_another(embed, redis_cache):
    from conftest import FakeLLM

    from pakka.flow import CachedAnswerer
    from pakka.policy import Policy

    bot = CachedAnswerer(redis_cache, FakeLLM(), threshold=0.9, policy=Policy(kb_version="v1"))
    bot.ask("mera wallet balance kitna hai", partition="ravi")
    assert bot.ask("mera wallet balance kitna hai", partition="ravi").hit
    assert not bot.ask("mera wallet balance kitna hai", partition="priya").hit
