import pytest
from conftest import fake_embed

from pakka.cache import NumpyCache


def test_empty_cache_has_no_nearest():
    assert NumpyCache(fake_embed).nearest("refund kab milega") is None


def test_identical_question_scores_one():
    cache = NumpyCache(fake_embed)
    cache.store("How long does a refund take?", "5-7 days")
    match = cache.nearest("How long does a refund take?")
    assert match.answer == "5-7 days"
    assert match.score == pytest.approx(1.0, abs=1e-5)


def test_nearest_picks_the_closest_entry_and_keeps_meta():
    cache = NumpyCache(fake_embed)
    cache.store("How long does a refund take?", "refund answer", {"id": "faq_01"})
    cache.store("How can I track my order?", "track answer", {"id": "faq_06"})
    match = cache.nearest("refund take how long")
    assert match.question == "How long does a refund take?"
    assert match.meta == {"id": "faq_01"}
    assert 0 < match.score < 1


# --- phase 3: expiry, filters, erasure

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_an_entry_past_its_ttl_is_never_returned():
    clock = Clock()
    cache = NumpyCache(fake_embed, clock=clock)
    cache.store("COD charges kitne hain?", "Rs 20", ttl=60)
    assert cache.nearest("COD charges kitne hain?").answer == "Rs 20"
    clock.now += 61
    assert cache.nearest("COD charges kitne hain?") is None


def test_allow_only_returns_entries_whose_meta_matches_every_field():
    cache = NumpyCache(fake_embed)
    cache.store("refund kab milega", "tenant A answer", {"partition": "a", "kb_version": "v1"})
    cache.store("refund kab milega?", "global answer", {"partition": "global", "kb_version": "v1"})
    cache.store("refund kab milega!", "old answer", {"partition": "global", "kb_version": "v0"})
    cache.store("refund kab milega.", "no tags")
    allow = {"partition": {"global", "b"}, "kb_version": {"v1"}}
    assert cache.nearest("refund kab milega", allow=allow).answer == "global answer"
    assert cache.nearest("refund kab milega", allow={"partition": {"z"}}) is None


def test_forget_deletes_every_entry_with_that_value_and_keeps_the_rest():
    cache = NumpyCache(fake_embed)
    cache.store("refund kab milega", "for ravi", {"partition": "ravi"})
    cache.store("order kab aayega", "for ravi too", {"partition": "ravi"})
    cache.store("COD hai kya", "global", {"partition": "global"})
    assert cache.forget("partition", "ravi") == 2
    assert cache.nearest("refund kab milega").answer == "global"
