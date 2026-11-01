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
