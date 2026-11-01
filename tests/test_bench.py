import pytest

from pakka.bench import sweep

ROWS = [
    {"label": "faq_01", "matched_id": "faq_01", "score": 0.92, "lookup_ms": 10},  # right FAQ
    {"label": "faq_02", "matched_id": "faq_01", "score": 0.88, "lookup_ms": 10},  # wrong FAQ
    {"label": None, "matched_id": "faq_03", "score": 0.80, "lookup_ms": 10},  # should have missed
    {"label": "faq_04", "matched_id": "faq_04", "score": 0.70, "lookup_ms": 10},  # paraphrase too far away
]


@pytest.fixture
def by_threshold():
    return {r["threshold"]: r for r in sweep(ROWS, [0.75, 0.85, 0.95], llm_ms=1000)}


def test_loose_threshold_serves_more_but_makes_false_hits(by_threshold):
    r = by_threshold[0.75]
    assert (r["hits"], r["correct"], r["false_hits"], r["missed"]) == (3, 1, 2, 1)
    assert r["hit_rate"] == 0.75
    assert r["avg_latency_ms"] == (4 * 10 + 1000) / 4


def test_a_wrong_faq_counts_as_a_false_hit_not_a_miss(by_threshold):
    r = by_threshold[0.85]
    assert (r["hits"], r["correct"], r["false_hits"], r["missed"]) == (2, 1, 1, 1)


def test_strict_threshold_misses_every_paraphrase(by_threshold):
    r = by_threshold[0.95]
    assert (r["hits"], r["false_hits"], r["missed"]) == (0, 0, 3)
    assert r["avg_latency_ms"] == 1010
