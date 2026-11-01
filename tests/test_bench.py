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


# --- phase 1: confidence intervals, threshold choice, ranking quality, exact-match baseline

from pakka.bench import evaluate, pchr_auc, pick_threshold, wilson  # noqa: E402
from pakka.cache import ExactCache  # noqa: E402


def test_zero_false_hits_in_40_still_allows_about_9_percent():
    assert wilson(0, 40)[1] == pytest.approx(0.0876, abs=1e-3)


def test_wilson_interval_matches_the_published_banking_faq_example():
    low, high = wilson(4, 49)
    assert low == pytest.approx(0.032, abs=2e-3) and high == pytest.approx(0.192, abs=2e-3)


def test_false_hit_rate_is_wrong_answers_among_served_answers():
    r = evaluate(ROWS, 0.75)
    assert r["false_hit_rate"] == pytest.approx(2 / 3)
    assert r["false_hit_upper"] == pytest.approx(wilson(2, 3)[1])


def test_false_hits_are_broken_down_by_kind():
    rows = [{**r, "kind": k} for r, k in zip(ROWS, ["paraphrase", "paraphrase", "negation", "paraphrase"])]
    assert evaluate(rows, 0.75)["false_by_kind"] == {"paraphrase": 1, "negation": 1}


def test_pchr_auc_averages_precision_over_every_hit_ratio():
    rows = [{"score": s, "label": "a", "matched_id": m} for s, m in [(0.9, "a"), (0.8, "a"), (0.7, "b"), (0.6, "a")]]
    assert pchr_auc(rows) == pytest.approx((1 + 1 + 2 / 3 + 3 / 4) / 4)


PICK_ROWS = [{"score": s, "label": "a", "matched_id": m}
             for s, m in [(0.95, "a"), (0.90, "a"), (0.85, "a"), (0.80, "b"), (0.70, "a")]]


def test_pick_threshold_takes_the_most_hits_within_the_false_hit_budget():
    assert pick_threshold(PICK_ROWS, budget=0.60) == 0.85  # 0 of 3 wrong, upper bound 0.56
    assert pick_threshold(PICK_ROWS, budget=0.65) == 0.70  # 1 of 5 wrong, upper bound 0.62


def test_pick_threshold_returns_none_when_nothing_fits_the_budget():
    assert pick_threshold(PICK_ROWS, budget=0.10) is None


def test_exact_cache_ignores_case_and_punctuation_only():
    cache = ExactCache()
    cache.store("COD available hai?", "Yes", {"id": "faq_05"})
    assert cache.nearest("cod available  hai").score == 1.0
    assert cache.nearest("COD milta hai?").score == 0.0
    assert ExactCache().nearest("anything") is None
