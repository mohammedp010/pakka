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


# --- phase 2: guards veto hits

from pakka.bench import score_rows  # noqa: E402


def test_a_blocked_hit_is_not_served_and_a_blocked_paraphrase_counts_as_missed():
    rows = [{**ROWS[0], "blocked": "negation"}, {**ROWS[2], "blocked": "slot"}, ROWS[3]]
    r = evaluate(rows, 0.75)
    assert (r["hits"], r["false_hits"], r["missed"]) == (0, 0, 2)
    assert r["blocked_by"] == {"negation": 1, "slot": 1}


def test_guards_let_the_threshold_go_lower_within_the_same_budget():
    trap = {"score": 0.97, "label": None, "matched_id": "a", "blocked": "negation"}
    rows = [trap] + [{"score": 0.9 - i / 100, "label": "a", "matched_id": "a"} for i in range(40)]
    assert pick_threshold(rows, budget=0.10) == pytest.approx(0.51)
    assert pick_threshold([{**trap, "blocked": None}] + rows[1:], budget=0.10) is None  # 1 of 41 wrong: upper 12.6%


def test_score_rows_records_the_guard_reason_against_the_matched_question():
    from pakka.cache import Match

    class OneEntry:
        def nearest(self, q):
            return Match("How long does a refund take?", "5-7 days", 0.9, {"id": "faq_01"})

    rows = score_rows(OneEntry(), [{"q": "refund nahi mila", "label": None}, {"q": "refund kab milega", "label": "faq_01"}],
                      guard=lambda q, c, a: "negation" if "nahi" in q else None)
    assert [r["blocked"] for r in rows] == ["negation", None]
    assert all(r["guard_ms"] >= 0 for r in rows)


def test_verifier_rescores_only_unblocked_candidates_at_or_above_the_floor():
    from pakka.bench import verify_rows

    rows = [{"q": "a", "matched_q": "x", "score": 0.9}, {"q": "b", "matched_q": "y", "score": 0.6},
            {"q": "c", "matched_q": "z", "score": 0.5}, {"q": "d", "matched_q": "w", "score": 0.95, "blocked": "slot"}]
    seen = []

    def verify(pairs):
        seen.extend(pairs)
        return [0.2, 0.8]

    out = verify_rows(rows, verify, floor=0.6)
    assert seen == [("a", "x"), ("b", "y")]
    assert [r["score"] for r in out] == [0.2, 0.8, -1.0, -1.0]


def test_score_rows_keeps_the_matched_question_for_verifiers():
    from pakka.cache import Match

    class OneEntry:
        def nearest(self, q):
            return Match("How long does a refund take?", "5-7 days", 0.9, {"id": "faq_01"})

    assert score_rows(OneEntry(), [{"q": "refund?", "label": "faq_01"}])[0]["matched_q"] == "How long does a refund take?"


def test_fail_above_exits_nonzero_when_the_false_hit_bound_is_too_high(tmp_path):
    import json as _json

    from pakka.bench import main

    faqs, data = tmp_path / "faqs.json", tmp_path / "data.jsonl"
    faqs.write_text(_json.dumps([{"id": "faq_01", "question": "How long does a refund take?", "answer": "5-7 days"}]))
    rows = [{"q": "How long does a refund take?", "label": label, "split": split, "kind": "paraphrase", "lang": "en"}
            for split in ("dev", "test") for label in ["faq_01"] * 20 + [None]]  # 1 wrong in 21 served
    data.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    args = ["--faqs", str(faqs), "--data", str(data), "--embedder", "exact", "--budget", "0.5"]
    main(args + ["--fail-above", "0.5"])  # upper bound about 23%: passes
    with pytest.raises(SystemExit) as exit_info:
        main(args + ["--fail-above", "0.1"])
    assert exit_info.value.code == 1


# --- phase 4: extra keys and per-FAQ thresholds, cross-fitted on dev

from pakka.bench import apply_floors, entry_floors, score_with_method  # noqa: E402


def test_entry_floor_is_the_highest_score_an_faq_was_wrongly_served_at():
    rows = [{"matched_id": "faq_01", "label": None, "score": 0.8}, {"matched_id": "faq_01", "label": "faq_02", "score": 0.7},
            {"matched_id": "faq_01", "label": "faq_01", "score": 0.9}, {"matched_id": "faq_02", "label": "faq_02", "score": 0.6},
            {"matched_id": "faq_03", "label": None, "score": 0.99, "blocked": "negation"}]  # a guard already stopped it
    assert entry_floors(rows) == {"faq_01": 0.8}


def test_hits_on_an_faq_must_beat_its_floor():
    rows = [{"matched_id": "faq_01", "score": 0.85}, {"matched_id": "faq_01", "score": 0.8},
            {"matched_id": "faq_02", "score": 0.5}, {"matched_id": "faq_01", "score": 0.7, "blocked": "slot"}]
    assert [r["blocked"] for r in apply_floors(rows, {"faq_01": 0.8})] == [None, "entry_floor", None, "slot"]


FAQ_ROWS = [{"id": "faq_01", "question": "How long does a refund take?", "answer": "5-7 days"},
            {"id": "faq_02", "question": "Is COD available?", "answer": "Yes"}]


def test_extra_keys_come_from_dev_and_a_dev_question_never_matches_itself():
    data = [{"q": "refund kab milega", "label": "faq_01", "split": "dev"},
            {"q": "cod hai kya", "label": "faq_02", "split": "dev"},
            {"q": "refund kab milega", "label": "faq_01", "split": "test"}]
    plain = score_with_method("exact", None, FAQ_ROWS, data)
    keyed = score_with_method("exact", None, FAQ_ROWS, data, extra_keys=True)
    assert [r["score"] for r in plain] == [0.0, 0.0, 0.0]
    dev = {r["q"]: r["score"] for r in keyed if r["split"] == "dev"}
    assert dev == {"refund kab milega": 0.0, "cod hai kya": 0.0}  # each dev half only sees the other half's keys
    assert [r["score"] for r in keyed if r["split"] == "test"] == [1.0]  # held-out sees every dev key
