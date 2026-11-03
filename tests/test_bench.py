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


# --- phase 6: public datasets: a disk cache for embeddings, and the stream and pairs protocols

import numpy as np  # noqa: E402

from pakka.bench import pair_rows, pick_stream_threshold, replay_stream  # noqa: E402
from pakka.embed import disk_cached  # noqa: E402


def test_pick_threshold_is_fast_on_tens_of_thousands_of_rows():
    import time

    rows = [{"score": i / 60000, "label": "a", "matched_id": "a" if i % 50 else "b"} for i in range(60000)]
    start = time.perf_counter()
    assert pick_threshold(rows, budget=0.03) == 0.0  # 2% wrong everywhere, upper bound about 2.1%
    assert time.perf_counter() - start < 2


def test_disk_cache_embeds_each_text_once_across_runs(tmp_path):
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)

    first = disk_cached(embed, tmp_path / "m" / "set.npz")
    assert first(["ab", "c", "ab"]).tolist() == [[2, 1], [1, 1], [2, 1]]
    first.save()
    again = disk_cached(embed, tmp_path / "m" / "set.npz")
    assert again(["c", "ab"]).tolist() == [[1, 1], [2, 1]]
    assert again(["new"]).tolist() == [[3, 1]]
    assert calls == [["ab", "c"], ["new"]]


def unit(*xy):
    v = np.array(xy, dtype=np.float32)
    return v / np.linalg.norm(v)


# class 0 near [1, 0], class 1 near [0, 1]; the 3rd item sits close to class 0's question but is class 1
STREAM = [{"q": q, "label": c, "split": "dev"} for q, c in [("a", 0), ("a2", 0), ("trap", 1), ("b", 1), ("b2", 1)]]
VECS = np.stack([unit(1, 0), unit(0.99, 0.141), unit(0.945, 0.327), unit(0, 1), unit(0.1, 0.995)])


def test_stream_hits_the_nearest_stored_question_and_stores_every_miss():
    rows = replay_stream(VECS, STREAM, 0.9)
    assert [(r["matched_id"], r["cache_size"]) for r in rows] == [(None, 0), (0, 1), (0, 1), (0, 1), (1, 2)]
    r = evaluate(rows, 0.9)
    assert (r["hits"], r["correct"], r["false_hits"]) == (3, 2, 1)
    assert rows[2]["matched_q"] == "a" and rows[4]["matched_q"] == "b"


def test_stream_gives_the_same_answer_for_any_block_size():
    for t in (0.5, 0.9, 0.99):
        assert replay_stream(VECS, STREAM, t, block=2) == replay_stream(VECS, STREAM, t)


def test_a_guard_veto_in_a_stream_is_a_miss_so_that_question_is_stored():
    rows = replay_stream(VECS, STREAM, 0.9, guard=lambda q, cached, answer: "slot" if q == "trap" else None)
    assert rows[2]["blocked"] == "slot" and rows[3]["cache_size"] == 2
    assert evaluate(rows, 0.9)["false_hits"] == 0


def test_stream_threshold_is_the_lowest_grid_point_that_fits_the_budget_on_dev():
    t, rows = pick_stream_threshold(VECS, STREAM, budget=0.7)
    assert t == pytest.approx(0.95)  # at 0.94 the trap is served: 1 wrong of 3, upper bound 79%
    assert evaluate(rows, t)["false_hits"] == 0
    assert pick_stream_threshold(VECS, STREAM, budget=0.01) == (None, None)


def test_pair_rows_count_a_high_score_on_a_different_pair_as_a_false_hit():
    pairs = [{"q": "x", "q2": "x2", "label": 1, "source": "qqp"}, {"q": "y", "q2": "y2", "label": 0, "source": "paws"}]
    rows = pair_rows(np.stack([unit(1, 0), unit(1, 0)]), np.stack([unit(1, 0.1), unit(1, 0.2)]), pairs,
                     guard=lambda q, cached, answer: None)
    r = evaluate(rows, 0.9)
    assert (r["hits"], r["correct"], r["false_hits"], r["missed"]) == (2, 1, 1, 0)
    assert [x["source"] for x in rows] == ["qqp", "paws"] and rows[0]["blocked"] is None

from pakka.bench import format_report  # noqa: E402


def test_report_shows_every_guard_reason_that_vetoed_something():
    line = {"encoder": "m + guards", "auc": 0.7, "threshold": 0.8, "hit_rate": 0.1, "correct": 9, "false_hits": 1,
            "false_hit_rate": 0.1, "false_hit_upper": 0.3, "en": "-", "hinglish": "-", "hi": "-", "lookup_p50_ms": 1,
            "false_by_kind": {}, "blocked_by": {"direction": 2, "entity": 1}, "blocked_correct": 1, "guard_p50_us": 60}
    report = format_report([line], 0.1, 10)
    assert "| Encoder | direction | entity | Good hits lost | Guard p50 |" in report
    assert "| m + guards | 2 | 1 | 1 | 60 µs |" in report


import json  # noqa: E402

from pakka.embed import load_embedder  # noqa: E402


def test_api_embedder_batches_in_order_cuts_long_text_adds_the_model_prefix_and_returns_unit_vectors(monkeypatch):
    import httpx

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = []

    def reply(request):
        body = json.loads(request.content)
        seen.append(body)
        data = [{"index": i, "embedding": [len(t), 1.0]} for i, t in enumerate(body["input"])][::-1]  # any order
        return httpx.Response(200, json={"data": data})

    embed = load_embedder("openrouter:qwen/qwen3-embedding-8b", max_length=4, transport=httpx.MockTransport(reply), batch=2)
    out = embed(["a", "bb", "x" * 50])
    assert [b["model"] for b in seen] == ["qwen/qwen3-embedding-8b"] * 2 and seen[0]["dimensions"] == 1024
    prefix = "Instruct: Retrieve semantically similar text.\nQuery: "
    assert seen[0]["input"] == [prefix + "a", prefix + "bb"] and seen[1]["input"] == [prefix + "x" * 16]  # 4 tokens ~ 16 chars
    assert out.shape == (3, 2) and np.allclose(np.linalg.norm(out, axis=1), 1)
    assert out[0][0] < out[1][0] < out[2][0]  # rows stay in input order


def fake_jev(seen, busy=0):
    """A stand-in for TypeSafe's API: the first `busy` calls get 429, then a noul of 0.9 when both questions
    share their first word, else 0.1."""
    import httpx

    def reply(request):
        seen.append(json.loads(request.content))
        if len(seen) <= busy:
            return httpx.Response(429, headers={"retry-after": "0"})
        state = seen[-1]["state"]
        same = state["new_question"].split()[0] == state["cached_question"].split()[0]
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"same_reply": {"type": "noul", "noul": 0.9 if same else 0.1}}})

    return httpx.MockTransport(reply)


def test_jev_verifier_asks_one_noul_per_pair_and_a_rerun_reads_its_cache(monkeypatch, tmp_path):
    from pakka.bench import jev_verifier

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen, cache = [], tmp_path / "jev.json"
    pairs = [("refund kab milega", "refund status?"), ("cancel my order", "track my order")]
    assert jev_verifier("jev-1.14.0", cache, transport=fake_jev(seen))(pairs) == [0.9, 0.1]
    assert [b["model"] for b in seen] == ["jev-1.14.0"] * 2 and seen[0]["questions"]["same_reply"]["type"] == "noul"
    assert seen[0]["state"] == {"new_question": "refund kab milega", "cached_question": "refund status?"}

    again = []
    assert jev_verifier("jev-1.14.0", cache, transport=fake_jev(again))(pairs[::-1]) == [0.1, 0.9]
    assert again == []  # every pair came from the cache


def test_jev_verifier_waits_out_rate_limits(monkeypatch):
    from pakka.bench import jev_verifier

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen = []
    assert jev_verifier("jev-1.14.0", transport=fake_jev(seen, busy=2))([("a b", "a c")]) == [0.9]
    assert len(seen) == 3
