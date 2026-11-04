from conftest import fake_embed

from scripts import phase10_bot as p10

FAQS = [{"id": "refund", "question": "how do I get a refund", "answer": "Refunds take 5 days."},
        {"id": "track", "question": "where is my order", "answer": "See Orders."}]
ROWS = [{"q": "how do I get a refund", "label": "refund"}, {"q": "tell me a joke", "label": None}]
BASE = {r["q"]: {"answer": "fresh", "tokens": 100, "cost": 0.001, "ms": 1000.0} for r in ROWS}


def test_replay_fills_new_verifier_pairs_once_and_adds_their_time(tmp_path, monkeypatch):
    monkeypatch.setattr(p10, "OUT", tmp_path)
    asked = []

    def fake_check(verifier, pairs, store):
        asked.extend(pairs)
        store.update({p10.key(verifier, *p): {"value": 0.9, "ms": 300.0} for p in pairs})

    monkeypatch.setattr(p10, "check_pairs", fake_check)
    checks = {}
    rows = p10.replay("tuned+jev", fake_embed, FAQS, ROWS, BASE, checks)
    assert asked == [("how do I get a refund", "how do I get a refund")]
    hit, miss = rows
    assert hit["hit"] and hit["checked"] and 300 <= hit["ms"] < 1000
    assert not miss["hit"] and miss["ms"] >= 1000  # a miss waits for the LLM
    assert p10.replay("tuned+jev", fake_embed, FAQS, ROWS, BASE, checks)[0]["hit"] and len(asked) == 1


def test_a_verifier_error_is_a_miss_and_the_summary_counts_it(tmp_path, monkeypatch):
    monkeypatch.setattr(p10, "OUT", tmp_path)
    checks = {p10.key("jev", "how do I get a refund", "how do I get a refund"): {"value": None, "ms": 5000.0}}
    rows = p10.replay("tuned+jev", fake_embed, FAQS, ROWS, BASE, checks)
    assert rows[0]["reason"] == "blocked:verify_error"
    s = p10.summarize(rows, BASE, {}, "jev", {f["id"]: f["answer"] for f in FAQS})
    assert s["hits"] == 0 and s["verify_errors"] == 1 and s["checks"] == 1 and s["net_dollars"] < 0


def test_judge_batch_wants_a_verdict_for_every_reply():
    items = [{"q": "a", "answer": "x"}, {"q": "b", "answer": "y"}]
    assert p10.judge_batch(items, lambda system, user: ('{"1": true, "2": false}', 0)) == [True, False]
    try:
        p10.judge_batch(items, lambda system, user: ('{"1": true}', 0))
    except ValueError:
        return
    raise AssertionError("a missing verdict must fail")
