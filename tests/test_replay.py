from types import SimpleNamespace as R

from examples.dukaan_bot.replay import headline, summarize


def test_summary_counts_false_hits_against_labels():
    data = [{"label": "faq_01"}, {"label": None}, {"label": "faq_02"}, {"label": None}]
    results = [
        R(hit=True, matched_id="faq_01", tokens=0, latency_ms=10),  # correct
        R(hit=True, matched_id="faq_05", tokens=0, latency_ms=10),  # should have missed
        R(hit=False, matched_id=None, tokens=50, latency_ms=1000),
        R(hit=False, matched_id=None, tokens=50, latency_ms=1000),
    ]
    s = summarize(results, data)
    assert (s["hits"], s["false_hits"], s["llm_calls"], s["tokens"]) == (2, 1, 2, 100)
    assert s["avg_ms"] == 505


def test_headline_line_matches_the_readme_format():
    before = {"n": 100, "llm_calls": 100, "avg_ms": 2000.0}
    after = {"n": 100, "hits": 40, "false_hits": 2, "llm_calls": 60, "avg_ms": 1210.0}
    assert headline(before, after, 0.85) == (
        "Cache hit rate 40% on 100 test questions at threshold 0.85 with 2 false hits; "
        "LLM calls down 40%; average latency down from 2000 ms to 1210 ms.")


def test_split_keeps_only_that_part_of_the_benchmark():
    from examples.dukaan_bot.replay import select

    rows = [{"q": "a", "split": "dev"}, {"q": "b", "split": "test"}, {"q": "c"}]
    assert [r["q"] for r in select(rows, "test")] == ["b"]
    assert select(rows, None) == rows
