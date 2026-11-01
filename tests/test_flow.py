import json

from conftest import FakeLLM, fake_embed

from pakka.cache import Match, NumpyCache
from pakka.flow import CachedAnswerer


class FixedScoreCache:
    """Always returns one match with a chosen score, to test the threshold edge exactly."""

    def __init__(self, score):
        self.score, self.stored = score, []

    def nearest(self, question):
        return Match("cached question", "cached answer", self.score, {"id": "faq_01"})

    def store(self, question, answer, meta=None):
        self.stored.append((question, answer))


def test_first_question_on_empty_cache_is_a_miss_that_calls_the_llm():
    llm = FakeLLM()
    result = CachedAnswerer(NumpyCache(fake_embed), llm, threshold=0.9).ask("refund kab milega")
    assert not result.hit and result.reason == "empty_cache"
    assert llm.calls == ["refund kab milega"] and result.tokens == 42


def test_repeat_question_comes_from_cache_without_calling_the_llm():
    llm = FakeLLM()
    answerer = CachedAnswerer(NumpyCache(fake_embed), llm, threshold=0.9)
    first = answerer.ask("refund kab milega")
    second = answerer.ask("refund kab milega")
    assert second.hit and second.reason == "hit"
    assert second.answer == first.answer and second.tokens == 0
    assert len(llm.calls) == 1


def test_score_equal_to_threshold_is_a_hit():
    result = CachedAnswerer(FixedScoreCache(0.85), FakeLLM(), threshold=0.85).ask("q")
    assert result.hit and result.matched_id == "faq_01" and result.answer == "cached answer"


def test_score_below_threshold_is_a_miss_that_stores_the_new_pair():
    cache = FixedScoreCache(0.8499)
    result = CachedAnswerer(cache, FakeLLM(), threshold=0.85).ask("q")
    assert not result.hit and result.reason == "below_threshold" and result.score == 0.8499
    assert cache.stored == [("q", "LLM answer: q")]


def test_store_misses_off_leaves_the_cache_untouched():
    cache = FixedScoreCache(0.1)
    CachedAnswerer(cache, FakeLLM(), threshold=0.85, store_misses=False).ask("q")
    assert cache.stored == []


def test_every_request_is_logged_as_one_json_line(tmp_path):
    log = tmp_path / "requests.jsonl"
    answerer = CachedAnswerer(NumpyCache(fake_embed), FakeLLM(), threshold=0.9, log_path=log)
    answerer.ask("refund kab milega")
    answerer.ask("refund kab milega")
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["hit"] for line in lines] == [False, True]
    assert set(lines[0]) == {"ts", "question", "hit", "reason", "score", "matched_id", "latency_ms", "tokens", "stored"}
    assert lines[1]["latency_ms"] >= 0 and lines[1]["tokens"] == 0


def test_a_guard_veto_turns_a_hit_into_a_logged_miss(tmp_path):
    llm, log = FakeLLM(), tmp_path / "log.jsonl"
    answerer = CachedAnswerer(FixedScoreCache(0.95), llm, threshold=0.85, log_path=log,
                              guard=lambda q, cached_q, cached_a: "negation")
    result = answerer.ask("refund nahi mila")
    assert not result.hit and result.reason == "blocked:negation" and result.score == 0.95
    assert llm.calls == ["refund nahi mila"]
    assert json.loads(log.read_text())["reason"] == "blocked:negation"


def test_guard_sees_the_cached_question_and_answer_and_can_allow_the_hit():
    seen = []
    answerer = CachedAnswerer(FixedScoreCache(0.95), FakeLLM(), threshold=0.85,
                              guard=lambda q, cached_q, cached_a: seen.append((q, cached_q, cached_a)))
    assert answerer.ask("refund kab milega").hit
    assert seen == [("refund kab milega", "cached question", "cached answer")]


def test_guard_is_not_consulted_below_the_threshold():
    seen = []
    CachedAnswerer(FixedScoreCache(0.5), FakeLLM(), threshold=0.85, guard=lambda *a: seen.append(a)).ask("q")
    assert seen == []
