import json

import httpx
import pytest


def fake_jev(seen, busy=0):
    """A stand-in for TypeSafe's API: the first `busy` calls get 429, then a noul of 0.9 when both questions
    share their first word, else 0.1."""

    def reply(request):
        seen.append(json.loads(request.content))
        if len(seen) <= busy:
            return httpx.Response(429, headers={"retry-after": "0"})
        state = seen[-1]["state"]
        same = state["new_question"].split()[0] == state["cached_question"].split()[0]
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"same_reply": {"type": "noul", "noul": 0.9 if same else 0.1}}})

    return httpx.MockTransport(reply)


def fake_openrouter(seen, busy=0):
    """A stand-in for OpenRouter's chat API: says "Yes" when both questions share their first word, else "no"."""

    def reply(request):
        seen.append(json.loads(request.content))
        if len(seen) <= busy:
            return httpx.Response(503)
        user = seen[-1]["messages"][1]["content"]
        cached, new = (line.split(": ", 1)[1] for line in user.splitlines())
        same = new.split()[0] == cached.split()[0]
        return httpx.Response(200, json={"choices": [{"message": {"content": "Yes" if same else "no"}}],
                                         "usage": {"total_tokens": 100}})

    return httpx.MockTransport(reply)


def test_jev_verifier_asks_one_noul_per_pair_and_a_rerun_reads_its_cache(monkeypatch, tmp_path):
    from pakka.verify import jev_verifier

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
    from pakka.verify import jev_verifier

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen = []
    assert jev_verifier("jev-1.14.0", transport=fake_jev(seen, busy=2))([("a b", "a c")]) == [0.9]
    assert len(seen) == 3


def test_openrouter_verifier_asks_yes_or_no_per_pair_and_caches(monkeypatch, tmp_path):
    from pakka.verify import openrouter_verifier

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen, cache = [], tmp_path / "or.json"
    pairs = [("refund kab milega", "refund status?"), ("cancel my order", "track my order")]
    assert openrouter_verifier("some/model", cache, transport=fake_openrouter(seen))(pairs) == [True, False]
    assert [b["model"] for b in seen] == ["some/model"] * 2 and seen[0]["temperature"] == 0
    assert seen[0]["messages"][1]["content"] == "cached_question: refund status?\nnew_question: refund kab milega"

    again = []
    assert openrouter_verifier("some/model", cache, transport=fake_openrouter(again))(pairs) == [True, False]
    assert again == []


def test_openrouter_verifier_retries_server_errors(monkeypatch):
    from pakka.verify import openrouter_verifier

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = []
    assert openrouter_verifier("m", transport=fake_openrouter(seen, busy=1))([("a b", "a c")]) == [True]
    assert len(seen) == 2


def test_load_verifier_jev_says_yes_at_or_above_min_score(monkeypatch):
    from pakka.verify import load_verifier

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen = []
    verify = load_verifier("jev", min_score=0.5, transport=fake_jev(seen))
    assert verify("refund kab milega", "refund status?", "5-7 days") is True
    assert verify("cancel my order", "track my order", "Use the app") is False
    assert seen[0]["model"].startswith("jev-")


def test_load_verifier_openrouter_tries_once_so_a_busy_api_is_an_error(monkeypatch):
    from pakka.verify import load_verifier

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen = []
    verify = load_verifier("openrouter:some/model", transport=fake_openrouter(seen, busy=1))
    with pytest.raises(httpx.HTTPStatusError):
        verify("a b", "a c", "answer")
    assert len(seen) == 1  # no retry: a live lookup can't wait
    assert verify("a b", "a c", "answer") is True


def test_load_verifier_needs_its_key(monkeypatch):
    from pakka.verify import load_verifier

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        load_verifier("jev")


def test_load_verifier_rejects_unknown_names():
    from pakka.verify import load_verifier

    with pytest.raises(ValueError):
        load_verifier("gpt")
