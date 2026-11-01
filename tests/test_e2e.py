"""Real embedder + real LLM through OpenRouter. Costs about $0.001. Run with: pytest -m e2e"""

import pytest

from examples.dukaan_bot.llm import api_key

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not api_key(), reason="OPENROUTER_API_KEY not set (env or pakka/.env)"),
]


def test_real_bot_misses_once_then_serves_the_repeat_from_cache(tmp_path):
    from examples.dukaan_bot.app import build

    answerer = build(log_path=tmp_path / "log.jsonl", pending_path=tmp_path / "pending.jsonl",
                     approved_path=tmp_path / "approved.jsonl")
    question = "Do you deliver on Sundays?"
    first = answerer.ask(question, partition="e2e-user")  # a named user's answers are cached for them
    second = answerer.ask(question, partition="e2e-user")
    assert not first.hit and first.tokens > 0 and first.answer and first.stored == "stored"
    assert second.hit and second.tokens == 0 and second.answer == first.answer
    assert second.latency_ms < first.latency_ms
