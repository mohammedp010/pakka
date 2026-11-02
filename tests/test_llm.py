import json

import httpx
import pytest
from conftest import FAQS

from examples.dukaan_bot.llm import make_llm


def fake_openrouter(content, finish_reason="stop"):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"total_tokens": 120},
        })

    return httpx.MockTransport(handler), sent


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key-for-fake-transport")


def test_returns_text_and_tokens_with_reasoning_off():
    transport, sent = fake_openrouter(" Refund 5-7 din mein aata hai. ")
    assert make_llm(FAQS, transport=transport)("refund kab milega?") == ("Refund 5-7 din mein aata hai.", 120)
    assert sent[0]["reasoning"] == {"enabled": False}
    assert FAQS[0]["answer"] in sent[0]["messages"][0]["content"]


def test_empty_reply_raises_a_clear_error():
    transport, _ = fake_openrouter(None, finish_reason="length")
    with pytest.raises(RuntimeError, match="length"):
        make_llm(FAQS, transport=transport)("q")


def test_chat_can_ask_for_some_reasoning():
    from examples.dukaan_bot.llm import make_chat

    transport, sent = fake_openrouter("ok")
    make_chat("openai/gpt-5.4", transport=transport, reasoning={"effort": "low"})("system", "user")
    assert sent[0]["reasoning"] == {"effort": "low"}
