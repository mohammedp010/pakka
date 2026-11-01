import pytest
from conftest import FAQS, FakeLLM, fake_embed
from fastapi.testclient import TestClient

from examples.dukaan_bot.app import create_app


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(tmp_path, llm):
    return TestClient(create_app(embed=fake_embed, llm=llm, threshold=0.9, log_path=tmp_path / "log.jsonl"))


def test_faq_question_is_answered_from_the_seeded_cache(client, llm):
    reply = client.post("/chat", json={"question": FAQS[0]["question"]}).json()
    assert reply["hit"] and reply["matched_id"] == "faq_01"
    assert reply["answer"] == FAQS[0]["answer"] and llm.calls == []


def test_unknown_question_goes_to_the_llm(client, llm):
    reply = client.post("/chat", json={"question": "Tell me a joke"}).json()
    assert not reply["hit"] and reply["tokens"] == 42 and llm.calls == ["Tell me a joke"]


@pytest.mark.parametrize("question", ["", "x" * 1001])
def test_empty_or_huge_questions_are_rejected(client, llm, question):
    assert client.post("/chat", json={"question": question}).status_code == 422
    assert llm.calls == []


def test_bot_guards_block_a_negated_faq_question_by_default(client, llm):
    reply = client.post("/chat", json={"question": FAQS[0]["question"] + " nahi"}).json()
    assert not reply["hit"] and reply["reason"] == "blocked:negation" and len(llm.calls) == 1
