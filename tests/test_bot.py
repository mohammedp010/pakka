import pytest
from conftest import FAQS, FakeLLM, fake_embed
from fastapi.testclient import TestClient

from examples.dukaan_bot.app import create_app


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(tmp_path, llm):
    return TestClient(create_app(embed=fake_embed, llm=llm, threshold=0.9, log_path=tmp_path / "log.jsonl",
                                 pending_path=tmp_path / "pending.jsonl", approved_path=tmp_path / "approved.jsonl"))


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


# --- phase 3: partitions, approval, versions, erasure

import json  # noqa: E402

from examples.dukaan_bot.app import kb_version  # noqa: E402


@pytest.fixture
def paths(tmp_path):
    return {"pending_path": tmp_path / "pending.jsonl", "approved_path": tmp_path / "approved.jsonl"}


def bot(llm, paths, tmp_path):
    return TestClient(create_app(embed=fake_embed, llm=llm, threshold=0.9, log_path=tmp_path / "log.jsonl", **paths))


def test_a_users_answer_is_served_back_to_them_only(llm, paths, tmp_path):
    client = bot(llm, paths, tmp_path)
    ask = lambda who: client.post("/chat", json={"question": "mera wallet balance kitna hai", "user_id": who}).json()
    assert ask("ravi")["stored"] == "stored"
    assert ask("ravi")["hit"]
    assert not ask("priya")["hit"]


def test_anonymous_answers_wait_in_the_queue(llm, paths, tmp_path):
    client = bot(llm, paths, tmp_path)
    assert client.post("/chat", json={"question": "gift wrap milta hai?"}).json()["stored"] == "queued"
    assert not client.post("/chat", json={"question": "gift wrap milta hai?"}).json()["hit"]
    assert json.loads(paths["pending_path"].read_text().splitlines()[0])["question"] == "gift wrap milta hai?"


def test_approved_answers_load_only_for_the_current_faq_version(llm, paths, tmp_path):
    current = kb_version(FAQS)
    paths["approved_path"].write_text(
        json.dumps({"question": "gift wrap milta hai?", "answer": "Yes, Rs 30", "kb_version": current}) + "\n"
        + json.dumps({"question": "old price of Plus", "answer": "Rs 499", "kb_version": "stale"}) + "\n")
    client = bot(llm, paths, tmp_path)
    assert client.post("/chat", json={"question": "gift wrap milta hai?"}).json()["answer"] == "Yes, Rs 30"
    assert not client.post("/chat", json={"question": "old price of Plus"}).json()["hit"]


def test_forget_erases_one_users_entries(llm, paths, tmp_path):
    client = bot(llm, paths, tmp_path)
    client.post("/chat", json={"question": "mera wallet balance kitna hai", "user_id": "ravi"})
    assert client.post("/forget", json={"user_id": "ravi"}).json() == {"deleted": 1}
    assert not client.post("/chat", json={"question": "mera wallet balance kitna hai", "user_id": "ravi"}).json()["hit"]


@pytest.mark.parametrize("user_id", ["", "global", "a b", "x" * 65, "../etc"])
def test_bad_or_reserved_user_ids_are_rejected(llm, paths, tmp_path, user_id):
    assert bot(llm, paths, tmp_path).post("/chat", json={"question": "hi", "user_id": user_id}).status_code == 422


def test_kb_version_changes_when_any_faq_changes():
    edited = [{**FAQS[0], "answer": "Refunds take 3-5 days."}] + FAQS[1:]
    assert kb_version(FAQS) == kb_version(list(FAQS)) != kb_version(edited)


def test_an_approved_answer_keeps_its_faq_id_so_hits_can_be_checked(llm, paths, tmp_path):
    paths["approved_path"].write_text(json.dumps(
        {"question": "refund kitne din me", "answer": FAQS[0]["answer"], "kb_version": kb_version(FAQS), "id": "faq_01"}) + "\n")
    reply = bot(llm, paths, tmp_path).post("/chat", json={"question": "refund kitne din me"}).json()
    assert reply["hit"] and reply["matched_id"] == "faq_01"


def test_threshold_defaults_to_the_one_picked_for_the_embedder(llm, paths, tmp_path, monkeypatch):
    from examples.dukaan_bot.app import build

    monkeypatch.delenv("PAKKA_THRESHOLD", raising=False)
    monkeypatch.delenv("PAKKA_EMBEDDER", raising=False)
    assert build(embed=fake_embed, llm=llm, log_path=tmp_path / "log.jsonl", **paths).threshold == 0.925
    monkeypatch.setenv("PAKKA_EMBEDDER", "paraphrase-multilingual-MiniLM-L12-v2")
    assert build(embed=fake_embed, llm=llm, log_path=tmp_path / "log.jsonl", **paths).threshold == 0.79
    monkeypatch.setenv("PAKKA_THRESHOLD", "0.8")
    assert build(embed=fake_embed, llm=llm, log_path=tmp_path / "log.jsonl", **paths).threshold == 0.8


def test_an_embedder_without_a_picked_threshold_needs_one_set(llm, paths, tmp_path, monkeypatch):
    from examples.dukaan_bot.app import build

    monkeypatch.delenv("PAKKA_THRESHOLD", raising=False)
    monkeypatch.setenv("PAKKA_EMBEDDER", "my/own-model")
    with pytest.raises(SystemExit, match="PAKKA_THRESHOLD"):
        build(embed=fake_embed, llm=llm, log_path=tmp_path / "log.jsonl", **paths)
