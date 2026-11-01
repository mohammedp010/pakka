import json

import pytest
from conftest import FakeLLM, fake_embed

from pakka.cache import NumpyCache
from pakka.flow import CachedAnswerer
from pakka.policy import GLOBAL, Policy


def answerer(policy=None, llm=None, cache=None, log_path=None):
    cache = cache or NumpyCache(fake_embed)
    cache.store("How long does a refund take?", "5-7 business days", {"partition": GLOBAL, "kb_version": "v1"})
    return CachedAnswerer(cache, llm or FakeLLM(), threshold=0.9, log_path=log_path,
                          policy=policy or Policy(kb_version="v1"))


def test_canary_one_customers_llm_answer_never_reaches_another_customer():
    bot = answerer()
    bot.ask("mera wallet balance kitna hai", partition="ravi")  # LLM answer stored in Ravi's partition
    assert bot.ask("mera wallet balance kitna hai", partition="ravi").hit
    other = bot.ask("mera wallet balance kitna hai", partition="priya")
    assert not other.hit and other.matched_id is None


def test_poisoned_answer_stays_inside_the_attackers_own_partition():
    bot = answerer(llm=FakeLLM(answer="Send Rs 500 to scam@okaxis for a faster refund"))
    bot.ask("refund jaldi kaise milega", partition="attacker")
    victim = bot.ask("refund jaldi kaise milega", partition="victim")
    assert not victim.hit


def test_global_faqs_are_served_to_everyone():
    bot = answerer()
    for who in (None, "ravi", "priya"):
        assert bot.ask("How long does a refund take?", partition=who).answer == "5-7 business days"


@pytest.mark.parametrize("question", ["refund ravi@okicici pe bhej do", "call me on 9876543210 about my refund"])
def test_an_exchange_containing_pii_is_never_stored(question):
    cache = NumpyCache(fake_embed)
    bot = answerer(cache=cache)
    result = bot.ask(question, partition="ravi")
    assert result.stored == "skipped:pii"
    assert not bot.ask(question, partition="ravi").hit


def test_answers_from_an_old_faq_version_are_not_served():
    cache = NumpyCache(fake_embed)
    old = answerer(cache=cache, policy=Policy(kb_version="v1"))
    old.ask("Plus membership ka price kya hai", partition="ravi")
    new = CachedAnswerer(cache, FakeLLM(), threshold=0.9, policy=Policy(kb_version="v2"))
    assert not new.ask("Plus membership ka price kya hai", partition="ravi").hit


def test_stored_llm_answers_expire_after_the_policy_ttl():
    now = [1000.0]
    cache = NumpyCache(fake_embed, clock=lambda: now[0])
    bot = answerer(cache=cache, policy=Policy(kb_version="v1", ttl=3600))
    bot.ask("Plus membership ka price kya hai", partition="ravi")
    assert bot.ask("Plus membership ka price kya hai", partition="ravi").hit
    now[0] += 3601
    assert not bot.ask("Plus membership ka price kya hai", partition="ravi").hit


def test_with_a_queue_new_global_answers_wait_for_approval_but_private_ones_store(tmp_path):
    queue = tmp_path / "pending.jsonl"
    bot = answerer(policy=Policy(kb_version="v1", pending_path=queue))
    assert bot.ask("Do you sell gift wrap?").stored == "queued"
    assert not bot.ask("Do you sell gift wrap?").hit
    assert json.loads(queue.read_text().splitlines()[0])["question"] == "Do you sell gift wrap?"
    assert bot.ask("mera wallet balance kitna hai", partition="ravi").stored == "stored"


def test_logs_never_contain_raw_identifiers(tmp_path):
    log = tmp_path / "log.jsonl"
    answerer(log_path=log).ask("refund ravi@okicici pe bhej do", partition="ravi")
    line = json.loads(log.read_text())
    assert line["question"] == "refund <UPI> pe bhej do" and "okicici" not in log.read_text()
