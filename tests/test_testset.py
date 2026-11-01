import json

import pytest
from conftest import FAQS

from pakka.testset import assign_splits, generate_for_faq, judge, parse_json, reconcile


class ScriptedChat:
    """Returns canned replies in order and records every prompt."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0), 100


def test_parse_json_accepts_code_fences_and_chatter():
    assert parse_json('Sure!\n```json\n{"a": [1]}\n```') == {"a": [1]}


def test_generator_labels_paraphrases_with_their_faq_and_keeps_valid_traps_only():
    reply = json.dumps({
        "paraphrases": [{"q": "refund kab tak aayega", "lang": "hinglish"}, {"q": " ", "lang": "en"}],
        "hard_negatives": [
            {"q": "refund nahi chahiye, exchange do", "lang": "hinglish", "kind": "negation", "why": "x"},
            {"q": "what is love", "lang": "en", "kind": "made_up_kind"},
        ],
    })
    chat = ScriptedChat(reply)
    items = generate_for_faq(FAQS[0], FAQS, chat)
    assert FAQS[0]["answer"] in chat.prompts[0]
    assert items == [
        {"q": "refund kab tak aayega", "lang": "hinglish", "kind": "paraphrase", "near": "faq_01", "intended": "faq_01"},
        {"q": "refund nahi chahiye, exchange do", "lang": "hinglish", "kind": "negation", "near": "faq_01", "intended": None},
    ]


def test_judge_labels_in_batches_and_maps_none():
    items = [{"q": f"q{i}"} for i in range(30)]
    chat = ScriptedChat(json.dumps({"labels": ["faq_02"] * 25}), json.dumps({"labels": ["none"] * 4 + ["faq_99"]}))
    labels = judge(items, FAQS, chat, batch=25)
    assert len(chat.prompts) == 2
    assert labels == ["faq_02"] * 25 + [None] * 5  # an unknown id counts as none


def test_judge_rejects_a_reply_with_the_wrong_number_of_labels():
    with pytest.raises(ValueError):
        judge([{"q": "a"}, {"q": "b"}], FAQS, ScriptedChat(json.dumps({"labels": ["none"]})))


@pytest.mark.parametrize("intended, judged, label, review", [
    ("faq_01", "faq_01", "faq_01", False),  # both models agree on the paraphrase
    ("faq_01", "faq_02", "faq_02", True),   # judge thinks another FAQ fits
    (None, None, None, False),              # both agree it should miss
    (None, "faq_01", None, True),           # judge thinks the trap is answered by its own FAQ
    (None, "faq_07", "faq_07", True),       # trap is really a paraphrase of another FAQ
])
def test_reconcile_flags_every_disagreement_for_review(intended, judged, label, review):
    item = reconcile({"q": "x", "near": "faq_01", "intended": intended}, judged)
    assert (item["label"], item["needs_review"], item["judged"]) == (label, review, judged)


def test_splits_are_deterministic_and_balanced_within_each_group():
    items = [{"q": f"q{i}", "kind": k, "lang": "en"} for i, k in enumerate(["paraphrase"] * 10 + ["negation"] * 4)]
    first, second = assign_splits(items, seed=1), assign_splits(items, seed=1)
    assert [i["split"] for i in first] == [i["split"] for i in second]
    for kind, size in [("paraphrase", 10), ("negation", 4)]:
        assert sum(i["split"] == "test" for i in first if i["kind"] == kind) == size // 2


def test_other_questions_come_one_per_line_so_quotes_cannot_break_parsing():
    from pakka.testset import generate_other

    chat = ScriptedChat('Here you go:\noff_topic | en | Who won the "IPL" final?\n'
                        "uncovered|hinglish| loyalty points kaise milte hain\n"
                        "garbage line\nweird | en | not a real kind\n")
    assert generate_other(FAQS, chat, n=4) == [
        {"q": 'Who won the "IPL" final?', "lang": "en", "kind": "off_topic", "near": None, "intended": None},
        {"q": "loyalty points kaise milte hain", "lang": "hinglish", "kind": "uncovered", "near": None, "intended": None},
    ]
