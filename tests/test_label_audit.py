"""Label audit: two LLMs re-judge sampled cache hits so label noise gets a number."""

import json
import random

import pytest


from scripts.label_audit import draw, judge, prompt, summarize


class ScriptedChat:
    """Returns canned replies in order and records every prompt."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, system, user):
        self.prompts.append(user)
        return self.replies.pop(0), 100


def item(dataset, wrong, vetoed=None, answer="", pool=None):
    return {"dataset": dataset, "q": "B?", "cached": "A?", "answer": answer, "label_wrong": wrong, "vetoed": vetoed,
            "pool": pool or ("false" if wrong else "true")}


def test_prompt_numbers_the_pairs_shows_a_stored_answer_only_when_there_is_one_and_cuts_long_text():
    text = prompt([item("ours", False, answer="Refunds take 5 days."), {**item("x", True), "q": "w" * 900}])
    assert "1. A: A?\n   B: B?\n   Stored answer for A: Refunds take 5 days." in text
    assert "2. A: A?\n   B: " + "w" * 500 + "…\n" in text and text.count("Stored answer") == 1
    assert '{"1": true or false, ... "2": ...}' in text


def test_judge_reads_one_numbered_verdict_per_pair_and_rejects_a_missing_one():
    chat = ScriptedChat('```json\n{"2": false, "1": true}\n```', json.dumps({"1": True}))
    assert judge([item("x", False), item("x", True)], chat) == [True, False]
    with pytest.raises(ValueError):  # a short reply must fail, so the batch is retried
        judge([item("x", False), item("x", True)], chat)


def test_summary_counts_labels_both_judges_overturn_and_how_often_judges_agree():
    items = [item("d", True), item("d", True), item("d", False), item("d", False), item("d", False, "slot", pool="veto")]
    a = [True, False, False, True, False]  # judge A: "the cached answer serves the new question"
    b = [True, True, False, True, True]
    s = summarize(items, a, b)["d"]
    assert s["false_hits"] == 2 and s["false_hits_overturned"] == 1  # both judges call the first one fine
    assert s["true_hits"] == 2 and s["true_hits_overturned"] == 1  # both judges call the third one wrong
    assert s["agreement"] == 3 / 5
    assert s["vetoed"] == 1 and s["vetoed_right"] == 0  # judges split on the veto, so it isn't counted right


def test_draw_takes_each_pool_once_and_never_repeats_a_hit():
    items = [{"q": str(i), "label_wrong": i < 3, "vetoed": "slot" if i % 2 else None} for i in range(10)]
    picked = draw(items, random.Random(0))
    assert len({p["q"] for p in picked}) == len(picked)  # no hit twice
    assert sorted(p["q"] for p in picked if p["pool"] == "false") == ["0", "1", "2"]
    assert all(not p["label_wrong"] and p["vetoed"] for p in picked if p["pool"] == "veto")


def test_a_third_judge_settles_the_pairs_the_two_split_on():
    items = [item("d", True), item("d", False), item("d", False)]
    a, b = [True, False, True], [False, False, True]  # they split on the first only
    s = summarize(items, a, b, third={0: True})["d"]
    assert s["false_hits_overturned"] == 1  # the third judge sides with "fine": majority overturns the label
    assert s["true_hits_overturned"] == 1  # the second: both say wrong
    s = summarize(items, a, b, third={0: False})["d"]
    assert s["false_hits_overturned"] == 0
