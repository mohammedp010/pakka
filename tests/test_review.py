import pytest

from pakka.review import DROP, decide

IDS = {f"faq_{i:02d}" for i in range(1, 31)}
ITEM = {"q": "warranty nahi chahiye, return kar sakta hoon?", "intended": None, "judged": "faq_03",
        "label": None, "needs_review": True}


@pytest.mark.parametrize("answer, label", [("w", None), ("j", "faq_03"), ("n", None), ("17", "faq_17"), ("3", "faq_03")])
def test_each_answer_sets_the_label_and_clears_the_review_flag(answer, label):
    item = decide(ITEM, answer, IDS)
    assert item["label"] == label and item["needs_review"] is False and item["reviewed"] is True


def test_d_drops_a_bad_question():
    assert decide(ITEM, "d", IDS) is DROP


@pytest.mark.parametrize("answer", ["", "x", "31", "0"])
def test_unknown_answers_are_rejected_so_the_prompt_repeats(answer):
    with pytest.raises(ValueError):
        decide(ITEM, answer, IDS)
