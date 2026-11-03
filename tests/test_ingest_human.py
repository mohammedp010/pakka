"""Phase 8: people's Google Forms answers become a labelled, human-written test split."""

import csv
import json
import re

import pytest
from conftest import FAQS, ROOT

from pakka.cache import normalize
from scripts.ingest_human import JUDGES, clean, lang_of, load_form, main, questions, read_responses, reconcile2, row, split_cell

FORM = load_form()
A = FORM["variants"]["A"]
CONSENT = FORM["consent"]


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)


class FakeJudge:
    """Labels each numbered message by the first keyword it contains, else "none", and counts calls."""

    def __init__(self, rules):
        self.rules, self.calls = rules, 0

    def __call__(self, system, user):
        self.calls += 1
        messages = re.findall(r"^\d+\. (.*)$", user.split("Messages:")[1], re.M)
        labels = [next((faq for word, faq in self.rules if word in m.lower()), "none") for m in messages]
        return json.dumps({"labels": labels}), 50


def test_form_asks_about_every_faq_once_in_three_variants_of_ten():
    situations = [q for v in FORM["variants"].values() for q in v["situations"]]
    assert sorted(q["faq"] for q in situations) == sorted(f["id"] for f in FAQS)
    assert all(q["kind"] == "paraphrase" for q in situations)
    for v in FORM["variants"].values():
        assert len(v["situations"]) == 10
        assert sorted(t["kind"] for t in v["twists"]) == ["entity", "negation", "number"]
        assert {t["faq"] for t in v["twists"]} <= {s["faq"] for s in v["situations"]}  # twists start from what they just saw
    assert FORM["other"] == {**FORM["other"], "faq": None, "kind": "uncovered"}


def test_form_titles_are_unique_and_never_quote_an_faq_question():
    titles = [q["title"] for q in questions(FORM)]
    assert len(titles) == len(set(titles))  # the CSV columns map back by title
    assert not [f["id"] for f in FAQS for t in titles if normalize(f["question"]) in normalize(t)]


def test_each_line_of_an_answer_is_one_message_without_list_markers():
    cell = '1. refund kab milega?\n\n- How many days for a refund\n  "2 din mein aayega?"  \n• ok'
    assert split_cell(cell) == ["refund kab milega?", "How many days for a refund", "2 din mein aayega?", "ok"]
    assert split_cell("") == []


@pytest.mark.parametrize("cell", ["NA", "n/a", "Nil.", "-", "none", "nothing"])
def test_placeholder_answers_are_not_messages(cell):
    assert split_cell(cell) == []


@pytest.mark.parametrize("q, lang", [
    ("How long will the refund take?", "en"),
    ("Do you sell Bata shoes?", "en"),
    ("refund kab milega", "hinglish"),
    ("COD available hai?", "hinglish"),
    ("Mujhe exchange chaiye", "hinglish"),
    ("रिफंड कब मिलेगा?", "hi"),
    ("मेरा refund कब आएगा", "hi"),
])
def test_language_is_tagged_from_script_and_common_hindi_words(q, lang):
    assert lang_of(q) == lang


def test_only_consented_rows_are_read_and_each_message_keeps_the_question_it_answered(tmp_path):
    situation, twist = A["situations"][0], A["twists"][0]
    header = ["Timestamp", CONSENT["title"], situation["title"], twist["title"], FORM["other"]["title"]]
    write_csv(tmp_path / "a.csv", header, [
        ["t1", CONSENT["choice"], "refund kab milega?\nHow many days for refund?", "", "Do you sell Bata shoes?"],
        ["t2", "", "no consent, never read", "", ""],
        ["t3", CONSENT["choice"], "- refund kitne din me aata hai", "", ""],
    ])
    items = read_responses([tmp_path / "a.csv"], FORM)
    faq = situation["faq"]
    assert items == [
        {"q": "refund kab milega?", "kind": "paraphrase", "near": faq, "intended": faq, "writer": 1},
        {"q": "How many days for refund?", "kind": "paraphrase", "near": faq, "intended": faq, "writer": 1},
        {"q": "Do you sell Bata shoes?", "kind": "uncovered", "near": None, "intended": None, "writer": 1},
        {"q": "refund kitne din me aata hai", "kind": "paraphrase", "near": faq, "intended": faq, "writer": 2},
    ]


def test_a_twist_column_gives_a_should_miss_intent_near_its_faq(tmp_path):
    twist = A["twists"][1]
    write_csv(tmp_path / "a.csv", ["Timestamp", CONSENT["title"], twist["title"]], [["t1", CONSENT["choice"], "x nahi chahiye"]])
    assert read_responses([tmp_path / "a.csv"], FORM) == [
        {"q": "x nahi chahiye", "kind": twist["kind"], "near": twist["faq"], "intended": None, "writer": 1}]


def test_a_column_the_form_never_had_stops_the_ingest(tmp_path):
    write_csv(tmp_path / "a.csv", ["Timestamp", "Email Address", CONSENT["title"]], [["t1", "x@y.com", CONSENT["choice"]]])
    with pytest.raises(ValueError, match="Email Address"):
        read_responses([tmp_path / "a.csv"], FORM)


def test_clean_drops_personal_data_and_repeats_and_tags_language():
    item = {"kind": "paraphrase", "near": "faq_01", "intended": "faq_01", "writer": 1}
    items = [{**item, "q": "Refund kab milega?"}, {**item, "q": "refund kab milega"},
             {**item, "q": "refund nahi aaya, call me on 9876543210"}, {**item, "q": "order 45678912 ka refund?"}]
    kept, dropped = clean(items)
    assert kept == [{**item, "q": "Refund kab milega?", "lang": "hinglish"}] and dropped == 2


@pytest.mark.parametrize("intended, near, a, b, label, review", [
    ("faq_01", "faq_01", "faq_01", "faq_01", "faq_01", False),  # everyone agrees
    ("faq_01", "faq_01", "faq_01", "faq_02", "faq_01", True),   # the judges split
    ("faq_01", "faq_01", None, None, None, True),               # judges agree, but not with what the form asked for
    (None, "faq_17", None, None, None, False),                  # a twist both judges call a miss
    (None, "faq_17", "faq_17", "faq_17", None, True),           # the twist didn't change the answer after all
    (None, None, "faq_05", "faq_05", "faq_05", True),           # "anything else" turned out to be an FAQ
])
def test_two_blind_judges_and_the_form_settle_a_label_or_flag_it(intended, near, a, b, label, review):
    out = reconcile2({"q": "x", "intended": intended, "near": near}, a, b)
    assert (out["label"], out["needs_review"], out["judged"], out["judged2"]) == (label, review, a, b)


def test_one_writer_in_five_goes_to_dev_and_every_row_is_marked_human():
    item = {"q": "x", "label": None, "kind": "uncovered", "near": None, "lang": "en", "intended": None,
            "judged": None, "judged2": None, "needs_review": False}
    assert [row({**item, "writer": w})["split"] for w in range(1, 11)] == ["test"] * 4 + ["dev"] + ["test"] * 4 + ["dev"]
    assert row({**item, "writer": 1})["source"] == "human"


def test_main_labels_writes_jsonl_and_a_rerun_asks_no_llm_again(tmp_path, capsys):
    situation = A["situations"][0]  # faq_01: how long a refund takes
    write_csv(tmp_path / "a.csv", ["Timestamp", CONSENT["title"], situation["title"], FORM["other"]["title"]], [
        ["t1", CONSENT["choice"], "refund kab milega?\nrefund in how many days", "cricket score kya hai"],
        ["t2", CONSENT["choice"], "mera refund kab aayega, 9876543210 pe call karo", ""],
    ])
    deepseek, haiku = FakeJudge([("refund", "faq_01")]), FakeJudge([("days", "faq_02"), ("refund", "faq_01")])
    out = tmp_path / "bench_human.jsonl"
    args = ["--private", str(tmp_path), "--out", str(out)]
    main(args, chats=dict(zip(JUDGES, [deepseek, haiku])))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["q"], r["label"], r["needs_review"]) for r in rows] == [
        ("refund kab milega?", "faq_01", False),
        ("refund in how many days", "faq_01", True),  # haiku says faq_02
        ("cricket score kya hai", None, False),
    ]
    assert {r["source"] for r in rows} == {"human"} and "pakka review" in capsys.readouterr().out
    calls = deepseek.calls + haiku.calls
    main(args + ["--force"], chats=dict(zip(JUDGES, [deepseek, haiku])))
    assert deepseek.calls + haiku.calls == calls  # every judge call came from the cache


def test_main_refuses_raw_answers_inside_the_repo_and_keeps_reviewed_labels(tmp_path):
    with pytest.raises(SystemExit, match="outside the repo"):
        main(["--private", str(ROOT / "data")])
    out = tmp_path / "bench_human.jsonl"
    out.write_text('{"q": "x", "reviewed": true}\n')
    with pytest.raises(SystemExit, match="reviewed"):
        main(["--private", str(tmp_path), "--out", str(out)])
