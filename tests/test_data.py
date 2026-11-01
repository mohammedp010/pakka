from collections import Counter

from conftest import FAQS, PILOT

FAQ_IDS = {f["id"] for f in FAQS}


def test_30_faqs_with_unique_ids_and_text():
    assert len(FAQS) == 30 and len(FAQ_IDS) == 30
    assert all(f["question"].strip() and f["answer"].strip() for f in FAQS)


def test_pilot_has_60_paraphrases_and_40_should_miss():
    assert len(PILOT) == 100
    assert sum(r["label"] is not None for r in PILOT) == 60
    assert sum(r["label"] is None for r in PILOT) == 40


def test_labels_point_at_real_faqs_and_cover_every_faq():
    counts = Counter(r["label"] for r in PILOT if r["label"])
    assert set(counts) == FAQ_IDS
    assert min(counts.values()) >= 2


def test_kinds_langs_and_no_duplicates():
    assert {r["kind"] for r in PILOT} <= {"paraphrase", "negation", "personal", "near_intent", "entity", "off_topic"}
    assert {r["lang"] for r in PILOT} <= {"en", "hinglish", "hi"}
    assert all((r["kind"] == "paraphrase") == (r["label"] is not None) for r in PILOT)
    assert len({r["q"].lower() for r in PILOT}) == 100
