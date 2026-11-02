"""Adapters that turn public datasets into pakka's formats, checked on tiny hand-made fixtures."""

import re

from scripts.fetch_public import (consonants, from_banking77, from_bitext, from_clinc150, from_hinglish_top,
                                  from_langcache, from_massive, from_stream, jsonl, massive_meaning,
                                  massive_translations, stratified, top_meaning)


def by_q(rows):
    return {r["q"]: r for r in rows}


def test_banking77_seeds_one_faq_per_intent_and_the_rest_should_hit_it():
    names = ["card_arrival", "lost_card"]
    train = [{"text": "Where is my card?", "label": 0}, {"text": "I lost my card", "label": 1},
             {"text": "card still not here", "label": 0}]
    test = [{"text": "when does my card arrive", "label": 0}]
    faqs, rows = from_banking77({"train": train, "test": test}, names)
    assert faqs == [{"id": "card_arrival", "question": "Where is my card?", "answer": "card_arrival"},
                    {"id": "lost_card", "question": "I lost my card", "answer": "lost_card"}]
    assert by_q(rows) == {
        "card still not here": {"q": "card still not here", "label": "card_arrival", "kind": "paraphrase",
                                "lang": "en", "split": "dev", "source": "banking77"},
        "when does my card arrive": {"q": "when does my card arrive", "label": "card_arrival", "kind": "paraphrase",
                                     "lang": "en", "split": "test", "source": "banking77"}}


def test_clinc150_out_of_scope_questions_must_miss_and_never_become_faqs():
    names = ["oos", "balance"]
    splits = {"train": [{"text": "how do i bake bread", "intent": 0}, {"text": "what's my balance", "intent": 1}],
              "validation": [{"text": "tell me a joke", "intent": 0}, {"text": "balance please", "intent": 1}],
              "test": [{"text": "how much money do i have", "intent": 1}]}
    faqs, rows = from_clinc150(splits, names)
    assert [f["id"] for f in faqs] == ["balance"]
    rows = by_q(rows)
    assert (rows["tell me a joke"]["label"], rows["tell me a joke"]["kind"]) == (None, "out_of_scope")
    assert (rows["balance please"]["label"], rows["balance please"]["split"]) == ("balance", "dev")
    assert rows["how much money do i have"]["split"] == "test"
    assert "how do i bake bread" not in rows  # out-of-scope training rows aren't used


def test_bitext_serves_the_seed_response_and_splits_the_rest_stably():
    data = [{"instruction": f"cancel order {i}", "intent": "cancel_order", "response": f"Sure {i}"} for i in range(40)]
    faqs, rows = from_bitext(data)
    assert faqs == [{"id": "cancel_order", "question": "cancel order 0", "answer": "Sure 0"}]
    assert len(rows) == 39 and {r["split"] for r in rows} == {"dev", "test"}
    assert from_bitext(data)[1] == rows  # same split every run


def test_a_row_identical_to_an_faq_question_is_dropped_and_duplicates_kept_once():
    train = [{"text": "Where is my card?", "label": 0}, {"text": "where is my card", "label": 0},
             {"text": "card not here", "label": 0}, {"text": "Card not here!", "label": 0}]
    _, rows = from_banking77({"train": train, "test": []}, ["card_arrival"])
    assert [r["q"] for r in rows] == ["card not here"]


def test_top_meaning_keeps_intent_and_slot_values_but_not_filler_words():
    a = top_meaning("[IN:CREATE_ALARM set an alarm [SL:DATE_TIME at 6 am ] ]")
    b = top_meaning("[IN:CREATE_ALARM create alarm please [SL:DATE_TIME at 6am ] ]")
    c = top_meaning("[IN:CREATE_ALARM set an alarm [SL:DATE_TIME at 7 am ] ]")
    assert a == b and a != c
    assert top_meaning("[IN:CREATE_ALARM x [SL:DATE_TIME at 6 am ] ]", values=False) == \
        top_meaning("[IN:CREATE_ALARM y [SL:DATE_TIME at 7 am ] ]", values=False)
    nested = "[IN:GET_EVENT [SL:LOCATION [IN:GET_LOCATION [SL:CATEGORY_LOCATION Library ] ] ] events ]"
    assert top_meaning(nested) != top_meaning(nested.replace("Library", "Museum"))


def test_massive_meaning_reads_slots_from_the_annotated_utterance():
    assert massive_meaning(3, "wake me up at [time : five am] [date : this week]") == \
        massive_meaning(3, "please wake me [date : This week] at [time : five am]")
    assert massive_meaning(3, "wake me at [time : six am]", values=False) == \
        massive_meaning(3, "wake me at [time : five am]", values=False)


HTOP = {
    "train": [{"en_query": "set an alarm at 6 am", "cs_query": "6 am ka alarm lagao",
               "en_parse": "[IN:CREATE_ALARM set an alarm [SL:DATE_TIME at 6 am ] ]"}],
    "validation": [{"en_query": "create alarm at 6 am", "cs_query": "alarm lagao 6 am pe",
                    "en_parse": "[IN:CREATE_ALARM create alarm [SL:DATE_TIME at 6 am ] ]"},
                   {"en_query": "set an alarm at 7 am", "cs_query": "7 am ka alarm lagao",
                    "en_parse": "[IN:CREATE_ALARM set an alarm [SL:DATE_TIME at 7 am ] ]"}],
    "test": [{"en_query": "pause my timer", "cs_query": "mera timer roko",
              "en_parse": "[IN:PAUSE_TIMER pause my [SL:METHOD_TIMER timer ] ]"}],
}


def test_hinglish_top_same_parse_hits_and_same_intent_with_other_slots_is_a_trap():
    faqs, rows = from_hinglish_top(HTOP)
    assert len(faqs) == 1 and faqs[0]["question"] == "set an alarm at 6 am" and faqs[0]["answer"] == "CREATE_ALARM"
    alarm = faqs[0]["id"]
    rows = by_q(rows)
    assert (rows["6 am ka alarm lagao"]["label"], rows["6 am ka alarm lagao"]["lang"]) == (alarm, "hinglish")
    assert (rows["create alarm at 6 am"]["label"], rows["create alarm at 6 am"]["split"]) == (alarm, "dev")
    assert rows["alarm lagao 6 am pe"]["label"] == alarm
    for q in ("set an alarm at 7 am", "7 am ka alarm lagao"):
        assert (rows[q]["label"], rows[q]["kind"]) == (None, "slot")
    assert (rows["pause my timer"]["label"], rows["pause my timer"]["kind"], rows["pause my timer"]["split"]) == \
        (None, "uncovered", "test")


def test_massive_hindi_rows_with_localized_slots_are_traps_not_hits():
    def row(i, part, utt, annot, method=None):
        return {"id": i, "partition": part, "intent": 7, "utt": utt, "annot_utt": annot,
                "slot_method": {"slot": ["place_name"] if method else [], "method": [method] if method else []}}

    en = [row("1", "train", "weather in boston", "weather in [place_name : boston]"),
          row("2", "dev", "boston weather", "[place_name : boston] weather")]
    hi = [row("1", "train", "बोस्टन में मौसम", "[place_name : बोस्टन] में मौसम", "translation"),
          row("2", "dev", "मुंबई का मौसम", "[place_name : मुंबई] का मौसम", "localization")]
    faqs, rows = from_massive({"en": en, "hi": hi})
    rows = by_q(rows)
    assert rows["बोस्टन में मौसम"]["label"] == faqs[0]["id"] and rows["बोस्टन में मौसम"]["lang"] == "hi"
    assert rows["boston weather"]["label"] == faqs[0]["id"]
    assert (rows["मुंबई का मौसम"]["label"], rows["मुंबई का मौसम"]["kind"]) == (None, "slot")


def test_consonants_match_a_name_written_in_latin_or_devanagari():
    for en, hi in (("boston", "बोस्टन"), ("new delhi", "नई दिल्ली"), ("mumbai", "मुंबई"), ("pizza", "पिज़्ज़ा"),
                   ("mahatma gandhi", "महात्मा गांधी"), ("mary's", "मैरी का"), ("taxi", "टैक्सी")):
        assert consonants(en) == consonants(hi), (en, hi)
    assert consonants("boston") != consonants("मुंबई")


def test_massive_translations_are_used_twice_and_mostly_for_that_value():
    pairs = [("time", "five am", "सुबह पांच बजे")] * 2 + [("time", "six am", "सुबह छह बजे")]  # once is not enough
    pairs += [("person", "john", "राम")] * 2 + [("person", "bob", "राम")] * 2  # राम replaces any name: localized
    english = {str(i): {"annot_utt": f"[{slot} : {x}]"} for i, (slot, x, _) in enumerate(pairs)}
    hindi = [{"id": str(i), "annot_utt": f"[{slot} : {y}]", "slot_method": {"slot": [slot], "method": ["translation"]}}
             for i, (slot, _, y) in enumerate(pairs)]
    assert massive_translations(hindi, english) == {("time", "fiveam", "सुबहपांचबजे")}


def test_massive_localized_slots_that_only_translate_or_transliterate_still_hit():
    def row(i, part, annot, methods=()):
        utt = re.sub(r"\[\w+ : ([^\]]+)\]", r"\1", annot)
        slots = list(dict.fromkeys(re.findall(r"\[(\w+) :", annot)))  # MASSIVE lists each slot once, in English order
        return {"id": i, "partition": part, "intent": 7, "utt": utt, "annot_utt": annot,
                "slot_method": {"slot": slots if methods else [], "method": list(methods)}}

    en = [row("1", "train", "wake me at [time : five am] in [place_name : boston]"),
          row("2", "train", "alarm at [time : five am]"), row("3", "train", "set an alarm for [time : five am]"),
          row("4", "dev", "in [place_name : boston] wake me at [time : five am]"),
          row("5", "dev", "wake me in [place_name : boston] at [time : five am]")]
    hi = [row("1", "train", "[place_name : बोस्टन] में [time : सुबह पांच बजे] जगाओ", ["translation", "translation"]),
          row("2", "train", "[time : सुबह पांच बजे] का अलार्म", ["translation"]),
          row("3", "train", "[time : सुबह पांच बजे] का अलार्म लगाओ", ["translation"]),
          # transliterated and translated, though flagged as localized
          row("4", "dev", "[place_name : बोस्टन] में मुझे [time : सुबह पांच बजे] जगाओ", ["localization"] * 2),
          # the flag sits on the wrong slot: the time is translated, Boston became Mumbai
          row("5", "dev", "[place_name : मुंबई] में मुझे [time : सुबह पांच बजे] उठाओ", ["unchanged", "localization"])]
    faqs, rows = from_massive({"en": en, "hi": hi})
    rows = by_q(rows)
    assert rows["बोस्टन में मुझे सुबह पांच बजे जगाओ"]["label"] == faqs[0]["id"]
    assert (rows["मुंबई में मुझे सुबह पांच बजे उठाओ"]["label"], rows["मुंबई में मुझे सुबह पांच बजे उठाओ"]["kind"]) == \
        (None, "slot")


def test_stream_keeps_arrival_order_and_tunes_on_the_first_fifth():
    rows = from_stream([{"prompt": f"p{i}", "set": i % 3} for i in range(10)], "prompt", "set", "vcache_x")
    assert [r["q"] for r in rows] == [f"p{i}" for i in range(10)]
    assert [r["split"] for r in rows] == ["dev"] * 2 + ["test"] * 8
    assert rows[4] == {"q": "p4", "label": 1, "kind": "stream", "lang": "en", "split": "test", "source": "vcache_x"}


def test_langcache_pairs_keep_their_source_and_split():
    splits = {"validation": [{"sentence1": "a", "sentence2": "b", "label": 1, "source": "mrpc"}],
              "test": [{"sentence1": "c", "sentence2": "d", "label": 0, "source": "paws"}]}
    assert from_langcache(splits) == [
        {"q": "a", "q2": "b", "label": 1, "kind": "pair", "lang": "en", "split": "dev", "source": "mrpc"},
        {"q": "c", "q2": "d", "label": 0, "kind": "pair", "lang": "en", "split": "test", "source": "paws"}]


def test_stratified_split_reuses_only_test_rows_with_every_source_in_the_same_share():
    rows = [{"q": "v", "q2": "w", "label": 1, "source": "mrpc", "split": "dev"}]
    rows += [{"q": f"{s}{i}", "q2": "x", "label": i % 2, "source": s, "split": "test"}
             for s, n in (("qqp", 50), ("paws", 10)) for i in range(n)]
    out = stratified(rows)
    assert len(out) == 60 and "v" not in by_q(out)
    dev = [r for r in out if r["split"] == "dev"]
    assert sorted(r["source"] for r in dev) == ["paws"] * 2 + ["qqp"] * 10  # a fifth of each source
    assert {r["q"]: r["split"] for r in stratified(rows[::-1])} == {r["q"]: r["split"] for r in out}  # order-free


def test_catch_all_intents_only_match_their_own_translation():
    def row(i, part, utt):
        return {"id": i, "partition": part, "intent": 0, "utt": utt, "annot_utt": utt,
                "slot_method": {"slot": [], "method": []}}

    en = [row("1", "train", "check my car is ready"), row("2", "dev", "check my laptop is working")]
    hi = [row("1", "train", "देखो मेरी कार तैयार है"), row("2", "dev", "देखो मेरा लैपटॉप चल रहा है")]
    faqs, rows = from_massive({"en": en, "hi": hi}, names=["general_quirky"])
    rows = by_q(rows)
    assert rows["देखो मेरी कार तैयार है"]["label"] == faqs[0]["id"]
    assert rows["check my laptop is working"]["label"] is None and rows["देखो मेरा लैपटॉप चल रहा है"]["label"] is None
    top = {"train": [{"en_query": "is it quicker by train", "cs_query": "train se jaldi hoga kya",
                      "en_parse": "[IN:UNSUPPORTED_NAVIGATION is it quicker by train ]"}],
           "validation": [{"en_query": "does this road have pot holes", "cs_query": "is road pe gadde hai kya",
                           "en_parse": "[IN:UNSUPPORTED_NAVIGATION does this road have pot holes ]"}], "test": []}
    faqs, rows = from_hinglish_top(top)
    assert {r["q"]: r["label"] for r in rows} == {"train se jaldi hoga kya": faqs[0]["id"],
                                                  "does this road have pot holes": None, "is road pe gadde hai kya": None}


def test_jsonl_keeps_one_row_per_line_even_with_unicode_line_separators():
    import json

    rows = [{"q": "first\u2028line"}, {"q": "next\x85one\u2029"}]
    assert [json.loads(line) for line in jsonl(rows).splitlines()] == rows
