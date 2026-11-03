from scripts.phase9_data import sample_pairs
from scripts.phase9_kaggle import QWEN_PREFIX, datasets_for

TRIPLETS = [{"source": s, "anchor": f"{s} faq", "positive": f"{s} para", "negative": f"{s} trap"}
            for s in ("hinglish_top", "massive", "ours")]
PAIRS = [{"source": "langcache", "s1": "a", "s2": "b", "label": 1.0}]


def test_pairs_sample_caps_each_source_and_never_changes():
    rows = [{"sentence1": f"{s} {i}", "sentence2": "x", "label": i % 2, "source": s} for s in "ab" for i in range(10)]
    picked = sample_pairs(rows, per_source=3)
    assert [r["source"] for r in picked] == ["a"] * 3 + ["b"] * 3
    assert picked == sample_pairs(list(reversed(rows)), per_source=3)


def test_leave_one_out_run_trains_on_no_part_of_that_dataset():
    data = datasets_for("no_massive", TRIPLETS, PAIRS)
    assert "massive para" not in data["faq"]["positive"] and len(data["faq"]) == 2
    assert "pairs" not in datasets_for("no_langcache", TRIPLETS, PAIRS)
    assert len(datasets_for("all", TRIPLETS, PAIRS)["faq"]) == 3


def test_qwen_texts_get_the_prefix_pakka_embeds_with():
    from pakka.embed import PREFIXES

    assert QWEN_PREFIX == PREFIXES["qwen3-embedding"]
    data = datasets_for("all", TRIPLETS, PAIRS, QWEN_PREFIX)
    assert data["pairs"]["sentence1"] == [QWEN_PREFIX + "a"] and data["faq"]["anchor"][0].startswith(QWEN_PREFIX)
