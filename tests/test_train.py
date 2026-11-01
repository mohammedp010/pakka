from pakka.train import training_triplets

FAQS = [{"id": "faq_01", "question": "How long does a refund take?"},
        {"id": "faq_02", "question": "Is Cash on Delivery available?"}]
ROWS = [
    {"q": "refund kab milega", "label": "faq_01"},
    {"q": "paisa wapas kab aayega", "label": "faq_01"},
    {"q": "refund nahi chahiye", "label": None, "near": "faq_01"},
    {"q": "refund in dollars?", "label": None, "near": "faq_01"},
    {"q": "COD hai kya", "label": "faq_02"},
    {"q": "IPL kisne jeeta", "label": None, "near": None},
]


def test_each_paraphrase_becomes_faq_question_paraphrase_and_a_trap_for_that_faq():
    assert training_triplets(ROWS, FAQS) == [
        ("How long does a refund take?", "refund kab milega", "refund nahi chahiye"),
        ("How long does a refund take?", "paisa wapas kab aayega", "refund in dollars?"),
        ("Is Cash on Delivery available?", "COD hai kya", "IPL kisne jeeta"),  # no faq_02 trap: any should-miss
    ]
