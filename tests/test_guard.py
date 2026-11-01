import pytest

from pakka.guard import check


@pytest.mark.parametrize("query, cached", [
    ("refund nahi mila", "refund kab milega?"),
    ("refund nhi aaya abhi tak", "How long does a refund take?"),
    ("payment kar diya par paisa nahin kata", "My payment failed but money was deducted. What now?"),
    ("I don't want to delete my account, how do I just deactivate it?", "How do I delete my account?"),
    ("Can I track my order without logging in?", "How can I track my order?"),
    ("मेरा रिफंड नहीं आया", "How long does a refund take?"),
    ("COD mat bhejo", "Is Cash on Delivery available?"),
])
def test_negation_on_one_side_only_blocks(query, cached):
    assert check(query, cached) == "negation"


@pytest.mark.parametrize("query, cached", [
    ("coupon apply kyun nahi ho raha?", "Why is my coupon not working?"),  # both negated
    ("refund abhi tak nahi aaya", "I haven't received my refund yet. What should I do?"),
    ("मेरा पैसा वापस नहीं आया", "I haven't received my refund yet. What should I do?"),
    ("refund kab milega?", "How long does a refund take?"),  # neither negated
    ("मतलब refund कब मिलेगा", "How long does a refund take?"),  # "मत" inside a longer word
])
def test_matching_polarity_passes(query, cached):
    assert check(query, cached) is None


@pytest.mark.parametrize("query, cached", [
    ("Order 4521 kab aayega?", "How long does delivery take?"),
    ("Where is my order #OD88213?", "How can I track my order?"),
    ("refund ravi@okicici pe bhej do", "How long does a refund take?"),
    ("Is COD available above Rs 60000?", "Is Cash on Delivery available?"),
])
def test_numbers_ids_and_upi_handles_the_cached_question_lacks_block(query, cached):
    assert check(query, cached) == "slot"


def test_the_same_number_on_both_sides_passes():
    assert check("Is delivery free above 499?", "Is delivery free above 499?") is None


@pytest.mark.parametrize("query, cached", [
    ("card unblock kaise kare", "How do I block my card?"),
    ("How do I deactivate my account?", "How do I delete my account?"),
    ("newsletter unsubscribe karna hai", "How do I subscribe to the newsletter?"),
])
def test_opposite_actions_block(query, cached):
    assert check(query, cached) == "action"


def test_identical_questions_pass():
    assert check("How do I cancel my order?", "How do I cancel my order?") is None


# --- refinements from the first bench run

@pytest.mark.parametrize("query, cached", [
    ("can u ship to my pin code or not", "Do you deliver to my pincode?"),
    ("card details save karna secure h ya nhi", "Is it safe to save my card details?"),
    ("मेरे पिन कोड में डिलीवरी उपलब्ध है या नहीं", "Do you deliver to my pincode?"),
    ("cash on delivery nhi h kya aapke yahan", "Is Cash on Delivery available?"),
    ("what products are non returnable", "Which items cannot be returned?"),
])
def test_yes_no_tags_and_non_words_do_not_count_as_a_flip(query, cached):
    assert check(query, cached) is None


def test_a_number_the_cached_answer_mentions_is_covered():
    answer = "Delivery is free on orders above Rs 499. Smaller orders have a Rs 40 delivery fee."
    assert check("Will I get free delivery if I order above 499?", "Is delivery free?", answer) is None
    assert check("free delivery above 999?", "Is delivery free?", answer) == "slot"


@pytest.mark.parametrize("query, cached", [
    ("Is Cash on Delivery available in Nepal?", "Is Cash on Delivery available?"),
    ("Is delivery free outside India?", "Is delivery free?"),
    ("how do I cancel my Flipkart Plus card?", "How do I cancel my Plus membership?"),
    ("Can I change the delivery address to a different country?", "Can I change my delivery address after placing an order?"),
])
def test_names_and_scope_words_the_cached_faq_never_mentions_block(query, cached):
    assert check(query, cached) == "entity"


@pytest.mark.parametrize("query, cached, answer", [
    ("kya EMI pe le sakte hain?", "Do you offer EMI?", "No-cost EMI is available on orders above Rs 3,000."),
    ("What do I get with Plus membership?", "What is Plus membership?", ""),
    ("Do you deliver all over India?", "Do you deliver to my pincode?", "We deliver to most pincodes in India."),
])
def test_names_the_cached_faq_does_mention_pass(query, cached, answer):
    assert check(query, cached, answer) is None
