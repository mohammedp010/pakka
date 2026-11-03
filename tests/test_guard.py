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


# --- refinements from public datasets (dev splits only): rules that fired on good hits, and traps they missed

@pytest.mark.parametrize("query, cached", [
    ("I don't know how to download my invoice", "How do I download my invoice?"),
    ("pata nahi order kaise cancel kare", "How do I cancel my order?"),
    ("I can't make a payment, where do I report it?", "How do I report a payment error?"),
    ("My card won't activate", "How do I activate my card?"),
    ("Why isn't my virtual card working?", "virtual card stopped, help"),
    ("I have not been able to use this card", "My card is blocked"),
])
def test_hedges_and_failure_reports_are_not_a_flip(query, cached):
    assert check(query, cached) is None


@pytest.mark.parametrize("query, cached", [
    ("Give me 5 title ideas for my essay", "Give me five title ideas for my essay"),
    ("set an alarm for 8 hours from now", "set an alarm for eight hours from now"),
    ("cost of 2nd class stamp", "cost of second class stamp"),
    ("६ बजे का अलार्म लगाओ", "set an alarm at 6"),
])
def test_a_number_written_as_a_word_or_in_devanagari_still_counts(query, cached):
    assert check(query, cached) is None


def test_a_number_word_in_the_query_is_a_slot_too():
    assert check("wake me up at seven am", "wake me up at six am") == "slot"


@pytest.mark.parametrize("query, cached", [
    ("will it snow today", "will it snow tomorrow"),
    ("Set an alarm for 6 PM Saturday", "set an alarm for 6 AM on Saturday"),
    ("Show me my alarms for next Wednesday", "Show me my alarms for next Saturday"),
    ("aaj ka mausam kaisa hai", "what's the weather tomorrow"),
    ("शनिवार का अलार्म हटाओ", "delete my sunday alarm"),
    ("alarm subah 7 baje", "alarm for 7 pm"),
])
def test_a_different_day_or_time_of_day_blocks(query, cached):
    assert check(query, cached) == "slot"


@pytest.mark.parametrize("query, cached", [
    ("kal ka mausam batao", "what's the weather tomorrow"),  # kal: tomorrow or yesterday
    ("मैं आज रात शहर में क्या कर सकता हूँ", "what can i do tonight in the city"),
    ("kal tak delivery ho jayegi?", "How long does delivery take?"),  # a time the FAQ doesn't name is no clash
])
def test_the_same_day_in_other_words_passes(query, cached):
    assert check(query, cached) is None


@pytest.mark.parametrize("query, cached", [
    ("turn off the bedroom lights", "turn on the bedroom lights"),
    ("light band karo", "light on karo"),
    ("how do I sell my old phone", "where can I buy an old phone"),
    ("purana phone bechna hai", "purana phone kharidna hai"),
    ("how do I close my savings account", "how do I open a savings account"),
])
def test_opposite_actions_from_public_data_block(query, cached):
    assert check(query, cached) == "action"


@pytest.mark.parametrize("query, cached", [
    ("convert jpg to pdf", "convert pdf to jpg"),
    ("converting pdf into word document", "convert word file to pdf"),
    ("Delhi se Mumbai ki flight", "flights from Mumbai to Delhi"),
    ("दिल्ली से मुंबई की फ्लाइट", "मुंबई से दिल्ली फ्लाइट"),
    ("use baking soda instead of baking powder", "use baking powder instead of baking soda"),
])
def test_swapped_direction_blocks(query, cached):
    assert check(query, cached) == "direction"


@pytest.mark.parametrize("query, cached", [
    ("Delhi se Mumbai ki flight", "flights from Delhi to Mumbai"),
    ("how to transfer money to my account", "how do I transfer money to my account"),
])
def test_the_same_direction_passes(query, cached):
    assert check(query, cached) is None


@pytest.mark.parametrize("query, cached", [
    ("What is your favourite anime? And why?", "What's your favourite anime?"),  # sentence start, not a name
    ("Pakistan's Musharraf Charged In Bhutto Death", "Pakistan's Musharraf charged over Bhutto murder"),  # title case
    ("Rudolf Kempe was born in Zurich", "Rudolf Kempe was born in Zürich"),
    ("new {{Account Category}} account for wife", "new {{Account Type}} account for wife"),  # template slot
])
def test_capitals_that_are_not_new_names_pass(query, cached):
    assert check(query, cached) is None


def test_long_prompts_are_left_to_the_score():
    """One different word means little in a long prompt: the rules only judge FAQ-length questions."""
    prompt = "Write a short story about a dragon who learns to cook, in the style of a fairy tale, for children " \
             "aged {} to eight, with a happy ending and a moral about patience"
    assert check(prompt.format(5) + " in London", prompt.format("five")) is None
    assert check("Write 5 titles in London", "Write five titles") == "entity"


# --- refinements from people's own messages (Phase 8 human split and public dev splits only): real messages put
# "nahi" or "not" in a side clause that describes the situation, not in what they ask for

@pytest.mark.parametrize("query, cached", [
    ("online pay nahi karaycha, parcel aala ki cash dein, possible ahe ka?", "Is Cash on Delivery available?"),
    ("ata hi shop nahi vaparaychi, account pura delete kara", "How do I delete my account?"),
    ("card save karne bol raha hai par darr lagta hai, safe hai kya, koi fraud toh nahi hoga",
     "Is it safe to save my card details?"),
    ("emi milegi iphone pe? no cost wali", "Do you offer EMI?"),
    ("I do not want this item, cancel my order", "How do I cancel my order?"),
])
def test_a_negation_about_something_else_passes(query, cached):
    assert check(query, cached) is None


@pytest.mark.parametrize("query, cached", [
    ("kisi ne gift card diya h iska kaise use kare samajh nhi aaya", "How do gift cards work?"),  # didn't understand
    ("Could not complete transfer, it says declined", "Transfer unable to be completed, states declined"),
    ("i dont khow how to cancel my order", "How do I cancel my order?"),
    ("I do not knowhow to cancel my order", "How do I cancel my order?"),
    ("Can I use my card no matter where I go?", "Is there anywhere I can't use my card?"),
])
def test_more_hedges_and_failure_reports_are_not_a_flip(query, cached):
    assert check(query, cached) is None


@pytest.mark.parametrize("query, cached", [
    ("Cash on delivery nahi chahiye, card se pay karna hai, kaise karun?", "Is Cash on Delivery available?"),
    ("I don't want to pay in cash at delivery, is there any way to avoid it?", "Is Cash on Delivery available?"),
    ("Mera order kab tak deliver hoga, abhi cancel nahi karna?", "How do I cancel my order?"),
    ("Plus membership? nahi chahiye", "I want Plus membership"),  # a negated clause that names nothing still counts
    ("I don't keep my card details, is that safe?", "Is it safe to save my card details?"),  # "keep" isn't "know"
])
def test_a_negation_about_what_both_ask_still_blocks(query, cached):
    assert check(query, cached) == "negation"
