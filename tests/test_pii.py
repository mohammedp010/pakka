import pytest

from pakka.pii import find_pii, luhn_ok, mask, verhoeff_digit


def aadhaar(first11):
    """A checksum-valid Aadhaar-format number built from 11 digits (not a real person's number)."""
    return first11 + str(verhoeff_digit(first11))


def test_verhoeff_matches_the_textbook_example():
    assert verhoeff_digit("236") == 3


def test_luhn_accepts_the_standard_test_card_and_rejects_a_typo():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")


@pytest.mark.parametrize("text, kind", [
    (f"mera aadhaar {aadhaar('23412341234')} hai", "aadhaar"),
    (f"Aadhaar: {aadhaar('98765432101')[:4]} {aadhaar('98765432101')[4:8]} {aadhaar('98765432101')[8:]}", "aadhaar"),
    ("PAN ABCPE1234F pe GST invoice chahiye", "pan"),
    ("pan abcpe1234f", "pan"),
    ("refund ravi@okicici pe bhej do", "upi"),
    ("mail me at ravi.k@example.com", "email"),
    ("call me on 9876543210", "phone"),
    ("mera number +91 98765 43210 hai", "phone"),
    ("IFSC HDFC0001234 hai", "ifsc"),
    ("card 4111 1111 1111 1111 se payment kiya", "card"),
    ("Order 4521 kab aayega?", "order_id"),
    ("Where is my order #88213?", "order_id"),
    ("mera order OD12345 cancel kar do", "order_id"),
])
def test_each_indian_identifier_is_found(text, kind):
    assert kind in {k for k, _ in find_pii(text)}


@pytest.mark.parametrize("text", [
    "refund kab milega?",
    "Delivery is free on orders above Rs 499.",
    "COD works on orders up to Rs 50,000 with a Rs 20 handling fee.",
    "call 1800-000-0000 between 8 am and 10 pm",  # the store's own toll-free number
    "Refunds take 5-7 business days.",
    "234123412341",  # 12 digits that fail the Aadhaar checksum
    "4111111111111112",  # 16 digits that fail Luhn
])
def test_ordinary_support_text_has_no_pii(text):
    assert find_pii(text) == []


def test_mask_replaces_each_identifier_with_its_kind():
    assert mask("refund ravi@okicici pe bhej do, order OD12345") == "refund <UPI> pe bhej do, order <ORDER_ID>"
    assert mask("refund kab milega?") == "refund kab milega?"
