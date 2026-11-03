"""Real verifiers (TypeSafe's Jev, an OpenRouter model). Four calls, well under $0.001. Run with: pytest -m e2e"""

import os

import pytest

pytestmark = pytest.mark.e2e
SAME = ("How many days does a refund take?", "How long does a refund take?", "Refunds take 5-7 business days.")
DIFFERENT = ("cancel my order", "How do I track my order?", "Use Track Order in the app.")


@pytest.mark.parametrize("name, env", [("jev", "TYPESAFE_API_KEY"), ("openrouter:qwen/qwen3.7-flash", "OPENROUTER_API_KEY")])
def test_real_verifier_passes_a_paraphrase_and_stops_a_different_question(name, env):
    if not os.environ.get(env):
        pytest.skip(f"{env} not set")
    from pakka.verify import load_verifier

    verify = load_verifier(name, timeout=30)
    assert verify(*SAME) is True
    assert verify(*DIFFERENT) is False
