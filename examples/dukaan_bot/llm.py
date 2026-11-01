"""OpenRouter chat client that answers from the FAQ. Called only on a cache miss."""

import os
from pathlib import Path

import httpx

# About $0.10 per 1M input and $0.50 per 1M output tokens on OpenRouter (Nov 2026).
DEFAULT_MODEL = "anthropic/claude-haiku-5.5"
URL = "https://openrouter.ai/api/v1/chat/completions"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

SYSTEM = (
    "You are the support assistant for an Indian online store. Reply in the customer's own language "
    "and style (English, Hinglish or Hindi) in at most 3 short sentences. Use only the FAQ below. "
    "If the FAQ does not answer the question, or it needs the customer's own order or account details, "
    "say you can't help with that here and point them to customer care (in-app chat under Help, "
    "or 1800-000-0000)."
)


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip("\"'")
    return None


def make_llm(faqs, model=None, max_tokens=300, transport=None):
    key = api_key()
    if not key:
        raise SystemExit(f"Set OPENROUTER_API_KEY, or add OPENROUTER_API_KEY=... to {ENV_FILE}")
    model = model or os.environ.get("PAKKA_MODEL", DEFAULT_MODEL)
    system = SYSTEM + "\n\nFAQ:\n" + "\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in faqs)
    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"}, transport=transport)

    def llm(question):
        response = client.post(URL, json={
            "model": model,
            "max_tokens": max_tokens,
            # FAQ answers need no extended thinking; it can eat the whole token budget.
            "reasoning": {"enabled": False},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        })
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        if not choice["message"].get("content"):
            raise RuntimeError(f"LLM returned no text (finish_reason={choice.get('finish_reason')})")
        return choice["message"]["content"].strip(), data["usage"]["total_tokens"]

    return llm
