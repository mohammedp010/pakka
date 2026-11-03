"""Optional verifiers: a second opinion on a cache hit the guards let through, "would the reply written for
the cached question also answer this one?". Off by default; pass one to `CachedAnswerer(verify=...)`.

    verify = load_verifier("jev")                        # TypeSafe's Jev, needs TYPESAFE_API_KEY
    verify = load_verifier("openrouter:<model>")         # any OpenRouter chat model, needs OPENROUTER_API_KEY

Both send the two questions to an API. `CachedAnswerer` masks PII first and counts an error or timeout as a miss.
`jev_verifier` and `openrouter_verifier` are the batch forms `pakka bench --verify` uses, cached on disk.
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

JEV_MODEL = "jev-1.14.0"
JEV_MIN_SCORE = 0.84  # picked on HinglishCacheBench v1 dev (Qwen3-0.6B + guards, similarity 0.6) for a 5% budget
JEV_URL = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Asks what a cache needs to know (does one reply fit both?), not how similar the texts are. Numbers and dates are
# Jev's known weak spot, so the guards run first and only what they let through gets here.
JEV_QUESTION = {
    "type": "noul",
    "instructions": "Would the support reply written for `cached_question` also correctly answer `new_question`?",
    "criteria": {
        "true": "Both ask for the same thing, so one reply fits both, even if the wording, language (English, "
                "Hinglish or Hindi) or spelling differ.",
        "false": "They ask for different things: another amount, number, day, place, product or person, the "
                 "opposite action (on/off, cancel/undo, not), or a related but separate question, so that reply "
                 "would be wrong or incomplete.",
    },
}
# The same question for a chat model, answered with one word.
YES_NO = (f"You check hits from a support bot's answer cache. {JEV_QUESTION['instructions']}\n"
          f"Answer yes if: {JEV_QUESTION['criteria']['true']}\nAnswer no if: {JEV_QUESTION['criteria']['false']}\n"
          "Reply with one word: yes or no.")


def _client(env, name, timeout, transport):
    import httpx

    key = os.environ.get(env)
    if not key:
        raise SystemExit(f"Set {env} to verify with {name}")
    return httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {key}"}, transport=transport)


def _post(client, url, body, attempts):
    import httpx

    for attempt in range(attempts):  # rate limits and overload (429, 5xx) are normal under load
        try:
            response = client.post(url, json=body)
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
            continue
        if (response.status_code == 429 or response.status_code >= 500) and attempt < attempts - 1:
            time.sleep(float(response.headers.get("retry-after") or 2 ** attempt))
            continue
        response.raise_for_status()
        return response.json()


def _batch(ask, tag, cache_path, workers):
    """`verify(pairs) -> answers` over `ask(pair)`, answers kept at `cache_path` (JSON keyed by a hash of `tag` and
    the pair) after every 200 pairs, so a rerun or a crash never pays twice."""
    path = Path(cache_path) if cache_path else None
    store = json.loads(path.read_text()) if path and path.exists() else {}

    def verify(pairs):
        keys = [hashlib.md5(json.dumps([*tag, *p]).encode()).hexdigest() for p in pairs]
        missing = list({k: p for k, p in zip(keys, pairs) if k not in store}.items())
        with ThreadPoolExecutor(workers) as pool:
            for i in range(0, len(missing), 200):
                chunk = missing[i:i + 200]
                store.update(zip([k for k, _ in chunk], pool.map(ask, [p for _, p in chunk])))
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(store))
        return [store[k] for k in keys]

    verify.ask = ask
    return verify


def jev_verifier(model=JEV_MODEL, cache_path=None, transport=None, workers=8, timeout=60, attempts=6):
    """TypeSafe's Jev as `verify(pairs) -> scores`: its probability that one reply fits both (new, cached)
    questions. `.ask(pair)` asks once, uncached."""
    client = _client("TYPESAFE_API_KEY", model, timeout, transport)

    def ask(pair):
        body = {"model": model, "state": {"new_question": pair[0], "cached_question": pair[1]},
                "questions": {"same_reply": JEV_QUESTION}}
        return _post(client, JEV_URL, body, attempts)["answers"]["same_reply"]["noul"]

    return _batch(ask, [model, JEV_QUESTION], cache_path, workers)


def openrouter_verifier(model, cache_path=None, transport=None, workers=8, timeout=60, attempts=6):
    """An OpenRouter chat model as `verify(pairs) -> [bool]`: does one reply fit both (new, cached) questions?
    Anything but a yes counts as no."""
    client = _client("OPENROUTER_API_KEY", model, timeout, transport)

    def ask(pair):
        body = {"model": model, "temperature": 0, "max_tokens": 5, "reasoning": {"enabled": False},
                "messages": [{"role": "system", "content": YES_NO},
                             {"role": "user", "content": f"cached_question: {pair[1]}\nnew_question: {pair[0]}"}]}
        content = _post(client, OPENROUTER_URL, body, attempts)["choices"][0]["message"].get("content") or ""
        return content.strip().lower().startswith("yes")

    return _batch(ask, [model, YES_NO], cache_path, workers)


def load_verifier(name="jev", min_score=JEV_MIN_SCORE, timeout=5, transport=None):
    """`verify(question, cached_question, cached_answer) -> bool` for `CachedAnswerer(verify=...)`: "jev" (or
    "jev-<version>", passes at `min_score`) or "openrouter:<model>". One try of at most `timeout` seconds, since a
    live lookup can't wait out retries; the answerer counts an error as a miss."""
    if name == "jev" or name.startswith("jev-"):
        ask = jev_verifier(JEV_MODEL if name == "jev" else name, transport=transport, timeout=timeout, attempts=1).ask
        return lambda question, cached_question, cached_answer: ask((question, cached_question)) >= min_score
    if name.startswith("openrouter:"):
        ask = openrouter_verifier(name.removeprefix("openrouter:"), transport=transport, timeout=timeout, attempts=1).ask
        return lambda question, cached_question, cached_answer: ask((question, cached_question))
    raise ValueError(f"Unknown verifier {name!r}: use 'jev' or 'openrouter:<model>', or pass your own function")
