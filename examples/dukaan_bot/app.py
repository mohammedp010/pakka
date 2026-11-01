"""Dukaan Bot: a small FAQ support bot with Pakka's cache in front of the LLM.

    uvicorn examples.dukaan_bot.app:create_app --factory

Settings (env vars): PAKKA_EMBEDDER, PAKKA_THRESHOLD, PAKKA_GUARD (1 or 0), PAKKA_BACKEND (numpy or redis),
PAKKA_REDIS_URL, PAKKA_MODEL, OPENROUTER_API_KEY.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pakka.cache import NumpyCache
from pakka.flow import CachedAnswerer
from pakka.guard import check

ROOT = Path(__file__).resolve().parents[2]
FAQS = json.loads((ROOT / "data" / "faqs.json").read_text())


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def build(embed=None, llm=None, threshold=None, log_path=None, store_misses=True):
    """Seed a cache with the 30 FAQs and wrap the LLM. Arguments left as None come from env vars."""
    if embed is None:
        from pakka.embed import load_embedder

        embed = load_embedder(os.environ.get("PAKKA_EMBEDDER", "paraphrase-multilingual-MiniLM-L12-v2"))
    if llm is None:
        from examples.dukaan_bot.llm import make_llm

        llm = make_llm(FAQS)
    if threshold is None:
        # Best pilot setting for the default embedder: 2 false hits on 100 questions (see README).
        threshold = float(os.environ.get("PAKKA_THRESHOLD", "0.70"))
    if log_path is None:
        log_path = ROOT / "logs" / "requests.jsonl"
        log_path.parent.mkdir(exist_ok=True)

    if os.environ.get("PAKKA_BACKEND") == "redis":
        from pakka.redis_cache import RedisVLCache

        cache = RedisVLCache(embed, name="dukaan", redis_url=os.environ.get("PAKKA_REDIS_URL", "redis://localhost:6380"))
    else:
        cache = NumpyCache(embed)
    for faq in FAQS:
        cache.store(faq["question"], faq["answer"], {"id": faq["id"]})
    guard = check if os.environ.get("PAKKA_GUARD", "1") == "1" else None
    return CachedAnswerer(cache, llm, threshold, log_path=log_path, store_misses=store_misses, guard=guard)


def create_app(**kwargs):
    answerer = build(**kwargs)
    app = FastAPI(title="Dukaan Bot")

    @app.post("/chat")
    def chat(body: Question):
        return asdict(answerer.ask(body.question))

    return app
