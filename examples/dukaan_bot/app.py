"""Dukaan Bot: a small FAQ support bot with Pakka's cache in front of the LLM.

    uvicorn examples.dukaan_bot.app:create_app --factory

Settings (env vars): PAKKA_EMBEDDER (default paraphrase-multilingual-MiniLM-L12-v2; Qwen/Qwen3-Embedding-0.6B with a
GPU), PAKKA_THRESHOLD (default: the one picked for that embedder, pakka.embed.THRESHOLDS), PAKKA_GUARD (1 or 0), PAKKA_TTL_DAYS,
PAKKA_BACKEND (numpy or redis), PAKKA_REDIS_URL, PAKKA_MODEL, OPENROUTER_API_KEY.

Answers the LLM writes for a `user_id` are cached for that user only. Answers for anonymous
questions wait in data/pending.jsonl until `python -m pakka.approve` moves them to
data/approved.jsonl, which is loaded at startup for as long as the FAQ is unchanged.
"""

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from pakka.cache import NumpyCache
from pakka.flow import CachedAnswerer
from pakka.guard import check
from pakka.policy import GLOBAL, Policy

ROOT = Path(__file__).resolve().parents[2]
FAQS = json.loads((ROOT / "data" / "faqs.json").read_text())


USER_ID = r"^[A-Za-z0-9_.-]{1,64}$"


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    user_id: str | None = Field(default=None, pattern=USER_ID)

    @field_validator("user_id")
    @classmethod
    def not_reserved(cls, value):
        if value == GLOBAL:
            raise ValueError("reserved user_id")
        return value


class Forget(BaseModel):
    user_id: str = Field(pattern=USER_ID)


def kb_version(faqs):
    """Short hash of the FAQ content: any edit gives a new version and retires older cached answers."""
    return hashlib.sha256(json.dumps(faqs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def build(embed=None, llm=None, threshold=None, log_path=None, store_misses=True,
          pending_path=ROOT / "data" / "pending.jsonl", approved_path=ROOT / "data" / "approved.jsonl"):
    """Seed a cache with the 30 FAQs and wrap the LLM. Arguments left as None come from env vars."""
    from pakka.embed import DEFAULT_EMBEDDER, THRESHOLDS, load_embedder

    name = os.environ.get("PAKKA_EMBEDDER", DEFAULT_EMBEDDER)
    if embed is None:
        embed = load_embedder(name)
    if llm is None:
        from examples.dukaan_bot.llm import make_llm

        llm = make_llm(FAQS)
    if threshold is None:
        if "PAKKA_THRESHOLD" not in os.environ and name not in THRESHOLDS:
            raise SystemExit(f"Set PAKKA_THRESHOLD for {name}: `pakka bench --embedder {name}` picks one")
        threshold = float(os.environ.get("PAKKA_THRESHOLD") or THRESHOLDS[name])
    if log_path is None:
        log_path = ROOT / "logs" / "requests.jsonl"
        log_path.parent.mkdir(exist_ok=True)

    if os.environ.get("PAKKA_BACKEND") == "redis":
        from pakka.redis_cache import RedisVLCache

        cache = RedisVLCache(embed, name="dukaan", redis_url=os.environ.get("PAKKA_REDIS_URL", "redis://localhost:6380"))
    else:
        cache = NumpyCache(embed)
    version = kb_version(FAQS)
    shared = {"partition": GLOBAL, "kb_version": version}
    for faq in FAQS:
        cache.store(faq["question"], faq["answer"], {"id": faq["id"], **shared})
    if Path(approved_path).exists():
        for line in Path(approved_path).read_text().splitlines():
            row = json.loads(line)
            if row.get("kb_version") == version:  # approved under an older FAQ: retired
                cache.store(row["question"], row["answer"], {**shared, **({"id": row["id"]} if row.get("id") else {})})
    policy = Policy(kb_version=version, ttl=float(os.environ.get("PAKKA_TTL_DAYS", "7")) * 86400,
                    pending_path=pending_path)
    guard = check if os.environ.get("PAKKA_GUARD", "1") == "1" else None
    return CachedAnswerer(cache, llm, threshold, log_path=log_path, store_misses=store_misses, guard=guard,
                          policy=policy)


def create_app(**kwargs):
    answerer = build(**kwargs)
    app = FastAPI(title="Dukaan Bot")

    @app.post("/chat")
    def chat(body: Question):
        return asdict(answerer.ask(body.question, partition=body.user_id))

    @app.post("/forget")
    def forget(body: Forget):
        # Erasure request (DPDP/GDPR). shortcut: no auth in this demo; check the caller owns user_id in production.
        return {"deleted": answerer.cache.forget("partition", body.user_id)}

    return app
