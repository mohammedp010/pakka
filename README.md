# pakka

**Tests, guards and safety rules for semantic caches, built for English and Hinglish support bots.**

A semantic cache answers a repeat question from memory when its meaning matches an earlier one.
Every gateway ships one now: Redis, Azure, Kong, LiteLLM and others. What none of them tells you is
**how often the cache serves a confident, wrong answer.** Pakka measures that, blocks the common
causes, and keeps the cache from leaking or going stale.

*Pakka* is Hinglish for "sure": *pakka sure hai?*

## The problem, in three numbers from this repo

- **No threshold is safe.** On 526 held-out support questions, a plain cache with its threshold tuned on
  separate data could not keep false hits under 10% at 95% confidence. At best it served 55 right answers
  and 12 wrong ones.
- **Negation fools embeddings.** `I don't want to delete my account, how do I just deactivate it?` matched
  "How do I delete my account?" at 0.88 similarity. So did `coupon apply kyun nahi ho raha?` and "How do I
  apply a coupon code?".
- **Hinglish is barely served.** The spec's multilingual model caught 12 of 200 Hinglish paraphrases.

## What pakka does

| Part | What it is |
|---|---|
| `pakka bench` | Replays a labelled question set through a cache. Reports hit rate and false hits with 95% Wilson intervals, picks the threshold on a dev split and reports on held-out, breaks errors down by trap type and language, and fails CI when the false-hit bound regresses |
| HinglishCacheBench (`data/bench_v1.jsonl`) | 1,059 support questions in English, Hinglish and Hindi with labelled trap questions (negation, entity, number, near intent, personal), written by one LLM, labelled blind by a second, with disagreements settled by a human ([dataset card](data/README.md)) |
| `pakka.guard` | Microsecond rule guards that veto a hit: negation (incl. nahi/nhi/nahin, नहीं), numbers and IDs, days and times (today/kal/आज, am/pm), opposite actions (on/off, buy/sell), swapped direction ("pdf to jpg"), and out-of-scope names. Tested on nine public datasets ([results](docs/public_results.md)) |
| `pakka.verify` | Optional, off by default: a second opinion on hits the guards let through, from TypeSafe's Jev, any OpenRouter model, or your own function. PII is masked first; an error or timeout counts as a miss |
| `pakka.policy` + `pakka.pii` | Never stores Indian PII (Aadhaar with checksum, PAN, UPI, +91 mobiles, IFSC, cards, order IDs), keeps each user's answers in their own partition, retires answers when the FAQ changes, expires stored answers, queues shared answers for approval, and supports erasure requests |
| `pakka.testset`, `pakka.train` | Generate a stress-test set for your own FAQ with two LLMs, and fine-tune an embedder on it |
| Dukaan Bot (`examples/dukaan_bot`) | A small FAQ support bot for a fictional Indian store, used as the end-to-end test |

## Results

Pakka's default setup: Qwen3-Embedding-0.6B with pakka's guards, threshold picked on HinglishCacheBench v1's dev
split and reported on held-out questions. Two test sets: v1's 526 LLM-written questions, and 258 messages real
people wrote in their own words (80% romanized Hinglish), which is the harder and more honest one. Cells are
right / wrong answers served from cache, with the 95% upper bound on the false-hit rate.

**Free (no API calls):**

| Setup | v1 test (526) | Human test (258) | Hinglish caught (human) |
|---|---|---|---|
| Guards, 10% budget (threshold 0.925) | 102 / 8 (13.7%) | 11 / 0 (25.9%) | 6/184 |
| Guards, 5% budget | no threshold fits | no threshold fits | - |

**With an optional verifier** (a second opinion on each hit the guards let through, threshold lowered to 0.6):

| Verifier | v1 test (526) | Human test (258) | Hinglish caught (human) | Extra time per checked hit | Cost per 100k checks |
|---|---|---|---|---|---|
| Cheap LLM, `openrouter:qwen/qwen3.7-flash` (yes / no) | 231 / 13 (8.9%) | 113 / 9 (13.4%) | 97/184 | ~0.9 s | ~$0.60 |
| Jev (TypeSafe `jev-1.14.0`), 5% budget (score ≥ 0.84) | 181 / 3 (4.7%) | 80 / 1 (6.7%) | 67/184 | ~0.3 s | ~$1.75 |

Without a verifier the cache answers 4% of real messages; with one it answers a third to a half of them. The
cheap LLM fits a 10% budget; Jev gives a score you can tighten, so it also fits 5%. The LLM call a hit saves took
1.3 s in the real bot. On the public parse datasets (Hinglish-TOP, MASSIVE) Jev cuts false hits, the cheap LLM
barely does, and no setup fits 5%: see [docs/public_results.md](docs/public_results.md#free-vs-with-a-verifier-phase-85).

v1 results with MiniLM-L12, the old default (526 held-out questions):

| Setup | Right answers | Wrong answers | False-hit rate (95% upper bound) | Hinglish caught |
|---|---|---|---|---|
| Plain cache, spec's model, 20% budget | 55 | 12 | 17.9% (28.7%) | 12/200 |
| + pakka guards, 10% budget | 49 | 5 | 9.3% (19.9%) | 11/200 |
| + fine-tuned encoder and approved keys, 10% budget | **70** | **2** | **2.8% (9.6%)** | **32/200** |

In the real bot with a real LLM on the same 526 questions (MiniLM): **13% of questions answered from cache
with 2 wrong answers, in 56 ms instead of 1,331 ms.** The set is built to be hard: 35% of it must miss and
55% is Hinglish.

| Safety check | Without pakka | With pakka |
|---|---|---|
| Hits served from before an FAQ edit (persistent cache) | 100% | 0% |
| One user's LLM answer served to another user | yes | never (canary test, in memory and on Redis) |
| A planted answer reaching other users | yes | stays in the attacker's partition |

**Honest caveats.** v1 overstated what a cache can answer: its questions were LLM-written, and on real
people's messages the free setup answers a tenth as much (the human test above). Part of v1's fine-tuning gain
is the generator's writing style too: on the 46 hand-written held-out questions, paraphrases caught only rose
from 11 to 13 of 27. Off-the-shelf cross-encoder verifiers made every result worse; the verifiers that help
are API calls, so they are optional and off by default. Every experiment is in [docs/results.md](docs/results.md)
and [docs/public_results.md](docs/public_results.md).

## Choosing an embedder

The default is Qwen3-Embedding-0.6B. It is much slower on a CPU, so **on a small CPU-only server use MiniLM-L12**
(`pakka.embed.FAST_EMBEDDER`, threshold 0.79). Measured on a MacBook Air M2 with the GPU switched off, one
question at a time, as a lookup embeds it:

| | Qwen3-Embedding-0.6B (default) | MiniLM-L12 (fast option) |
|---|---|---|
| Download | 1.2 GB | 0.48 GB |
| Memory (process peak) | ~1.0 GB | ~0.7 GB |
| CPU, 8 threads (median / p90) | 235 / 312 ms | 9 / 10 ms |
| CPU, 2 threads, like a small VPS (median / p90) | 538 / 834 ms | 8 / 8 ms |
| M2 GPU (median) | ~36 ms | ~9 ms |
| Threshold, guards on (`pakka.embed.THRESHOLDS`) | 0.925 | 0.79 |
| v1 test, free, 10% budget | 102 / 8 | 47 / 5 |
| Human test, free, 10% budget | 11 / 0 | 2 / 1 |

Mean P-CHR AUC over the public datasets: 0.647 for Qwen3-0.6B against 0.571 for MiniLM. A threshold belongs to
one embedder; for any other model, pick one with `pakka bench --embedder <model>`.

## Quickstart

```bash
pip install pakka            # not yet published; until then: pip install -e .
```

Guard a cache you already have:

```python
from pakka.guard import check

check("refund nahi mila", "How long does a refund take?")                    # "negation"
check("Is Cash on Delivery available in Nepal?", "Is Cash on Delivery available?")  # "entity"
check("COD available hai kya?", "Is Cash on Delivery available?")           # None: safe to serve
```

Or use the full flow:

```python
from pakka.cache import NumpyCache            # or pakka.redis_cache.RedisVLCache
from pakka.embed import load_embedder
from pakka.flow import CachedAnswerer
from pakka.guard import check
from pakka.policy import GLOBAL, Policy

def my_llm(question):                         # your LLM call: returns (answer, tokens used)
    ...

cache = NumpyCache(load_embedder())          # Qwen3-Embedding-0.6B, a 1.2 GB download on first use
cache.store("How long does a refund take?", "Refunds take 5-7 business days.",
            {"id": "faq_01", "partition": GLOBAL, "kb_version": "v1"})

bot = CachedAnswerer(cache, my_llm, threshold=0.925, guard=check, policy=Policy(kb_version="v1"))
r = bot.ask("How many days does a refund take?", partition="user-42")
r.hit, r.reason, r.score                      # (True, 'hit', 0.977)
```

Measure your cache:

```bash
pakka bench --data data/bench_v1.jsonl --guard --budget 0.10 --fail-above 0.25
```

## Turn on a verifier (optional)

Off by default: pakka works fully offline without one. A verifier is asked one question about each hit the
guards let through, "would the reply for the cached question also answer this one?", and the hit is served only
on a yes. Since it catches what similarity misses, lower the threshold to 0.6 so it sees the grey band:

```python
from pakka.verify import load_verifier

bot = CachedAnswerer(cache, my_llm, threshold=0.6, guard=check, policy=Policy(kb_version="v1"),
                     verify=load_verifier("openrouter:qwen/qwen3.7-flash"))   # needs OPENROUTER_API_KEY
# or verify=load_verifier("jev")              TypeSafe's Jev, needs TYPESAFE_API_KEY, passes at score 0.84
# or verify=my_check                          any function (question, cached_question, cached_answer) -> bool
```

- `pip install "pakka[verify]"` adds the HTTP client.
- The verifier only ever sees text masked with `pakka.pii.mask` (Aadhaar, PAN, phone numbers, order IDs and so on).
- One try with a 5-second timeout (`load_verifier(..., timeout=)`). An error or a timeout counts as a miss, so the
  LLM answers and no unchecked hit is served. `Result.reason` says `blocked:verify` or `blocked:verify_error`.
- Any OpenRouter chat model works after `openrouter:`; `qwen/qwen3.7-flash` was the cheapest of five tried, and
  as accurate. Measure your own pick before trusting it:

```bash
pakka bench --data data/bench_v1.jsonl --guard --budget 0.05 --verify jev-1.14.0 --verify openrouter:qwen/qwen3.7-flash
```

## How a question flows

```mermaid
flowchart TD
    Q[Question + user_id] --> E[Embed and find the nearest entry<br/>in this user's partition or the shared FAQs,<br/>current FAQ version, not expired]
    E --> T{Similarity at or above threshold?}
    T -- no --> L[Call the LLM]
    T -- yes --> G{Guards: negation, numbers, days and times,<br/>opposite actions, direction, out-of-scope names}
    G -- veto --> L
    G -- pass --> V{Optional verifier, PII masked:<br/>does the cached reply fit?}
    V -- no, error or timeout --> L
    V -- yes, or no verifier --> H[Serve the cached answer]
    L --> P{Contains PII?}
    P -- yes --> N[Answer, never store]
    P -- no --> S{Shared or per user?}
    S -- user --> U[Store in the user's partition with a TTL]
    S -- shared --> A[Queue for approval]
```

## Run everything

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[redis,bot,dev,train,verify]"
.venv/bin/python -m pytest                                  # unit tests, under a second
docker run -d --name pakka-redis -p 6380:6379 redis:8.2.2-alpine3.22
.venv/bin/python -m pytest -m integration                   # real model + Redis
echo "OPENROUTER_API_KEY=..." > .env                        # for the bot and e2e test
.venv/bin/python -m examples.dukaan_bot.chat --user ravi    # chat with Dukaan Bot
```

More commands (building the benchmark, fine-tuning, the approval queue, the stale-answer demo) are in
[docs/results.md](docs/results.md).

## Limits

Qwen3-Embedding-0.6B, the default, takes a quarter to half a second per question on a CPU (MiniLM-L12 is the
fast option). PII detection is regex plus checksums and doesn't catch names, so a verifier can see a name. Partitions isolate answers but not
timing: a hit is still faster than a miss. `/forget` in the demo bot has no authentication. The
benchmark covers one domain (Indian e-commerce support) with 30 FAQs.

## License

Code: MIT. Dataset: CC BY 4.0.
