# pakka

Tests, guards and safety rules for semantic caches, so a cache can prove its answers are right.

**Status: phase 2 (guard).** A semantic cache with rule guards that veto wrong hits, a reference FAQ
bot ("Dukaan Bot"), a 1,059-question English/Hinglish/Hindi test set and a bench that reports false
hits with confidence intervals. Safety rules (privacy, expiry, poisoning) come in phase 3.

> **Pilot result:** Cache hit rate 32% on 100 test questions at threshold 0.70 with 2 false hits;
> LLM calls down 32%; average latency down from 1258 ms to 899 ms.
> Hits answer in 27 ms (median) against 1228 ms for an LLM call. Tokens fell from 211k to 144k.
> Measured with Dukaan Bot, paraphrase-multilingual-MiniLM-L12-v2 and anthropic/claude-haiku-5.5 on OpenRouter.

```mermaid
flowchart LR
    Q[Question] --> E[Embed] --> L{Nearest cached question<br/>similarity >= threshold?}
    L -- hit --> A[Saved answer, ~10 ms]
    L -- miss --> M[LLM call, answers from the FAQ] --> S[Store the new pair] --> B[Answer]
```

## Quickstart

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[redis,bot,dev]"
.venv/bin/python -m pytest                    # unit tests, no model or network, under a second
.venv/bin/python -m pytest -m integration     # real embedding model + Redis on port 6380
.venv/bin/python -m pakka.bench --embedder paraphrase-multilingual-MiniLM-L12-v2 --thresholds 0.6,0.65,0.7,0.75
```

The bot needs an OpenRouter key in `.env` (`OPENROUTER_API_KEY=...`, gitignored):

```bash
.venv/bin/python -m examples.dukaan_bot.chat                          # chat in the terminal
.venv/bin/uvicorn examples.dukaan_bot.app:create_app --factory         # POST /chat {"question": "..."}
.venv/bin/python -m examples.dukaan_bot.replay --no-cache && .venv/bin/python -m examples.dukaan_bot.replay
.venv/bin/python -m pytest -m e2e                                      # real LLM, costs about $0.001
```

Build, review and measure the benchmark:

```bash
.venv/bin/python scripts/build_bench.py                  # LLM-written questions, labelled by a second model (~$0.05)
.venv/bin/python -m pakka.review data/bench_v1.jsonl     # check the items the two models disagreed on
.venv/bin/python -m pakka.bench --data data/bench_v1.jsonl --budget 0.15 \
    --embedder exact --embedder paraphrase-multilingual-MiniLM-L12-v2 --guard --fail-above 0.30
```

Add `--verify <cross-encoder>` to re-score grey-band hits (bi-encoder score at or above `--floor`).

Redis with search for the `redis` backend (`PAKKA_BACKEND=redis`):
`docker run -d --name pakka-redis -p 6380:6379 redis:8.2.2-alpine3.22`

## Tests

| Layer | What runs | When |
|---|---|---|
| Unit | Cache, flow, sweep, bot API, data checks, with a fake embedder and a fake LLM | Every change |
| Integration (`-m integration`) | Real sentence-transformers model; RedisVL scores match numpy within 0.001 | Before each phase ends |
| End to end (`-m e2e`) | Dukaan Bot with the real LLM | Before each phase ends |

## Data

- `data/faqs.json`: 30 FAQs for a fictional Indian online store.
- `data/pilot.jsonl`: 100 labelled questions. 60 paraphrases (30 English, 30 Hinglish) carry the FAQ
  they should match; 40 should miss, tagged `negation`, `personal`, `near_intent`, `entity` or `off_topic`.

## Pilot sweep (100 questions; misses charged an assumed 1,500 ms of LLM time, measured later at about 1,260 ms)

Thresholds don't transfer between models. In the spec's 0.75-0.95 range, the MiniLM models
catch few paraphrases, while multilingual-e5-small lets almost everything through:

| Model | 0.75 | 0.85 | 0.95 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 19% hits, 1 false | 7%, 0 | 4%, 0 |
| paraphrase-multilingual-MiniLM-L12-v2 | 25%, 1 | 12%, 0 | 2%, 0 |
| multilingual-e5-small | 100%, 48 | 70%, 22 | 8%, 1 |

Swept over each model's own range, the best "almost safe" settings were:

| Model | Threshold | Hit rate | Correct hits | False hits | Hinglish paraphrases caught |
|---|---|---|---|---|---|
| **paraphrase-multilingual-MiniLM-L12-v2** | **0.70** | **32%** | **30** | **2** | **8/30** |
| all-MiniLM-L6-v2 | 0.70 | 24% | 22 | 2 | - |
| multilingual-e5-small | 0.91 | 25% | 22 | 3 | 1/30 |

False hits the next phases have to stop:

- `coupon apply kyun nahi ho raha?` was served the answer to "How do I apply a coupon code?" (negation, 0.714)
- `Is Cash on Delivery available in Nepal?` got "Yes, COD is available" from every model (entity)
- `How do I cancel my Netflix subscription?` matched "How do I cancel my order?" on e5-small (near intent)

100 questions is a pilot: one false hit moves the rate by a full point, so these numbers only show
direction. Phase 1 grows the set to about 1,000 and adds confidence intervals.

## Phase 1: the benchmark (`data/bench_v1.jsonl`)

1,059 questions: the 100 hand-written pilot questions plus 965 written by `anthropic/claude-haiku-5.5`
(20 paraphrases and 10 trap questions per FAQ, plus 97 off-topic or uncovered questions). A second
model family, `deepseek/deepseek-v4.1-flash`, labelled every generated question without seeing the
first model's intent. They agreed on 880 of 965 (91%); a human decided the other 85 with `pakka.review`.

| | Count |
|---|---|
| Should hit | 678 |
| Should miss | 381 |
| Questions written as traps, by type (some turned out to be paraphrases of another FAQ) | negation 89, near intent 74, entity 66, number 64, personal 41, uncovered 49, off topic 48 |
| English / Hinglish / Hindi (Devanagari) | 367 / 582 / 110 |
| Dev / held-out | 533 / 526 |

**Results before guards.** The threshold is picked on dev as the most hits whose false-hit rate stays
under a budget at 95% confidence, then reported on the 526 held-out questions. At a 5%, 10% or 15%
budget, no model fits except the multilingual MiniLM at 15%. At a 20% budget:

| Encoder | Threshold | Hit rate | Correct | False hits | False-hit rate (95% upper) | English | Hinglish | Hindi |
|---|---|---|---|---|---|---|---|---|
| exact match | none fits | | | | | | | |
| all-MiniLM-L6-v2 | 0.875 | 3% | 18 | 0 | 0.0% (17.6%) | 18/96 | 0/200 | 0/45 |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.776 | 13% | 55 | 12 | 17.9% (28.7%) | 35/96 | 12/200 | 8/45 |
| intfloat/multilingual-e5-small | 0.952 | 4% | 18 | 2 | 10.0% (30.1%) | 18/96 | 0/200 | 0/45 |
| redis/langcache-embed-v3-small | 0.709 | 10% | 46 | 6 | 11.5% (23.0%) | 44/96 | 2/200 | 0/45 |

Two findings. First, Hinglish and Hindi are almost unserved: the best model catches 12 of 200 Hinglish
paraphrases. Second, the top of the ranking is clean (the top 5% of scores are 96% correct), but
negation traps score as high as real paraphrases, for example:

- `I don't want to delete my account, how do I just deactivate it?` matched "How do I delete my account?" at 0.880
- `Can I track my order without logging in?` matched "How can I track my order?" at 0.866

So no plain cosine threshold is both useful and safe here. The guards in phase 2 have to change that.

## Phase 2: guards (`pakka/guard.py`)

`check(query, cached_question, cached_answer)` vetoes a hit for one of four reasons, in about 30 µs:

| Guard | Blocks | Example it catches |
|---|---|---|
| `negation` | A negator on one side only (English, Hinglish spellings like nahi/nhi/nahin, Devanagari नहीं/मत/बिना). Yes/no tails such as "or not", "ya nahi" and "nhi h kya" don't count. | "I don't want to delete my account, how do I just deactivate it?" vs "How do I delete my account?" |
| `slot` | A number, order ID or UPI ID the cached question and answer don't mention | "Order 4521 kab aayega?" |
| `action` | Opposite or confusable actions (block/unblock, delete/deactivate, ...) | "card unblock kaise kare" vs "How do I block my card?" |
| `entity` | A name or scope word (Nepal, Flipkart, abroad, country) the cached FAQ never mentions | "Is Cash on Delivery available in Nepal?" |

Dukaan Bot runs them by default (`PAKKA_GUARD=0` turns them off); a vetoed hit goes to the LLM and is
logged with a reason such as `blocked:negation`.

**Results on the 526 held-out questions** (paraphrase-multilingual-MiniLM-L12-v2, threshold picked on dev):

| Budget | | Threshold | Correct | False hits | False-hit rate (95% upper) | English | Hinglish | Hindi |
|---|---|---|---|---|---|---|---|---|
| 10% | without guards | none fits | | | | | | |
| 10% | **with guards** | 0.790 | 46 | 4 | 8.0% (18.8%) | 29/96 | 11/200 | 6/45 |
| 20% | without guards | 0.776 | 55 | 12 | 17.9% (28.7%) | 35/96 | 12/200 | 8/45 |
| 20% | **with guards** | 0.740 | 66 | 6 | 8.3% (17.0%) | 38/96 | 20/200 | 8/45 |

At the 20% budget, guards halve the false hits (negation false hits 6 to 2, entity 2 to 0) and still
add hits, because a safer cache can use a lower threshold. They also throw away 6 good hits, mostly
real negations that keep the same answer ("i dont remember my password" vs "How do I reset my password?").
Guards don't transfer to every encoder: with redis/langcache-embed-v3-small the dev-picked threshold
dropped and held-out false hits rose from 6 to 15.

**Cross-encoder verifiers didn't help.** Re-scoring grey-band hits (similarity 0.6 and above) on top of guards:

| Verifier | Correct | False hits | False-hit rate (95% upper) | Hinglish | Added latency |
|---|---|---|---|---|---|
| none (guards only) | 66 | 6 | 8.3% (17.0%) | 20/200 | 0 ms |
| redis/langcache-reranker-v1-bce | 28 | 6 | 17.6% (33.5%) | 4/200 | 19 ms |
| cross-encoder/quora-distilroberta-base | 96 | 17 | 15.0% (22.8%) | 38/200 | 9 ms |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 15 | 1 | 6.2% (28.3%) | 2/200 | 11 ms |

All three lowered P-CHR AUC (0.706 with guards alone, 0.372 to 0.634 with a verifier), matching Redis's
EMNLP 2026 finding that off-the-shelf rerankers often hurt cache precision. The Quora model nearly
doubles Hinglish hits but serves almost three times as many wrong answers, so no verifier ships by default.

No setup fits a 5% budget yet. The limit is recall: even with guards, the best setup catches 20 of 200
Hinglish paraphrases.
