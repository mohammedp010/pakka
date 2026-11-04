---
license: cc-by-4.0
language:
  - en
  - hi
multilinguality: multilingual
size_categories:
  - 1K<n<10K
task_categories:
  - sentence-similarity
  - text-classification
tags:
  - semantic-cache
  - hinglish
  - code-mixed
  - customer-support
  - hard-negatives
pretty_name: HinglishCacheBench
---

# HinglishCacheBench v1

A test set for **semantic caches** in a customer-support bot: for each customer message, should the
cache serve one of 30 FAQ answers, and which one, or must it miss and call the LLM? Every question is
labelled, and a third of them are **traps** that look like an FAQ but must not get its answer.

Messages are in English, romanized Hinglish (*refund kab milega?*, with real spelling variety such as
nahi/nhi/nahin) and Hindi in Devanagari. The domain is a fictional Indian online store.

## Files

| File | What it holds |
|---|---|
| `faqs.json` | The 30 FAQs: `id`, `question`, `answer` |
| `bench_v1.jsonl` | 1,059 labelled messages (below) |
| `bench_human.jsonl` | 315 messages written by 23 people (v1.1, below) |
| `pilot.jsonl` | The 100 hand-written messages the set started from (also inside `bench_v1.jsonl`) |
| `raw/` | Every LLM generation and label, cached, so the set can be rebuilt with `scripts/build_bench.py` |

Each line of `bench_v1.jsonl`:

| Field | Meaning |
|---|---|
| `q` | The customer message |
| `label` | The FAQ id whose answer fully and correctly answers it, or `null` if the cache must miss |
| `kind` | `paraphrase`, or how the message was built to trap a cache: `negation`, `entity`, `number`, `near_intent`, `personal`, plus `uncovered` and `off_topic` |
| `near` | The FAQ a trap was written against (it looks like that FAQ but needs a different answer) |
| `lang` | `en`, `hinglish` or `hi` |
| `split` | `dev` (choose thresholds, train) or `test` (report results) |
| `source` | `hand` (written by a person) or `generated` |
| `intended`, `judged`, `needs_review`, `reviewed` | Labelling provenance (below) |

## Counts

| | Count |
|---|---|
| Messages | 1,059 |
| Should hit / should miss | 678 / 381 |
| English / Hinglish / Hindi | 367 / 582 / 110 |
| Dev / test | 533 / 526 |
| Written as: paraphrase / negation / near intent / entity / number / personal / uncovered / off topic | 628 / 89 / 74 / 66 / 64 / 41 / 49 / 48 |

A message's `kind` records how it was written; its `label` is the final answer. Some traps turned out to
be paraphrases of a different FAQ, and they carry that FAQ as their label.

## How it was built

1. 100 messages written by hand (`source: hand`).
2. For each FAQ, `anthropic/claude-haiku-5.5` wrote 20 paraphrases and 10 trap messages, plus 97
   off-topic or uncovered messages.
3. `deepseek/deepseek-v4.1-flash`, a different model family, labelled every generated message
   without seeing what the writer intended.
4. The two agreed on 880 of 965. A person decided the other 85 (`reviewed: true`).
5. Each (kind, language) group was split in half, reproducibly, into dev and test.

## Known issues

- **Style leakage.** One model wrote both splits in the same prompt, so the splits share its phrasing.
  A model fine-tuned on dev scores much better on generated test messages than on the 46 hand-written
  ones. Report results on `source: hand` messages and on `bench_human.jsonl` as well.
- **Labels from agreeing models are not hand-checked.** About 25 of the 880 were spot-checked.
- **One domain, 30 FAQs.** Thresholds and guards tuned here may not transfer.
- **Personal identifiers are fake.** Order numbers, UPI IDs and names are invented.

## Human-written split (v1.1): `bench_human.jsonl`

315 messages written by 23 people through a Google Form (`scripts/make_form.gs`), in their own words, the
way they'd text a shop on WhatsApp. People saw a situation ("your parcel came and nothing is wrong with
it, you just don't like it"), never the FAQ text, so they couldn't copy its wording. Each form had 10
situations, 3 trap situations (a different number, a "not", a different place or product) and one
optional question of their own. Every writer ticked a box releasing their answers under CC BY 4.0; no
name, email or phone number was collected, and messages with personal identifiers were dropped.

| | Count |
|---|---|
| Messages / writers | 315 / 23 |
| Should hit / should miss | 254 / 61 |
| English / Hinglish / Hindi | 31 / 280 / 4 |
| Dev / test | 57 / 258 (by writer: one writer in five is dev) |
| Written as: paraphrase / negation / entity / number / uncovered | 239 / 24 / 24 / 23 / 5 |

Labels: `deepseek/deepseek-v4.1-flash` (`judged`) and `anthropic/claude-haiku-5.5` (`judged2`) labelled
each message blind. They agreed on 245; a person decided 90 (`reviewed: true`), including every message
where the judges agreed with each other but not with the form question it answered. Extra fields:
`writer` (a number per person), `source: human`. Rebuild from the form exports with
`scripts/ingest_human.py` (the raw exports are not published).

This split is the style-leakage check for v1: with the threshold picked on v1 dev, the cache answers
10–21% of v1 test but only 1–4% of these messages. Real messages are longer and add context ("password
yaad nhi aa raha, naya kaise banaye"), so they score lower against the FAQ questions, and the negation
guard read that context as a negation (Phase 8 fixed most of it). Results: [docs/public_results.md](../docs/public_results.md).

## Use

```bash
pakka bench --data data/bench_v1.jsonl --faqs data/faqs.json --guard --budget 0.10   # default embedder, MiniLM-L12
```

## License

CC BY 4.0.
