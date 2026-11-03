---
language: [en, hi]
license: apache-2.0
base_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags: [semantic-cache, hinglish, code-mixed, pakka]
---

# pakka-minilm-l12-hinglish

[paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
fine-tuned for one job: deciding whether a support-bot question can reuse a cached answer, in English, Hinglish
and Hindi. It is the embedder of [pakka](https://github.com/mohammedp010/pakka)'s Phase 9. It is the same size and
speed as its base (about 9 ms a question on a laptop CPU, 0.47 GB), and **it helps only with a verifier.**

## Use it

```python
from pakka.cache import NumpyCache
from pakka.embed import TUNED_EMBEDDER, load_embedder
from pakka.flow import CachedAnswerer
from pakka.guard import check
from pakka.verify import load_verifier

cache = NumpyCache(load_embedder(TUNED_EMBEDDER))
bot = CachedAnswerer(cache, my_llm, threshold=0.6, guard=check, verify=load_verifier("jev"))  # or "openrouter:<model>"
```

Or with sentence-transformers alone: `SentenceTransformer("mohd010/pakka-minilm-l12-hinglish")`, then cosine
similarity between the new question and each cached one.

Without a verifier its threshold is 0.906 (`pakka.embed.THRESHOLDS`), but then it is no better than the base model.

## Results on data it never saw

Thresholds are picked on dev splits and reported on held-out test splits, with pakka's guards on. "Never saw v1"
is a second model from the same recipe, trained without any part of pakka's own set. It checks that the gain is not
v1's writing style leaking in (the v1 lesson). Cells: right / wrong answers served from cache (false-hit rate, 95%
upper bound).

**258 messages real people wrote** (pakka's human test split, 80% romanized Hinglish; never trained on), at
pakka's library settings: similarity 0.6 with a verifier, Jev passing at 0.84:

| Setup | Base MiniLM-L12 | This model | Same recipe, never saw v1 | Qwen3-Embedding-0.6B (reference) |
|---|---|---|---|---|
| Free, 10% budget | 2 / 1 | 9 / 2 | 8 / 1 | 11 / 0 |
| + Jev (`jev-1.14.0`, ≥ 0.84) | 19 / 0 (0.0%, 16.8%) | **54 / 1 (1.8%, 9.6%)** | 54 / 0 (0.0%, 6.6%) | 80 / 1 (1.2%, 6.7%) |
| + cheap LLM (`qwen/qwen3.7-flash`, yes / no) | 23 / 0 (0.0%, 14.3%) | 76 / 9 (10.6%, 18.9%) | 74 / 5 (6.3%, 14.0%) | 113 / 9 (7.4%, 13.4%) |
| Hinglish caught, + Jev | 10 / 184 | 44 / 184 | 43 / 184 | 67 / 184 |

Free, at the same number of answers, it is no better on real messages: AUC 0.603 → 0.589, and its 50
top-scoring answers hold 11 wrong against the base model's 7. What changes is that **165 of 258 real messages now
score above the verifier's 0.6 floor, against 26 for the base model**, so the verifier gets to see them.

On the leave-one-dataset-out runs and the public sets no training run touched (free, guards, 5% budget), it is
mixed:

| Held-out set | AUC base → tuned | Base | Tuned | False hits in the top tenth, base → tuned |
|---|---|---|---|---|
| Hinglish-TOP (left out) | 0.222 → 0.257 | none fits | none fits | 63.7% → 56.6% |
| MASSIVE en+hi (left out) | 0.453 → 0.455 | none fits | none fits | 26.5% → 28.3% |
| LangCache pairs (left out) | 0.595 → 0.591 | 8,715 / 4,674 | 8,649 / 4,730 | 34.5% → 35.3% |
| Bitext (never trained on) | 0.710 → 0.720 | 163 / 1 | 614 / 20 (3.2%, 4.8%) | 13.8% → 10.2% |
| Banking77 (never trained on) | 0.624 → 0.604 | 84 / 4 | 88 / 5 | 13.1% → 13.8% |
| CLINC150 (never trained on) | 0.692 → 0.650 | 404 / 19 | none fits | 4.9% → 9.5% |
| vCache Classification (never trained on) | - | 7,654 / 190 | 7,680 / 186 | - |

The verifier was not run on the public sets with this model.

## Training

- **Data** (about 7.7 MB): 11,527 (FAQ question, paraphrase, trap) triplets from the dev splits of Hinglish-TOP
  (2,048), MASSIVE en+hi (9,296) and half of pakka's v1 dev set (183), plus 21,064 labelled pairs from LangCache's
  train split (up to 3,000 per source). Traps are the same request with other slot values ("alarm 6 baje" against
  "alarm 7 baje") or v1's hand-written traps. Any training text that also appears in any test split of the
  benchmark, or in the human-written messages, was dropped. Hinglish-TOP's and MASSIVE's Hugging Face train
  splits were not used directly, because pakka's test split reuses half of their rows.
- **Recipe:** one epoch, learning rate 2e-5, batch 64, MultipleNegativesRankingLoss on triplets (no duplicate text
  in a batch) plus CoSENT on the pairs. 1.4 minutes on a Kaggle T4. Script: `scripts/phase9_kaggle.py`; data:
  `scripts/phase9_data.py`; measurements: `scripts/phase9_eval.py` and `pakka bench`.

## Limits

- **On its own it does not help.** Use it with a verifier, or keep the base model.
- With the cheap yes / no LLM verifier it serves more wrong answers than with Jev (9 of 85 on real messages), and
  that is above a 5% budget. Jev's score can be tightened; the LLM's yes / no can't.
- It is worse than its base on CLINC150 and slightly worse on Banking77, MASSIVE and LangCache, without a verifier.
- The human test set is small (258 messages, 23 writers, one shop's 30 FAQs), so the 95% bounds are wide.
- Thresholds for this model were picked on the half of v1 dev it was not trained on (252 questions).
