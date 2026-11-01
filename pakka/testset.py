"""Generate a labelled cache test set from an FAQ with two different LLMs.

One model writes paraphrases and trap questions for each FAQ. A second model, ideally from another
family, labels every question without seeing the first model's intent. Where they disagree, the
item is flagged for a human to check. `chat(system, user) -> (text, tokens)` is any LLM call.
"""

import json
import random

from pakka.cache import normalize

LANGS = {"en", "hinglish", "hi"}
TRAP_KINDS = {"negation", "entity", "number", "near_intent", "personal"}
OTHER_KINDS = {"off_topic", "uncovered"}

GEN_SYSTEM = "You write test data for the semantic cache of an Indian e-commerce support bot. Reply with JSON only."

GEN_PROMPT = """All FAQ questions in the bot, so you know the neighbours:
{faq_list}

Target FAQ {id}:
Q: {question}
A: {answer}

Write JSON {{"paraphrases": [...], "hard_negatives": [...]}}.

paraphrases: 20 customer messages that this exact answer fully and correctly answers.
- 6 in English, 11 in romanized Hinglish, 3 in Hindi written in Devanagari.
- Sound like real customers on chat or WhatsApp: short, informal, some typos, varied word order and
  varied Hinglish spellings (nahi/nhi/nahin, kya/kyaa, kaise/kese, hai/h).
- Vary the wording a lot; do not just swap one word.
- No order numbers, names, phone numbers or other personal details.
Each item: {{"q": "...", "lang": "en" | "hinglish" | "hi"}}

hard_negatives: 10 messages that look or sound very close to the target question but must NOT get this
answer, because it would be wrong, incomplete or misleading for them.
- 3 "negation": change the meaning with not/no/never/nahi/mat/bina/didn't, or an opposite action.
- 2 "entity": a country, brand, product type or service this answer does not cover.
- 2 "number": a specific amount, day count, date or quantity this answer does not cover.
- 2 "near_intent": a related but different request.
- 1 "personal": needs this customer's own order, account or payment details.
Mix English and Hinglish. Each item: {{"q": "...", "lang": "...", "kind": "...", "why": "short reason"}}"""

OTHER_PROMPT = """These are all the FAQ questions an e-commerce support bot can answer:
{faq_list}

Write {n} customer messages that NONE of these FAQs answers:
- half "uncovered": shopping, product or account questions a store's support bot might get but this
  list does not answer;
- half "off_topic": unrelated messages (cricket, weather, coding, recipes, general knowledge, small talk).
Mix English, romanized Hinglish and some Hindi in Devanagari.
One message per line, nothing else, in this exact format:
kind | lang | message
where kind is uncovered or off_topic, and lang is en, hinglish or hi."""

JUDGE_SYSTEM = "You label test data for an e-commerce support bot's FAQ cache. Reply with JSON only."

JUDGE_PROMPT = """FAQ:
{faqs}

For each customer message below, give the id of the single FAQ whose answer fully and correctly answers
it as written, or "none" if no answer fits, if one only partly fits, or if it needs the customer's own
order, account or payment details. Be strict: a related topic is not enough.

Messages:
{messages}

Reply {{"labels": [...]}} with exactly {n} labels, in order."""


def parse_json(text):
    """Parse the first JSON object in an LLM reply, ignoring code fences and chatter."""
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def _faq_list(faqs):
    return "\n".join(f"{f['id']}: {f['question']}" for f in faqs)


def _clean(entry, kinds):
    q, lang, kind = str(entry.get("q", "")).strip(), entry.get("lang"), entry.get("kind")
    return q and lang in LANGS and kind in kinds


def generate_for_faq(faq, faqs, chat):
    prompt = GEN_PROMPT.format(faq_list=_faq_list(faqs), **faq)
    reply = parse_json(chat(GEN_SYSTEM, prompt)[0])
    items = [{"q": p["q"].strip(), "lang": p["lang"], "kind": "paraphrase", "near": faq["id"], "intended": faq["id"]}
             for p in reply.get("paraphrases", []) if _clean({**p, "kind": "paraphrase"}, {"paraphrase"})]
    items += [{"q": n["q"].strip(), "lang": n["lang"], "kind": n["kind"], "near": faq["id"], "intended": None}
              for n in reply.get("hard_negatives", []) if _clean(n, TRAP_KINDS)]
    return items


def generate_other(faqs, chat, n=100):
    # One "kind | lang | message" per line: a stray quote can't break a whole reply the way it breaks JSON.
    reply = chat(GEN_SYSTEM, OTHER_PROMPT.format(faq_list=_faq_list(faqs), n=n))[0]
    items = []
    for line in reply.splitlines():
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) == 3 and _clean({"kind": parts[0], "lang": parts[1], "q": parts[2]}, OTHER_KINDS):
            items.append({"q": parts[2], "lang": parts[1], "kind": parts[0], "near": None, "intended": None})
    return items


def judge(items, faqs, chat, batch=25):
    """The second model's label for each item: an FAQ id or None."""
    ids = {f["id"] for f in faqs}
    faq_text = "\n".join(f"{f['id']} | Q: {f['question']} | A: {f['answer']}" for f in faqs)
    labels = []
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        messages = "\n".join(f"{i}. {item['q']}" for i, item in enumerate(chunk, 1))
        reply = parse_json(chat(JUDGE_SYSTEM, JUDGE_PROMPT.format(faqs=faq_text, messages=messages, n=len(chunk)))[0])
        got = reply.get("labels", [])
        if len(got) != len(chunk):
            raise ValueError(f"judge returned {len(got)} labels for {len(chunk)} messages")
        labels += [x if x in ids else None for x in got]
    return labels


def reconcile(item, judged):
    """Final label plus a review flag whenever the two models disagree."""
    intended = item["intended"]
    if intended is not None:  # a paraphrase: the judge has to pick the same FAQ
        label, review = judged, judged != intended
    elif judged is None:  # a should-miss question both models agree on
        label, review = None, False
    elif judged == item.get("near"):  # judge says the trap's own FAQ answers it after all
        label, review = None, True
    else:  # the trap is really a question for another FAQ
        label, review = judged, True
    return {**item, "label": label, "judged": judged, "needs_review": review}


def assign_splits(items, seed=0):
    """Half of each (kind, lang) group goes to dev and half to test, reproducibly."""
    groups = {}
    for item in items:
        groups.setdefault((item["kind"], item["lang"]), []).append(item)
    rng, out = random.Random(seed), []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        out += [{**item, "split": "test" if i % 2 else "dev"} for i, item in enumerate(group)]
    return out


def dedupe(items):
    """Drop repeats after normalising case and punctuation; the first copy wins."""
    seen, out = set(), []
    for item in items:
        key = normalize(item["q"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
