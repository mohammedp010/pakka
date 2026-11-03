"""Rule guards: cheap checks that veto a cache hit when the two questions can't want the same answer.

`check(query, cached_question, cached_answer)` returns None when the hit may be served, or the reason
it may not: "negation", "slot", "action", "direction" or "entity". Embeddings score these pairs as
near-identical, so the score alone can't catch them.

The rules read single words, so they only judge FAQ-length questions: in a long prompt one different word
means little, and on public datasets every rule vetoed good hits there about as often as bad ones.
"""

import re
import unicodedata

MAX_WORDS = 24  # longer questions are left to the score

_EN_NEG = (r"not|no|non|never|none|nothing|without|cannot|nor|\w+n't|dont|didnt|doesnt|isnt|wasnt|werent|cant|wont"
           r"|havent|hasnt|hadnt|arent|shouldnt|couldnt|wouldnt|aint")
# Romanized Hindi spellings. shortcut: "nai" can also mean "new"; drop it if that costs real hits.
_HINGLISH_NEG = r"nahi|nahin|nahee|nahii|nhi|nhin|nai|mat|bina"
_NEGATION = re.compile(rf"\b(?:{_EN_NEG}|{_HINGLISH_NEG})\b")
# Devanagari has no \b for whole words, so require a non-Devanagari character (or edge) on both sides.
_NEGATION_HI = re.compile(r"(?<![ऀ-ॿ])(?:नहीं|नही|मत|बिना)(?![ऀ-ॿ])")

# Yes/no tails ("or not", "ya nahi", "nhi h kya") ask a question; they don't flip its meaning.
_TAGS = re.compile(r"\bor not\b|\bya (?:nahi|nahin|nhi|nai)\b|\b(?:nahi|nahin|nhi) (?:hai|h|he) (?:kya|kyaa)\b|या नहीं")
# Neither do hedges ("I don't know how to...", "pata nahi") or failure reports ("can't pay", "won't activate",
# "isn't working"): they ask about the same thing as the plain question. "cannot be returned" still flips.
# "don't khow", "dont knwo": any word after "don't" that starts with k but "keep" is "know" misspelled.
_NEUTRAL = re.compile(r"\b(?:do not|don't|dont|didn't|didnt|did not|d not) (?!keep\b)k\w*|\bnot sure\b|\bno idea\b"
                      r"|\bno matter\b|\b(?:pata|samajh|samaj|samjh|smjh) (?:nahi|nahin|nhi|nai)\b|पता नहीं"
                      r"|\b(?:can't|cannot|cant|can not|could not|couldn't|couldnt|unable|won't|wont)\b(?! be\b)"
                      r"|\bnot (?:\w+ ){0,2}able\b"
                      r"|\b(?:not|\w+n't|doesnt|isnt|didnt|hasnt|havent|wasnt|arent)\b(?: [\w']+){0,4}? work(?:s|ed|ing)?\b"
                      r"|\b(?:nahi|nhi|nahin) (?:ho|chal) (?:raha|rahi|rha|rhi)\b|नहीं (?:हो|चल) रह")

# People put "nahi" or "not" in side clauses that set the scene ("online pay nahi karunga, COD ka option do",
# "I do not want this item, cancel my order"). Such a negation is about something else, so it doesn't flip what
# they ask. Clauses end at punctuation or a "but"/"because" word; the stopwords say nothing about a topic.
_CLAUSE = re.compile(r"[,.;:!?।]+|\b(?:but|lekin|magar|because|kyunki|kyuki|kyonki|however|although|though)\b")
_STOPWORDS = set("""
the and are was were been have has had can could will would should shall may might must for from with about its
this that these those there here what which who how when where why any some get got want need please just also yet
still then than our you your they them their his her one ask tell know like help helpful
hai hain hota hoti hote hua hui hue tha thi kya kyaa mein mai main mujhe muje mera meri mere aap aapka aapko apna
apne hum yeh woh kar karna karni karo karu karun kare karein kardo krna krdo raha rahi rahe rha rhi gaya gayi gaye
diya liya dena lena sakta sakti sakte chahiye bhi toh abhi aur par sir bhai yaar kaise kese kab kahan kaha kitna
kitne kitni kyun kyu kyon koi kuch sab bas wala wali wale hoga hogi hoge
""".split())  # words of 2 letters or fewer never count

# Numbers, order IDs, amounts and UPI IDs / emails: if they differ, so does the answer.
_SLOTS = re.compile(r"[\w.\-]+@[\w.\-]+|\d+")
_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_WORD = re.compile(r"(?:[^\W\d_]|[ऀ-ॣॱ-ॿ])+")  # letters, Devanagari vowel signs included (\w drops them)
_EN_NUMBERS = {w: str(n) for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                               "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_EN_NUMBERS.update({w: str(n * 10) for n, w in enumerate("thirty forty fifty sixty seventy eighty ninety".split(), 3)})
_EN_NUMBERS.update({w: str(n) for n, w in enumerate("first second third fourth fifth sixth seventh eighth ninth tenth".split(), 1)})
# A query's own number words are slots only when unambiguous: not "one", "first" or "second".
_QUERY_NUMBERS = {w for w, n in _EN_NUMBERS.items() if int(n) > 1} - {"second"}
# Hinglish and Hindi number words only ever excuse a number ("do" also means "give").
_NUMBER_WORDS = {**_EN_NUMBERS, **{w: str(n) for n, ws in enumerate(
    ["", "ek एक", "do दो", "teen तीन", "char chaar चार", "paanch panch पांच पाँच", "chhe chah छह छः", "saat सात",
     "aath आठ", "nau नौ", "das दस"]) for w in ws.split()}}

# Days and times of day are slots too ("alarm 6 baje" today vs tomorrow). Each word names a set of values in
# a group; a hit is vetoed when both questions name the group with no value in common. "kal" is tomorrow or
# yesterday, "tonight" is today and night.
_TIMES = {}
for _group, _words_of in {
    "day": {"today": "today aaj आज", "tomorrow": "tomorrow", "yesterday": "yesterday", "tomorrow yesterday": "kal कल",
            "dayafter": "parso parson परसों", "today tonight": "tonight", "weekend": "weekend weekends",
            "monday": "monday mondays somvar somwar सोमवार", "tuesday": "tuesday tuesdays mangalvar mangalwar मंगलवार",
            "wednesday": "wednesday wednesdays budhvar budhwar बुधवार",
            "thursday": "thursday thursdays guruvar guruwar गुरुवार बृहस्पतिवार",
            "friday": "friday fridays shukravar shukrawar शुक्रवार",
            "saturday": "saturday saturdays shanivar shaniwar शनिवार", "sunday": "sunday sundays ravivar raviwar itwar रविवार"},
    "part": {"morning": "morning mornings subah सुबह", "afternoon": "afternoon afternoons dopahar दोपहर",
             "evening": "evening evenings shaam sham शाम", "night tonight": "night nights raat रात tonight"},
    "meridiem": {"am": "am subah सुबह", "pm": "pm shaam sham शाम"},  # "subah 7 baje" is 7 am
    # "march" and "may" are left out: as verbs they are far more common in questions
    "month": {m: m for m in "january february april june july august september october november december".split()},
}.items():
    for _values, _ws in _words_of.items():
        for _w in _ws.split():
            _TIMES.setdefault(_w, []).append((_group, set(_values.split())))
_MERIDIEM = re.compile(r"\b([ap])\. ?m\.?")

# Actions that read alike but need different answers; each side lists its word forms.
_OPPOSITES = [("block", "unblock"), ("lock", "unlock"), ("subscribe", "unsubscribe"), ("link", "unlink"),
              ("activate", "deactivate"), ("enable", "disable"), ("delete", "deactivate"),
              ("add", "remove"), ("start", "stop"), ("upgrade", "downgrade"), ("increase", "decrease"),
              ("buy buying bought kharid kharidna kharido khareed khareedna खरीद खरीदना खरीदें",
               "sell selling sold bech bechna becho बेच बेचना बेचें"),
              ("open opening", "close closing"), ("import", "export"), ("upload", "download"),
              ("deposit", "withdraw withdrawal"), ("credit credited", "debit debited"), ("login", "logout")]
_OPPOSITES = [(set(a.split()), set(b.split())) for a, b in _OPPOSITES]
# "on" is mostly a preposition, so on/off counts only when one side says off and the other on.
_OFF = re.compile(r"\boff\b|\bband (?:karo|kar|kr|kardo|krdo|ho|hai)\b|बंद|ऑफ")
_ON = re.compile(r"\bon\b|\bchalu\b|\bchaalu\b|चालू|ऑन")

# Direction: "jpg to pdf" vs "pdf to jpg", "Delhi se Mumbai" vs "Mumbai to Delhi" (se = from).
_CONNECTORS = re.compile(r"\b(?:to|into|than|se|instead of)\b|(?<![ऀ-ॿ])से(?![ऀ-ॿ])")
_SAME_CONNECTOR = {"into": "to", "se": "to", "से": "to", "instead of": "instead"}
_FILLER = {"the", "a", "an", "my", "your", "i", "me", "how", "do", "convert", "converting", "conversion", "change",
           "from", "go", "get", "is", "it", "can", "you", "and", "of", "in", "on", "for", "ka", "ki", "ke", "ko",
           "mein", "want", "need", "way", "ways", "what", "best", "better", "more", "less", "rather", "use"}

# Scope words that move a question outside what a domestic FAQ answer covers.
_SCOPE = {"country", "countries", "international", "abroad", "overseas", "outside", "foreign", "videsh"}
_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")  # placeholders such as {{Order Number}}


def _polar(text):
    """Lowercase, with yes/no tails, hedges and failure reports blanked: a negation still left flips the meaning."""
    return _NEUTRAL.sub(" ", _TAGS.sub(" ", text.lower().replace("’", "'")))


def _negated(text):
    text = _polar(text)
    return bool(_NEGATION.search(text) or _NEGATION_HI.search(text))


def _topics(text):
    """Content words cut to 5 letters, so "cancel" meets "cancelling" and "deliver" meets "delivery"."""
    text = _NEGATION_HI.sub(" ", _NEGATION.sub(" ", text.lower()))  # "don't" must not leave "don" behind
    return {w[:5] for w in _WORD.findall(text) if len(w) > 2 and w not in _STOPWORDS}


def _share(a, b):
    # short words (pin, cod, emi) must match exactly; longer ones on a common prefix
    return any(x == y or (min(len(x), len(y)) > 3 and (x.startswith(y) or y.startswith(x))) for x in a for y in b)


def _negation_about(text, other):
    """Is a negation in `text` about what it asks in common with `other`? A negated clause must use a word `other`
    uses too, or name nothing at all ("Plus membership? nahi chahiye"). Texts with no word in common (Hindi vs
    English) give nothing to tell by, so their negation counts."""
    about = _topics(other)
    if not _share(_topics(text), about):
        return True
    return any((_NEGATION.search(c) or _NEGATION_HI.search(c)) and (not _topics(c) or _share(_topics(c), about))
               for c in _CLAUSE.split(_polar(text)))


def _tokens(text):
    return set(_WORD.findall(_MERIDIEM.sub(r"\1m", text.lower())))


def _numbers(text, query=False):
    """Numbers in `text`: digits (any script), plus number words; a query's own words only when unambiguous."""
    text = text.lower().translate(_DIGITS)
    words = _QUERY_NUMBERS if query else _NUMBER_WORDS
    return set(_SLOTS.findall(text)) | {_NUMBER_WORDS[w] for w in _WORD.findall(text) if w in words}


def _times(text):
    found = {}
    for word in _tokens(text):
        for group, values in _TIMES.get(word, ()):
            found.setdefault(group, set()).update(values)
    return found


def _time_clash(query, known):
    k = _times(known)
    return any(group in k and not values & k[group] for group, values in _times(query).items())


def _opposite(query, cached):
    q, c = _tokens(query), _tokens(cached)
    for a, b in _OPPOSITES:
        if (a & q and b & c and not b & q and not a & c) or (b & q and a & c and not a & q and not b & c):
            return True
    off_q, off_c = bool(_OFF.search(query.lower())), bool(_OFF.search(cached.lower()))
    return off_q != off_c and bool(_ON.search((cached if off_q else query).lower()))


def _sides(text):
    """(connector, words just before, words just after) for every direction word in `text`."""
    marked = _CONNECTORS.sub(lambda m: f" →{_SAME_CONNECTOR.get(m.group(), m.group())} ", text.lower())
    tokens = re.findall(rf"{_WORD.pattern}|\d+|→\w+", marked)
    words = lambda part: [w for w in part if w not in _FILLER and not w.startswith("→")]  # noqa: E731
    return [(t, set(words(tokens[max(0, i - 3):i])[-2:]), set(words(tokens[i + 1:i + 4])[:2]))
            for i, t in enumerate(tokens) if t.startswith("→")]


def _swapped(query, cached):
    """A word before the connector in one question is after it in the other, and another word moved the other way."""
    sides_c = _sides(cached)
    for conn, before, after in _sides(query):
        for conn_c, before_c, after_c in sides_c:
            x, y = (before - after) & (after_c - before_c), (after - before) & (before_c - after_c)
            if conn == conn_c and x and y and x != y and not any(
                    w == conn and x & b and y & a for w, b, a in sides_c):  # the cached text says it both ways
                return True
    return False


def _plain(word):
    """Lowercase, without accents: Zürich and Zurich match."""
    return unicodedata.normalize("NFKD", word.lower()).encode("ascii", "ignore").decode() or word.lower()


def _names(text):
    """Capitalised words that don't start a sentence (likely names: Nepal, Flipkart), plus scope words. A sentence
    in Title Case (a headline or product title: no lowercase word longer than three letters) has no names."""
    names = set()
    for sentence in re.split(r"[.?!:;\n\"(]+", _TEMPLATE.sub(" ", text)):
        words = re.findall(r"[^\W\d_][^\W\d_']*", sentence)
        if len(words) < 4 or any(w[0].islower() and len(w) > 3 for w in words):
            names |= {w.lower() for w in words[1:] if w[0].isupper() and w != "I"}
    return names | (set(re.findall(r"[a-z]+", text.lower())) & _SCOPE)


def _unknown_names(query, known):
    """Names in the query that the cached question and answer never mention (matching on the first five letters,
    so Chechnya/Chechen and Parkways/Parkway count as mentioned)."""
    names = _names(query)
    seen = {_plain(w) for w in re.findall(r"[^\W\d_]+", known)} if names else ()
    return {n for n in names if not any(k.startswith(_plain(n)[:5]) for k in seen)}


def check(query, cached, answer=""):
    if max(len(query.split()), len(cached.split())) > MAX_WORDS:
        return None
    known = f"{cached} {answer}"
    q_neg, c_neg = _negated(query), _negated(cached)
    if q_neg != c_neg and _negation_about(*((query, cached) if q_neg else (cached, query))):
        return "negation"
    if _numbers(query, query=True) - _numbers(known) or _time_clash(query, known):
        return "slot"
    if _opposite(query, cached):
        return "action"
    if _swapped(query, cached):
        return "direction"
    if _unknown_names(query, known):
        return "entity"
    return None
