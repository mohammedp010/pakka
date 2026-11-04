import math

from scripts import phase10b as b


def cand(sim, jev, llm, wrong, q="q"):
    return {"q": q, "label": "x", "id": "x", "cand": True, "sim": sim, "lookup_ms": 10.0, "wrong": wrong,
            "v": {"jev": (jev, 300.0), "m": (llm, 800.0), "m:cost": 0.0001}}


def test_p_yes_normalises_yes_against_no():
    top = [{"token": "yes", "logprob": math.log(0.6)}, {"token": " Yes", "logprob": math.log(0.2)},
           {"token": "no", "logprob": math.log(0.2)}]
    assert abs(b.p_yes(top) - 0.8) < 1e-9
    assert b.p_yes([{"token": "maybe", "logprob": 0.0}]) is None


def test_pick_serves_the_most_while_the_upper_bound_fits():
    # 200 right answers from 0.7 up, 10 wrong ones at low similarity: the pick must cut those off
    cands = [cand(0.7 + i / 1000, 0.99, 0.99, False) for i in range(200)] + [cand(0.61, 0.99, 0.99, True)] * 10
    sim, bars = b.pick(cands, [], lambda c: c["wrong"], 0.03)
    assert sim == 0.7 and bars == ()
    assert b.pick(cands[200:], [], lambda c: c["wrong"], 0.03) is None


def test_agree_needs_both_verifiers_and_waits_for_the_slower():
    cands = [cand(0.9, 0.95, 0.99, False), cand(0.9, 0.95, 0.1, True), cand(0.9, 0.2, 0.99, True)]
    m = b.measure(cands, ["jev", "m"], 0.6, (0.84, 0.5), lambda c: c["wrong"])
    assert m["hits"] == 1 and m["wrong"] == 0 and m["checks"] == 3
    assert m["ms_hits"] is None or m["ms_hits"][0] >= 800  # one hit: p50_p90 needs two
    assert b.served(cands, ["jev", "m"], 0.6, (0.84, 0.5)) == cands[:1]


def test_a_verifier_error_never_serves():
    c = cand(0.9, None, 0.99, False)
    assert not b.passes(c, ["jev"], (0.5,)) and b.passes(c, ["m"], (0.5,))
