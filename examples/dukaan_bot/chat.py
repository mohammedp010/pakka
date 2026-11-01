"""Chat with Dukaan Bot in the terminal: python -m examples.dukaan_bot.chat [--user ravi]"""

import argparse

from examples.dukaan_bot.app import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user", help="cache this user's LLM answers for them only (default: shared, needs approval)")
    user = p.parse_args().user
    answerer = build()
    print("Dukaan Bot ready. Type a question (Ctrl+C to quit).\n")
    while True:
        try:
            question = input("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not question:
            continue
        r = answerer.ask(question, partition=user)
        score = "-" if r.score is None else f"{r.score:.2f}"
        tag = f"HIT {r.matched_id or 'stored'}" if r.hit else f"MISS ({r.reason}), answer {r.stored}"
        print(f"bot> {r.answer}\n     [{tag} | similarity {score} | {r.latency_ms:.0f} ms | {r.tokens} tokens]\n")


if __name__ == "__main__":
    main()
