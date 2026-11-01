"""Chat with Dukaan Bot in the terminal: python -m examples.dukaan_bot.chat"""

from examples.dukaan_bot.app import build


def main():
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
        r = answerer.ask(question)
        score = "-" if r.score is None else f"{r.score:.2f}"
        tag = f"HIT {r.matched_id or 'stored'}" if r.hit else f"MISS ({r.reason})"
        print(f"bot> {r.answer}\n     [{tag} | similarity {score} | {r.latency_ms:.0f} ms | {r.tokens} tokens]\n")


if __name__ == "__main__":
    main()
