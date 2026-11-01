"""`pakka <command>`: bench a cache, review generated labels, approve queued answers."""

import sys

COMMANDS = {
    "bench": ("pakka.bench", "measure hit rate and false hits on a labelled question set"),
    "review": ("pakka.review", "settle labels two LLMs disagreed on"),
    "approve": ("pakka.approve", "approve answers waiting to enter the shared cache"),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        listing = "\n".join(f"  {name:8s} {help}" for name, (_, help) in COMMANDS.items())
        print(f"usage: pakka <command> [options]\n\ncommands:\n{listing}", file=sys.stderr)
        sys.exit(2)
    module, _ = COMMANDS[argv[0]]
    sys.argv[0] = f"pakka {argv[0]}"  # so each command's --help shows the right name
    __import__(module, fromlist=["main"]).main(argv[1:])
