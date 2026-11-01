import json

from pakka.approve import main


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_yes_moves_to_approved_no_discards_and_quit_keeps_the_rest(tmp_path, monkeypatch):
    pending, approved = tmp_path / "pending.jsonl", tmp_path / "approved.jsonl"
    rows = [{"question": f"q{i}", "answer": f"a{i}", "partition": "global", "kb_version": "v1"} for i in range(3)]
    write(pending, rows)
    answers = iter(["y", "n", "q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    main([str(pending), "--to", str(approved)])
    assert [json.loads(line)["question"] for line in approved.read_text().splitlines()] == ["q0"]
    assert [json.loads(line)["question"] for line in pending.read_text().splitlines()] == ["q2"]
