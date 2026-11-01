import pytest

from pakka.cli import main


@pytest.mark.parametrize("command", ["bench", "review", "approve"])
def test_each_subcommand_has_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_no_or_unknown_command_lists_the_commands(capsys):
    for argv in ([], ["frobnicate"]):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
        assert "bench" in capsys.readouterr().err
