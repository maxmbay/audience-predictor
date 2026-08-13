from audience_predictor.cli import main


def test_cli_help(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "coverage" in capsys.readouterr().out
