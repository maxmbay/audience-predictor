from audience_predictor.cli import main


def test_cli_help(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "coverage" in out
    assert "ingest-moviesum" in out
    assert "ingest-sublens" in out
    assert "predict" in out
