from smartops import __version__
from smartops.cli import main
from smartops.config import PATHS


def test_version():
    assert __version__


def test_paths_resolve_under_data_dir():
    assert PATHS.raw.parent == PATHS.data


def test_cli_info(capsys):
    main(["info"])
    assert "smartops" in capsys.readouterr().out
