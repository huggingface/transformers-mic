import tomllib
from pathlib import Path

import transformers_mic
from transformers_mic import _version


def test_version_comes_from_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert transformers_mic.__version__.split("+")[0] == pyproject["project"]["version"]


def test_git_hash_is_appended_once():
    assert _version._append_git_hash("1.2.3", "abc1234") == "1.2.3+gabc1234"
    assert _version._append_git_hash("1.2.3+gabc1234", "def5678") == "1.2.3+gabc1234"
    assert _version._append_git_hash("1.2.3", None) == "1.2.3"
