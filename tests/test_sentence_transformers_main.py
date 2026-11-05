import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installer():
    path = (
        Path(__file__).parents[1]
        / ".github/scripts/install_sentence_transformers_main.py"
    )
    spec = importlib.util.spec_from_file_location("st_main_installer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_install_resolves_new_heads_and_verifies_installed_commits(
    tmp_path, monkeypatch, installer
):
    calls = []
    current = {"sha": "a" * 40}
    installed = {}

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            assert command[1:] in [
                ["ls-remote", url, "refs/heads/main"]
                for url in installer.PROJECTS.values()
            ]
            return SimpleNamespace(stdout=current["sha"] + "\trefs/heads/main\n")
        if "install" in command:
            assert "--upgrade" in command and "--no-cache-dir" in command
            for package, url in installer.PROJECTS.items():
                assert f"{package} @ git+{url}@{current['sha']}" in command
                installed[package] = {
                    "url": url,
                    "vcs_info": {"commit_id": current["sha"]},
                }
        return SimpleNamespace(stdout="transformers @ git+source\n")

    def distribution(package):
        return SimpleNamespace(read_text=lambda name: json.dumps(installed[package]))

    monkeypatch.setattr(installer.subprocess, "run", run)
    monkeypatch.setattr(installer.importlib.metadata, "distribution", distribution)
    for sha in ["a" * 40, "b" * 40]:
        current["sha"] = sha
        installer.install(tmp_path)
        assert {
            pin["commit"]
            for pin in json.loads(
                (tmp_path / "upstream-commits.json").read_text()
            ).values()
        } == {sha}
    assert len([c for c in calls if c[0] == "git"]) == 4
    assert (tmp_path / "pip-freeze.txt").exists()


@pytest.mark.parametrize(
    "provenance",
    [
        {},
        {"url": "wrong", "vcs_info": {"commit_id": "a" * 40}},
        {"url": "correct", "vcs_info": {"commit_id": "old"}},
    ],
)
def test_stale_or_release_install_is_rejected(monkeypatch, installer, provenance):
    monkeypatch.setattr(
        installer.importlib.metadata,
        "distribution",
        lambda package: SimpleNamespace(read_text=lambda name: json.dumps(provenance)),
    )
    with pytest.raises(ValueError, match="not the resolved main commit"):
        installer.verify_installed(
            {"transformers": {"url": "correct", "commit": "a" * 40}}
        )


def test_unresolved_main_stops_before_install(monkeypatch, installer, tmp_path):
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="")
    )
    with pytest.raises(ValueError, match="Could not resolve"):
        installer.install(tmp_path)
    assert not (tmp_path / "pip-freeze.txt").exists()
