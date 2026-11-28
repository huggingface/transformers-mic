import importlib.util
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("backfill", HERE / ".github/scripts/backfill.py")
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


def run(monkeypatch, table, installed, pinned):
    monkeypatch.setattr(backfill, "git", lambda repo, *args: table)
    monkeypatch.setattr(backfill.md, "version", lambda name: installed[name])
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd[5:]))
    return backfill.dependency_overrides("repo", "sha", pinned), calls


def test_a_git_installed_dev_pin_is_not_put_back(monkeypatch):
    # The image installs hf-doc-builder from git (0.6.0.dev0); an older commit overrode it to 0.6.0.
    table = '"hf-doc-builder": "hf-doc-builder>=0.3.0",'
    overrides, calls = run(monkeypatch, table, {"hf-doc-builder": "0.6.0"}, {"hf-doc-builder": "0.6.0.dev0"})
    assert overrides == {} and calls == []


def test_a_release_pin_is_still_put_back(monkeypatch):
    table = '"tokenizers": "tokenizers>=0.22",'
    overrides, calls = run(monkeypatch, table, {"tokenizers": "0.22.1"}, {"tokenizers": "0.23.1"})
    assert overrides == {} and calls == [["tokenizers==0.23.1"]]
