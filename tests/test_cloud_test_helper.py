"""Check test-helper retry behavior without submitting real GPU work."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from transformers_mic.workspace import read, write

spec = importlib.util.spec_from_file_location(
    "cloud_test_helper", Path(__file__).parents[1] / "tools/try_cloud_baselines.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def args():
    return SimpleNamespace(
        fleet="control",
        include_checkpoints=False,
        upload_repo="",
        transformers_ref="main",
    )


def state():
    options = {
        "fleets": ["control"],
        "fixtures_only": True,
        "upload_repo": "",
        "transformers_ref": "main",
        "profiles": ["cpu", "cuda"],
    }
    return {
        "transport": "github-test",
        "request": options,
        "job": {"id": "a" * 32},
        "dispatch_attempted": True,
    }


def test_ambiguous_dispatch_is_not_repeated(tmp_path, monkeypatch):
    write(tmp_path / "cloud.json", state())
    monkeypatch.setattr(helper, "find_run", lambda _: None)
    monkeypatch.setattr(helper.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        helper.subprocess, "run", lambda *a, **k: pytest.fail("redispatch")
    )
    with pytest.raises(ValueError, match="Dispatch is unresolved"):
        helper.run_locked(args(), tmp_path)


def test_dispatch_attempt_saved_before_network_error(tmp_path, monkeypatch):
    existing = state()
    del existing["dispatch_attempted"]
    existing["job"].update(
        fleet_revision="b" * 40,
        mic_sha="c" * 40,
        transformers_sha="d" * 40,
        fleets=["control"],
        fixtures_only=True,
    )
    write(tmp_path / "cloud.json", existing)
    monkeypatch.setattr(helper, "find_run", lambda _: None)

    def fail(*a, **kw):
        assert read(tmp_path / "cloud.json")["dispatch_attempted"] is True
        raise subprocess.CalledProcessError(1, a[0])

    monkeypatch.setattr(helper.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        helper.run_locked(args(), tmp_path)
    assert read(tmp_path / "cloud.json")["job"]["mic_sha"] == "c" * 40


def test_changed_options_cannot_resume_existing_run(tmp_path):
    write(tmp_path / "cloud.json", state())
    changed = args()
    changed.include_checkpoints = True
    with pytest.raises(ValueError, match="Different collection/options"):
        helper.run_locked(changed, tmp_path)
