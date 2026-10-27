"""Check test-helper retry behavior without submitting real GPU work."""

from types import SimpleNamespace
import subprocess

import pytest

from transformers_mic import github_baselines as helper
from transformers_mic.workspace import read, write


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


def test_watcher_reports_actual_step_and_queue_without_native_watch(monkeypatch):
    calls, views = [], []
    states = iter(
        [
            {
                "status": "in_progress",
                "conclusion": None,
                "html_url": "https://github.com/run",
            },
            {
                "status": "completed",
                "conclusion": "success",
                "html_url": "https://github.com/run",
            },
        ]
    )

    def api(path):
        calls.append(path)
        if "/jobs?" in path:
            return {
                "jobs": [{"name": "measure (cpu)", "status": "queued", "steps": []}]
            }
        return next(states)

    monkeypatch.setattr(helper, "api", api)
    monkeypatch.setattr(helper.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        helper.cloud_progress, "display", lambda view, previous: views.append(view)
    )
    monkeypatch.setattr(
        helper.subprocess,
        "run",
        lambda *a, **k: pytest.fail("native watch repeats job sections"),
    )
    assert helper.watch(123)["conclusion"] == "success"
    assert len(views) == 2 and views[0]["workflow_jobs"][0]["total_steps"] == 0
    assert all("123" in call for call in calls)


def test_download_retries_a_dropped_connection_on_a_clean_directory(
    tmp_path, monkeypatch
):
    calls = []

    def run(command, check):
        calls.append(sorted(p.name for p in tmp_path.iterdir()))
        (tmp_path / "partial").mkdir(exist_ok=True)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(helper.subprocess, "run", run)
    monkeypatch.setattr(helper.time, "sleep", lambda seconds: None)
    helper.download(7, tmp_path)
    assert calls == [[], []]
