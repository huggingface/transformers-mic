import io
import json
import subprocess
import sys
import pytest

from transformers_mic import batch, cloud_progress


def test_relay_preserves_carriage_returns_instead_of_appending_lines(
    tmp_path, monkeypatch
):
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        return real_popen(
            [sys.executable, "-c", "import os; os.write(2, b'first\\rsecond\\n')"],
            **kwargs,
        )

    monkeypatch.setattr(batch.subprocess, "Popen", popen)
    output = io.StringIO()
    monkeypatch.setattr(batch.sys, "stderr", output)
    result = batch.invoke(tmp_path, "record", ["baselines"], 5)
    assert result["exit_code"] == 0
    assert output.getvalue() == "first\rsecond\n"
    assert "first\nsecond" not in output.getvalue()


def test_parent_animation_advances_timer_without_another_server_poll(monkeypatch):
    class TTY(io.StringIO):
        def isatty(self):
            return True

    output = TTY()
    monkeypatch.setattr(cloud_progress.sys, "stderr", output)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.delenv("TMIC_PROGRESS_EVENTS", raising=False)
    clock = [100.0]
    monkeypatch.setattr(cloud_progress.time, "monotonic", lambda: clock[0])
    live = cloud_progress.LiveProgress()
    live.update({"status": "queued", "jobs": [], "progress": []})
    assert "00:00" in output.getvalue()
    start = output.tell()
    clock[0] += 65
    live.render()
    update = output.getvalue()[start:]
    assert "01:05" in update and "\x1b[1A\r\x1b[J" in update
    assert "checked" not in output.getvalue()


def test_child_sends_only_changed_state_and_never_animation_frames(monkeypatch, capsys):
    monkeypatch.setenv("TMIC_PROGRESS_EVENTS", "1")
    job = {"status": "queued"}
    first = cloud_progress.display(job)
    message = capsys.readouterr().err
    assert message.startswith("MIC_CLOUD_VIEW ")
    assert json.loads(message.removeprefix("MIC_CLOUD_VIEW "))["status"] == "queued"
    assert "\x1b" not in message and "checked" not in message
    cloud_progress.display(job, first)
    assert capsys.readouterr().err == ""


def test_progress_events_are_rendered_and_saved_without_animation(
    tmp_path, monkeypatch
):
    real_popen = subprocess.Popen
    view = {"status": "queued", "jobs": [], "progress": []}
    script = (
        "import sys; print("
        + repr("MIC_CLOUD_VIEW " + json.dumps(view))
        + ", file=sys.stderr)"
    )

    def popen(command, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(batch.subprocess, "Popen", popen)
    output = io.StringIO()
    monkeypatch.setattr(batch.sys, "stderr", output)
    result = batch.invoke(tmp_path, "record", ["baselines"], 5)
    assert result["exit_code"] == 0
    assert "[cloud] queued" in output.getvalue()
    assert "MIC_CLOUD_VIEW" not in output.getvalue()
    log = (tmp_path / result["log"]).read_text()
    assert "MIC_CLOUD_VIEW" in log and "\x1b" not in log


def test_interrupt_requests_cancellation_only_after_explicit_yes(
    tmp_path, monkeypatch, capsys
):
    from transformers_mic import github_baselines as github
    from transformers_mic.workspace import write, read

    write(
        tmp_path / "cloud.json",
        {"transport": "github-test", "job": {"id": "job", "run_id": 123}},
    )
    monkeypatch.setattr(
        github,
        "api",
        lambda path: {
            "id": 123,
            "status": "in_progress",
            "html_url": "https://github.com/run/123",
        },
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda: "y")
    calls = []
    monkeypatch.setattr(github, "gh", lambda *args: calls.append(args))
    github.interrupt(tmp_path)
    assert calls == [
        (
            "api",
            "--method",
            "POST",
            "repos/huggingface/transformers-mic/actions/runs/123/cancel",
        )
    ]
    assert read(tmp_path / "cloud.json")["cancel_requested"] is True
    assert "Cancellation requested" in capsys.readouterr().err


def test_interrupt_no_or_noninteractive_leaves_remote_running(
    tmp_path, monkeypatch, capsys
):
    from transformers_mic import github_baselines as github
    from transformers_mic.workspace import write

    write(
        tmp_path / "cloud.json",
        {"transport": "github-test", "job": {"id": "job", "run_id": 123}},
    )
    monkeypatch.setattr(
        github,
        "api",
        lambda path: {
            "id": 123,
            "status": "queued",
            "html_url": "https://github.com/run/123",
        },
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    answers = []
    monkeypatch.setattr("builtins.input", lambda: answers.append(True) or "n")
    calls = []
    monkeypatch.setattr(github, "gh", lambda *args: calls.append(args))
    github.interrupt(tmp_path)
    github.interrupt(tmp_path, interactive=False)
    assert answers == [True]
    assert calls == []
    assert "Remote run continues" in capsys.readouterr().err


def test_interrupt_finished_run_does_not_prompt(tmp_path, monkeypatch):
    from transformers_mic import github_baselines as github
    from transformers_mic.workspace import write

    write(tmp_path / "cloud.json", {"transport": "github-test", "job": {"run_id": 123}})
    monkeypatch.setattr(github, "api", lambda path: {"status": "completed"})
    monkeypatch.setattr(
        "builtins.input", lambda: pytest.fail("must not prompt for finished run")
    )
    github.interrupt(tmp_path)


def test_cancelled_collection_is_archived_before_fresh_submission(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from transformers_mic import github_baselines as github
    from transformers_mic.workspace import write

    options = {
        "fleets": ["control"],
        "fixtures_only": True,
        "upload_repo": "",
        "transformers_ref": "main",
        "profiles": ["cpu", "cuda"],
    }
    write(
        tmp_path / "cloud.json",
        {
            "transport": "github-test",
            "request": options,
            "job": {"id": "old-job", "run_id": 123},
        },
    )
    (tmp_path / "partial.txt").write_text("preserve evidence")
    (tmp_path / ".cloud.lock").touch()
    monkeypatch.setattr(
        github, "api", lambda path: {"status": "completed", "conclusion": "cancelled"}
    )
    run_locked = github.run_locked

    def fresh(args, root):
        assert not (root / "cloud.json").exists()
        assert (root / "cancelled/old-job/cloud.json").exists()
        assert (
            root / "cancelled/old-job/partial.txt"
        ).read_text() == "preserve evidence"
        assert (root / ".cloud.lock").exists()
        return 42

    monkeypatch.setattr(github, "run_locked", fresh)
    args = SimpleNamespace(
        fleet="control",
        include_checkpoints=False,
        upload_repo="",
        transformers_ref="main",
    )
    assert run_locked(args, tmp_path) == 42


def test_cloud_failure_reports_worker_cause(tmp_path):
    from transformers_mic.github_baselines import failure_summary
    from transformers_mic.workspace import write

    write(
        tmp_path / "artifacts/cpu/collection.json",
        {
            "cases": [
                {
                    "status": "failed",
                    "variant": "bfloat16",
                    "target": "fixture",
                    "record": {
                        "failures": ["traceback tail\nRuntimeError: dtype mismatch"]
                    },
                }
            ]
        },
    )
    assert (
        failure_summary(tmp_path)
        == "cpu/bfloat16/fixture: RuntimeError: dtype mismatch"
    )
