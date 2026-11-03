import subprocess

from transformers_mic import terminal
from transformers_mic.terminal import failure


def test_gate_failure_shows_cause_threshold_and_log_instead_of_large_dictionaries():
    text = "COMMAND [build]\nW1025 warning\nrandom weights fail the gate: {'random': {'pass': False}, 'overfit': {'weakest': 'pixel_values:halve', 'threshold': 0.001, 'deltas': {'pixel_values:halve': 0.000001}}}"
    summary = failure(text)
    assert "not sensitive enough" in summary
    assert "pixel_values:halve" in summary and "required 0.001" in summary
    assert "'deltas'" not in summary and "COMMAND" not in summary


def test_traceback_shows_exception_without_stack_frames():
    text = 'Traceback (most recent call last):\n  File "usage.md", line 53\n    named[bad]\nKeyError: missing.parameter'
    assert failure(text) == "KeyError: missing.parameter"


def test_color_respects_terminal_and_no_color(monkeypatch):
    monkeypatch.setattr(terminal, "interactive", lambda stream=None: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert "\x1b[" in terminal.color("Question", "question")
    monkeypatch.setenv("NO_COLOR", "")
    assert terminal.color("Question", "question") == "Question"


def test_phase_reads_only_current_attempt(tmp_path):
    log = tmp_path / "build.log"
    log.write_text('MIC_PROGRESS {"phase":"Old build"}\n')
    offset = log.stat().st_size
    assert terminal.phase(log, offset) == ""
    with log.open("a") as handle:
        handle.write(
            'MIC_PROGRESS {"phase":"Fit fixture weights","detail":"step 20/500 max"}\n'
        )
    assert terminal.phase(log, offset) == "Fit fixture weights · step 20/500 max"


def test_prompt_collapses_long_details_and_indents_question(capsys):
    terminal.prompt(
        "Long question", "\n".join(f"detail {n}" for n in range(20)), "confirmation", {}
    )
    output = capsys.readouterr().out
    assert "  ? Long question" in output and "    detail 0" in output
    assert "detail 19" not in output
    assert "D=details" in terminal.choices("confirmation")


def test_choices_name_each_kind_s_extra_keys():
    assert "E=edit config" in terminal.choices("fixture-review")
    assert "D=show code" in terminal.choices("source-review")
    assert "R=reject" in terminal.choices("contract-repair")


def test_short_details_print_inline_instead_of_paging(monkeypatch, capsys):
    monkeypatch.setattr(terminal, "interactive", lambda stream=None: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    terminal.details("line one\nline two")
    assert "    line two" in capsys.readouterr().out


def test_editor_falls_back_like_git(monkeypatch):
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", "code --wait")
    assert terminal.editor_command() == ["code", "--wait"]
    monkeypatch.setenv("EDITOR", "true")  # what non-interactive tools export
    # No git core.editor configured.
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})()
    )
    monkeypatch.setattr(terminal.shutil, "which", lambda name: "/usr/bin/" + name if name == "vim" else None)
    assert terminal.editor_command() == ["vim"]


def test_edit_waits_for_enter_when_the_editor_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal, "editor_command", lambda: ["missing-editor-binary"])
    waited = []
    monkeypatch.setattr("builtins.input", waited.append)
    terminal.edit(tmp_path / "config.json")
    assert waited and "press Enter" in waited[0]
