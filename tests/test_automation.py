import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformers_mic import automation, batch
from transformers_mic.cli import main
from transformers_mic.workspace import Workspace, read

MODEL = "google-bert/bert-base-uncased"
SHA = "b" * 40


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    root = tmp_path / "project"
    calls = []
    failures = {}
    exports = []

    def invoke(out, name, arguments, timeout):
        calls.append((name, list(arguments)))
        log = out / "logs" / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        code = failures.get(name, []).pop(0) if failures.get(name) else 0
        payload = {"stage": name}
        if name == "create" and not code:
            Workspace(root).from_catalog(MODEL, "BERT", "auto", False)
        if name == "export-preview":
            payload = {
                "revision": "a" * 40,
                "contract": "BERT",
                "contracts": {
                    "BERT": {"integration": {"repo": "huggingface-mic/bert-mic"}}
                },
            }
        if name == "export" and not code:
            payload = {"revision": SHA}
            exports.append(SHA)
        if name == "pin" and not code:
            target = Path(arguments[arguments.index("--transformers-checkout") + 1])
            pin = target / "tests/contracts.lock"
            pin.parent.mkdir(parents=True, exist_ok=True)
            pin.write_text(SHA + "\n")
        log.write_text(json.dumps(payload) + "\n")
        return {"exit_code": code, "log": str(log.relative_to(out))}

    monkeypatch.setattr(batch, "invoke", invoke)
    monkeypatch.setattr(automation.sys.stdin, "isatty", lambda: True)
    return SimpleNamespace(
        root=root,
        calls=calls,
        failures=failures,
        exports=exports,
        argv=["run", MODEL, "--out", str(root), "--no-history"],
    )


def confirm(monkeypatch, replies):
    answers = iter(replies)
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))


def test_full_workflow_uploads_only_after_decisions_and_resumes_without_reupload(
    workflow, monkeypatch
):
    confirm(monkeypatch, ["yes", "yes", "yes"])
    target = workflow.root.parent / "transformers"
    args = [*workflow.argv, "--transformers-checkout", str(target)]
    assert main(args) == 0
    names = [name for name, _ in workflow.calls]
    assert names == [
        "create",
        "build",
        "auto",
        "record",
        "review",
        "check",
        "publish-preview",
        "publish",
        "published-check",
        "export-preview",
        "export",
        "pin",
    ]
    assert (target / "tests/contracts.lock").read_text() == SHA + "\n"
    state = read(workflow.root / ".mic/automation.json")
    assert state["status"] == "complete" and state["dataset_revision"] == SHA
    export = dict(workflow.calls)["export"]
    assert export[export.index("--expected-revision") + 1] == "a" * 40
    before = list(workflow.calls)
    assert main(args) == 0
    assert workflow.calls == before


def test_noninteractive_stops_at_source_and_records_resumable_question(
    workflow, monkeypatch, capsys
):
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("must not read stdin"))
    assert main([*workflow.argv, "--non-interactive"]) == 3
    state = read(workflow.root / ".mic/automation.json")
    assert state["next_stage"] == "source" and state["status"] == "needs-input"
    assert "Review the executable" in state["question"]
    assert "tmic run" in state["resume"]
    assert [name for name, _ in workflow.calls] == ["create"]
    assert (
        json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "needs-input"
    )


def test_local_only_checks_without_publication_and_can_resume_to_publish(
    workflow, monkeypatch
):
    confirm(monkeypatch, ["yes"])
    assert main([*workflow.argv, "--local-only"]) == 0
    assert [name for name, _ in workflow.calls][-1] == "check"
    assert read(workflow.root / ".mic/automation.json")["status"] == "validated"
    confirm(monkeypatch, ["yes", "yes"])
    assert main(workflow.argv) == 0
    assert [name for name, _ in workflow.calls].count("record") == 1
    assert len(workflow.exports) == 1


def test_declining_upload_preserves_validation_and_resumes_from_upload(
    workflow, monkeypatch
):
    confirm(monkeypatch, ["yes", "no"])
    assert main(workflow.argv) == 3
    assert not workflow.exports
    assert read(workflow.root / ".mic/automation.json")["next_stage"] == "publish"
    confirm(monkeypatch, ["yes", "yes"])
    assert main(workflow.argv) == 0
    assert [n for n, _ in workflow.calls].count("check") == 1


def test_source_edits_invalidate_prior_progress(workflow, monkeypatch):
    confirm(monkeypatch, ["yes", "no"])
    assert main(workflow.argv) == 3
    workspace = Workspace(workflow.root)
    usage = workspace.contract_root / "integrations/transformers/usage.md"
    usage.write_text(usage.read_text() + "\n# reviewed fix\n")
    confirm(monkeypatch, ["yes", "yes", "yes"])
    assert main(workflow.argv) == 0
    assert [n for n, _ in workflow.calls].count("record") == 2


def test_check_failure_never_uploads_and_resumes_same_stage(workflow, monkeypatch):
    workflow.failures["check"] = [1]
    confirm(monkeypatch, ["yes", "no"])
    assert main(workflow.argv) == 3
    assert not any(n == "publish" for n, _ in workflow.calls)
    assert read(workflow.root / ".mic/automation.json")["next_stage"] == "check"
    confirm(monkeypatch, ["yes", "yes"])
    assert main(workflow.argv) == 0
    assert [n for n, _ in workflow.calls].count("check") == 2
    assert [n for n, _ in workflow.calls].count("record") == 1


def test_dataset_conflict_requires_fresh_preview_and_new_decision(
    workflow, monkeypatch
):
    workflow.failures["export"] = [2]
    confirm(monkeypatch, ["yes", "yes", "yes", "yes", "yes"])
    assert main(workflow.argv) == 0
    names = [n for n, _ in workflow.calls]
    assert names.count("export-preview") == 2 and names.count("export") == 2
    assert names.count("publish") == 1
    assert len(workflow.exports) == 1


def test_new_checkout_pin_does_not_republish_completed_run(workflow, monkeypatch):
    confirm(monkeypatch, ["yes", "yes", "yes"])
    assert main(workflow.argv) == 0
    before = list(workflow.calls)
    target = workflow.root.parent / "other-transformers"
    assert main([*workflow.argv, "--transformers-checkout", str(target)]) == 0
    assert workflow.calls[:-1] == before
    assert workflow.calls[-1][0] == "pin"
    assert (target / "tests/contracts.lock").read_text().strip() == SHA


def test_changed_baseline_needs_reason_and_keeps_hash_receipt(workflow, monkeypatch):
    Workspace(workflow.root).from_catalog(MODEL, "BERT", "auto", False)
    workspace = Workspace(workflow.root)
    baseline = next(workspace.expectations.rglob("*.json"))
    candidate = baseline.with_name(baseline.stem + ".candidate.json")
    original = baseline.read_text()
    replacement = original + "\n"
    candidate.write_text(replacement)
    args = SimpleNamespace(non_interactive=False)
    confirm(monkeypatch, ["yes", ""])
    with pytest.raises(automation.Pending):
        automation.review_changed(args, workspace)
    assert baseline.read_text() == original and candidate.exists()
    confirm(monkeypatch, ["yes", "Measured change in the checked runtime"])
    automation.review_changed(args, workspace)
    assert baseline.read_text() == replacement and not candidate.exists()
    receipts = read(workflow.root / ".mic/automation-reviews.json")
    assert receipts[-1]["reason"] == "Measured change in the checked runtime"
    assert receipts[-1]["old_sha256"] != receipts[-1]["new_sha256"]
    assert workspace.state["validated"] is False


def test_review_path_requires_human_before_check(workflow, monkeypatch):
    Workspace(workflow.root).from_catalog(MODEL, "BERT", "auto", False)
    workspace = Workspace(workflow.root)
    baseline = next(workspace.expectations.rglob("*.json"))
    candidate = baseline.with_name(baseline.stem + ".candidate.json")
    candidate.write_text(baseline.read_text() + "\n")
    workflow.failures["review"] = [1]
    confirm(monkeypatch, ["yes", "no"])
    assert main(workflow.argv) == 3
    assert candidate.exists()
    assert not any(n == "check" for n, _ in workflow.calls)


def test_interrupt_stops_subprocess_group(tmp_path, monkeypatch):
    killed = []

    class Process:
        pid = 12345
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt
            return 0

    monkeypatch.setattr(batch.subprocess, "Popen", lambda *a, **kw: Process())
    monkeypatch.setattr(batch.os, "killpg", lambda *args: killed.append(args))
    with pytest.raises(KeyboardInterrupt):
        batch.invoke(tmp_path, "record", ["record"], 1)
    assert killed == [(12345, batch.signal.SIGTERM)]


def test_edit_during_upload_question_restarts_validation(workflow, monkeypatch):
    confirmations = []
    edited = False

    def reply(prompt):
        nonlocal edited
        current = read(workflow.root / ".mic/automation.json")["next_stage"]
        confirmations.append(current)
        if current == "publish" and not edited:
            path = (
                Workspace(workflow.root).contract_root
                / "integrations/transformers/usage.md"
            )
            path.write_text(path.read_text() + "\n# corrected before publication\n")
            edited = True
        return "yes"

    monkeypatch.setattr("builtins.input", reply)
    assert main(workflow.argv) == 0
    assert confirmations.count("source") == 2
    assert [n for n, _ in workflow.calls].count("check") == 2
    assert [n for n, _ in workflow.calls].count("publish") == 1


def test_destination_change_reuses_validated_published_model(workflow, monkeypatch):
    confirm(monkeypatch, ["yes", "yes", "yes"])
    assert main(workflow.argv) == 0
    confirm(monkeypatch, ["yes"])
    assert (
        main([*workflow.argv, "--fleet", "control", "--contract-name", "new-name"]) == 0
    )
    names = [n for n, _ in workflow.calls]
    assert names.count("publish") == 1 and names.count("check") == 1
    assert names.count("export") == 2


def test_different_model_never_overwrites_workspace(workflow, monkeypatch):
    confirm(monkeypatch, ["yes"])
    assert main([*workflow.argv, "--local-only"]) == 0
    before = list(workflow.calls)
    changed = list(workflow.argv)
    changed[1] = "another/model"
    assert main(changed) == 2
    assert workflow.calls == before
