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
        argv=[
            "run",
            MODEL,
            "--out",
            str(root),
            "--no-history",
            "--baseline-execution",
            "local",
        ],
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


def test_default_run_measures_its_model_in_cloud_before_publication(
    workflow, monkeypatch
):
    monkeypatch.setattr(
        automation.model_baselines, "inputs", lambda _: ("f" * 64, None, {})
    )
    confirm(monkeypatch, ["yes", "yes", "yes", "yes"])
    argv = list(workflow.argv)
    argv.remove("--baseline-execution")
    argv.remove("local")
    assert main([*argv, "--baseline-transport", "github"]) == 0
    commands = dict(workflow.calls)
    for name in ("record", "check", "published-check"):
        command = commands[name]
        assert "baselines" in command and "--model-workspace" in command
        assert command[command.index("--model-workspace") + 1] == str(workflow.root)
    assert "--transport" in commands["record"] and "github" in commands["record"]
    assert "--fixtures-only" not in commands["record"]
    assert (
        "--baseline-transport github"
        in read(workflow.root / ".mic/automation.json")["resume"]
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


def agent_call(workflow, capsys, *, answer=None, extra=()):
    args = [*workflow.argv, "--mode", "agent", *extra]
    if answer is not None:
        path = workflow.root.parent / "answer.json"
        path.write_text(json.dumps(answer))
        args += ["--answer", str(path)]
    code = main(args)
    captured = capsys.readouterr()
    # Agent stdout must be exactly one JSON value, with no prose or progress lines.
    return code, json.loads(captured.out), captured.err


def approval(payload, answer="approve", **fields):
    return {
        "question_id": payload["pending_question"]["question_id"],
        "answer": answer,
        **fields,
    }


@pytest.fixture
def repair_workflow(workflow, monkeypatch):
    from transformers_mic import contract_repair

    original = batch.invoke

    def invoke(out, name, arguments, timeout):
        result = original(out, name, arguments, timeout)
        if name == "create":
            Workspace(workflow.root).save_state(origin="llm")
        if result["exit_code"]:
            log = out / result["log"]
            log.write_text(
                log.read_text()
                + "\nthe contract fails on the random fixture: KeyError: invented.parameter\n"
            )
        return result

    def propose(context, **kw):
        return {
            "name": "BERT",
            "manifest": context["existing_manifest"],
            "usage": context["existing_usage"] + "\n<!-- repaired input -->\n",
            "rationale": "Repair the input using inspected names.",
            "historical_candidates": [],
        }, {"model": "test"}

    monkeypatch.setattr(batch, "invoke", invoke)
    monkeypatch.setattr(contract_repair, "inspect_model", lambda _: {"parameters": []})
    monkeypatch.setattr(contract_repair.planner, "propose", propose)
    workflow.failures["build"] = [1]
    return workflow


def test_reviewed_repair_rebuilds_fixture_before_measurements(
    repair_workflow, monkeypatch
):
    workflow = repair_workflow
    confirm(monkeypatch, ["yes", "yes"])
    assert main([*workflow.argv, "--local-only"]) == 0
    state = read(workflow.root / ".mic/automation.json")
    assert state["repairs"][0]["status"] == "applied"
    builds = [args for name, args in workflow.calls if name == "build"]
    assert len(builds) == 2 and "--rebuild" in builds[-1]
    assert "--max-contract-repairs 2" in state["resume"]


def test_agent_repair_waits_for_exact_approval_and_reuses_saved_proposal(
    repair_workflow, capsys
):
    workflow = repair_workflow
    _, source, _ = agent_call(workflow, capsys, extra=["--local-only"])
    code, proposed, _ = agent_call(
        workflow, capsys, answer=approval(source), extra=["--local-only"]
    )
    assert code == 3 and proposed["pending_question"]["kind"] == "contract-repair"
    calls = list(workflow.calls)
    code, same, _ = agent_call(workflow, capsys, extra=["--local-only"])
    assert (
        code == 3
        and same["pending_question"]["question_id"]
        == proposed["pending_question"]["question_id"]
    )
    assert workflow.calls == calls
    code, done, _ = agent_call(
        workflow, capsys, answer=approval(proposed), extra=["--local-only"]
    )
    assert code == 0 and done["status"] == "validated"
    assert len(read(workflow.root / ".mic/automation.json")["repairs"]) == 1


def test_repair_budget_exhaustion_returns_to_retry(repair_workflow, monkeypatch):
    workflow = repair_workflow
    workflow.failures["build"] = [1, 1, 1]
    confirm(monkeypatch, ["yes", "yes", "no"])
    assert main([*workflow.argv, "--local-only", "--max-contract-repairs", "1"]) == 3
    state = read(workflow.root / ".mic/automation.json")
    assert len(state["repairs"]) == 1
    assert state["pending_question"]["kind"] == "retry"


def test_human_always_completes_workflow_with_readable_summary(
    workflow, monkeypatch, capsys
):
    prompts = []

    def always(prompt):
        prompts.append(prompt)
        return "A"

    monkeypatch.setattr("builtins.input", always)
    assert main(workflow.argv) == 0
    output = capsys.readouterr().out
    assert len(prompts) == 1
    assert "[2/11] Build fixture" in output
    assert "Complete. Model:" in output
    assert '"protocol_version"' not in output
    state = read(workflow.root / ".mic/automation.json")
    assert all(d["always_yes"] for d in state["decisions"])


def test_human_always_stops_repeated_failed_builds(workflow, monkeypatch, capsys):
    workflow.failures["build"] = [1] * 10
    monkeypatch.setattr("builtins.input", lambda _: "A")
    assert main(workflow.argv) == 3
    assert sum(name == "build" for name, _ in workflow.calls) == 3
    assert not workflow.exports
    output = capsys.readouterr().out
    assert "Stopped: this stage keeps failing" in output
    assert "Resume:" in output


def test_resumed_build_starts_at_one_and_omits_unrequested_pin(
    workflow, monkeypatch, capsys
):
    workflow.failures["build"] = [1]
    confirm(monkeypatch, ["yes", "N"])
    assert main(workflow.argv) == 3
    capsys.readouterr()
    confirm(monkeypatch, ["yes", "yes"])
    assert main(workflow.argv) == 0
    output = capsys.readouterr().out
    assert "[1/10] Build fixture" in output
    assert "[10/10] Update fleet" in output
    assert "[11/" not in output


def test_agent_drives_full_workflow_without_a_terminal(workflow, monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("agent must never read stdin")
    )
    code, source, logs = agent_call(workflow, capsys)
    assert code == 3 and source["protocol_version"] == 1
    assert source["pending_question"]["kind"] == "source-review"
    assert source["pending_question"]["context"]["files"][0]["sha256"]
    assert "tmic: create" in logs
    assert "--mode agent" in source["resume"] and "--answer" not in source["resume"]
    code, publish, _ = agent_call(workflow, capsys, answer=approval(source))
    assert code == 3 and publish["pending_question"]["kind"] == "model-publication"
    code, fleet, _ = agent_call(workflow, capsys, answer=approval(publish))
    assert code == 3 and fleet["pending_question"]["kind"] == "fleet-update"
    assert fleet["pending_question"]["context"]["base_revision"] == "a" * 40
    code, result, _ = agent_call(workflow, capsys, answer=approval(fleet))
    assert code == 0 and result["status"] == "complete"
    assert result["dataset_revision"] == SHA
    decisions = read(workflow.root / ".mic/automation.json")["decisions"]
    assert [d["kind"] for d in decisions] == [
        "source-review",
        "model-publication",
        "fleet-update",
    ]
    assert all(d["mode"] == "agent" for d in decisions)


def test_agent_polling_keeps_question_id_and_does_no_work(workflow, capsys):
    _, first, _ = agent_call(workflow, capsys)
    before = list(workflow.calls)
    _, second, _ = agent_call(workflow, capsys)
    assert first["pending_question"] == second["pending_question"]
    assert workflow.calls == before


def test_stale_answer_cannot_approve_edited_source(workflow, capsys):
    _, old, _ = agent_call(workflow, capsys)
    usage = (
        Workspace(workflow.root).contract_root / "integrations/transformers/usage.md"
    )
    usage.write_text(usage.read_text() + "\n# changed after review\n")
    code, current, _ = agent_call(workflow, capsys, answer=approval(old))
    assert code == 3 and "does not match" in current["answer_error"]
    assert (
        current["pending_question"]["question_id"]
        != old["pending_question"]["question_id"]
    )
    assert not any(n == "build" for n, _ in workflow.calls)


def test_agent_replay_never_reuses_source_approval_for_publication(workflow, capsys):
    _, source, _ = agent_call(workflow, capsys)
    _, publish, _ = agent_call(workflow, capsys, answer=approval(source))
    before = list(workflow.calls)
    code, current, _ = agent_call(workflow, capsys, answer=approval(source))
    assert code == 3 and "already been used" in current["answer_error"]
    assert current["pending_question"] == publish["pending_question"]
    assert workflow.calls == before


def test_agent_retry_waits_for_answer_before_reexecuting_failure(workflow, capsys):
    workflow.failures["check"] = [1]
    _, source, _ = agent_call(workflow, capsys)
    code, failed, _ = agent_call(workflow, capsys, answer=approval(source))
    assert code == 3 and failed["pending_question"]["kind"] == "retry"
    assert failed["pending_question"]["choices"] == ["retry", "defer"]
    code, same, _ = agent_call(workflow, capsys)
    assert code == 3 and same["pending_question"] == failed["pending_question"]
    assert [n for n, _ in workflow.calls].count("check") == 1
    code, ready, _ = agent_call(workflow, capsys, answer=approval(failed, "retry"))
    assert code == 3 and ready["pending_question"]["kind"] == "model-publication"
    assert [n for n, _ in workflow.calls].count("check") == 2


def test_agent_can_retry_failed_creation(workflow, capsys):
    workflow.failures["create"] = [1]
    code, failed, _ = agent_call(workflow, capsys)
    assert code == 3 and failed["pending_question"]["kind"] == "retry"
    agent_call(workflow, capsys)
    assert [n for n, _ in workflow.calls].count("create") == 1
    code, source, _ = agent_call(workflow, capsys, answer=approval(failed, "retry"))
    assert code == 3 and source["pending_question"]["kind"] == "source-review"
    assert [n for n, _ in workflow.calls].count("create") == 2


def test_agent_local_only_remains_local_when_resuming(workflow, capsys):
    _, source, _ = agent_call(workflow, capsys, extra=["--local-only"])
    assert "--local-only" in source["resume"]
    code, result, _ = agent_call(
        workflow, capsys, answer=approval(source), extra=["--local-only"]
    )
    assert code == 0 and result["status"] == "validated"
    assert not any(n == "publish" for n, _ in workflow.calls)


def test_agent_can_take_over_a_human_question(workflow, monkeypatch, capsys):
    monkeypatch.setattr(automation.sys.stdin, "isatty", lambda: False)
    assert main(workflow.argv) == 3
    capsys.readouterr()
    human = read(workflow.root / ".mic/automation.json")
    code, agent, _ = agent_call(workflow, capsys)
    assert code == 3 and agent["pending_question"] == human["pending_question"]


def test_agent_rejects_malformed_answer_and_reports_json(workflow, capsys):
    code, result, _ = agent_call(workflow, capsys, answer={"answer": True})
    assert code == 2 and result["status"] == "error"
    assert not workflow.calls


def test_workspace_lock_prevents_concurrent_consumption(workflow, capsys):
    from filelock import FileLock

    with FileLock(workflow.root / ".mic/automation.lock"):
        code, result, _ = agent_call(workflow, capsys)
    assert code == 2 and "Another tmic run" in result["error"]
    assert not workflow.calls


def test_agent_protocol_is_discoverable_in_help(capsys):
    with pytest.raises(SystemExit) as root:
        main(["--help"])
    assert root.value.code == 0
    help = capsys.readouterr().out
    assert "--mode human" in help and "--mode agent" in help
    assert "--answer answer.json" in help and "pending_question" in help
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    help = capsys.readouterr().out
    for word in (
        "question_id",
        "context_sha256",
        "requires_reason",
        "retry",
        "defer",
        "Exit codes",
        "130",
    ):
        assert word in help


def test_agent_changed_baseline_poll_and_reason_do_not_repeat_work(workflow, capsys):
    Workspace(workflow.root).from_catalog(MODEL, "BERT", "auto", False)
    workspace = Workspace(workflow.root)
    baseline = next(workspace.expectations.rglob("*.json"))
    candidate = baseline.with_name(baseline.stem + ".candidate.json")
    candidate.write_text(baseline.read_text() + "\n")
    workflow.failures["review"] = [1]
    _, source, _ = agent_call(workflow, capsys)
    code, changed, _ = agent_call(workflow, capsys, answer=approval(source))
    assert code == 3 and changed["pending_question"]["kind"] == "baseline-change"
    before = list(workflow.calls)
    _, same, _ = agent_call(workflow, capsys)
    assert same["pending_question"] == changed["pending_question"]
    code, missing, _ = agent_call(workflow, capsys, answer=approval(changed))
    assert code == 3 and "reason" in missing["answer_error"]
    assert workflow.calls == before and candidate.exists()
    code, ready, _ = agent_call(
        workflow,
        capsys,
        answer=approval(changed, reason="Measured reference correction"),
    )
    assert code == 3 and ready["pending_question"]["kind"] == "model-publication"
    assert not candidate.exists()
    assert (
        read(workflow.root / ".mic/automation-reviews.json")[-1]["reason"]
        == "Measured reference correction"
    )


def test_source_edit_during_baseline_review_is_not_approved(workflow, monkeypatch):
    Workspace(workflow.root).from_catalog(MODEL, "BERT", "auto", False)
    workspace = Workspace(workflow.root)
    baseline = next(workspace.expectations.rglob("*.json"))
    candidate = baseline.with_name(baseline.stem + ".candidate.json")
    candidate.write_text(baseline.read_text() + "\n")
    usage = workspace.contract_root / "integrations/transformers/usage.md"

    def reply(prompt):
        if "Reason" in prompt:
            usage.write_text(usage.read_text() + "\n# altered executable\n")
            return "Reviewed baseline correction"
        return "yes"

    monkeypatch.setattr("builtins.input", reply)
    with pytest.raises(ValueError, match="changed during review"):
        automation.review_changed(SimpleNamespace(non_interactive=False), workspace)
    assert candidate.exists()
