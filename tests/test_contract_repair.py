import copy
from types import SimpleNamespace

import pytest

from transformers_mic import contract_repair as repair
from transformers_mic.workspace import Workspace
from transformers_mic.workspace import write


@pytest.fixture
def setup(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "project")
    workspace.from_catalog("google-bert/bert-base-uncased", "BERT", "auto", False)
    workspace.save_state(origin="llm")
    usage = (
        workspace.contract_root
        / workspace.manifest["integrations"]["transformers"]["entry_point"]
    )
    log = tmp_path / "build.log"
    log.write_text(
        "the contract fails on the random fixture:\nKeyError: bad.parameter\nhf_notasecret"
    )
    plan = {
        "name": "BERT",
        "manifest": workspace.manifest,
        "usage": usage.read_text() + "\n<!-- Corrected usage -->\n",
        "rationale": "Use the actual inspected parameter name.",
        "historical_candidates": [],
    }
    monkeypatch.setattr(
        repair,
        "inspect_model",
        lambda _: {"parameters": [{"name": "bert.embeddings.word_embeddings.weight"}]},
    )
    contexts = []

    def propose(context, **kw):
        contexts.append(context)
        return copy.deepcopy(plan), {"model": "test"}

    monkeypatch.setattr(repair.planner, "propose", propose)
    return SimpleNamespace(
        workspace=workspace,
        log=log,
        plan=plan,
        contexts=contexts,
        folder=tmp_path / "automation",
        state={},
        args=SimpleNamespace(max_contract_repairs=2, llm_model="test", provider="auto", bill_to=""),
    )


def propose(s):
    return repair.propose(
        s.args, s.workspace, "build", s.log, s.folder, s.state, lambda: None
    )


def test_proposal_uses_inspection_and_log_without_mutating_contract(setup):
    s = setup
    original = s.workspace.contract_digest()
    attempt = propose(s)
    assert attempt["status"] == "proposed"
    assert s.workspace.contract_digest() == original
    assert "hf_notasecret" not in s.contexts[0]["failure_log"]
    assert s.contexts[0]["model_inspection"]["parameters"]
    repair.apply(s.workspace, attempt)
    assert attempt["status"] == "applied" and attempt["previous_source"]
    assert s.workspace.manifest == s.plan["manifest"]
    assert not s.workspace.state["validated"]


@pytest.mark.parametrize("change", ["tolerance", "variant", "gate", "output"])
def test_repairs_cannot_weaken_manifest(setup, change):
    s = setup
    manifest = s.plan["manifest"]
    if change == "tolerance":
        manifest["outputs"]["answer_logits"]["atol"] = 1
    elif change == "variant":
        del manifest["integrations"]["transformers"]["variants"]["eager"]
    elif change == "gate":
        next(iter(manifest["fixtures"].values()))["build"]["accept_insensitive"] = {
            "input_ids:reverse": "ignore"
        }
    else:
        del manifest["outputs"]["answer_logits"]
    before = s.workspace.contract_digest()
    assert propose(s) is None
    assert s.state["repairs"][0]["status"] == "rejected"
    assert s.workspace.contract_digest() == before


def test_repair_keeps_the_contract_name(setup):
    s = setup
    s.plan["name"] = "bert-base-uncased.v2"  # invalid slug, and a rename
    attempt = propose(s)
    assert attempt["status"] == "proposed"
    repair.apply(s.workspace, attempt)
    assert s.workspace.state["contract"] == "BERT"


def test_changed_source_or_proposal_invalidates_approval(setup):
    s = setup
    attempt = propose(s)
    path = s.folder / "repairs/1/proposal.json"
    path.write_text(path.read_text() + "\n")
    assert not repair.current(s.workspace, attempt)
    with pytest.raises(ValueError, match="changed"):
        repair.apply(s.workspace, attempt)


def test_repair_budget_survives_failed_proposals(setup):
    s = setup
    s.args.max_contract_repairs = 1
    s.plan["usage"] = "invalid syntax"
    assert propose(s) is None
    assert propose(s) is None
    assert len(s.contexts) == 1 and len(s.state["repairs"]) == 1


def test_explicit_assertions_and_gradient_checks_are_preserved():
    old = """```py runnable:usage
assert loss.isfinite()
result = {"training": {"gradients": {n: bool(named[n].grad is not None and named[n].grad.isfinite().all()) for n in ("wrong.parameter",)}, "updated": any(changed)}}
```"""
    fixed = old.replace("wrong.parameter", "actual.parameter")
    assert not repair.guards(old) - repair.guards(fixed)
    assert repair.guards(old) - repair.guards(
        fixed.replace("assert loss.isfinite()", "")
    )
    assert repair.guards(old) - repair.guards(
        fixed.replace("named[n].grad.isfinite().all()", "True")
    )


def test_dependency_network_errors_do_not_trigger_contract_repair(setup):
    s = setup
    s.log.write_text("Traceback\nConnectError: cannot reach Hub")
    assert not repair.eligible(s.workspace, "build", s.log)


def test_cloud_failure_uses_downloaded_worker_traceback_and_binds_it_to_review(setup):
    s = setup
    s.log.write_text("Cloud job failed; available partial artifacts were downloaded")
    root = s.folder / "cloud/model-input"
    report = root / "artifacts/cpu/collection.json"
    write(
        report,
        {
            "cases": [
                {
                    "status": "failed",
                    "fleet": "prototypes",
                    "contract": "Demo",
                    "variant": "default",
                }
            ]
        },
    )
    worker_log = report.parent / "logs/prototypes-Demo-default-record.log"
    worker_log.parent.mkdir(parents=True)
    worker_log.write_text("Traceback\nKeyError: invented.parameter")
    command = ["baselines", "run", "--out", str(root)]
    assert repair.eligible(s.workspace, "record", s.log, command)
    attempt = repair.propose(
        s.args, s.workspace, "record", s.log, s.folder, s.state, lambda: None, command
    )
    assert "invented.parameter" in s.contexts[0]["failure_log"]
    assert repair.current(s.workspace, attempt)
    worker_log.write_text("different failure")
    assert not repair.current(s.workspace, attempt)


def test_truncated_runtime_error_is_eligible_for_repair(setup):
    setup.log.write_text(
        "RuntimeError: Input type FloatTensor and weight type BFloat16Type should be the same"
    )
    assert repair.eligible(setup.workspace, "auto", setup.log)
