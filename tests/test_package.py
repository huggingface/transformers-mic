import copy
import json
from types import SimpleNamespace

import pytest

from transformers_mic import commands, history, planner, policy, resources
from transformers_mic.cli import main, parser
from transformers_mic.workspace import Workspace, read, write


@pytest.fixture
def workspace(tmp_path):
    project = Workspace(tmp_path / "project")
    project.from_catalog("google-bert/bert-base-uncased", "BERT", "auto", False)
    return project


def measured(weights, passed=True, finite=True):
    return {
        "weights": weights,
        "gate": {weights: {"pass": passed}},
        "numeric_health": {"logits": {"finite": finite}},
    }


@pytest.mark.parametrize(
    "weights,passed,finite,requested,readable,accepted",
    [
        ("random", True, True, "auto", False, True),
        ("random", False, True, "auto", False, False),
        ("random", True, False, "auto", False, False),
        ("random", True, True, "overfit", False, False),
        ("random", True, True, "auto", True, False),
        ("overfit", True, True, "auto", True, True),
    ],
)
def test_weight_policy(weights, passed, finite, requested, readable, accepted):
    assert (
        policy.decision(measured(weights, passed, finite), requested, readable)[
            "accepted"
        ]
        is accepted
    )


def test_legacy_fixture_can_only_be_reused():
    metadata = {"training": {"steps": 20}}
    assert policy.decision(metadata, reused=True)["accepted"]
    assert not policy.decision(metadata)["accepted"]


def test_default_provider_and_model():
    args = parser().parse_args(["create", "model/name", "--out", "project"])
    assert args.provider == "auto"
    assert args.llm_model == "Qwen/Qwen3.8-27B"


def test_planner_requests_json_and_never_generates_values():
    plan = {
        "name": "Bert",
        "manifest": {},
        "usage": "x" * 100,
        "rationale": "",
        "historical_candidates": [],
    }

    class Client:
        def chat_completion(self, **kwargs):
            assert kwargs["response_format"] == {"type": "json_object"}
            assert "Never invent baseline" in kwargs["messages"][0]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=json.dumps(plan)))
                ],
                usage=None,
            )

    result, provenance = planner.propose({}, client=Client())
    assert result == plan
    assert provenance["model"] == planner.DEFAULT_MODEL


def test_plan_rejects_relaxed_rules_and_removed_assertions(workspace):
    manifest = workspace.manifest
    plan = {
        "name": "BERT",
        "manifest": manifest,
        "usage": (
            workspace.contract_root / "integrations/transformers/usage.md"
        ).read_text(),
        "rationale": "",
        "historical_candidates": [],
    }
    commands.validate_plan(plan, workspace.state["model"], workspace.fixture)
    relaxed = copy.deepcopy(plan)
    relaxed["manifest"]["outputs"]["answer_logits"]["atol"] = 1
    with pytest.raises(ValueError, match="tolerances"):
        commands.validate_plan(relaxed, workspace.state["model"], workspace.fixture)
    removed = copy.deepcopy(plan)
    del removed["manifest"]["outputs"]["answer_logits"]
    with pytest.raises(ValueError, match="removes"):
        commands.validate_plan(
            removed, workspace.state["model"], workspace.fixture, manifest
        )


def test_history_import_and_query_dedup(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    write(path, [{"url": "https://example.org/issue", "title": "failure"}])
    assert not history.collect("org/model", imported=path)["complete"]
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"url": "https://example.org/issue"}])
        )

    monkeypatch.setattr(history.subprocess, "run", run)
    result = history.collect("org/model", "bert")
    assert len(calls) == 4
    assert len(result["items"]) == 1
    assert result["complete"] is False


def approve_receipts(workspace):
    import hashlib

    paths = {
        "lock_sha256": workspace.lock_path,
        "manifest_sha256": workspace.contract_root / "integration.json",
        "usage_sha256": workspace.contract_root / "integrations/transformers/usage.md",
    }
    proof = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
    }
    validations = {
        v: {"fingerprint": workspace.validation_fingerprint()}
        for v in workspace.manifest["integrations"]["transformers"]["variants"]
    }
    workspace.save_state(validated=True, validation=proof, validations=validations)


def test_publish_requires_all_variants_and_invalidates_baseline_edits(workspace):
    with pytest.raises(ValueError, match="Run mic check"):
        commands.publication_ready(workspace)
    approve_receipts(workspace)
    commands.publication_ready(workspace)
    path = next((workspace.lock_path.parent / "expectations").rglob("*.json"))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Every declared variant"):
        commands.publication_ready(workspace)


def test_snapshot_keeps_previous_usage(workspace):
    backup = workspace.snapshot()
    original = (backup / "integrations/transformers/usage.md").read_text()
    (workspace.contract_root / "integrations/transformers/usage.md").write_text(
        "new usage"
    )
    assert (backup / "integrations/transformers/usage.md").read_text() == original


def test_export_keeps_other_contracts_and_removes_local_paths(workspace, tmp_path):
    approve_receipts(workspace)
    target = tmp_path / "transformers"
    path = target / "tests/contracts/fleet.lock"
    write(
        path,
        {
            "framework": "transformers",
            "environment": workspace.lock["environment"],
            "contracts": {"Other": {"keep": True}},
        },
    )
    commands.export(
        SimpleNamespace(workspace=workspace.root, transformers_checkout=target)
    )
    lock = read(path)
    assert lock["contracts"]["Other"] == {"keep": True}
    assert "path" not in lock["contracts"]["BERT"]["integration"]
    assert all("path" not in p for p in lock["contracts"]["BERT"]["fixtures"].values())


def test_create_offline_and_status(tmp_path, capsys):
    path = tmp_path / "new"
    assert (
        main(
            ["create", "google-bert/bert-base-uncased", "--offline", "--out", str(path)]
        )
        == 0
    )
    assert main(["--workspace", str(path), "build"]) == 0
    assert Workspace(path).state["weight_decision"]["reused"]
    assert main(["--workspace", str(path), "status"]) == 0
    assert "catalog" in capsys.readouterr().out


def test_backend_schemas_and_catalog_present():
    assert (resources.backend() / "schemas/integration.schema.json").is_file()
    assert (resources.backend() / "run.py").is_file()
    assert resources.catalog_lock().is_file()


def test_planner_repairs_invalid_draft(workspace, monkeypatch):
    plan = {
        "name": "BERT",
        "manifest": workspace.manifest,
        "usage": (
            workspace.contract_root / "integrations/transformers/usage.md"
        ).read_text(),
        "rationale": "",
        "historical_candidates": [],
    }
    invalid = copy.deepcopy(plan)
    del invalid["manifest"]["outputs"]["training.updated"]
    calls = []

    def propose(context, **kwargs):
        calls.append(context)
        return (invalid if len(calls) == 1 else plan), {"model": planner.DEFAULT_MODEL}

    monkeypatch.setattr(planner, "propose", propose)
    result, _ = commands.propose_valid(
        {},
        SimpleNamespace(llm_model=planner.DEFAULT_MODEL, provider="auto"),
        workspace.state["model"],
        workspace.fixture,
    )
    assert result == plan
    assert "Training contracts require" in calls[1]["validation_error"]


def test_update_keeps_fixture_pin_and_old_source(workspace, monkeypatch):
    plan = {
        "name": "BERT",
        "manifest": workspace.manifest,
        "usage": (
            workspace.contract_root / "integrations/transformers/usage.md"
        ).read_text()
        + "\nAdditional path documentation.\n",
        "rationale": "",
        "historical_candidates": [],
    }
    old_lock = workspace.lock_path.read_bytes()
    monkeypatch.setattr(planner, "propose", lambda *a, **k: (plan, {}))
    commands.update(
        SimpleNamespace(
            workspace=workspace.root,
            request="Add a path",
            llm_model=planner.DEFAULT_MODEL,
            provider="auto",
            retrain=False,
        )
    )
    assert workspace.lock_path.read_bytes() == old_lock
    assert list((workspace.root / ".mic/history").glob("*/integration.json"))
    assert not workspace.state["validated"]


def test_export_refuses_unpublished_local_source(workspace, tmp_path):
    path = workspace.contract_root / "integrations/transformers/usage.md"
    path.write_text(path.read_text() + "\nNew source.\n")
    approve_receipts(workspace)
    with pytest.raises(ValueError, match="differs from its published pin"):
        commands.export(
            SimpleNamespace(
                workspace=workspace.root,
                transformers_checkout=tmp_path / "transformers",
            )
        )


def test_publication_rejects_pending_candidates_and_missing_variant(workspace):
    approve_receipts(workspace)
    validations = workspace.state["validations"]
    del validations["eager"]
    workspace.save_state(validations=validations)
    with pytest.raises(ValueError, match="Every declared variant"):
        commands.publication_ready(workspace)
    path = (
        workspace.lock_path.parent
        / "expectations/BERT/cpu-arm64/example.candidate.json"
    )
    write(path, {})
    approve_receipts(workspace)
    with pytest.raises(ValueError, match="Unreviewed"):
        commands.publication_ready(workspace)


def test_external_fleet_uses_each_contract_variants(tmp_path, monkeypatch):
    from transformers_mic import cli

    manifests = {
        "a": {
            "integrations": {"transformers": {"variants": {"default": {}, "eager": {}}}}
        },
        "b": {"integrations": {"transformers": {"variants": {"default": {}}}}},
    }
    for key, manifest in manifests.items():
        write(tmp_path / f"{key}.json", manifest)
    lock = tmp_path / "fleet.lock"
    write(
        lock,
        {
            "contracts": {
                key: {"integration": {"repo": key, "revision": "pinned"}}
                for key in manifests
            }
        },
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, *a, **k: tmp_path / f"{repo}.json",
    )
    calls = []
    monkeypatch.setattr(
        cli.backend,
        "invoke",
        lambda tool, workspace, options: calls.append(options)
        or SimpleNamespace(returncode=0),
    )
    assert cli.main(["check", "--lock", str(lock), "--all-variants"]) == 0
    assert len(calls) == 3
    assert not any(
        options[options.index("--contract") + 1] == "b" and options[-1] == "eager"
        for options in calls
    )
