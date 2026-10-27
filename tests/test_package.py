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
    assert args.bill_to == "huggingface"


def test_bill_to_follows_environment_and_can_be_personal(monkeypatch):
    monkeypatch.setenv("TMIC_BILL_TO", "my-org")
    args = parser().parse_args(["create", "model/name", "--out", "project"])
    assert args.bill_to == "my-org"
    args = parser().parse_args(
        ["create", "model/name", "--out", "project", "--bill-to", ""]
    )
    assert args.bill_to == ""


def test_planner_bills_the_organization_and_never_falls_back(monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    created = []

    class Client:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def chat_completion(self, **kwargs):
            raise HfHubHTTPError(
                "forbidden",
                response=SimpleNamespace(status_code=403, headers={}, request=None),
            )

    monkeypatch.setattr(planner, "InferenceClient", Client)
    with pytest.raises(ValueError, match="--bill-to ''"):
        planner.propose({}, bill_to="huggingface")
    assert len(created) == 1 and created[0]["bill_to"] == "huggingface"
    with pytest.raises(HfHubHTTPError):
        planner.propose({}, bill_to="")
    assert created[1]["bill_to"] is None


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


def test_history_rate_limit_keeps_partial_candidates(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=0, stdout=json.dumps([{"url": "https://example.org/1"}])
            )
        return SimpleNamespace(
            returncode=1, stdout="", stderr="HTTP 403: API rate limit exceeded"
        )

    monkeypatch.setattr(history.subprocess, "run", run)
    result = history.collect("org/model", "bert")
    assert len(calls) == 2
    assert len(result["items"]) == 1
    assert "rate limit" in result["error"]

    def denied(command, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="HTTP 401: Bad credentials"
        )

    monkeypatch.setattr(history.subprocess, "run", denied)
    with pytest.raises(ValueError, match="Bad credentials"):
        history.collect("org/model", "bert")


def test_planner_transport_errors_are_retryable():
    import httpx2

    class Client:
        def chat_completion(self, **kwargs):
            raise httpx2.ReadTimeout("The read operation timed out")

    with pytest.raises(ValueError, match="ReadTimeout"):
        planner.propose({}, client=Client())


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
    with pytest.raises(ValueError, match="Run tmic check"):
        commands.publication_ready(workspace)
    approve_receipts(workspace)
    commands.publication_ready(workspace)
    path = next(workspace.expectations.rglob("*.json"))
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


@pytest.fixture
def dataset(workspace, monkeypatch, tmp_path):
    from transformers_mic import fleet

    path = tmp_path / "remote-fleet.json"
    write(
        path,
        {
            "framework": "transformers",
            "environment": workspace.lock["environment"],
            "contracts": {"Other": {"keep": True}},
        },
    )
    monkeypatch.setattr(fleet, "download", lambda *args: (path, "a" * 40))
    calls = []

    class API:
        def create_commit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(oid="b" * 40)

    monkeypatch.setattr(commands, "HfApi", API)
    return path, calls


def test_export_publishes_dataset_and_writes_only_sha(workspace, dataset, tmp_path):
    approve_receipts(workspace)
    target = tmp_path / "transformers"
    args = SimpleNamespace(
        workspace=workspace.root, transformers_checkout=target, execute=False
    )
    commands.export(args)
    assert dataset[1] == []
    assert not target.exists()
    args.execute = True
    commands.export(args)
    call = dataset[1][0]
    assert call["repo_id"] == "huggingface-mic/transformers-fleet"
    assert call["repo_type"] == "dataset"
    assert call["parent_commit"] == "a" * 40
    operation = call["operations"][0]
    assert operation.path_in_repo == "fleet-lock.json"
    lock = json.loads(operation.path_or_fileobj)
    assert lock["contracts"]["Other"] == {"keep": True}
    assert "path" not in lock["contracts"]["BERT"]["integration"]
    assert all("path" not in p for p in lock["contracts"]["BERT"]["fixtures"].values())
    assert list(target.rglob("*")) == [
        target / "tests",
        target / "tests/contracts.lock",
    ]
    assert (target / "tests/contracts.lock").read_text() == "b" * 40 + "\n"
    assert (
        "BERT" not in read(dataset[0])["contracts"]
    )  # cached download is never edited


def test_export_rejects_case_collision_and_allows_explicit_name(workspace, dataset):
    approve_receipts(workspace)
    old = read(dataset[0])
    old["contracts"]["bert"] = {}
    write(dataset[0], old)
    args = SimpleNamespace(workspace=workspace.root, execute=True)
    with pytest.raises(ValueError, match="differs only by case"):
        commands.export(args)
    args.contract_name = "mic-bert"
    commands.export(args)
    lock = json.loads(dataset[1][0]["operations"][0].path_or_fileobj)
    assert "mic-bert" in lock["contracts"]
    assert "bert" in lock["contracts"]


def test_dataset_export_conflict_does_not_update_framework_pin(
    workspace, dataset, tmp_path, monkeypatch
):
    approve_receipts(workspace)
    target = tmp_path / "transformers"
    pin = target / "tests/contracts.lock"
    pin.parent.mkdir(parents=True)
    pin.write_text("a" * 40 + "\n")

    class API:
        def create_commit(self, **kwargs):
            raise ValueError("remote changed since preview")

    monkeypatch.setattr(commands, "HfApi", API)
    with pytest.raises(ValueError, match="remote changed"):
        commands.export(
            SimpleNamespace(
                workspace=workspace.root, execute=True, transformers_checkout=target
            )
        )
    assert pin.read_text() == "a" * 40 + "\n"


def test_dataset_check_uses_framework_pin(dataset, tmp_path, monkeypatch):
    from transformers_mic import cli, fleet

    monkeypatch.chdir(tmp_path)
    pin = tmp_path / "tests/contracts.lock"
    pin.parent.mkdir()
    pin.write_text("c" * 40 + "\n")
    calls = []

    def download(name, revision):
        calls.append((name, revision))
        return dataset[0], revision

    monkeypatch.setattr(fleet, "download", download)
    monkeypatch.setattr(cli, "external_check", lambda args: 0)
    assert main(["check", "--fleet", "prototypes"]) == 0
    assert calls == [("prototypes", "c" * 40)]
    assert main(["check"]) == 0
    assert calls[-1] == ("control", "c" * 40)
    assert main(["check", "--lock", str(pin)]) == 0
    assert calls[-1] == ("control", "c" * 40)
    pin.write_text("main\n")
    assert main(["check", "--fleet", "control"]) == 2


def test_external_check_defaults_to_full_fleet(monkeypatch, tmp_path):
    from transformers_mic import backend

    calls = []
    monkeypatch.setattr(
        backend,
        "invoke",
        lambda tool, root, options: (
            calls.append(options) or SimpleNamespace(returncode=0)
        ),
    )
    assert main(["check", "--lock", str(tmp_path / "fleet.lock")]) == 0
    assert "--run-all" in calls[-1]
    assert (
        main(
            [
                "check",
                "--lock",
                str(tmp_path / "fleet.lock"),
                "--changed-files",
                str(tmp_path / "changes"),
            ]
        )
        == 0
    )
    assert "--run-all" not in calls[-1]


def test_contract_infrastructure_and_baseline_changes_select_cases():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mic_selector_test", resources.backend() / "select_contracts.py"
    )
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    cases = {"BERT": ["bert"], "mic-vit": ["vit"]}
    assert set(selector.select(["tests/contracts.lock"], cases)) == set(cases)
    assert set(selector.select(["tests/contracts/prototype-fleet.lock"], cases)) == set(
        cases
    )
    assert set(
        selector.select(
            ["tests/contracts/expectations/mic-vit/cpu-x86_64/a.json"], cases
        )
    ) == {"mic-vit"}


def test_export_rejects_patched_validation(workspace, tmp_path):
    lock = workspace.lock
    lock["local_patches"] = {"masked-spec-embed-init": "reference control"}
    write(workspace.lock_path, lock)
    approve_receipts(workspace)
    with pytest.raises(ValueError, match="without local patches"):
        commands.export(
            SimpleNamespace(
                workspace=workspace.root, transformers_checkout=tmp_path / "tf"
            )
        )


def test_cpu_precision_baselines_follow_hardware(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mic_platform_test", resources.backend() / "run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.os, "uname", lambda: SimpleNamespace(machine="x86_64"))
    monkeypatch.setattr(runner, "native_bfloat16", lambda: True)
    monkeypatch.setattr(runner, "native_float16", lambda: True)
    assert runner.platform_class("cpu", "float32") == "cpu-x86_64"
    assert runner.platform_class("cpu", "bfloat16") == "cpu-x86_64-bf16"
    assert runner.platform_class("cpu", "float16") == "cpu-x86_64-fp16"
    assert runner.platform_class("cuda", "float16") == "cuda"


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
        SimpleNamespace(llm_model=planner.DEFAULT_MODEL, provider="auto", bill_to=""),
        workspace.state["model"],
        workspace.fixture,
    )
    assert result == plan
    assert "Training contracts require" in calls[1]["validation_error"]


def test_planner_repairs_invented_parameter_names(workspace, monkeypatch):
    pytest.importorskip("torch")
    from transformers import BertConfig

    usage = (workspace.contract_root / "integrations/transformers/usage.md").read_text()
    plan = {
        "name": "BERT",
        "manifest": workspace.manifest,
        "usage": usage
        + '\nchecked = ("bert.encoder.layer.0.attention.self.query.weight",)\n',
        "rationale": "",
        "historical_candidates": [],
    }
    invented = copy.deepcopy(plan)
    invented["usage"] = usage + '\nchecked = ("bert.encoder.layers.0.q_proj.weight",)\n'
    calls = []

    def propose(context, **kwargs):
        calls.append(context)
        return (invented if len(calls) == 1 else plan), {}

    monkeypatch.setattr(planner, "propose", propose)
    config = BertConfig(
        hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=8
    ).to_dict()
    result, _ = commands.propose_valid(
        {"config": config},
        SimpleNamespace(llm_model=planner.DEFAULT_MODEL, provider="auto", bill_to=""),
        workspace.state["model"],
        workspace.fixture,
    )
    assert result == plan
    error = calls[1]["validation_error"]
    assert "bert.encoder.layers.0.q_proj.weight" in error
    assert "attention.self.query.weight" in error


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
            bill_to="",
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
    path = workspace.expectations / "cpu-arm64/example.candidate.json"
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
        lambda tool, workspace, options: (
            calls.append(options) or SimpleNamespace(returncode=0)
        ),
    )
    assert cli.main(["check", "--lock", str(lock), "--all-variants"]) == 0
    assert len(calls) == 3
    assert not any(
        options[options.index("--contract") + 1] == "b" and options[-1] == "eager"
        for options in calls
    )


def test_regeneration_creates_new_mic_artifacts_without_old_baselines(tmp_path):
    path = tmp_path / "regenerated"
    assert (
        main(
            [
                "create",
                "google-bert/bert-base-uncased",
                "--regenerate",
                "--fixtures-only",
                "--namespace",
                "hf-internal-testing",
                "--name",
                "bert",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    project = Workspace(path)
    assert project.fixture == "hf-internal-testing/bert-mic"
    assert (
        project.lock["contracts"]["bert"]["integration"]["repo"]
        == "hf-internal-testing/bert-mic"
    )
    assert project.manifest["integrations"]["transformers"]["targets"] == ["fixtures"]
    assert not (project.lock_path.parent / "expectations").exists()


def test_metadata_missing_keys_handles_ties_and_legacy_norm_names():
    from transformers_mic.inspection import missing_from_keys

    tied = object()

    class Model:
        base_model_prefix = "model"
        all_tied_weights_keys = {}

        def state_dict(self):
            return {
                "model.norm.weight": None,
                "model.norm.bias": None,
                "embedding.weight": None,
                "lm_head.weight": None,
                "masked_spec_embed": None,
                "vision.layer_scale.gamma": None,
            }

        def named_parameters(self, **kwargs):
            return [("embedding.weight", tied), ("lm_head.weight", tied)]

    assert missing_from_keys(
        Model(),
        {"norm.gamma", "norm.beta", "embedding.weight", "vision.layer_scale.gamma"},
    ) == ["masked_spec_embed"]


def test_mxfp4_presence_requires_paired_blocks_and_scales():
    from transformers_mic.inspection import logical_checkpoint_keys

    config = SimpleNamespace(
        model_type="gpt_oss", quantization_config={"quant_method": "mxfp4"}
    )
    keys = {
        "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "model.layers.0.mlp.experts.gate_up_proj_scales",
    }
    assert "model.layers.0.mlp.experts.gate_up_proj" in logical_checkpoint_keys(
        config, keys
    )
    with pytest.raises(ValueError, match="lacks scales"):
        logical_checkpoint_keys(
            config, {next(k for k in keys if k.endswith("_blocks"))}
        )
    config.quantization_config = {"quant_method": "other"}
    with pytest.raises(ValueError, match="reviewed parameter-name adapter"):
        logical_checkpoint_keys(config, keys)


def test_fixtures_only_manifest_never_downloads_checkpoints(monkeypatch, tmp_path):
    import importlib.util

    source = resources.backend() / "run.py"
    spec = importlib.util.spec_from_file_location("mic_test_runner", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(
        runner,
        "checkpoint_path",
        lambda *a: pytest.fail("Full checkpoint download requested"),
    )
    manifest = {
        "fixtures": {},
        "checkpoints": {"org/model": {}},
        "integrations": {"transformers": {"targets": ["fixtures"]}},
    }
    contract = {"fixtures": {}, "checkpoints": {"org/model": "sha"}}
    assert list(runner.targets(tmp_path, contract, manifest, "transformers")) == []


def test_batch_keeps_running_after_a_model_failure(tmp_path, monkeypatch):
    from transformers_mic import batch

    root = tmp_path / "batch"
    for name in ("bert", "other"):
        Workspace(root / "projects" / name).from_catalog(
            "google-bert/bert-base-uncased", "BERT", "auto", False
        )
    path = tmp_path / "models.json"
    write(
        path,
        {
            "models": [
                {"name": "bert", "checkpoint": "org/bert"},
                {"name": "other", "checkpoint": "org/other"},
            ]
        },
    )

    def invoke(out, name, command, timeout):
        return {"exit_code": 1 if name == "bert" else 0}

    monkeypatch.setattr(batch, "invoke", invoke)
    args = SimpleNamespace(
        models_file=path,
        out=root,
        phase="run",
        select=None,
        resume=False,
        publish=False,
        timeout=1,
    )
    assert batch.run(args) == 1
    report = read(root / "report.json")
    assert report["models"]["bert"]["status"] == "build-failed"
    assert report["models"]["other"]["status"] == "validated"


def test_new_artifacts_default_to_mic_organization(tmp_path):
    args = parser().parse_args(["create", "org/model", "--out", "project"])
    assert args.namespace == "huggingface-mic"
    args = parser().parse_args(
        ["batch", "--models-file", "models.json", "--out", "project"]
    )
    assert args.namespace == "huggingface-mic"
    path = tmp_path / "mic-org"
    assert (
        main(
            [
                "create",
                "google-bert/bert-base-uncased",
                "--regenerate",
                "--name",
                "bert",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    project = Workspace(path)
    assert project.fixture == "huggingface-mic/bert-mic"
    assert (
        project.lock["contracts"]["bert"]["integration"]["repo"]
        == "huggingface-mic/bert-mic"
    )


def test_hub_aliases_preserve_pins_and_unrelated_owners():
    from transformers_mic.hub import canonical_references

    original = {
        "integration": {
            "repo": "hf-internal-testing/mic-bert-integration",
            "revision": "same",
            "files_sha256": "hash",
        },
        "fixtures": {"hf-internal-testing/mic-bert-slim": {"revision": "same"}},
        "checkpoints": {"hf-internal-testing/tiny-random-bert": "same"},
    }
    moved = canonical_references(original)
    assert moved["integration"] == {
        **original["integration"],
        "repo": "huggingface-mic/mic-bert-integration",
    }
    assert "huggingface-mic/mic-bert-slim" in moved["fixtures"]
    assert moved["checkpoints"] == original["checkpoints"]
    assert original["integration"]["repo"].startswith("hf-internal-testing/")


def test_umbrella_baselines_live_next_to_usage_and_invalidate_validation(tmp_path):
    path = tmp_path / "umbrella"
    assert (
        main(
            [
                "create",
                "google-bert/bert-base-uncased",
                "--regenerate",
                "--name",
                "bert",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    project = Workspace(path)
    assert project.contract_root.name == "bert-mic"
    assert (
        project.expectations
        == project.contract_root / "integrations/transformers/expectations"
    )
    digest_before = project.contract_digest()
    fingerprint_before = project.validation_fingerprint()
    write(project.expectations / "cpu-test/bert-mic.json", {"outputs": {"value": 1}})
    assert project.contract_digest() == digest_before
    assert project.validation_fingerprint() != fingerprint_before


def test_umbrella_publish_uploads_one_snapshot_and_pins_both_roles(
    tmp_path, monkeypatch
):
    import importlib.util
    import sys
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "example"
    monkeypatch.syspath_prepend(str(example))
    spec = importlib.util.spec_from_file_location(
        "publish_mic_test", example / "publish_mic.py"
    )
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    path = tmp_path / "umbrella"
    assert (
        main(
            [
                "create",
                "google-bert/bert-base-uncased",
                "--regenerate",
                "--name",
                "bert",
                "--out",
                str(path),
            ]
        )
        == 0
    )
    project = Workspace(path)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "config.json").write_text("{}")
    (fixture / "model.safetensors").write_bytes(b"weights")
    write(
        fixture / "build.json",
        {"sha256": {"config.json": "hash", "model.safetensors": "hash"}},
    )
    write(project.expectations / "cpu-test/bert-mic.json", {"reviewed": True})
    lock = project.lock
    pin = lock["contracts"]["bert"]["fixtures"][project.fixture]
    pin.update(path=str(fixture), files_sha256="hash")
    write(project.lock_path, lock)
    monkeypatch.setattr(publisher, "HERE", project.root / "example")
    monkeypatch.setattr(publisher.run, "load", lambda path, schema: read(path))
    monkeypatch.setattr(publisher.run, "fixture_problem", lambda *a: None)
    calls = []

    class API:
        def repo_exists(self, repo):
            return False

        def create_repo(self, repo, **kwargs):
            assert repo == "huggingface-mic/bert-mic"

        def upload_folder(self, repo_id, folder_path, **kwargs):
            assert repo_id == "huggingface-mic/bert-mic"
            assert (folder_path / "model.safetensors").read_bytes() == b"weights"
            assert (folder_path / "integrations/transformers/usage.md").exists()
            assert read(
                folder_path
                / "integrations/transformers/expectations/cpu-test/bert-mic.json"
            ) == {"reviewed": True}
            assert (folder_path / ".mic-repo.json").exists()
            calls.append(repo_id)
            return SimpleNamespace(oid="c" * 40)

    monkeypatch.setattr(publisher, "HfApi", API)
    monkeypatch.setattr(sys, "argv", ["publish_mic"])
    assert publisher.main() == 0
    contract = project.lock["contracts"]["bert"]
    assert (
        contract["integration"]["revision"]
        == contract["fixtures"][project.fixture]["revision"]
        == "c" * 40
    )
    assert calls == ["huggingface-mic/bert-mic"]


def test_merged_hub_card_keeps_both_descriptions(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "example"))
    from mic_card import merged_card

    card = merged_card(
        "huggingface-mic/example-mic",
        "# Usage\n\nOriginal usage explanation.",
        "---\ntags: [mic]\n---\n# Weights\n\nOriginal parameter count and build details.",
        "owner/model",
    )
    assert "Original usage explanation." in card
    assert "Original parameter count and build details." in card
    assert "integrations/transformers/expectations" in card
    assert card.startswith("---\n")


def test_dataset_pin_resolves_main_once_and_checks_both_fleets(monkeypatch, tmp_path):
    from transformers_mic import fleet

    calls = []

    class API:
        def dataset_info(self, repo, revision):
            calls.append((repo, revision))
            return SimpleNamespace(sha="d" * 40)

    def download(repo, filename, **kwargs):
        assert repo == fleet.REPOSITORY
        assert kwargs == {"repo_type": "dataset", "revision": "d" * 40}
        return str(tmp_path / filename)

    monkeypatch.setattr(fleet, "HfApi", API)
    monkeypatch.setattr(fleet, "hf_hub_download", download)
    checkout = tmp_path / "transformers"
    assert main(["pin", "--transformers-checkout", str(checkout)]) == 0
    assert calls == [(fleet.REPOSITORY, "main")]
    assert (checkout / "tests/contracts.lock").read_text() == "d" * 40 + "\n"


def test_dataset_export_scopes_known_failures_and_uses_prototype_file(
    workspace, dataset
):
    approve_receipts(workspace)
    original = read(dataset[0])
    original["known_failures"] = {"Other/target": {"issue": "keep"}}
    write(dataset[0], original)
    commands.export(
        SimpleNamespace(workspace=workspace.root, fleet="prototypes", execute=True)
    )
    op = dataset[1][0]["operations"][0]
    assert op.path_in_repo == "prototype-fleet.lock"
    lock = json.loads(op.path_or_fileobj)
    assert lock["known_failures"]["Other/target"] == {"issue": "keep"}
    assert all(k.startswith(("Other/", "BERT/")) for k in lock["known_failures"])


def test_dataset_export_rejects_environment_change(workspace, dataset):
    approve_receipts(workspace)
    lock = read(dataset[0])
    lock["environment"] = {"torch": "changed"}
    write(dataset[0], lock)
    with pytest.raises(ValueError, match="environment differs") as error:
        commands.export(SimpleNamespace(workspace=workspace.root, execute=True))
    assert "torch: fleet=changed, model=" in str(error.value)
    assert "measure the model" in str(error.value)
    assert not dataset[1]


def test_export_rejects_a_dataset_changed_since_human_preview(workspace, dataset):
    approve_receipts(workspace)
    with pytest.raises(ValueError, match="changed since preview"):
        commands.export(
            SimpleNamespace(
                workspace=workspace.root, execute=True, expected_revision="f" * 40
            )
        )
    assert not dataset[1]


def test_export_moves_known_failure_to_the_renamed_target(
    workspace, dataset, monkeypatch
):
    source = workspace.lock
    fixture, pin = next(iter(source["contracts"]["BERT"]["fixtures"].items()))
    old_target = pin.get("target", fixture.split("/")[-1])
    source["known_failures"] = {
        f"BERT/{old_target}": {"issue": "tracked", "skip_outputs": ["training.loss"]}
    }
    write(workspace.lock_path, source)
    approve_receipts(workspace)
    original = commands.upgrade_contract

    def canonical(contract):
        result = original(contract)
        for repo, pin in result["fixtures"].items():
            pin["target"] = repo.split("/")[-1]
        return result

    monkeypatch.setattr(commands, "upgrade_contract", canonical)
    commands.export(SimpleNamespace(workspace=workspace.root, execute=True))
    exported = json.loads(dataset[1][0]["operations"][0].path_or_fileobj)
    target = fixture.split("/")[-1]
    assert exported["known_failures"][f"BERT/{target}"]["skip_outputs"] == [
        "training.loss"
    ]
    assert f"BERT/{old_target}" not in exported["known_failures"]
