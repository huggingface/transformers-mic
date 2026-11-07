import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformers_mic import commands, history, planner, policy, resources
from transformers_mic.cli import main, parser
from transformers_mic.workspace import Workspace, fleet_environment, read, write


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


def test_weight_policy_skips_dtypes_declared_unhealthy():
    build = {
        "weights": "random",
        "gate": {"random": {"pass": True}},
        "numeric_health": {"float32": {"finite": True}, "bfloat16": {"finite": False}},
        "recipe": {"accept_unhealthy": {"bfloat16": "FNet's FFT rejects half precision on CPU"}},
    }
    assert policy.decision(build)["accepted"]
    build["recipe"] = {}
    assert not policy.decision(build)["accepted"]


def test_usage_must_not_enable_remote_code():
    with pytest.raises(ValueError, match="trust_remote_code"):
        commands.check_no_remote_code({"usage": "AutoModel.from_pretrained(model_id, trust_remote_code=True)"})
    commands.check_no_remote_code({"usage": "AutoModel.from_pretrained(model_id)"})


def test_a_large_checkpoint_must_train_a_subset():
    full = {"usage": "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)"}
    with pytest.raises(ValueError, match="4.7B parameters"):
        commands.check_partial_training(full, {"parameters": 4_676_271_104})
    commands.check_partial_training(full, {"parameters": 4_676_271_104, "fixtures_only": True})
    commands.check_partial_training(full, {"parameters": 2_000_000_000})
    commands.check_partial_training({"usage": "model.requires_grad_(False)"}, {"parameters": 4_676_271_104})


def test_a_frozen_model_must_train_parameters_the_fixture_has():
    # jetmoe/jetmoe-8b, run 42017175194: "optimizer got an empty parameter list".
    jetmoe = """model.requires_grad_(False)
for name, param in model.named_parameters():
    if name.startswith("model.layers.23.") or name.startswith("lm_head."):
        param.requires_grad = True
"""
    with pytest.raises(ValueError, match="layer 23 does not exist"):
        commands.check_trainable_selection({"usage": jetmoe}, {"tie_word_embeddings": True})
    head = jetmoe.replace('name.startswith("model.layers.23.") or ', "")
    with pytest.raises(ValueError, match="tie_word_embeddings"):
        commands.check_trainable_selection({"usage": head}, {"tie_word_embeddings": True})
    commands.check_trainable_selection({"usage": head}, {"tie_word_embeddings": False})
    last = 'model.requires_grad_(False)\nlast = f"model.layers.{model.config.num_hidden_layers - 1}."\n'
    commands.check_trainable_selection({"usage": last}, {"tie_word_embeddings": True})
    commands.check_trainable_selection({"usage": 'name.startswith("model.layers.23.")'}, {})  # nothing frozen
    # Run 42019643731: every layer's experts trained, CUDA out of memory on the real checkpoint.
    every = """model.requires_grad_(False)
for name, param in model.named_parameters():
    if name.endswith("mlp.input_linear.weight"):
        param.requires_grad_(True)
"""
    with pytest.raises(ValueError, match="every layer"):
        commands.check_trainable_selection({"usage": every}, {}, 8.5e9)
    commands.check_trainable_selection({"usage": every}, {}, 1e9)  # a full step fits
    head = every.replace("mlp.input_linear.weight", "lm_head.weight")
    commands.check_trainable_selection({"usage": head}, {}, 8.5e9)


def test_legacy_fixture_can_only_be_reused():
    metadata = {"training": {"steps": 20}}
    assert policy.decision(metadata, reused=True)["accepted"]
    assert not policy.decision(metadata)["accepted"]


def test_default_provider_and_model():
    args = parser().parse_args(["create", "model/name", "--out", "project"])
    assert args.provider == "together"
    assert args.llm_model == "deepseek-ai/DeepSeek-V4-Pro-0813"
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
    assets = copy.deepcopy(plan)
    assets["manifest"]["fixtures"][workspace.fixture]["build"]["copy_assets"] = [
        "tokenizer.json"
    ]
    with pytest.raises(ValueError, match="remove copy_assets from the build recipe"):
        commands.validate_plan(assets, workspace.state["model"], workspace.fixture)
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
            "contracts": {
                "Other": {
                    "integration": {
                        "repo": "huggingface-mic/other-mic",
                        "revision": "c" * 40,
                        "files_sha256": "d" * 64,
                    },
                    "fixtures": {
                        "huggingface-mic/other-mic": {
                            "revision": "c" * 40,
                            "files_sha256": "e" * 64,
                        }
                    },
                    "checkpoints": {"org/other": "f" * 40},
                }
            },
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


def exported(call, name):
    """The lock a fleet export uploaded, read back from its tables."""
    from transformers_mic import fleet

    files = {op.path_in_repo: op.path_or_fileobj.decode() for op in call["operations"]}
    assert {fleet.table_path(name, t) for t in fleet.COLUMNS} <= set(files)
    return fleet.from_tables({Path(path).stem: value for path, value in files.items()})


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
    lock = exported(call, "control")
    assert lock["contracts"]["Other"] == read(dataset[0])["contracts"]["Other"]
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


def test_retire_removes_one_contract_after_a_preview(workspace, dataset):
    args = SimpleNamespace(fleet="prototypes", contract="Other", reason="test source", execute=False)
    commands.retire(args)
    assert dataset[1] == []
    args.execute = True
    commands.retire(args)
    call = dataset[1][0]
    assert call["parent_commit"] == "a" * 40
    assert "Other" not in exported(call, "prototypes")["contracts"]
    with pytest.raises(ValueError, match="not in the prototypes fleet"):
        commands.retire(SimpleNamespace(fleet="prototypes", contract="Missing", reason="x", execute=False))


def test_export_rejects_case_collision_and_allows_explicit_name(workspace, dataset):
    approve_receipts(workspace)
    old = read(dataset[0])
    old["contracts"]["bert"] = old["contracts"]["Other"]
    write(dataset[0], old)
    args = SimpleNamespace(workspace=workspace.root, execute=True)
    with pytest.raises(ValueError, match="differs only by case"):
        commands.export(args)
    args.contract_name = "mic-bert"
    commands.export(args)
    lock = exported(dataset[1][0], "control")
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


@pytest.mark.requires_runtime
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


def test_parameter_names_accept_suffixes_and_hint_every_layer_kind():
    pytest.importorskip("torch")
    from transformers import DeepseekV3Config

    # Block 0 dense, block 1 MoE: the shape of Mistral-Large-4.
    config = DeepseekV3Config(
        vocab_size=32, hidden_size=16, intermediate_size=16, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        n_routed_experts=2, num_experts_per_tok=1, first_k_dense_replace=1,
        q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, qk_nope_head_dim=4, v_head_dim=4,
    ).to_dict()
    plan = {"manifest": {"architecture": "DeepseekV3ForCausalLM"}}
    commands.check_parameter_names({**plan, "usage": 'next(n for n in names if n.endswith("mlp.gate_proj.weight"))'}, config)
    with pytest.raises(ValueError) as error:
        commands.check_parameter_names({**plan, "usage": '"model.layers.1.mlp.down_proj.weight"'}, config)
    assert "model.layers.1.mlp.experts." in str(error.value)


def test_index_only_misses_name_the_suffix_lookup():
    # Switch Transformers: routers only in odd blocks; drafts copied T5's block.0.
    real = {"decoder.block.0.layer.2.mlp.wi.weight",
            "decoder.block.1.layer.2.mlp.router.classifier.weight",
            "decoder.block.3.layer.2.mlp.router.classifier.weight"}
    lookup = "n.startswith('decoder.block.') and n.endswith('.layer.2.mlp.router.classifier.weight')"
    (hint,) = commands.index_only_misses(["decoder.block.0.layer.2.mlp.router.classifier.weight"], real)
    assert "decoder.block.1.layer.2.mlp.router.classifier.weight is (layers 1, 3)" in hint and lookup in hint
    # A dropped sublayer: the lookup still uses the real suffix.
    (hint,) = commands.index_only_misses(["decoder.block.1.mlp.router.classifier.weight"], real)
    assert lookup in hint
    assert commands.index_only_misses(["decoder.block.0.layer.0.nope.weight"], real) == []


def test_a_reply_that_is_not_a_draft_does_not_use_a_draft(monkeypatch, capsys):
    plan = {"name": "M"}
    replies = [json.JSONDecodeError("Unterminated string", "{", 0),
               planner.MalformedDraft("'rationale' is a required property"), (plan, {})]

    def propose(context, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(planner, "propose", propose)
    args = SimpleNamespace(llm_model="m", provider="p", bill_to="")
    assert commands.propose_well_formed({}, args, attempt=0, retries=2) == (plan, {})
    assert capsys.readouterr().err.count("not a draft") == 2
    replies[:] = [planner.MalformedDraft("empty")] * 3
    with pytest.raises(planner.MalformedDraft):
        commands.propose_well_formed({}, args, attempt=0, retries=2)


def test_planner_logs_the_last_rejected_draft(monkeypatch, capsys):
    monkeypatch.setattr(planner, "propose", lambda context, **kwargs: ({"name": "M", "manifest": {}}, {}))
    monkeypatch.setattr(commands, "validate_plan", lambda *a: (_ for _ in ()).throw(ValueError("bad draft")))
    with pytest.raises(ValueError, match="bad draft"):
        commands.propose_valid({}, SimpleNamespace(llm_model="m", provider="p", bill_to=""), "org/m", "f")
    assert "planner draft 3/3 rejected: bad draft" in capsys.readouterr().err


def test_parse_reports_a_missing_top_level_key_as_not_a_draft():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            {"name": "M", "manifest": {}, "usage": "x" * 100, "historical_candidates": []})), finish_reason="stop")],
        usage=None,
    )
    with pytest.raises(planner.MalformedDraft, match="rationale"):
        planner.parse(response, "m", "p", None)


def test_compare_rule_spelling_slips_are_normalized():
    plan = {"manifest": {"outputs": {"g": {"compare": "alltrue"}, "l": {"compare": "AllClose"}, "x": {"compare": "fuzzy"}}}}
    commands.normalize_compare(plan)
    assert [r["compare"] for r in plan["manifest"]["outputs"].values()] == ["all_true", "allclose", "fuzzy"]


def test_unknown_manifest_keys_are_dropped_from_drafts():
    plan = {"manifest": {"mic": "0.3", "name": "M", "checkpoints_note": "placeholder"}}
    assert commands.drop_unknown_manifest_keys(plan) == ["checkpoints_note"]
    assert plan["manifest"] == {"mic": "0.3", "name": "M"}


def test_checkpoints_above_the_gpu_limit_are_not_loaded():
    context = {"parameters": 1_053_000_000_000}
    assert commands.loaded_parameters(context) is None
    commands.check_partial_training({"usage": "model(**inputs).loss.backward()"}, context)
    with pytest.raises(ValueError, match="requires_grad_"):
        commands.check_partial_training({"usage": "model(**inputs).loss.backward()"}, {"parameters": 8e9})


def test_planner_context_names_one_parameter_per_layer_kind():
    names = {"model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.experts.down_proj",
             "model.layers.2.mlp.experts.down_proj", "lm_head.weight"}
    assert commands.names_per_layer_kind(names) == [
        "lm_head.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.experts.down_proj"]
    assert commands.names_per_layer_kind(names, pattern=True) == [
        "lm_head.weight", "model.layers.N.mlp.down_proj.weight", "model.layers.N.mlp.experts.down_proj"]


def test_parameter_patterns_keep_sublayer_indices():
    # Switch Transformers: the router is in every other block, always in sublayer 1.
    names = {"encoder.block.0.layer.0.SelfAttention.q.weight",
             "encoder.block.0.layer.1.mlp.wi.weight",
             "encoder.block.1.layer.1.mlp.router.classifier.weight",
             "encoder.block.3.layer.1.mlp.router.classifier.weight",
             "decoder.block.0.layer.1.EncDecAttention.q.weight",
             "decoder.block.2.layer.1.EncDecAttention.q.weight"}
    assert commands.names_per_layer_kind(names, pattern=True) == [
        "decoder.block.N.layer.1.EncDecAttention.q.weight",
        "encoder.block.N.layer.0.SelfAttention.q.weight",
        "encoder.block.N.layer.1.mlp.router.classifier.weight",
        "encoder.block.N.layer.1.mlp.wi.weight"]


@pytest.mark.requires_runtime
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
    assert "mic-coverage.static.hf.space/shrink/example-mic.html" in card
    assert card.startswith("---\n")


def test_dataset_pin_resolves_main_once_and_checks_both_fleets(monkeypatch, tmp_path):
    from transformers_mic import fleet

    calls = []

    class API:
        def dataset_info(self, repo, revision):
            calls.append((repo, revision))
            return SimpleNamespace(sha="d" * 40)

    lock = {"framework": "transformers", "environment": {}, "contracts": {}}
    tables = {
        **fleet.to_tables(lock),
        **dict.fromkeys(fleet.FILES.values(), json.dumps(lock)),
    }

    def download(repo, filename, **kwargs):
        assert repo == fleet.REPOSITORY
        assert kwargs == {"repo_type": "dataset", "revision": "d" * 40}
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if Path(filename).stem == "integrations":
            from huggingface_hub.errors import EntryNotFoundError
            raise EntryNotFoundError(filename)
        path.write_text(tables.get(filename) or tables[Path(filename).stem])
        return str(path)

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
    lock = exported(dataset[1][0], "prototypes")
    # An issue-only entry is the schema's plain string form
    assert lock["known_failures"]["Other/target"] == "keep"
    assert all(k.startswith(("Other/", "BERT/")) for k in lock["known_failures"])


def test_new_contracts_carry_the_fleet_environment(tmp_path, monkeypatch, capsys):
    from transformers_mic import fleet

    lock = tmp_path / "prototype-fleet.lock"
    write(lock, {"environment": {"torch": "2.14.0"}})
    monkeypatch.setattr(fleet, "download", lambda name: (lock, "a" * 40))
    assert fleet_environment({"environment": {"torch": "2.11.0"}}) == {
        "torch": "2.14.0"
    }

    def offline(name):
        raise OSError("no network")

    monkeypatch.setattr(fleet, "download", offline)
    assert fleet_environment({"environment": {"torch": "2.11.0"}}) == {
        "torch": "2.11.0"
    }
    assert "fleet environment unavailable (no network)" in capsys.readouterr().err


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
    lock = exported(dataset[1][0], "control")
    target = fixture.split("/")[-1]
    assert lock["known_failures"][f"BERT/{target}"]["skip_outputs"] == ["training.loss"]
    assert f"BERT/{old_target}" not in lock["known_failures"]


def test_binary_only_checkpoint_reads_names_from_conversion_pr(tmp_path, monkeypatch):
    from transformers_mic import inspection

    class Api:
        def file_exists(self, repo, filename, revision=None):
            return False

        def get_repo_discussions(self, repo, **filters):
            assert filters["author"] == "SFconvertbot"
            return [
                SimpleNamespace(num=1, title="Something else"),
                SimpleNamespace(
                    num=4, title="Adding `safetensors` variant of this model"
                ),
            ]

        def get_safetensors_metadata(self, repo, revision=None):
            assert revision == "refs/pr/4"
            return SimpleNamespace(weight_map={"encoder.weight": "model.safetensors"})

        def model_info(self, *args, **kwargs):
            raise AssertionError("must not fall back to the binary checkpoint")

    monkeypatch.setattr(inspection, "HfApi", Api)
    assert inspection.checkpoint_keys("org/model", "a" * 40, tmp_path) == {
        "encoder.weight"
    }


@pytest.mark.requires_runtime
@pytest.mark.parametrize("zipped", [True, False])
def test_binary_keys_read_names_without_tensor_data(monkeypatch, tmp_path, zipped):
    import torch

    from transformers_mic import inspection

    state = {"encoder.weight": torch.ones(2, 2), "head.bias": torch.zeros(2), "config": {"a": 1}}
    path = tmp_path / "pytorch_model.bin"
    torch.save(state, path, _use_new_zipfile_serialization=zipped)

    class FileSystem:
        def open(self, name, revision, block_size):
            assert name == "org/model/pytorch_model.bin" and revision == "a" * 40
            return path.open("rb")

    monkeypatch.setattr(inspection, "HfFileSystem", FileSystem)
    assert inspection.binary_keys("org/model", "a" * 40) == {"encoder.weight", "head.bias"}


def test_checkpoint_keys_read_shard_headers_over_a_stale_index(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from transformers_mic import inspection

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm.weight": "a.safetensors"}}))

    class Api:  # MOSS-Transcribe-Diarize-HF: the index omits the audio tower its shards hold
        def get_safetensors_metadata(self, repo, revision=None):
            return SimpleNamespace(weight_map={"lm.weight": "a.safetensors", "audio.weight": "b.safetensors"})

    monkeypatch.setattr(inspection, "HfApi", Api)
    assert inspection.checkpoint_keys("org/model", "a" * 40, tmp_path) == {"lm.weight", "audio.weight"}


def test_checkpoint_keys_follow_transformers_weights(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from transformers_mic import inspection

    # Mistral names its shards consolidated-*; config.json points the loader at them.
    name = "consolidated.safetensors.index.json"
    (tmp_path / "config.json").write_text(json.dumps({"transformers_weights": name}))
    (tmp_path / name).write_text(json.dumps({"weight_map": {"a.weight": "consolidated-1.safetensors", "b.weight": "consolidated-2.safetensors"}}))
    headers = {"consolidated-1.safetensors": {"a.weight"}, "consolidated-2.safetensors": {"b.weight", "c.weight"}}

    class Api:
        def parse_safetensors_file_metadata(self, repo, filename, revision=None, timeout=None):
            return SimpleNamespace(tensors=dict.fromkeys(headers[filename]))

        def file_exists(self, *args, **kwargs):
            raise AssertionError("must not look for model.safetensors or pytorch_model.bin")

    monkeypatch.setattr(inspection, "HfApi", Api)
    assert inspection.checkpoint_keys("org/model", "a" * 40, tmp_path) == {"a.weight", "b.weight", "c.weight"}
    assert inspection.loads_binary("org/model", "a" * 40, tmp_path) is False


@pytest.mark.parametrize("name", ["../x.safetensors", "pytorch_model.bin", 3])
def test_transformers_weights_must_be_a_local_safetensors_name(tmp_path, name):
    from transformers_mic import inspection

    (tmp_path / "config.json").write_text(json.dumps({"transformers_weights": name}))
    with pytest.raises(ValueError, match="transformers_weights"):
        inspection.explicit_weights(tmp_path)


def test_checkpoint_keys_name_an_unknown_weight_layout(monkeypatch, tmp_path):
    from transformers_mic import inspection

    class Api:
        def file_exists(self, repo, filename, revision=None):
            return False

        def get_repo_discussions(self, repo, **filters):
            return []

        def list_repo_files(self, repo, revision=None):
            return ["config.json", "weights-00001.safetensors", "weights.index.json"]

    monkeypatch.setattr(inspection, "HfApi", Api)
    with pytest.raises(ValueError, match="weights-00001.safetensors"):
        inspection.checkpoint_keys("org/model", "a" * 40, tmp_path)


def test_parameter_count_sums_explicit_shards_the_hub_does_not_count(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import huggingface_hub

    from transformers_mic import inspection

    name = "consolidated.safetensors.index.json"
    (tmp_path / "config.json").write_text(json.dumps({"transformers_weights": name}))
    (tmp_path / name).write_text(json.dumps({"weight_map": {"a": "c-1.safetensors", "b": "c-2.safetensors", "c": "c-2.safetensors"}}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, f, revision=None: str(tmp_path / f))

    class Api:
        def parse_safetensors_file_metadata(self, repo, filename, revision=None, timeout=None):
            return SimpleNamespace(parameter_count={"F8_E4M3": 100, "BF16": 5} if filename == "c-2.safetensors" else {"BF16": 7})

    monkeypatch.setattr(inspection, "HfApi", Api)
    assert inspection.explicit_parameter_count("org/model", "a" * 40) == 112


def test_visible_usage_never_shows_runner_plumbing():
    blocks = [
        "model = AutoModel.from_pretrained(model_id).to(device)\n"
        "if variant:  # doc-builder: hide\n"
        "    model = AutoModel.from_pretrained(model_id, **variant)\n"
        "print(model)\n"
    ]
    assert commands.visible_plumbing(blocks) == []
    blocks = ["model = AutoModel.from_pretrained(model_id, **variant).to(device)\n"]
    assert commands.visible_plumbing(blocks) == [
        "model = AutoModel.from_pretrained(model_id, **variant).to(device)"
    ]


def test_create_rejects_unknown_override_fields_and_adds_missing_requires():
    plan = {
        "manifest": {
            "fixtures": {
                "org/fixture": {
                    "build": {"config_overrides": {"fields": {"hidden_size": 64}}}
                }
            },
            "integrations": {"transformers": {"requires": ["torch"]}},
        }
    }
    config = {
        "model_type": "not-a-registered-type",
        "text_config": {"hidden_size": 1024},
        "vision_config": {"model_type": "timm_wrapper"},
    }
    with pytest.raises(ValueError, match="text_config"):
        commands.check_config_facts(plan, config)
    fields = plan["manifest"]["fixtures"]["org/fixture"]["build"]["config_overrides"]
    fields["fields"] = {"text_config.hidden_size": 64}
    commands.check_config_facts(plan, config)
    requires = plan["manifest"]["integrations"]["transformers"]["requires"]
    assert requires == ["torch", "timm", "torchvision"]
    commands.check_config_facts(plan, config)  # idempotent
    assert requires == ["torch", "timm", "torchvision"]
    plan["usage"] = 'model = AutoModel.from_pretrained(model_id, device_map="auto")'
    commands.check_config_facts(plan, config)
    assert plan["manifest"]["integrations"]["transformers"]["requires"][-1] == "accelerate"


def test_overrides_never_shrink_the_checkpoint_vocabulary():
    fields = {"hidden_size": 64, "vocab_size": 1024, "text_config.vocab_size": 512}
    plan = {
        "manifest": {
            "fixtures": {"org/f": {"build": {"config_overrides": {"fields": fields}}}},
            "integrations": {"transformers": {"requires": ["torch"]}},
        }
    }
    config = {"model_type": "x", "hidden_size": 4096, "vocab_size": 32000, "text_config": {"vocab_size": 256}}
    commands.check_config_facts(plan, config)
    assert fields == {"hidden_size": 64, "text_config.vocab_size": 512}


@pytest.mark.requires_runtime
def test_sentencepiece_tokenizers_add_their_backends_to_requires():
    plan = {"manifest": {"integrations": {"transformers": {"requires": ["torch"]}}}}
    commands.check_config_facts(plan, {"model_type": "siglip"})
    requires = plan["manifest"]["integrations"]["transformers"]["requires"]
    assert requires == ["torch", "sentencepiece", "protobuf"]
    plan = {"manifest": {"integrations": {"transformers": {"requires": ["torch"]}}}}
    commands.check_config_facts(plan, {"model_type": "bert"})
    assert plan["manifest"]["integrations"]["transformers"]["requires"] == ["torch"]


def test_create_rejects_usage_reading_names_it_never_defines():
    usage = (
        "```py runnable:a\nimport torch\nlogits = torch.ones(2)\n```\n"
        "```py runnable:a:2\nchecked = ('w',)\nresult = {n: logits[0] for n in checked}\n```\n"
    )
    commands.check_undefined_names({"usage": usage})  # state carries across :2
    with pytest.raises(ValueError, match="gradients"):
        commands.check_undefined_names(
            {"usage": usage.replace("for n in checked", "for n in gradients")}
        )
    # A separate runnable id does not see the first block's names.
    with pytest.raises(ValueError, match="logits"):
        commands.check_undefined_names(
            {"usage": usage.replace("runnable:a:2", "runnable:b")}
        )


@pytest.mark.requires_runtime
def test_create_rejects_model_classes_that_cannot_load_the_type():
    def plan(model_class, auto):
        return {
            "manifest": {"fixtures": {"org/f": {"build": {"model_class": model_class}}}},
            "usage": f"model = {auto}.from_pretrained(model_id)",
        }

    config = {"model_type": "m2m_100"}
    commands.check_model_classes(plan("M2M100ForConditionalGeneration", "AutoModelForSeq2SeqLM"), config)
    with pytest.raises(ValueError, match="M2M100ForConditionalGeneration"):
        commands.check_model_classes(plan("M2m100ForConditionalGeneration", "AutoModelForSeq2SeqLM"), config)
    with pytest.raises(ValueError, match="AutoModelForCausalLM does not map"):
        commands.check_model_classes(plan("M2M100ForConditionalGeneration", "AutoModelForCausalLM"), config)


def test_create_rejects_targets_without_the_drafted_fixture():
    def plan(targets):
        return {
            "manifest": {
                "fixtures": {"org/f": {}},
                "integrations": {"transformers": {"targets": targets}},
            }
        }

    commands.check_targets(plan(["fixtures", "checkpoints"]))
    with pytest.raises(ValueError, match="repeat"):
        commands.check_targets(plan(["checkpoints", "checkpoints"]))
    with pytest.raises(ValueError, match="omit 'fixtures'"):
        commands.check_targets(plan(["checkpoints"]))


def test_create_keeps_large_checkpoints_off_the_cpu_runner():
    manifest = {
        "checkpoints": {"org/big": {}, "org/small": {}, "org/gpu": {}},
        "integrations": {"transformers": {
            "variants": {"default": {}, "eager": {"attn_implementation": "eager"}, "bfloat16": {"dtype": "bfloat16"}},
            "resources": {
                "org/big": {"memory_gb": 16},
                "org/gpu": {"devices": ["cuda"]},
            },
        }},
    }
    parameters = {"org/big": 2_600_000_000, "org/small": 500_000_000, "org/gpu": 9e9}
    assert commands.keep_large_checkpoints_off_cpu(manifest, parameters) == ["org/big", "org/gpu"]
    resources = manifest["integrations"]["transformers"]["resources"]
    float32_off = {"default": {"devices": []}, "eager": {"devices": []}}
    assert resources["org/big"] == {"memory_gb": 16, "devices": ["cuda"], "variants": float32_off}
    assert resources["org/gpu"] == {"devices": ["cuda"], "variants": float32_off}
    assert "org/small" not in resources
    # The checkpoint loads visibly in bfloat16, so the fixture's float32 variants say so explicitly.
    assert manifest["integrations"]["transformers"]["variants"] == {
        "default": {"dtype": "float32"},
        "eager": {"dtype": "float32", "attn_implementation": "eager"},
        "bfloat16": {"dtype": "bfloat16"},
    }
    assert commands.keep_large_checkpoints_off_cpu(manifest, parameters) == []
    assert commands.keep_large_checkpoints_off_cpu(manifest, {"org/big": None}) == []
    # Above 10B even the bfloat16 weights do not fit the GPU: every variant is declared unmeasured.
    manifest["checkpoints"]["org/huge"] = {}
    assert commands.keep_large_checkpoints_off_cpu(manifest, {"org/huge": 15_750_057_456}) == ["org/huge"]
    assert resources["org/huge"]["variants"] == {v: {"devices": []} for v in ("default", "eager", "bfloat16")}
    jsonschema = pytest.importorskip("jsonschema")
    schema = commands.manifest_schema()["properties"]["integrations"]["additionalProperties"]["properties"]["resources"]
    jsonschema.validate(resources, schema)


def test_github_api_reads_retry_transient_failures(monkeypatch):
    import subprocess

    from transformers_mic import github_baselines

    calls = []

    def flaky(*args):
        calls.append(args)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(1, ["gh", *args])
        return '{"status": "in_progress"}'

    monkeypatch.setattr(github_baselines, "gh", flaky)
    monkeypatch.setattr(github_baselines.time, "sleep", lambda s: None)
    assert github_baselines.api("repos/x/runs/1") == {"status": "in_progress"}
    assert len(calls) == 3
    monkeypatch.setattr(github_baselines, "gh", lambda *a: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "gh")))
    with pytest.raises(subprocess.CalledProcessError):
        github_baselines.api("repos/x/runs/1")


def test_fp8_checkpoint_names_drop_their_scales():
    from transformers_mic.inspection import logical_checkpoint_keys

    config = SimpleNamespace(
        model_type="minimax_m3_vl", quantization_config={"quant_method": "mxfp8"}
    )
    keys = {"layer.w1.weight", "layer.w1.weight_scale_inv", "norm.weight"}
    assert logical_checkpoint_keys(config, keys) == {"layer.w1.weight", "norm.weight"}
    with pytest.raises(ValueError, match="scale without its weight"):
        logical_checkpoint_keys(config, {"layer.w2.weight_scale_inv"})


@pytest.mark.requires_runtime
def test_split_weights_are_expanded_after_renaming(tmp_path, monkeypatch):
    # nomic_bert stores a fused encoder.layers.N.attn.Wqkv; Transformers renames
    # encoder.layers -> layers and chunks it into q/k/v_proj.
    import torch
    from transformers import NomicBertConfig, NomicBertModel

    from transformers_mic import inspection

    NomicBertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    ).save_pretrained(tmp_path)
    with torch.device("meta"):
        model = NomicBertModel(NomicBertConfig.from_pretrained(tmp_path))
    legacy = {
        name.replace("layers.", "encoder.layers.")
        .replace("self_attn.q_proj", "attn.Wqkv")
        .replace("self_attn.o_proj", "attn.out_proj")
        .replace("up_proj", "fc11")
        .replace("gate_proj", "fc12")
        .replace("down_proj", "fc2")
        .replace("post_attention_layernorm", "norm1")
        .replace("post_mlp_layernorm", "norm2")
        .replace("embeddings.LayerNorm", "emb_ln")
        for name in model.state_dict()
        if "k_proj" not in name and "v_proj" not in name
    }
    monkeypatch.setattr(inspection, "checkpoint_keys", lambda *args: legacy)
    assert (
        inspection.inspect_checkpoint(NomicBertModel, tmp_path, "org/m", "a" * 40) == []
    )


@pytest.mark.requires_runtime
def test_scoped_split_weights_are_expanded(tmp_path, monkeypatch):
    # Unlimited-OCR's fused vision qkv_proj is split by a converter scoped to the
    # renamed sub-model (vision_model -> vision_tower.vision_encoder), which never
    # matches the checkpoint's original key.
    import torch
    from transformers import (
        UnlimitedOcrConfig,
        UnlimitedOcrForConditionalGeneration as Model,
    )

    from transformers_mic import inspection

    UnlimitedOcrConfig().save_pretrained(tmp_path)
    with torch.device("meta"):
        model = Model._from_config(UnlimitedOcrConfig.from_pretrained(tmp_path))
    encoder = "model.vision_tower.vision_encoder.encoder.layers."
    original = {
        name.replace(encoder, "model.vision_model.transformer.layers.").replace(
            "self_attn.q_proj", "self_attn.qkv_proj"
        )
        if name.startswith(encoder)
        else name
        for name in model.state_dict()
        if not (name.startswith(encoder) and ("k_proj" in name or "v_proj" in name))
    }
    monkeypatch.setattr(inspection, "checkpoint_keys", lambda *args: original)
    assert inspection.inspect_checkpoint(Model, tmp_path, "org/m", "a" * 40) == []


def test_planner_backs_off_on_rate_limits_then_gives_up(monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    calls, waits = [], []

    class Client:
        def chat_completion(self, **kwargs):
            calls.append(kwargs)
            raise HfHubHTTPError(
                "busy",
                response=SimpleNamespace(
                    status_code=429, headers={"retry-after": "7"}, request=None
                ),
            )

    monkeypatch.setattr(planner.time, "sleep", waits.append)
    with pytest.raises(HfHubHTTPError):
        planner.complete_with_backoff(Client(), {}, "m", 10)
    assert len(calls) == 4 and waits == [7, 7, 7]


@pytest.mark.requires_runtime
def test_planner_context_names_loaded_parameters_without_architectures():
    from transformers import BertConfig

    from transformers_mic import references

    config = BertConfig(
        num_hidden_layers=2, hidden_size=32, num_attention_heads=2, intermediate_size=64
    ).to_dict()
    config.pop("architectures", None)
    classes = references.auto_classes(config)
    assert "BertModel" in classes
    names = references.parameter_names("BertModel", config)
    assert "encoder.layer.0.attention.self.query.weight" in names
    assert not any(".1." in name for name in names)  # one block shows the scheme


@pytest.mark.requires_runtime
def test_planner_context_replaces_a_config_class_transformers_lacks(monkeypatch):
    from transformers import BertConfig

    from transformers_mic import references

    monkeypatch.setattr(references, "hf_hub_download", lambda *a, **kw: "/missing")
    config = BertConfig(
        num_hidden_layers=2, hidden_size=32, num_attention_heads=2, intermediate_size=64
    ).to_dict()
    config["architectures"] = ["BertForEverything"]
    context = references.collect("org/bert", "main", config)
    assert "BertModel" in context["transformers_classes"]
    assert "BertForEverything" not in context["transformers_classes"]


def test_fixture_resources_drop_hardware_requirements():
    manifest = {"integrations": {"transformers": {"resources": {
        "fixture": {"devices": ["cuda"], "gpus": 1, "gpu_memory_gb": 22, "memory_gb": 64, "download_mb": 200},
        "org/large": {"devices": ["cuda"], "gpus": 1},
    }}}}
    assert commands.fixture_runs_anywhere(manifest) == ["devices", "gpu_memory_gb", "gpus", "memory_gb"]
    resources = manifest["integrations"]["transformers"]["resources"]
    assert resources["fixture"] == {"download_mb": 200}
    assert resources["org/large"] == {"devices": ["cuda"], "gpus": 1}
    assert commands.fixture_runs_anywhere({"integrations": {"transformers": {}}}) == []


@pytest.mark.requires_runtime
def test_create_rejects_imports_transformers_does_not_export():
    usage = "```py runnable:x\nfrom transformers import (\n    AutoModelForConditionalGeneration,\n    AutoProcessor,\n)\n```\n"
    with pytest.raises(ValueError, match="AutoModelForConditionalGeneration"):
        commands.check_imports({"usage": usage})
    commands.check_imports({"usage": usage.replace("AutoModelForConditionalGeneration", "AutoModelForImageTextToText")})


def test_planner_slugs_a_dotted_contract_name():
    from types import SimpleNamespace

    from transformers_mic import planner

    content = json.dumps(
        {"name": "Kimi-K2.5", "manifest": {}, "usage": "x" * 100, "rationale": "", "historical_candidates": []}
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=None,
    )
    plan, _ = planner.parse(response, "m", "p", None)
    assert plan["name"] == "Kimi-K2-5"


def test_planner_renames_a_historical_key_slip_and_keeps_invalid_drafts():
    from types import SimpleNamespace

    import jsonschema

    from transformers_mic import planner

    def parse(draft):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(draft)), finish_reason="stop")],
            usage=None,
        )
        return planner.parse(response, "m", "p", None)

    draft = {"name": "M", "manifest": {}, "usage": "x" * 100, "rationale": ""}
    plan, _ = parse({**draft, "historical_issues": [{"url": "u"}]})
    assert plan["historical_candidates"] == [{"url": "u"}] and "historical_issues" not in plan
    with pytest.raises(jsonschema.ValidationError) as error:
        parse({**draft, "historical_candidates": [], "usage": "short"})
    assert error.value.plan["usage"] == "short"


@pytest.mark.requires_runtime
def test_create_rejects_labels_a_forward_does_not_take():
    def plan(load, call):
        return {"usage": f"model = {load}.from_pretrained(model_id)\nloss = model(**inputs, {call}).loss\n"}

    owlv2 = {"model_type": "owlv2"}
    with pytest.raises(ValueError, match="takes no labels"):
        commands.check_label_arguments(plan("Owlv2ForObjectDetection", "labels=labels"), owlv2)
    mask2former = {"model_type": "mask2former"}
    with pytest.raises(ValueError, match="mask_labels"):
        commands.check_label_arguments(plan("AutoModelForUniversalSegmentation", "labels=labels"), mask2former)
    commands.check_label_arguments(plan("AutoModelForUniversalSegmentation", "mask_labels=m"), mask2former)
    commands.check_label_arguments(plan("AutoModelForObjectDetection", "labels=labels"), {"model_type": "yolos"})


def test_create_rejects_plain_fences_and_float16_loads():
    usage = "```py runnable:a\nmodel = AutoModel.from_pretrained(model_id)\n```\n\n```python\nx = 1\n```\n"
    with pytest.raises(ValueError, match="not runnable"):
        commands.check_code_fences({"usage": usage})
    commands.check_code_fences({"usage": usage.replace("```python", "```py runnable:a:2")})
    with pytest.raises(ValueError, match="dtype=torch.float32"):
        commands.check_float16_load({"usage": usage}, {"torch_dtype": "float16"})
    commands.check_float16_load({"usage": usage}, {"dtype": "bfloat16"})
    fixed = usage.replace("(model_id)", "(model_id, dtype=torch.float32)")
    commands.check_float16_load({"usage": fixed}, {"torch_dtype": "float16"})
    # Above 1B the contract runs in bfloat16 only and a float32 copy does not fit the GPU.
    with pytest.raises(ValueError, match="dtype=torch.bfloat16"):
        commands.check_float16_load({"usage": fixed}, {"torch_dtype": "float16"}, 8_000_000_000)
    large = usage.replace("(model_id)", "(model_id, dtype=torch.bfloat16)")
    commands.check_float16_load({"usage": large}, {"torch_dtype": "float16"}, 8_000_000_000)
    with pytest.raises(ValueError, match="dtype=torch.bfloat16"):  # Audio Flamingo 3: stored float32
        commands.check_float16_load({"usage": usage}, {"dtype": "float32"}, 8_000_000_000)
    # A fixtures-only contract loads only its fixture.
    huge = {"parameters": 1_026_879_376_368, "fixtures_only": True}
    commands.check_float16_load({"usage": usage}, {"dtype": "bfloat16"}, commands.loaded_parameters(huge))
    # Above GPU_PARAMETER_LIMIT only the fixture is measured, fixtures-only or not.
    assert commands.loaded_parameters({**huge, "fixtures_only": False}) is None
    assert commands.loaded_parameters({"parameters": 8_000_000_000, "fixtures_only": False}) == 8_000_000_000


def test_create_rejects_the_broadcast_image_idiom():
    bad = "y = a.view(H, 1, 1)\nx = b.view(1, W, 1)\nimage = (x + y).expand(3, H, W)\n"
    with pytest.raises(ValueError, match="view"):
        commands.check_image_shapes({"usage": bad})
    commands.check_image_shapes({"usage": bad.replace("view(H, 1, 1)", "view(1, H, 1)").replace("view(1, W, 1)", "view(1, 1, W)")})


def test_create_rejects_a_loss_built_only_from_no_grad_outputs():
    usage = (
        "```py runnable:a\nwith torch.no_grad():\n    outputs = model(**inputs)\n"
        "features = outputs.last_hidden_state.mean(1)\n```\n\n"
        "```py runnable:a:2\nlogits = probe(features)\nloss = ce(logits, labels)\n```\n"
    )
    with pytest.raises(ValueError, match="no_grad"):
        commands.check_loss_has_gradients({"usage": usage})
    fresh = usage.replace("probe(features)", "probe(model(**inputs).last_hidden_state.mean(1))")
    commands.check_loss_has_gradients({"usage": fresh})
    # A constant target computed under no_grad is fine.
    target = fresh.replace("loss = ce(logits, labels)", "loss = ce(logits, features)")
    commands.check_loss_has_gradients({"usage": target})


def test_nearest_contract_prefers_shared_structure(monkeypatch, tmp_path):
    from transformers_mic import examples

    configs = {
        "org/llama": {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
        "org/mistral3": {"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
                         "vision_config": {"model_type": "pixtral"}, "text_config": {"model_type": "ministral3"}},
        "org/bert": {"architectures": ["BertForMaskedLM"], "model_type": "bert"},
    }

    def download(repo, filename, revision=None):
        path = tmp_path / repo.replace("/", "--") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "config.json":
            path.write_text(json.dumps(configs[repo]))
        elif filename == "integration.json":
            path.write_text(json.dumps({"integrations": {"transformers": {"entry_point": "usage.md"}}}))
        else:
            path.write_text(f"usage of {repo}")
        return str(path)

    monkeypatch.setattr(examples, "hf_hub_download", download)
    rows = [("prototypes", name, {"repo": f"{repo}-mic", "revision": "r"}, {repo: "r"})
            for name, repo in (("Llama", "org/llama"), ("Mistral3", "org/mistral3"), ("BERT", "org/bert"))]
    for repo in list(configs):
        configs[f"{repo}-mic"] = configs[repo]
    large4 = {"architectures": ["MistralLarge4ForConditionalGeneration"], "model_type": "mistral_large4",
              "vision_config": {"model_type": "pixtral"}, "text_config": {"model_type": "mistral_large4_text"}}
    plan, reason = examples.nearest_contract(large4, rows=rows)
    assert plan["name"] == "Mistral3" and "pixtral" in reason
    plan, reason = examples.nearest_contract({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}, rows=rows)
    assert plan["name"] == "Llama" and "same architecture" in reason
    assert examples.nearest_contract({"architectures": ["WhisperModel"], "model_type": "whisper"}, rows=rows) is None
    monkeypatch.setenv("TMIC_EXAMPLE_SKIP_SELF", "1")
    # Without its own contract, nothing else here shares Llama's structure.
    assert examples.nearest_contract(configs["org/llama"], rows=rows, model="org/llama") is None


@pytest.mark.requires_runtime
def test_encoder_decoder_calls_need_decoder_inputs():
    whisper = {"is_encoder_decoder": True, "architectures": ["WhisperForConditionalGeneration"]}
    usage = "```py runnable:t\nlogits = model(inputs.input_features).logits\n```\n"
    with pytest.raises(ValueError, match="decoder_input_ids"):
        commands.check_encoder_decoder_calls({"usage": usage}, whisper)
    for ok in ("model(inputs.input_features, decoder_input_ids=ids)", "model(x, labels=labels)", "model(**inputs)"):
        commands.check_encoder_decoder_calls({"usage": f"```py runnable:t\nout = {ok}\n```\n"}, whisper)
    commands.check_encoder_decoder_calls({"usage": usage}, {"is_encoder_decoder": False})
    # RT-DETR is an encoder-decoder over pixel_values that builds its own queries.
    commands.check_encoder_decoder_calls({"usage": usage}, {"is_encoder_decoder": True, "architectures": ["RTDetrForObjectDetection"]})



def test_vlm_training_text_must_carry_image_tokens():
    # Salesforce/blip2-opt-2.7b (2026-11-07): captions tokenized alone, so pixel_values moved nothing.
    blip2 = {"image_token_index": 50265}
    usage = (
        "```py runnable:t\n"
        "inputs = processor(images=images, return_tensors=\"pt\").to(device, model.dtype)\n"
        "targets = processor(text=captions, return_tensors=\"pt\", padding=True).to(device)\n"
        "loss = model(pixel_values=inputs.pixel_values, input_ids=targets.input_ids, labels=labels).loss\n"
        "```\n"
    )
    with pytest.raises(ValueError, match="no image token"):
        commands.check_image_tokens_in_text({"usage": usage}, blip2)
    with pytest.raises(ValueError, match="no image token"):
        commands.check_image_tokens_in_text({"usage": usage.replace("targets.input_ids", 'targets["input_ids"]')}, blip2)
    joint = usage.replace("processor(text=captions,", "processor(images=images, text=captions,")
    commands.check_image_tokens_in_text({"usage": joint}, blip2)
    commands.check_image_tokens_in_text({"usage": usage}, {"model_type": "clip"})  # no image token to carry
    # A later reassignment with images clears the name.
    fixed = usage.replace("loss = ", "targets = processor(images=images, text=captions)\nloss = ")
    commands.check_image_tokens_in_text({"usage": fixed}, blip2)


def test_an_image_processor_takes_clips_as_images():
    # MCG-NJU/videomae-base (2026-11-07): VideoMAEImageProcessor(videos=...) raised "missing 'images'".
    usage = (
        "```py runnable:t\n"
        "processor = VideoMAEImageProcessor.from_pretrained(model_id)\n"
        "inputs = processor(videos=videos, return_tensors=\"pt\")\n"
        "```\n"
    )
    with pytest.raises(ValueError, match="takes no videos="):
        commands.check_image_processor_videos({"usage": usage})
    commands.check_image_processor_videos({"usage": usage.replace("videos=videos", "images=videos")})
    commands.check_image_processor_videos({"usage": usage.replace("VideoMAEImageProcessor", "AutoProcessor")})


def test_any_requires_grad_selection_by_a_deep_layer_name_is_rejected():
    # Salesforce/blip2-opt-2.7b (2026-11-07): frozen per parameter, without requires_grad_(False).
    usage = (
        "```py runnable:t\n"
        "for name, param in model.named_parameters():\n"
        "    param.requires_grad_(name.startswith(\"language_model.model.decoder.layers.31.\"))\n"
        "```\n"
    )
    with pytest.raises(ValueError, match="layer 31 does not exist"):
        commands.check_trainable_selection({"usage": usage}, {})
    commands.check_trainable_selection({"usage": usage.replace("layers.31.", "layers.1.")}, {})


def test_gradient_checks_must_name_the_trained_last_layer():
    # Salesforce/blip2-opt-2.7b (2026-11-07): trained layers[-1], checked layers.0.
    usage = (
        "```py runnable:t\n"
        "model.requires_grad_(False)\n"
        "model.language_model.model.decoder.layers[-1].requires_grad_(True)\n"
        "checked = (\"language_model.model.decoder.layers.0.fc1.weight\",)\n"
        "```\n"
    )
    with pytest.raises(ValueError, match="names layer 0"):
        commands.check_trainable_selection({"usage": usage}, {})
    trained = usage.replace('\"language_model.model.decoder.layers.0.fc1.weight\"',
                            'f\"language_model.model.decoder.layers.{last}.fc1.weight\"')
    commands.check_trainable_selection({"usage": trained}, {})


def test_blip2_q_former_first_query_gets_no_fixture_gradient():
    usage = '```py runnable:t\nchecked = ("qformer.encoder.layer.0.attention.attention.query.weight",)\n```\n'
    with pytest.raises(ValueError, match="zero-initialized"):
        commands.check_zero_query_gradients({"usage": usage}, {"num_query_tokens": 32})
    commands.check_zero_query_gradients({"usage": usage}, {"model_type": "bert"})
    commands.check_zero_query_gradients({"usage": usage.replace("layer.0.attention.attention", "layer.0.crossattention.attention")}, {"num_query_tokens": 32})


def test_planner_records_every_call_with_its_listed_price(monkeypatch):
    monkeypatch.setattr(planner, "CALLS", [])
    monkeypatch.setattr(planner, "_prices", {("m", "p"): (0.3, 1.2)})
    usage = SimpleNamespace(prompt_tokens=10_000, completion_tokens=2_000,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=4_000),
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=0))
    planner.record_call(SimpleNamespace(usage=usage), "m", "p", 3.21)
    planner.record_call(SimpleNamespace(usage=usage), "m", "unlisted", 1.0)
    first, second = planner.CALLS
    assert first == {"model": "m", "provider": "p", "prompt_tokens": 10_000, "cached_tokens": 4_000,
                     "completion_tokens": 2_000, "reasoning_tokens": 0, "seconds": 3.2, "listed_usd": 0.0054}
    assert "listed_usd" not in second


@pytest.mark.requires_runtime
def test_a_batch_encoding_is_not_cast_with_dtype():
    # microsoft/layoutlmv3-base: LayoutLMv3Processor returns its tokenizer's BatchEncoding.
    bad = 'inputs = processor(images=images, text=words, boxes=boxes, return_tensors="pt")\ninputs = inputs.to(device, dtype=model.dtype)\n'
    with pytest.raises(ValueError, match="BatchEncoding"):
        commands.check_batch_encoding_dtype({"usage": bad}, {"model_type": "layoutlmv3"})
    good = bad.replace("inputs.to(device, dtype=model.dtype)", 'inputs.to(device)\ninputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)')
    commands.check_batch_encoding_dtype({"usage": good}, {"model_type": "layoutlmv3"})
    commands.check_batch_encoding_dtype({"usage": bad}, {"model_type": "clip"})  # a BatchFeature takes dtype
    with pytest.raises(ValueError, match="BatchEncoding"):
        commands.check_batch_encoding_dtype({"usage": 'batch = tokenizer(texts, return_tensors="pt").to(device, dtype=torch.bfloat16)\n'}, {})


def test_a_transducer_forward_needs_decoder_inputs():
    # nvidia/parakeet-tdt-0.6b-v3: labels alone leave the prediction network without input.
    bad = "model = AutoModelForTDT.from_pretrained(model_id)\ninputs = processor(audio=audio, return_tensors='pt')\nloss = model(**inputs, labels=labels).loss\n"
    with pytest.raises(ValueError, match="decoder_input_ids"):
        commands.check_transducer_inputs({"usage": bad})
    with_ids = bad.replace("labels=labels)", "decoder_input_ids=decoder_input_ids, labels=labels)")
    commands.check_transducer_inputs({"usage": with_ids})
    from_text = bad.replace("processor(audio=audio,", "processor(audio=audio, text=transcripts,")
    commands.check_transducer_inputs({"usage": from_text})
    commands.check_transducer_inputs({"usage": bad.replace("AutoModelForTDT", "AutoModelForCTC")})
    nested = "model = AutoModelForTDT.from_pretrained(model_id)\nbatch = processor(\n    audio=[np.zeros(16000), np.ones(16000)],\n    text=['a', 'b'],\n)\nloss = model(**batch).loss\n"
    commands.check_transducer_inputs({"usage": nested})  # a ")" before text= is still a text call
    # nemotron-3.5-asr: training fixed, inference still model(**inputs) on audio only.
    inference = "model = Nemotron3_5AsrForRNNT.from_pretrained(m)\ninputs = processor(audio=a)\noutputs = model(**inputs)\nbatch = processor(audio=a, text=t)\nloss = model(**batch).loss\n"
    with pytest.raises(ValueError, match=r"line 3: model\(\*\*inputs\)"):
        commands.check_transducer_inputs({"usage": inference})


@pytest.mark.requires_runtime
def test_apply_chat_template_needs_a_template(monkeypatch):
    from transformers import ProcessorMixin

    class NoTemplate(ProcessorMixin):
        attributes = []
        chat_template = None

        def __init__(self):
            pass

    import transformers

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: NoTemplate())
    usage = {"usage": "inputs = processor.apply_chat_template(conversation)"}
    with pytest.raises(ValueError, match="no chat template"):
        commands.check_chat_template(usage, {"model": "mistralai/Voxtral-Mini-4B-Realtime-2602"})
    NoTemplate.chat_template = "{{ messages }}"
    commands.check_chat_template(usage, {"model": "o/m"})


def test_audio_reaches_the_processor_as_waveforms():
    # Voxtral-Mini-4B-Realtime: WAV paths written in the block, then handed to the processor.
    bad = 'sf.write("a.wav", wave, sr)\ninputs = processor(text=texts, audio=["a.wav"], sampling_rate=sr)\n'
    with pytest.raises(ValueError, match="waveforms"):
        commands.check_audio_waveforms({"usage": bad})
    commands.check_audio_waveforms({"usage": "inputs = processor(audio=[wave_a, wave_b], sampling_rate=sr)\n"})
    multiline = 'path_a = os.path.join(tmp, "a.wav")\ninputs = processor(\n    text=["x"],\n    audio=[path_a],\n)\n'
    with pytest.raises(ValueError, match="waveforms"):
        commands.check_audio_waveforms({"usage": multiline})
    # A chat template's audio entries load files (Voxtral-Mini-3B).
    commands.check_audio_waveforms({"usage": bad + 'processor.apply_chat_template([{"type": "audio", "path": "a.wav"}])\n'})


def test_every_draft_problem_is_reported_at_once(monkeypatch):
    # Voxtral Realtime: three problems, one per draft, ran out of drafts.
    calls = []

    def fake(context, args, attempt, retries):
        calls.append(context.get("validation_error"))
        return {"usage": "", "manifest": {}}, None

    monkeypatch.setattr(commands, "propose_well_formed", fake)
    monkeypatch.setattr(commands, "normalize_compare", lambda plan: None)
    monkeypatch.setattr(commands, "drop_unknown_manifest_keys", lambda plan: None)
    monkeypatch.setattr(commands, "validate_plan", lambda *a: None)
    for name in ("check_imports", "check_undefined_names", "check_targets", "check_code_fences", "check_no_remote_code",
                 "check_image_shapes", "check_loss_has_gradients", "check_transducer_inputs", "check_partial_training"):
        monkeypatch.setattr(commands, name, lambda *a: None)
    monkeypatch.setattr(commands, "check_chat_template", lambda *a: (_ for _ in ()).throw(ValueError("no chat template")))
    monkeypatch.setattr(commands, "check_audio_waveforms", lambda *a: (_ for _ in ()).throw(ValueError("audio files")))
    with pytest.raises(ValueError, match=r"2 problems: \(1\) no chat template \(2\) audio files"):
        commands.propose_valid({}, SimpleNamespace(), "o/m", "o/m-mic")
    assert "no chat template" in calls[1] and "audio files" in calls[1]
