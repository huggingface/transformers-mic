import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformers_mic import automation, backend, batch, fixture_review
from transformers_mic.cli import main
from transformers_mic.interaction import Interaction
from transformers_mic.workspace import Workspace, read, write

MODEL = "google-bert/bert-base-uncased"
SOURCE_KEYS = [
    "bert.embeddings.word_embeddings.weight",
    *(f"bert.encoder.layer.{i}.attention.self.query.weight" for i in range(12)),
    "cls.predictions.bias",
]


def unpublish(root):
    """A model whose fixture this run builds, as `tmic create` leaves a new one."""
    workspace = Workspace(root)
    lock = workspace.lock
    lock["contracts"][workspace.state["contract"]]["fixtures"][workspace.fixture] = {
        "revision": "unpublished",
        "path": "../../../hub/" + workspace.fixture.split("/")[-1],
    }
    write(workspace.lock_path, lock)


def review_data(workspace, out, layers=2):
    """What the backend's fixture-stage reports, for a recipe's overrides."""
    recipe = workspace.manifest["fixtures"][workspace.fixture]["build"]
    hidden = recipe.get("config_overrides", {}).get("fields", {}).get("hidden_size", 32)
    config = {
        "model_type": "bert",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "transformers_version": "5.0",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config))
    (out / "model.safetensors").write_bytes(b"weights")
    return {
        "fixture": workspace.fixture,
        "source": {"repo": MODEL, "revision": "a" * 40},
        "model_class": "BertForMaskedLM",
        "weights": "random",
        "weights_file": "model.safetensors",
        "files": ["config.json", "model.safetensors"],
        "config": config,
        "source_keys": SOURCE_KEYS,
        "fixture_keys": [
            k
            for k in SOURCE_KEYS
            if ".layer." not in k or int(k.split(".")[3]) < layers
        ],
        "missing_parameters": [],
        "parameters": {"original": 110_000_000, "fixture": 400_000},
        "config_changes": [
            {"field": "hidden_size", "original": "768", "fixture": str(hidden)},
            {"field": "num_hidden_layers", "original": "12", "fixture": str(layers)},
        ],
        "config_unchanged": 20,
        "findings": [],
        "tester_notes": [],
        "page": "<html></html>",
    }


@pytest.fixture
def fake_backend(monkeypatch):
    """Stands in for the torch process: writes the fixture and its comparison."""
    calls = []

    def invoke(tool, root, options=(), **kwargs):
        assert tool == "fixture-stage"
        calls.append(list(options))
        _, _, out, result, *_ = options
        workspace = Workspace(root)
        Path(result).write_text(json.dumps(review_data(workspace, Path(out))))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(backend, "invoke", invoke)
    return calls


@pytest.fixture
def workflow(tmp_path, monkeypatch, fake_backend):
    root = tmp_path / "project"
    calls = []

    def invoke(out, name, arguments, timeout):
        calls.append((name, list(arguments)))
        log = out / "logs" / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        if name == "create":
            Workspace(root).from_catalog(MODEL, "BERT", "auto", False)
            unpublish(root)
        if arguments[-1:] == ["fixture"]:
            # The real stage runs `tmic --workspace ROOT fixture` in a subprocess.
            assert main(arguments) == 0
        log.write_text(json.dumps({"stage": name}) + "\n")
        return {"exit_code": 0, "log": str(log.relative_to(out))}

    monkeypatch.setattr(batch, "invoke", invoke)
    monkeypatch.setattr(automation.sys.stdin, "isatty", lambda: True)
    return SimpleNamespace(
        root=root,
        calls=calls,
        argv=[
            "run",
            MODEL,
            "--out",
            str(root),
            "--no-history",
            "--baseline-execution",
            "local",
            "--local-only",
        ],
    )


def answer(tmp_path, result, choice="approve"):
    path = tmp_path / f"answer-{result['pending_question']['question_id']}.json"
    path.write_text(
        json.dumps(
            {
                "question_id": result["pending_question"]["question_id"],
                "answer": choice,
                "reason": "checked",
            }
        )
    )
    return path


def agent(workflow, capsys, *extra):
    code = main([*workflow.argv, "--mode", "agent", *extra])
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_tiny_model_is_reviewed_before_the_contract_and_its_code_runs(
    workflow, capsys, tmp_path
):
    code, result = agent(workflow, capsys)
    assert code == 3
    question = result["pending_question"]
    assert question["kind"] == "fixture-review"
    # Nothing that runs contract code has started yet.
    assert [name for name, _ in workflow.calls] == ["create", "shrink"]
    folder = workflow.root / "fixture"
    assert (folder / "config.json").exists() and (folder / "model.safetensors").exists()
    summary = question["context"]["fixture"]
    assert summary["folder"] == str(folder)
    assert summary["weight_names"]["only_source"] == []
    assert "bert.encoder.layer.{n}.attention.self.query.weight" in question["details"]
    assert (
        "hidden_size" in question["details"]
        and "edit config.json" in question["details"]
    )

    code, result = agent(workflow, capsys, "--answer", str(answer(tmp_path, result)))
    assert code == 3 and result["pending_question"]["kind"] == "source-review"
    # The contract review shows the code itself, not only its path.
    assert "AutoModelForMaskedLM" in result["pending_question"]["details"]

    code, result = agent(workflow, capsys, "--answer", str(answer(tmp_path, result)))
    assert code == 0 and result["status"] == "validated"
    names = [name for name, _ in workflow.calls]
    assert names[:4] == ["create", "shrink", "build", "build"]
    assert workflow.calls[3][1][-1] == "fixture"  # fixture/ now shows the built model


def test_editing_config_json_reshrinks_with_the_values_as_overrides(
    workflow, capsys, tmp_path
):
    code, first = agent(workflow, capsys)
    config = workflow.root / "fixture/config.json"
    values = json.loads(config.read_text())
    values["hidden_size"] = 64
    config.write_text(json.dumps(values))

    # A stale approval names the old snapshot: refused, and the new one is attached.
    code, result = agent(workflow, capsys, "--answer", str(answer(tmp_path, first)))
    assert code == 3 and result["answer_error"]
    second = result["pending_question"]
    assert (
        second["kind"] == "fixture-review"
        and second["question_id"] != first["pending_question"]["question_id"]
    )
    workspace = Workspace(workflow.root)
    overrides = workspace.manifest["fixtures"][workspace.fixture]["build"][
        "config_overrides"
    ]
    assert overrides["fields"] == {"hidden_size": 64} and overrides["reason"]
    assert json.loads(config.read_text())["hidden_size"] == 64
    assert [name for name, _ in workflow.calls].count("shrink") == 2
    assert not read(workflow.root / ".mic/automation.json").get(
        "reviewed_fixture_sha256"
    )


def test_an_edit_after_approval_sends_the_run_back_to_the_review(
    workflow, capsys, tmp_path
):
    _, result = agent(workflow, capsys)
    code, result = agent(workflow, capsys, "--answer", str(answer(tmp_path, result)))
    assert code == 3 and result["pending_question"]["kind"] == "source-review"
    config = workflow.root / "fixture/config.json"
    config.write_text(json.dumps({**json.loads(config.read_text()), "hidden_size": 48}))
    _, result = agent(workflow, capsys)
    assert result["pending_question"]["kind"] == "fixture-review"
    assert "48" in result["pending_question"]["details"]


def test_human_presses_e_edits_then_approves(workflow, monkeypatch, capsys):
    replies = iter(["d", "e", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    def edit(path):
        values = json.loads(Path(path).read_text())
        Path(path).write_text(json.dumps({**values, "hidden_size": 128}))

    monkeypatch.setattr(automation.terminal, "edit", edit)
    assert main(workflow.argv) == 0
    output = capsys.readouterr().out
    assert "Review tiny model" in output and "Folder:" in output
    assert "Matching the source" in output  # D printed the full comparison inline
    state = read(workflow.root / ".mic/automation.json")
    kinds = [d["kind"] for d in state["decisions"]]
    assert kinds == ["fixture-review", "source-review"]  # E was not a decision
    workspace = Workspace(workflow.root)
    fields = workspace.manifest["fixtures"][workspace.fixture]["build"][
        "config_overrides"
    ]["fields"]
    assert fields == {"hidden_size": 128}


def test_published_fixtures_skip_the_tiny_model_review(tmp_path, monkeypatch, capsys):
    root = tmp_path / "project"

    def invoke(out, name, arguments, timeout):
        if name == "create":
            Workspace(root).from_catalog(MODEL, "BERT", "auto", False)
        log = out / "logs" / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("{}\n")
        return {"exit_code": 0, "log": str(log.relative_to(out))}

    monkeypatch.setattr(batch, "invoke", invoke)
    assert (
        main(
            [
                "run",
                MODEL,
                "--out",
                str(root),
                "--no-history",
                "--mode",
                "agent",
                "--baseline-execution",
                "local",
                "--local-only",
            ]
        )
        == 3
    )
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["pending_question"]["kind"] == "source-review"
    assert not (root / "fixture").exists()


def test_fixture_command_reuses_until_something_changes(tmp_path, fake_backend, capsys):
    root = tmp_path / "project"
    Workspace(root).from_catalog(MODEL, "BERT", "auto", False)
    unpublish(root)
    assert main(["--workspace", str(root), "fixture", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["weight_names"] == {
        "matching": 3,
        "only_source": [],
        "only_fixture": [],
    }
    assert (root / fixture_review.PAGE).read_text() == "<html></html>"
    assert main(["--workspace", str(root), "fixture"]) == 0
    assert len(fake_backend) == 1  # nothing changed: no second shrink
    assert main(["--workspace", str(root), "fixture", "--force"]) == 0
    assert len(fake_backend) == 2


def test_invalid_json_edit_is_reported_with_the_path(tmp_path, fake_backend):
    root = tmp_path / "project"
    Workspace(root).from_catalog(MODEL, "BERT", "auto", False)
    unpublish(root)
    fixture_review.refresh(Workspace(root))
    (root / "fixture/config.json").write_text("{not json")
    with pytest.raises(ValueError, match="fixture/config.json"):
        fixture_review.edits(Workspace(root))


def test_nested_edits_become_dotted_overrides(tmp_path, fake_backend):
    root = tmp_path / "project"
    Workspace(root).from_catalog(MODEL, "BERT", "auto", False)
    unpublish(root)
    workspace = Workspace(root)
    data, _ = fixture_review.refresh(workspace)
    data["config"]["text_config"] = {"hidden_size": 32, "vocab_size": 100}
    write(root / fixture_review.RECORD, data)
    edited = {
        **data["config"],
        "text_config": {"hidden_size": 16, "vocab_size": 100},
        "transformers_version": "9.9",
    }
    (root / "fixture/config.json").write_text(json.dumps(edited))
    assert fixture_review.edits(workspace) == {"text_config.hidden_size": 16}


def test_weight_names_collapse_layer_numbers_and_flag_differences():
    data = {
        "source_keys": ["a.layers.0.w", "a.layers.1.w", "lm_head.weight"],
        "fixture_keys": ["a.layers.0.w", "model.lm_head.weight"],
        "missing_parameters": [],
    }
    names = fixture_review.weight_names(data)
    assert names["matching"] == ["a.layers.{n}.w"]
    assert names["only_source"] == ["lm_head.weight"]
    assert names["only_fixture"] == ["model.lm_head.weight"]


def test_edit_is_not_a_decision(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "e")
    edited = []
    monkeypatch.setattr(automation.terminal, "edit", edited.append)
    state = {}
    interaction = Interaction("human", state, lambda: None, dict)
    config = {"path": str(tmp_path / "config.json"), "sha256": "x"}
    assert (
        interaction.ask(
            "Tiny model?", "d", kind="fixture-review", context={"config": config}
        )
        == "edit"
    )
    assert edited == [config["path"]]
    assert not state.get("decisions") and "pending_question" not in state


def test_an_overfit_build_still_counts_as_the_current_fixture(tmp_path, fake_backend):
    root = tmp_path / "project"
    Workspace(root).from_catalog(MODEL, "BERT", "auto", False)
    unpublish(root)
    workspace = Workspace(root)
    built = root / ".mic/artifacts/bert/abc"
    built.mkdir(parents=True)
    recipe = workspace.manifest["fixtures"][workspace.fixture]["build"]
    write(
        built / "build.json",
        {"recipe": {**recipe, "overfit": "requested"}, "built_from": {"repo": MODEL}},
    )
    lock = workspace.lock
    pin = lock["contracts"][workspace.state["contract"]]["fixtures"][workspace.fixture]
    pin.update(path="../../../../.mic/artifacts/bert/abc", files_sha256="abc")
    write(workspace.lock_path, lock)
    assert fixture_review.artifact(workspace) == built.resolve()
    data, _ = fixture_review.refresh(workspace)
    assert fake_backend[-1][-1] == str(built.resolve())  # described, not shrunk again
    assert data["artifact"] == str(built.resolve())
