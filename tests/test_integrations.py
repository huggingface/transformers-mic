import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformers_mic import commands, fleet, integrations, resources
from transformers_mic.cli import main
from transformers_mic.workspace import Workspace, write


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace(tmp_path)
    root = tmp_path / "example/hub/embeddinggemma-2-mic"
    write(
        ws.state_path,
        {"model": "google/embeddinggemma-2", "contract": "EmbeddingGemma"},
    )
    write(
        ws.lock_path,
        {
            "framework": "transformers",
            "environment": {},
            "contracts": {
                "EmbeddingGemma": {
                    "integration": {
                        "repo": "huggingface-mic/embeddinggemma-2-mic",
                        "revision": "unpublished",
                        "path": str(root),
                    },
                    "fixtures": {},
                    "checkpoints": {"google/embeddinggemma-2": "a" * 40},
                }
            },
        },
    )
    manifest = {
        "mic": "0.3",
        "name": "EmbeddingGemma",
        "usage": "Embeddings",
        "checkpoints": {
            "google/embeddinggemma-2": {
                "fixture": "huggingface-mic/embeddinggemma-2-mic"
            }
        },
        "fixtures": {},
        "integrations": {
            "transformers": {
                "entry_point": "integrations/transformers/usage.md",
                "format": "runnable-markdown",
                "targets": ["fixtures"],
                "variants": {"default": {}},
                "training": False,
            }
        },
        "outputs": {
            "runtime": {"compare": "exact"},
            "text_embeddings": {"compare": "exact"},
        },
    }
    write(root / "integration.json", manifest)
    write(root / ".mic-repo.json", {"format": "mic-umbrella-v1", "contract_files": []})
    path = root / "integrations/transformers/usage.md"
    path.parent.mkdir(parents=True)
    path.write_text("Existing usage")
    return ws


def test_generate_both_preserves_transformers_and_is_idempotent(workspace):
    original = workspace.manifest
    args = [
        "--workspace",
        str(workspace.root),
        "integrations",
        "--integration",
        "transformers",
        "--integration",
        "sentence_transformers",
    ]
    assert main(args) == 0
    assert workspace.manifest["outputs"] == original["outputs"]
    assert (
        workspace.manifest["integrations"]["transformers"]
        == original["integrations"]["transformers"]
    )
    assert (
        workspace.manifest["integrations"][integrations.NAME]["outputs"]["scores"][
            "compare"
        ]
        == "allclose"
    )
    path = workspace.contract_root / f"integrations/{integrations.NAME}/usage.md"
    path.write_text(path.read_text() + "\nReviewed extra text\n")
    assert main(args) == 0
    assert path.read_text().endswith("Reviewed extra text\n")
    assert not workspace.state["validated"]
    assert (
        main(
            [
                "--workspace",
                str(workspace.root),
                "integrations",
                "--integration",
                integrations.NAME,
            ]
        )
        == 0
    )
    assert list(workspace.manifest["integrations"]) == [integrations.NAME]


def test_generation_refuses_other_models_before_writing(workspace):
    workspace.save_state(model="org/unreviewed")
    original = workspace.manifest
    with pytest.raises(ValueError, match="currently supports"):
        integrations.generate(
            SimpleNamespace(workspace=workspace.root, integration=[integrations.NAME])
        )
    assert workspace.manifest == original


def test_fleet_optional_metadata_keeps_legacy_tables_unchanged():
    from test_fleet_tables import LOCK

    lock = copy.deepcopy(LOCK)
    name, contract = next(iter(lock["contracts"].items()))
    contract["integrations"] = ["transformers", integrations.NAME]
    tables = fleet.to_tables(lock)
    assert all(tables[key] == fleet.to_tables(LOCK)[key] for key in fleet.COLUMNS)
    assert fleet.from_tables(tables) == lock
    assert fleet.from_tables({key: tables[key] for key in fleet.COLUMNS}) == LOCK
    tables["integrations"] = tables["integrations"].replace(
        contract["integration"]["revision"], "0" * 40
    )
    assert fleet.from_tables(tables) == LOCK


@pytest.mark.requires_runtime
def test_runner_uses_only_selected_outputs_and_checks_numeric_shape(
    workspace, monkeypatch
):
    integrations.generate(
        SimpleNamespace(
            workspace=workspace.root, integration=["transformers", integrations.NAME]
        )
    )
    sys.path.insert(0, str(resources.backend()))
    import run

    monkeypatch.setattr(
        run, "contract_folder", lambda *a: (workspace.contract_root, "local")
    )
    monkeypatch.setattr(
        run,
        "targets",
        lambda *a: iter([("fixture", "fixture", "/fixture", "a" * 40, None)]),
    )
    values = {
        "runtime": {"device": "cpu", "dtype": "float32"},
        "scores": [0.1, 0.2, 0.3, 0.4],
    }

    class Executor:
        def __init__(self, *a):
            pass

        def environment(self):
            return {"python": "3.11", "sentence-transformers": "test"}

        def run(self, *a, **kw):
            return True, json.dumps(values), ""

    monkeypatch.setattr(run, "Executor", Executor)
    options = [
        "run",
        "--framework",
        integrations.NAME,
        "--lock",
        str(workspace.lock_path),
        "--record",
    ]
    monkeypatch.setattr(sys, "argv", options)
    assert run.main() == 0
    candidates = list(workspace.contract_root.rglob("*.candidate.json"))
    assert len(candidates) == 1
    candidate = candidates[0]
    candidate.rename(candidate.with_name(candidate.name.replace(".candidate", "")))
    monkeypatch.setattr(sys, "argv", options[:-1])
    assert run.main() == 0
    values["scores"] = [[0.1, 0.2, 0.3, 0.4]]
    assert run.main() == 1
    values.pop("scores")
    assert run.main() == 1
    # Transformers uses its original assertions; no ST output becomes mandatory.
    values["text_embeddings"] = [1, 2]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--framework",
            "transformers",
            "--lock",
            str(workspace.lock_path),
            "--no-baseline",
        ],
    )
    monkeypatch.setattr(
        run.template_date
        if hasattr(run, "template_date")
        else __import__("template_date"),
        "freeze",
        lambda *a: None,
    )
    assert run.main() == 0


def test_publication_requires_every_integration(workspace):
    import hashlib

    integrations.generate(
        SimpleNamespace(
            workspace=workspace.root, integration=["transformers", integrations.NAME]
        )
    )
    proof = {
        "lock_sha256": hashlib.sha256(workspace.lock_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (workspace.contract_root / "integration.json").read_bytes()
        ).hexdigest(),
        "usage_sha256": hashlib.sha256(
            (
                workspace.contract_root / "integrations/transformers/usage.md"
            ).read_bytes()
        ).hexdigest(),
    }
    fingerprint = workspace.validation_fingerprint()
    workspace.save_state(
        validated=True,
        validation=proof,
        validations={"default": {"fingerprint": fingerprint}},
    )
    with pytest.raises(ValueError, match="Every declared"):
        commands.publication_ready(workspace)
    workspace.save_state(
        validations={
            key: {"fingerprint": fingerprint}
            for key in ("default", "sentence_transformers:default")
        }
    )
    commands.publication_ready(workspace)


def test_nightly_selects_st_fleet_and_advertised_contracts(tmp_path, monkeypatch):
    source = (
        Path(__file__).parents[1] / ".github/scripts/sentence_transformers_nightly.py"
    )
    spec = importlib.util.spec_from_file_location("st_nightly", source)
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    revision, st_revision = "a" * 40, "b" * 40
    monkeypatch.setattr(fleet, "resolve", lambda ref: revision)
    calls = []

    def contract(repo, *advertised):
        value = {"integration": {"repo": repo}}
        if advertised:
            value["integrations"] = list(advertised)
        return value

    def download(name, pin):
        assert pin == revision
        path = tmp_path / f"{name}.json"
        write(
            path,
            {
                "framework": "transformers",
                "contracts": {
                    "Selected": contract(
                        "org/selected-mic", "transformers", integrations.NAME
                    ),
                    "Legacy": contract("org/legacy-mic"),
                    "Other": contract("org/other-mic", "transformers"),
                    # Also in the ST fleet: checked once.
                    "Twice": contract(
                        "org/st-a-mic", "transformers", integrations.NAME
                    ),
                },
                "environment": {
                    "transformers": "old",
                    "sentence-transformers": "old",
                    "sentence_transformers": "old",
                    "torch": "2.14.0",
                },
                "known_failures": {"Selected/fixture": "TF failure"},
                "local_patches": {"old-patch": {}},
            },
        )
        return path, pin

    def download_st(pin):
        assert pin == "main"
        path = tmp_path / "st.json"
        write(
            path,
            {
                "framework": fleet.ST_FRAMEWORK,
                "contracts": {
                    "st-a": contract("org/st-a-mic"),
                    "st-b": contract("org/st-b-mic"),
                },
            },
        )
        return path, st_revision

    def invoke(tool, cwd, options):
        lock = json.loads(Path(options[options.index("--lock") + 1]).read_text())
        assert "known_failures" not in lock and "local_patches" not in lock
        assert (
            not {"transformers", "sentence-transformers", "sentence_transformers"}
            & lock["environment"].keys()
        )
        if "Selected" in lock["contracts"]:
            assert lock["environment"]["torch"] == "2.14.0"
        assert options[options.index("--framework") + 1] == integrations.NAME
        calls.append(options[options.index("--contract") + 1])
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(fleet, "download", download)
    monkeypatch.setattr(fleet, "download_st", download_st)
    monkeypatch.setattr(nightly.backend, "invoke", invoke)
    monkeypatch.setattr(sys, "argv", ["nightly", "--out", str(tmp_path / "out")])
    assert nightly.main() == 1
    # A MIC repo in several fleets is checked once, from the first that lists it.
    assert calls == ["st-a", "st-b", "Selected"]
    assert (tmp_path / "out/fleet-revision.txt").read_text().strip() == revision
    assert (tmp_path / "out/st-fleet-revision.txt").read_text().strip() == st_revision


def test_scenario_uses_query_and_passages_and_returns_numpy_vector(monkeypatch):
    import types

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    seen = {}

    class Model:
        def __init__(self, model, **kwargs):
            seen["model"] = model

        def eval(self):
            pass

        def encode_query(self, text):
            seen["query"] = text
            return np.zeros(3)

        def encode_document(self, texts):
            seen["passages"] = texts
            return np.zeros((len(texts), 3))

        def similarity(self, query, documents):
            return torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    monkeypatch.delenv("MIC_RUNNER", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=Model),
    )
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(set_seed=lambda seed: None)
    )
    code = integrations.USAGE.split("```py runnable:similarity\n")[1].split("```")[0]
    namespace = {}
    exec(compile(code, "usage.md", "exec"), namespace)
    assert namespace["scores"].shape == (4,)
    assert isinstance(namespace["scores"], np.ndarray)
    assert len(seen["passages"]) == 4 and "Red Planet" in seen["query"]


def test_preview_requires_an_immutable_pin():
    source = (
        Path(__file__).parents[1] / ".github/scripts/sentence_transformers_nightly.py"
    )
    spec = importlib.util.spec_from_file_location("st_preview", source)
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    with pytest.raises(ValueError, match="immutable"):
        nightly.preview_lock("huggingface-mic/embeddinggemma-2-mic", "main")
