from pathlib import Path
from types import SimpleNamespace
import hashlib
import shutil

import pytest

from transformers_mic import cloud_inputs, model_baselines
from transformers_mic.workspace import read, write
from test_baselines import make_collection


@pytest.fixture
def measured(tmp_path, monkeypatch):
    root = tmp_path / "collection"
    state = make_collection(root)
    snapshot = {"repo": "huggingface-mic/demo-mic", "revision": "9" * 40}
    state["job"]["workspace"] = snapshot
    write(root / "cloud.json", state)
    write(root / "model-input.json", {"source": "source", "workspace": snapshot})
    for profile in ("cpu", "cuda"):
        path = root / "artifacts" / profile / "collection.json"
        report = read(path)
        report["request"] = {**report["request"], "workspace": snapshot}
        report["cases"][0].update(contract="Demo", variant="default")
        write(path, report)
    workspace = SimpleNamespace(
        root=tmp_path / "model",
        state={"contract": "Demo"},
        expectations=tmp_path / "model/expectations",
        lock_path=tmp_path / "model/lock.json",
        contract_root=tmp_path / "model/contract",
        manifest={
            "integrations": {
                "transformers": {"variants": {"default": {}}, "entry_point": "usage.md"}
            }
        },
        validation_fingerprint=lambda: "fingerprint",
    )
    workspace.save_state = lambda **kw: workspace.state.update(kw)
    write(workspace.lock_path, {})
    write(workspace.contract_root / "integration.json", workspace.manifest)
    (workspace.contract_root / "usage.md").write_text("reviewed usage")
    monkeypatch.setattr(model_baselines, "inputs", lambda _: ("source", None, {}))
    model_baselines.import_candidates(workspace, root)
    return workspace, root


def approve(workspace):
    for path in workspace.expectations.rglob("*.candidate.json"):
        path.rename(path.with_name(path.name.replace(".candidate.json", ".json")))


def test_model_requires_review_then_records_both_hardware_profiles(measured):
    workspace, root = measured
    with pytest.raises(ValueError, match="Review every"):
        model_baselines.verify(workspace, root)
    approve(workspace)
    assert model_baselines.verify(workspace, root) == 0
    assert workspace.state["validated"] is True
    assert workspace.state["validations"]["default"]["cloud_collection"] == str(root)
    assert {p.name for p in workspace.expectations.iterdir()} == {"cpu-x86_64", "cuda"}


def test_changed_source_or_reviewed_measurement_cannot_be_published(
    measured, monkeypatch
):
    workspace, root = measured
    approve(workspace)
    baseline = workspace.expectations / "cuda/mic-demo.json"
    original = baseline.read_bytes()
    baseline.write_text("changed")
    with pytest.raises(ValueError, match="Reviewed expectations differ"):
        model_baselines.verify(workspace, root)
    baseline.write_bytes(original)
    monkeypatch.setattr(
        model_baselines, "inputs", lambda _: ("edited-source", None, {})
    )
    with pytest.raises(ValueError, match="current model inputs"):
        model_baselines.verify(workspace, root)


def test_missing_cuda_evidence_cannot_complete(measured):
    workspace, root = measured
    approve(workspace)
    shutil.rmtree(root / "artifacts/cuda")
    with pytest.raises(FileNotFoundError):
        model_baselines.verify(workspace, root)


def test_staged_model_snapshot_is_separate_and_immutable(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "weights.safetensors").write_bytes(b"weights")
    weights = hashlib.sha256(b"weights").hexdigest()
    write(fixture / "build.json", {"sha256": {"weights.safetensors": weights}})
    source = tmp_path / "contract"
    source.mkdir()
    (source / "usage.md").write_text("reviewed source")
    repo = "huggingface-mic/demo-mic"
    lock = {
        "environment": {"python": "3.11"},
        "contracts": {
            "Demo": {
                "integration": {
                    "repo": repo,
                    "revision": "unpublished",
                    "path": "/local",
                },
                "fixtures": {repo: {"revision": "unpublished", "path": "/local"}},
                "checkpoints": {"org/model": "a" * 40},
            }
        },
    }
    workspace = SimpleNamespace(
        fixture=repo,
        contract_root=source,
        expectations=source / "expectations",
        lock=lock,
        state={"contract": "Demo"},
    )
    monkeypatch.setattr(
        model_baselines, "inputs", lambda _: ("f" * 64, fixture, {"usage.md": "0" * 64})
    )
    calls = []

    class API:
        def create_repo(self, *a, **kw):
            calls.append("create")

        def create_branch(self, *a, **kw):
            assert kw["branch"] == "tmic-baselines/" + "f" * 64

        def upload_folder(self, **kw):
            assert kw["revision"].startswith("tmic-baselines/")
            uploaded = read(Path(kw["folder_path"]) / ".mic/baseline-lock.json")
            assert "path" not in uploaded["contracts"]["Demo"]["integration"]
            assert (
                Path(kw["folder_path"]) / "weights.safetensors"
            ).read_bytes() == b"weights"
            return SimpleNamespace(oid="b" * 40)

    monkeypatch.setattr(model_baselines, "HfApi", API)
    root = tmp_path / "result"
    assert model_baselines.stage(workspace, root) == {
        "repo": repo,
        "revision": "b" * 40,
    }
    assert model_baselines.stage(workspace, root)["revision"] == "b" * 40
    assert calls == ["create"]
    assert workspace.lock["contracts"]["Demo"]["integration"]["path"] == "/local"


def test_mutable_or_unapproved_inputs_never_reach_workers(monkeypatch):
    monkeypatch.setenv("MIC_WORKSPACE_REPO", "someone/demo-mic")
    monkeypatch.setenv("MIC_WORKSPACE_REVISION", "a" * 40)
    with pytest.raises(ValueError, match="approved MIC"):
        cloud_inputs.from_env()
    monkeypatch.setenv("MIC_WORKSPACE_REPO", "huggingface-mic/demo-mic")
    monkeypatch.setenv("MIC_WORKSPACE_REVISION", "main")
    with pytest.raises(ValueError, match="immutable"):
        cloud_inputs.from_env()
