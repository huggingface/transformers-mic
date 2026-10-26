"""Regression coverage for non-language-model fixture inputs."""

import hashlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
import build_fixtures  # noqa: E402


def test_weight_named_processor_asset_is_copied(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "attention_mask.safetensors").write_bytes(b"processor tensor")
    (tmp_path / "model.safetensors").write_bytes(b"model weights")
    recipe = {"copy_assets": ["attention_mask.safetensors"]}

    assert "attention_mask.safetensors" in build_fixtures.copied_files(tmp_path, recipe)
    assert "model.safetensors" not in build_fixtures.copied_files(tmp_path, recipe)


def test_batch_norm_calibration_recovers_evaluation_sensitivity_without_changing_weights():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.BatchNorm1d(4, eps=1e-8),
        torch.nn.Dropout(0.9),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(4) * 0.001)
    batch = {"input": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])}
    weights = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    before = model(**batch)
    assert build_fixtures.calibrate_batch_norm(model, batch) == 1
    after = model(**batch)
    assert torch.max(torch.abs(after[0] - after[1])) > 100 * torch.max(
        torch.abs(before[0] - before[1])
    )
    assert torch.isfinite(after).all()
    assert all(
        torch.equal(weights[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert not model[1].training and not model[2].training
    assert model[1].momentum == 0.1
    assert torch.equal(model(**batch), after)


def test_calibration_does_not_forward_models_without_batch_norm():
    class Model(torch.nn.Module):
        def forward(self, **kw):
            raise AssertionError("No calibration forward is needed")

    assert build_fixtures.calibrate_batch_norm(Model(), {}) == 0


def test_selected_fixed_tensors_survive_random_initialization(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.geometry = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.weights = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("mesh_index", torch.tensor(0))
            self.register_buffer("scratch", torch.tensor(0), persistent=False)

    source = tmp_path / "source"
    source.mkdir()
    safetensors.save_file(
        {
            "geometry": torch.tensor([2.0, 3.0]),
            "weights": torch.tensor([8.0, 9.0]),
            "mesh_index": torch.tensor(7),
        },
        source / "model.safetensors",
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, **kwargs: source / filename,
    )
    model = Model()
    hashes = build_fixtures.restore_frozen(
        model,
        source,
        "org/checkpoint",
        "pinned-sha",
        {"preserve_frozen_prefixes": ["geometry", "mesh_"]},
    )
    assert model.geometry.tolist() == [2.0, 3.0]
    assert model.weights.tolist() == [0.0, 0.0]
    assert model.mesh_index.item() == 7
    assert (
        hashes["mesh_index"]
        == hashlib.sha256(
            torch.tensor([7]).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    assert (
        hashes["geometry"]
        == hashlib.sha256(
            torch.tensor([2.0, 3.0]).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="trainable"):
        build_fixtures.restore_frozen(
            model,
            source,
            "org/checkpoint",
            "pinned-sha",
            {"preserve_frozen_prefixes": ["weights"]},
        )
    with pytest.raises(ValueError, match="No frozen parameters or persistent buffers"):
        build_fixtures.restore_frozen(
            model,
            source,
            "org/checkpoint",
            "pinned-sha",
            {"preserve_frozen_prefixes": ["scratch"]},
        )
