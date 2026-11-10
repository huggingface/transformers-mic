"""tmic reserialize: a fixture's .bin becomes model.safetensors with the same tensors."""

import json
from types import SimpleNamespace

import pytest
from huggingface_hub.errors import HfHubHTTPError

from transformers_mic import reserialize


def fixture_folder(tmp_path, state, indent=2):
    import torch

    runner = reserialize.runner()
    (tmp_path / "config.json").write_text("{}")
    torch.save(state, tmp_path / "pytorch_model.bin")
    build = {
        "fixture": "huggingface-mic/x-mic",
        "key": "0" * 64,
        "covers": {},
        "built_from": {"repo": "org/x", "revision": "r"},
        "seed": 0,
        "missing_parameters": [],
        "serialization": {"file": "pytorch_model.bin", "reverted_conversions": [], "migrated_from": "model.safetensors"},
        "sha256": runner.file_hashes(tmp_path),
    }
    (tmp_path / "build.json").write_text(json.dumps(build, indent=indent) + "\n")
    return runner, runner.digest(build["sha256"])


@pytest.mark.requires_runtime
def test_convert_keeps_every_name_and_bit(tmp_path):
    import torch
    from safetensors.torch import load_file

    embed = torch.randn(8, 4).half()
    state = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": embed,  # a tied alias sharing storage: both names stay
        "head.weight": torch.randn(4, 3).t(),  # not contiguous
        "bn.num_batches_tracked": torch.tensor(7),
        "nan": torch.tensor([float("nan"), -0.0]),
    }
    runner, before = fixture_folder(tmp_path, state)

    assert reserialize.convert(tmp_path, "pytorch_model.bin.index.json")["tensors"] == 5

    assert not (tmp_path / "pytorch_model.bin").exists()
    assert reserialize.differences(state, load_file(tmp_path / "model.safetensors")) == []
    text = (tmp_path / "build.json").read_text()
    build = json.loads(text)
    assert text == json.dumps(build, indent=2) + "\n"  # the file's own indentation
    assert build["serialization"] == {
        "file": "model.safetensors",
        "reverted_conversions": [],
        "migrated_from": "pytorch_model.bin",
        "source_file": "pytorch_model.bin.index.json",
    }
    assert list(build["sha256"]) == sorted(build["sha256"])
    after = runner.digest(build["sha256"])
    assert after != before
    assert runner.fixture_problem(tmp_path, {"files_sha256": after}) is None


@pytest.mark.requires_runtime
def test_differences_names_renames_and_changed_bits():
    import torch

    a = {"x": torch.zeros(2), "y": torch.zeros(1)}
    assert reserialize.differences(a, {"x": torch.zeros(2), "z": torch.zeros(1)})[0].startswith("tensor names differ")
    assert reserialize.differences(a, {"x": torch.tensor([0.0, -0.0]), "y": torch.zeros(1)}) == ["x: values differ"]
    assert "became" in reserialize.differences(a, {"x": torch.zeros(2).half(), "y": torch.zeros(1)})[0]


@pytest.mark.requires_runtime
def test_convert_refuses_a_fixture_without_a_single_bin(tmp_path):
    import torch

    fixture_folder(tmp_path, {"w": torch.zeros(1)})
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="already has"):
        reserialize.convert(tmp_path)


@pytest.mark.requires_runtime
def test_loaded_model_is_unchanged(tmp_path):
    import shutil

    import torch
    from transformers import BertConfig, BertForMaskedLM

    before, after = tmp_path / "before", tmp_path / "after"
    before.mkdir()
    torch.manual_seed(0)
    config = BertConfig(vocab_size=32, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    model = BertForMaskedLM(config)
    config.save_pretrained(before)
    fixture_folder(before, model.state_dict())
    (before / "config.json").write_text(config.to_json_string())
    runner = reserialize.runner()
    build = json.loads((before / "build.json").read_text())
    build["sha256"] = runner.file_hashes(before)
    (before / "build.json").write_text(json.dumps(build, indent=2) + "\n")
    shutil.copytree(before, after)
    reserialize.convert(after)

    assert reserialize.loaded_differences(before, after, "BertForMaskedLM") == []


def lock(repo="huggingface-mic/x-mic"):
    return {
        "framework": "transformers",
        "environment": {"python": "3.11"},
        "contracts": {
            "X": {
                "integration": {"repo": repo, "revision": "a" * 40, "files_sha256": "c" * 64},
                "fixtures": {repo: {"revision": "a" * 40, "files_sha256": "1" * 64}},
                "checkpoints": {"org/x": "b" * 40},
            },
            "Shared": {
                "integration": {"repo": "huggingface-mic/y-mic", "revision": "d" * 40, "files_sha256": "e" * 64},
                "fixtures": {
                    "huggingface-mic/y-mic": {"revision": "d" * 40, "files_sha256": "3" * 64},
                    repo: {"revision": "a" * 40, "files_sha256": "1" * 64, "target": "x"},
                },
                "checkpoints": {},
            },
        },
    }


def test_repin_moves_every_pin_and_keeps_the_contract_hash():
    fleet_lock = lock()
    changed = reserialize.repin(fleet_lock, "huggingface-mic/x-mic", "1" * 64, "f" * 40, "2" * 64)

    assert changed == ["X", "Shared"]
    x, shared = fleet_lock["contracts"]["X"], fleet_lock["contracts"]["Shared"]
    assert x["integration"] == {"repo": "huggingface-mic/x-mic", "revision": "f" * 40, "files_sha256": "c" * 64}
    assert x["fixtures"]["huggingface-mic/x-mic"] == {"revision": "f" * 40, "files_sha256": "2" * 64}
    assert shared["integration"]["revision"] == "d" * 40
    assert shared["fixtures"]["huggingface-mic/x-mic"]["target"] == "x"


def test_repin_refuses_a_fleet_pinning_another_build():
    with pytest.raises(ValueError, match="another build"):
        reserialize.repin(lock(), "huggingface-mic/x-mic", "9" * 64, "f" * 40, "2" * 64)


def test_export_retries_a_concurrent_dataset_update(monkeypatch, tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock()))
    parents, uploads = iter(["p1", "p2"]), []
    monkeypatch.setattr(reserialize.fleet, "download", lambda name: (path, next(parents)))

    class Api:
        def create_commit(self, **kwargs):
            uploads.append(kwargs["parent_commit"])
            if len(uploads) == 1:
                raise HfHubHTTPError("moved", response=SimpleNamespace(status_code=412, headers={}, request=None))
            return SimpleNamespace(oid="n" * 40)

    monkeypatch.setattr(reserialize, "HfApi", Api)
    result = reserialize.export("huggingface-mic/x-mic", "1" * 64, "f" * 40, "2" * 64, ["prototypes"])

    assert uploads == ["p1", "p2"]
    assert result == {"prototypes": {"revision": "n" * 40, "contracts": ["X", "Shared"]}}
