"""The baseline environment takes its torch stack from the Transformers source."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/install_baseline_environment.py"
)


@pytest.fixture
def installer():
    spec = importlib.util.spec_from_file_location(
        "install_baseline_environment", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_torch_stack_follows_transformers_ci_images(tmp_path, monkeypatch, installer):
    (tmp_path / "docker/transformers-all-latest-gpu").mkdir(parents=True)
    (tmp_path / "docker/torch-light.dockerfile").write_text(
        "FROM ci-base\n"
        "RUN uv pip install --no-cache-dir 'torch<=2.14.0' 'torchaudio' 'torchvision' "
        "'torchcodec<=0.16.0' --index-url https://download.pytorch.org/whl/cpu\n"
        "RUN uv pip install --no-deps timm accelerate\n"
    )
    (tmp_path / "docker/transformers-all-latest-gpu/Dockerfile").write_text(
        "ARG PYTORCH='2.14.0'\nARG CUDA='cu130'\n"
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert installer.transformers_torch_stack(tmp_path) == (
        ["torch<=2.14.0", "torchaudio", "torchvision", "torchcodec<=0.16.0"],
        "cpu",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert installer.transformers_torch_stack(tmp_path) == (
        ["torch==2.14.0", "torchvision", "torchaudio"],
        "cu130",
    )
