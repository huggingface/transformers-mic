import hashlib
from pathlib import Path

import pytest

from transformers_mic import github_baselines as github
from transformers_mic.workspace import write


def setup_collection(root, name=".mic/baseline-lock.json"):
    revision = "a" * 40
    relative = f"models/org--model/{revision}/{name}"
    write(root / "cloud.json", {"request": {"profiles": ["cuda"]}})
    write(
        root / "artifacts/cuda/collection.json",
        {
            "models": [
                {
                    "repo": "org/model",
                    "revision": revision,
                    "files": {relative: hashlib.sha256(b"original").hexdigest()},
                }
            ]
        },
    )
    return relative


def test_restore_missing_hidden_source_from_exact_pin(tmp_path, monkeypatch):
    relative = setup_collection(tmp_path)
    cached = tmp_path / "cached"
    cached.write_bytes(b"original")
    calls = []

    def download(repo, filename, *, revision):
        calls.append((repo, filename, revision))
        return cached

    monkeypatch.setattr(github, "hf_hub_download", download)
    github.restore_hidden_sources(tmp_path)
    github.restore_hidden_sources(tmp_path)
    assert calls == [("org/model", ".mic/baseline-lock.json", "a" * 40)]
    assert (tmp_path / "artifacts/cuda" / relative).read_bytes() == b"original"


def test_restore_rejects_changed_content(tmp_path, monkeypatch):
    relative = setup_collection(tmp_path)
    cached = tmp_path / "cached"
    cached.write_bytes(b"changed")
    monkeypatch.setattr(github, "hf_hub_download", lambda *a, **kw: cached)
    with pytest.raises(ValueError, match="differs"):
        github.restore_hidden_sources(tmp_path)
    assert not (tmp_path / "artifacts/cuda" / relative).exists()


@pytest.mark.parametrize("name", ["weights.safetensors", "../.secret"])
def test_restore_does_not_repair_measurements_or_unsafe_paths(
    tmp_path, monkeypatch, name
):
    setup_collection(tmp_path, name)
    monkeypatch.setattr(
        github, "hf_hub_download", lambda *a, **kw: pytest.fail("unexpected fetch")
    )
    if ".." in Path(name).parts:
        with pytest.raises(ValueError, match="Unsafe"):
            github.restore_hidden_sources(tmp_path)
    else:
        github.restore_hidden_sources(tmp_path)
