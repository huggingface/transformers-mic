"""Nested integrations: one framework (vLLM) with several implementations under integrations/vllm/."""

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformers_mic import backend, carryover, commands, resources
from transformers_mic.cli import main, parser
from transformers_mic.workspace import Workspace, digest, primary_integration, read, write

NAMES = ("vllm/transformers", "vllm/vllm")
FIXTURE = "huggingface-mic/qwen2-vllm-mic"


@pytest.fixture
def workspace(tmp_path):
    """An unpublished vLLM-only MIC: both vLLM implementations, no Transformers integration."""
    ws = Workspace(tmp_path / "work")
    root = ws.root / "example/hub/qwen2-vllm-mic"
    fixture = tmp_path / "fixture"
    build = {"sha256": {"model.safetensors": hashlib.sha256(b"weights").hexdigest()}}
    write(fixture / "build.json", build)
    (fixture / "model.safetensors").write_bytes(b"weights")
    write(ws.state_path, {"model": "Qwen/Qwen2-0.5B", "contract": "qwen2-vllm"})
    write(
        ws.lock_path,
        {
            "framework": "transformers",
            "environment": {},
            "contracts": {
                "qwen2-vllm": {
                    "integration": {"repo": FIXTURE, "revision": "unpublished", "path": str(root)},
                    "fixtures": {
                        FIXTURE: {
                            "revision": "unpublished",
                            "path": str(fixture),
                            "files_sha256": digest(build["sha256"]),
                        }
                    },
                    "checkpoints": {"Qwen/Qwen2-0.5B": "a" * 40},
                }
            },
        },
    )
    integration = {
        "format": "runnable-markdown",
        "requires": ["vllm"],
        "targets": ["fixtures"],
        "training": False,
        "outputs": {"runtime": {"compare": "exact"}, "tokens": {"compare": "exact"}},
    }
    write(
        root / "integration.json",
        {
            "mic": "0.3",
            "name": "Qwen2 (vLLM)",
            "usage": "vLLM greedy generation",
            "checkpoints": {"Qwen/Qwen2-0.5B": {"fixture": FIXTURE}},
            "fixtures": {FIXTURE: {"built_from": "Qwen/Qwen2-0.5B", "covers": ["Qwen/Qwen2-0.5B"]}},
            "integrations": {
                "vllm/transformers": {
                    **integration,
                    "entry_point": "integrations/vllm/transformers/usage.md",
                    "variants": {"default": {}},
                },
                "vllm/vllm": {
                    **integration,
                    "entry_point": "integrations/vllm/vllm/usage.md",
                    "variants": {"default": {}, "batched": {}},
                },
            },
            "outputs": {"runtime": {"compare": "exact"}},
        },
    )
    for name in NAMES:
        usage = root / f"integrations/{name}/usage.md"
        usage.parent.mkdir(parents=True)
        usage.write_text(f"{name} usage")
    write(root / ".mic-repo.json", {"format": "mic-umbrella-v1", "contract_files": []})
    ws.update_layout()
    return ws


@pytest.fixture
def values(monkeypatch):
    """Run the backend in this process; the usage's outputs are whatever `values` holds."""
    sys.path.insert(0, str(resources.backend()))
    import run

    outputs = {"runtime": {"device": "cpu", "dtype": "float32"}, "tokens": [1, 2]}

    class Executor:
        def __init__(self, *a):
            pass

        def environment(self):
            return {"python": "3.11", "vllm": "test"}

        def run(self, *a, **kw):
            return True, json.dumps(outputs), ""

    monkeypatch.setattr(run, "Executor", Executor)
    monkeypatch.setattr(
        run, "targets", lambda *a, **kw: iter([("qwen2-vllm-mic", "fixture", "/fixture", "a" * 40, None)])
    )
    monkeypatch.setattr(
        commands.backend,
        "invoke",
        lambda tool, root, options: SimpleNamespace(
            returncode=backend.dispatch(tool, Path(root), [str(o) for o in options])
        ),
    )
    return outputs


def tmic(workspace, *argv):
    try:
        return main(["--workspace", str(workspace.root), *argv])
    except SystemExit as exc:  # the backend exits non-zero on failures
        return exc.code


def expectations(workspace, name):
    folder = workspace.contract_root / "integrations" / name / "expectations"
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.json")) if folder.exists() else []


@pytest.mark.parametrize("command", ["record", "check", "review"])
@pytest.mark.parametrize("name", NAMES)
def test_cli_accepts_nested_integrations(command, name):
    assert parser().parse_args([command, "--integration", name]).integration == name


def test_primary_integration_of_one_framework_is_its_first():
    assert primary_integration({"integrations": dict.fromkeys(NAMES, {})}) == "vllm/transformers"
    assert primary_integration({"integrations": dict.fromkeys(reversed(NAMES), {})}) == "vllm/vllm"
    assert primary_integration({"integrations": dict.fromkeys(["transformers", *NAMES], {})}) == "transformers"
    with pytest.raises(ValueError, match="--integration"):
        primary_integration({"integrations": dict.fromkeys(["vllm/transformers", "sglang"], {})})


@pytest.mark.requires_runtime
def test_record_review_check_keep_each_integration_in_its_own_folder(workspace, values):
    # record writes candidates under the nested integration's folder only
    assert tmic(workspace, "record", "--integration", "vllm/transformers") == 0
    assert expectations(workspace, "vllm/transformers") == ["cpu-x86_64/qwen2-vllm-mic.candidate.json"]
    assert expectations(workspace, "vllm/vllm") == []
    assert (workspace.root / ".mic/record.vllm-transformers.default.jsonl").exists()
    assert tmic(workspace, "review", "--integration", "vllm/transformers", "--approve") == 0
    assert expectations(workspace, "vllm/transformers") == ["cpu-x86_64/qwen2-vllm-mic.json"]
    baseline = read(
        workspace.contract_root / "integrations/vllm/transformers/expectations/cpu-x86_64/qwen2-vllm-mic.json"
    )
    assert baseline["recorded_with"]["framework"] == "vllm/transformers"
    assert baseline["recorded_with"]["baseline_path"] == (
        "integrations/vllm/transformers/expectations/cpu-x86_64/qwen2-vllm-mic.json"
    )
    assert tmic(workspace, "check", "--integration", "vllm/transformers") == 0
    assert "vllm/transformers:default" in workspace.state["validations"]
    # the sibling does not read that baseline
    assert tmic(workspace, "check", "--integration", "vllm/vllm") == 1
    # every variant of the other implementation, with its own values
    values["tokens"] = [1, 3]
    assert tmic(workspace, "record", "--integration", "vllm/vllm", "--all-variants") == 0
    assert tmic(workspace, "review", "--integration", "vllm/vllm", "--approve") == 0
    assert expectations(workspace, "vllm/vllm") == [
        "cpu-x86_64/qwen2-vllm-mic.batched.json",
        "cpu-x86_64/qwen2-vllm-mic.json",
    ]
    assert tmic(workspace, "check", "--integration", "vllm/vllm", "--all-variants") == 0
    assert tmic(workspace, "check", "--integration", "vllm/transformers") == 1
    values["tokens"] = [1, 2]
    assert tmic(workspace, "check", "--integration", "all", "--all-variants") == 1
    values["tokens"] = [1, 3]
    assert tmic(workspace, "check", "--integration", "vllm/vllm", "--all-variants") == 0
    # without --integration: the first declared one
    assert tmic(workspace, "check") == 1
    assert (workspace.root / ".mic/check.vllm-transformers.default.jsonl").exists()


@pytest.mark.requires_runtime
def test_publication_needs_every_variant_of_every_nested_integration(workspace, values):
    for name in NAMES:
        assert tmic(workspace, "record", "--integration", name, "--all-variants") == 0
        assert tmic(workspace, "review", "--integration", name, "--approve") == 0
    assert tmic(workspace, "check", "--integration", "vllm/transformers") == 0
    assert tmic(workspace, "check", "--integration", "vllm/vllm") == 0
    with pytest.raises(ValueError, match="Every declared variant"):
        commands.publication_ready(workspace)
    assert tmic(workspace, "check", "--integration", "all", "--all-variants") == 0
    assert set(workspace.state["validations"]) == {
        "vllm/transformers:default",
        "vllm/vllm:default",
        "vllm/vllm:batched",
    }
    assert commands.publication_ready(workspace) == {
        "vllm/transformers": "primary",
        "vllm/vllm": "not published yet",
    }
    # an unreviewed candidate in a nested folder blocks publication
    candidate = workspace.contract_root / "integrations/vllm/transformers/expectations/cpu-x86_64/x.candidate.json"
    write(candidate, {})
    with pytest.raises(ValueError, match="Unreviewed vllm/transformers candidate"):
        commands.publication_ready(workspace)


def published(workspace, tmp_path):
    """Pin the workspace to a published revision whose Hub snapshot matches it."""
    for name in NAMES:
        write(workspace.contract_root / f"integrations/{name}/expectations/cpu-x86_64/qwen2-vllm-mic.json", {})
    lock = workspace.lock
    contract = lock["contracts"]["qwen2-vllm"]
    contract["integration"]["revision"] = "b" * 40
    write(workspace.lock_path, lock)
    build = read(Path(contract["fixtures"][FIXTURE]["path"]) / "build.json")
    hub = {}
    for revision in ("b" * 40, "main"):
        folder = tmp_path / "hub" / revision
        shutil.copytree(workspace.contract_root, folder)
        write(folder / "build.json", build)
        hub[revision] = folder
    return lambda repo, revision, frameworks: hub[revision]


def test_carryover_compares_each_nested_folder_alone(workspace, tmp_path):
    fetch = published(workspace, tmp_path)
    files = carryover.integration_files(workspace.contract_root, workspace.manifest, "vllm/vllm")
    assert sorted(files) == [
        "integrations/vllm/vllm/expectations/cpu-x86_64/qwen2-vllm-mic.json",
        "integrations/vllm/vllm/usage.md",
    ]
    assert carryover.plan(workspace, fetch=fetch) == {"vllm/transformers": "primary", "vllm/vllm": None}
    # a change in the primary's folder does not touch the sibling's carry-over ...
    (workspace.contract_root / "integrations/vllm/transformers/usage.md").write_text("changed")
    assert carryover.plan(workspace, fetch=fetch)["vllm/vllm"] is None
    # ... one in its own folder does
    (workspace.contract_root / "integrations/vllm/vllm/usage.md").write_text("changed")
    assert carryover.plan(workspace, fetch=fetch)["vllm/vllm"].startswith(
        "files differ: integrations/vllm/vllm/usage.md vs bbbbbbbbbbbb"
    )


def test_carryover_counts_the_files_both_vllm_integrations_share(workspace, tmp_path):
    shared = workspace.contract_root / "integrations/vllm/requirements.txt"
    shared.write_text("vllm>=0.34\n")
    fetch = published(workspace, tmp_path)
    # the parent folder's own files belong to both; its subfolders are the siblings, not shared
    assert sorted(carryover.integration_files(workspace.contract_root, workspace.manifest, "vllm/vllm")) == [
        "integrations/vllm/requirements.txt",
        "integrations/vllm/vllm/expectations/cpu-x86_64/qwen2-vllm-mic.json",
        "integrations/vllm/vllm/usage.md",
    ]
    assert "integrations/vllm/requirements.txt" in carryover.integration_files(
        workspace.contract_root, workspace.manifest, "vllm/transformers"
    )
    assert carryover.plan(workspace, fetch=fetch) == {"vllm/transformers": "primary", "vllm/vllm": None}
    # a change to a shared file is a change of every nested integration: none is carried over
    shared.write_text("vllm>=0.35\n")
    assert carryover.plan(workspace, fetch=fetch)["vllm/vllm"].startswith(
        "files differ: integrations/vllm/requirements.txt vs bbbbbbbbbbbb"
    )
    # so is a shared file the Hub does not have yet
    shared.write_text("vllm>=0.34\n")
    (workspace.contract_root / "integrations/vllm/constraints.txt").write_text("torch<3\n")
    assert carryover.plan(workspace, fetch=fetch)["vllm/vllm"].startswith(
        "files differ: integrations/vllm/constraints.txt vs bbbbbbbbbbbb"
    )


def test_carryover_fetches_the_shared_files_of_nested_integrations(monkeypatch):
    import huggingface_hub

    fetched = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo, revision, allow_patterns: fetched.append(allow_patterns) or "snapshot",
    )
    carryover.published("huggingface-mic/qwen2-vllm-mic", "b" * 40, ["vllm/vllm", "sentence_transformers"])
    assert fetched == [
        [
            "integration.json",
            "build.json",
            "integrations/vllm/vllm/**",
            "integrations/sentence_transformers/**",
            "integrations/vllm/*",
        ]
    ]
    # a flat integration has no folder above it
    assert carryover.shared_folders("sentence_transformers") == []
    assert carryover.shared_folders("vllm/vllm") == ["vllm"]


@pytest.mark.requires_runtime
def test_publish_stages_nested_expectations_and_carries_one(workspace, tmp_path, monkeypatch):
    for name in NAMES:
        write(
            workspace.contract_root / f"integrations/{name}/expectations/cpu-x86_64/qwen2-vllm-mic.json",
            {"reviewed": name},
        )
    write(workspace.contract_root / "integrations/vllm/vllm/expectations/cpu-x86_64/new.candidate.json", {})
    example = Path(__file__).resolve().parents[1] / "example"
    monkeypatch.syspath_prepend(str(example))
    spec = importlib.util.spec_from_file_location("publish_mic_nested_test", example / "publish_mic.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    monkeypatch.setattr(publisher, "HERE", workspace.root / "example")
    load = publisher.run.load
    monkeypatch.setattr(
        publisher.run, "load", lambda path, schema: read(path) if schema == "fixture-build" else load(path, schema)
    )
    monkeypatch.setattr(publisher.run, "fixture_problem", lambda *a: None)
    previous = tmp_path / "previous"
    write(previous / ".mic-repo.json", read(workspace.contract_root / ".mic-repo.json"))
    write(previous / "build.json", read(tmp_path / "fixture/build.json"))
    monkeypatch.setattr(publisher, "hf_hub_download", lambda repo, name: str(previous / name))
    uploads = []

    class API:
        def repo_exists(self, repo):
            return True

        def list_repo_files(self, repo):
            return [".mic-repo.json", "build.json", "integrations/vllm/vllm/usage.md"]

        def create_repo(self, repo, **kwargs):
            pass

        def upload_folder(self, repo_id, folder_path, delete_patterns, **kwargs):
            staged = {p.relative_to(folder_path).as_posix() for p in folder_path.rglob("*") if p.is_file()}
            uploads.append((staged, delete_patterns))
            return SimpleNamespace(oid="c" * 40)

    monkeypatch.setattr(publisher, "HfApi", API)
    unpublished = workspace.lock_path.read_text()
    monkeypatch.setattr(sys, "argv", ["publish_mic"])
    assert publisher.main() == 0
    workspace.lock_path.write_text(unpublished)  # the publish pinned the lock to the upload
    monkeypatch.setattr(sys, "argv", ["publish_mic", "--carry", "vllm/vllm"])
    assert publisher.main() == 0
    (both, _), (carried, deleted) = uploads
    nested = {p for p in both if p.startswith("integrations/")}
    assert nested == {
        "integrations/vllm/transformers/usage.md",
        "integrations/vllm/transformers/expectations/cpu-x86_64/qwen2-vllm-mic.json",
        "integrations/vllm/vllm/usage.md",
        "integrations/vllm/vllm/expectations/cpu-x86_64/qwen2-vllm-mic.json",
    }
    # the carried implementation is neither uploaded nor deleted; its sibling is published
    assert {p for p in carried if p.startswith("integrations/")} == {
        "integrations/vllm/transformers/usage.md",
        "integrations/vllm/transformers/expectations/cpu-x86_64/qwen2-vllm-mic.json",
    }
    assert not [p for p in deleted if p.startswith("integrations/vllm/vllm/")]


def test_fleet_check_runs_every_variant_of_a_nested_integration(tmp_path, monkeypatch):
    import huggingface_hub

    lock = tmp_path / "fleet.lock"
    write(lock, {"contracts": {"qwen2-vllm": {"integration": {"repo": FIXTURE, "revision": "b" * 40}}}})
    manifest = tmp_path / "integration.json"
    write(
        manifest, {"integrations": {"vllm/transformers": {}, "vllm/vllm": {"variants": {"default": {}, "batched": {}}}}}
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(manifest))
    invoked = []
    monkeypatch.setattr(
        backend, "invoke", lambda tool, cwd, options: invoked.append(options) or SimpleNamespace(returncode=0)
    )
    assert main(["check", "--lock", str(lock), "--integration", "vllm/vllm", "--all-variants"]) == 0
    assert [options[options.index("--variant") + 1] for options in invoked] == ["default", "batched"]
    assert {options[options.index("--framework") + 1] for options in invoked} == {"vllm/vllm"}


def test_an_integration_folder_holds_one_integration(monkeypatch):
    sys.path.insert(0, str(resources.backend()))
    import run

    monkeypatch.setattr(run, "schema", lambda name: {})  # only the folder rule, not the full manifest schema

    def manifest(*names):
        return {"integrations": dict.fromkeys(names, {})}

    with pytest.raises(SystemExit, match="vllm/transformers is inside another declared integration's folder"):
        run.validate(manifest("vllm", "vllm/transformers"), "integration", "integration.json")
    # siblings and look-alike names are fine
    assert run.validate(manifest(*NAMES), "integration", "x")
    assert run.validate(manifest("transformers", "transformers.js", "vllm", "vllm_extra"), "integration", "x")
