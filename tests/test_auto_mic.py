"""tmic-run.yml's automatic MIC creation: when a draft is ready, and what publication refuses."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / ".github" / "scripts" / "auto_mic.py"
spec = importlib.util.spec_from_file_location("auto_mic", SOURCE)
auto_mic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_mic)


def record_log(tmp_path, *cases):
    log = tmp_path / ".mic/automation/logs/record.log"
    log.parent.mkdir(parents=True)
    lines = [
        "tmic: record (noise that is not JSON)",
        *map(json.dumps, cases),
        json.dumps({"summary": {}}),
    ]
    log.write_text("\n".join(lines) + "\n")
    return tmp_path


def stopped_at(stage, kind="retry"):
    return {
        "status": "needs-input",
        "pending_question": {"kind": kind, "context": {"failed_stage": stage}},
    }


RECORDED = {"target": "a-mic", "kind": "fixture", "status": "recorded"}
NO_GPU = {
    "target": "a",
    "kind": "checkpoint",
    "status": "unsupported",
    "skip": "hardware",
    "failures": ["needs cuda"],
}
DECLARED = {
    "target": "a",
    "kind": "checkpoint",
    "status": "unsupported",
    "skip": "declared",
}
FAILED = {
    "target": "a-mic",
    "kind": "fixture",
    "status": "fail",
    "failures": ["logits mismatch"],
}


def test_a_validated_draft_is_ready(tmp_path):
    assert auto_mic.readiness({"status": "validated"}, tmp_path)[0]


def test_a_cuda_only_checkpoint_does_not_stop_a_cpu_draft(tmp_path):
    # MiniCPM-OCR: every fixture case recorded on CPU; the real checkpoint needs CUDA.
    workspace = record_log(tmp_path, RECORDED, NO_GPU, DECLARED)
    ok, reason = auto_mic.readiness(stopped_at("record"), workspace)
    assert ok and "hardware" in reason


@pytest.mark.parametrize("cases", [(RECORDED, FAILED, NO_GPU), (NO_GPU,), ()])
def test_a_failed_or_empty_recording_is_not_ready(tmp_path, cases):
    workspace = record_log(tmp_path, *cases)
    assert not auto_mic.readiness(stopped_at("record"), workspace)[0]


def test_a_draft_stopped_before_record_is_not_ready(tmp_path):
    ok, reason = auto_mic.readiness(stopped_at("build"), tmp_path)
    assert not ok and "build" in reason
    assert not auto_mic.readiness(
        {"status": "needs-input", "pending_question": {"kind": "source"}}, tmp_path
    )[0]


class Fleet:
    def __init__(self, contracts):
        self.contracts, self.exports = contracts, []

    def download(self, name):
        path = Path(self.dir) / f"{name}.json"
        path.write_text(
            json.dumps({"contracts": self.contracts if name == "prototypes" else {}})
        )
        return path, "rev0"

    def resolve(self, ref):
        return "rev1"


def workspace(repo="huggingface-mic/a-mic", revision="r1"):
    return SimpleNamespace(
        root=Path("."),
        fixture=repo,
        state={"contract": "A"},
        lock={
            "contracts": {"A": {"integration": {"repo": repo, "revision": revision}}}
        },
    )


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    from transformers_mic import commands
    from transformers_mic import fleet as fleet_module

    fake = Fleet({})
    fake.dir = tmp_path
    monkeypatch.setattr(fleet_module, "download", fake.download)
    monkeypatch.setattr(fleet_module, "resolve", fake.resolve)
    monkeypatch.setattr(
        commands,
        "export",
        lambda options: fake.exports.append(options.expected_revision),
    )
    return fake


def test_export_adds_a_new_contract_against_the_previewed_revision(fleet):
    assert auto_mic.export_new(workspace()) == "rev1"
    assert fleet.exports == ["rev0"]


def test_export_is_a_no_op_when_an_earlier_attempt_landed(fleet):
    fleet.contracts["A"] = {
        "integration": {"repo": "huggingface-mic/a-mic", "revision": "r1"}
    }
    assert auto_mic.export_new(workspace()) == "rev0" and fleet.exports == []


def test_export_never_overwrites_another_pin(fleet):
    fleet.contracts["A"] = {
        "integration": {"repo": "huggingface-mic/a-mic", "revision": "other"}
    }
    with pytest.raises(ValueError, match="refusing to overwrite"):
        auto_mic.export_new(workspace())


def test_export_retries_when_another_export_landed_first(fleet, monkeypatch):
    from transformers_mic import commands

    calls = []

    def export(options):
        calls.append(options.expected_revision)
        if len(calls) == 1:
            raise ValueError(
                "Dataset changed since preview; review the current fleet before exporting"
            )

    monkeypatch.setattr(commands, "export", export)
    monkeypatch.setattr(auto_mic.time, "sleep", lambda seconds: None)
    assert auto_mic.export_new(workspace()) == "rev1" and len(calls) == 2


def test_publication_refuses_a_contract_already_in_a_fleet_or_on_the_hub(
    fleet, monkeypatch
):
    import huggingface_hub

    files = ["README.md", ".gitattributes"]
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(list_repo_files=lambda repo, revision: files),
    )
    auto_mic.require_new(workspace())  # an empty repo, from the staging step, is fine
    files.append("integration.json")
    with pytest.raises(ValueError, match="already has published content"):
        auto_mic.require_new(workspace())
    fleet.contracts["a"] = {"integration": {"repo": "huggingface-mic/other-mic"}}
    with pytest.raises(ValueError, match="Already in the prototypes fleet"):
        auto_mic.require_new(workspace())
