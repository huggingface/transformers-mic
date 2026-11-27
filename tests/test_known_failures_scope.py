"""known_failures entries scoped to variants and platforms (torch 2.15 bfloat16 drift, #247)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))

import run  # noqa: E402

CATALOG = ROOT / "example/transformers/tests/contracts/fleet.lock"


def test_a_scoped_entry_covers_only_its_variants_and_platforms():
    known = {"WhisperSpeech/whisper-tiny": {"issue": "drift", "variants": ["bfloat16"], "platforms": ["cuda"]}}
    contract = "hf-internal-testing/WhisperSpeech-integration"
    assert run.known_entry(known, contract, "whisper-tiny", "bfloat16", "cuda")["issue"] == "drift"
    assert run.known_entry(known, "WhisperSpeech", "whisper-tiny", "bfloat16", "cuda")["issue"] == "drift"
    assert run.known_entry(known, contract, "whisper-tiny", "default", "cuda") is None
    assert run.known_entry(known, contract, "whisper-tiny", "bfloat16", "cpu-arm64") is None
    # Failed before its platform was measured: not covered.
    assert run.known_entry(known, contract, "whisper-tiny", "bfloat16", None) is None


def test_unscoped_entries_keep_covering_the_whole_target():
    known = {"BERT/bert-base-uncased": "bug", "T5/t5-small": {"issue": "bug", "match": "x"}}
    assert run.known_entry(known, "BERT", "bert-base-uncased", "eager", "cuda") == {"issue": "bug"}
    assert run.known_entry(known, "T5", "t5-small")["match"] == "x"
    assert run.known_entry(known, "T5", "other") is None


def test_the_catalog_lock_validates_and_scopes_every_known_failure():
    pytest.importorskip("jsonschema")
    lock = run.load(CATALOG, "fleet-lock")
    for key, entry in lock["known_failures"].items():
        assert key.split("/")[0] in lock["contracts"]
        assert entry["variants"] and entry["platforms"], f"{key} must stay narrowly scoped"
        assert "torch 2.15" in entry["issue"]


def test_the_schema_rejects_an_empty_scope():
    pytest.importorskip("jsonschema")
    lock = run.load(CATALOG, "fleet-lock")
    lock["known_failures"]["SmolLM/SmolLM-135M"]["platforms"] = []
    with pytest.raises(SystemExit, match="known_failures"):
        run.validate(lock, "fleet-lock", "catalog")


def test_an_integration_scoped_entry_covers_only_its_integrations():
    known = {"Qwen2/qwen2-mic": {"issue": "backend bug", "integrations": ["vllm"], "variants": ["default"]}}
    assert run.known_entry(known, "Qwen2", "qwen2-mic", "default", "cpu-x86_64", "vllm")["issue"] == "backend bug"
    assert run.known_entry(known, "Qwen2", "qwen2-mic", "default", "cpu-x86_64", "transformers") is None
    # every scope must match: the right integration with another variant is not covered
    assert run.known_entry(known, "Qwen2", "qwen2-mic", "bfloat16", "cpu-x86_64", "vllm") is None
    # an entry without integrations keeps covering every integration
    unscoped = {"Qwen2/qwen2-mic": {"issue": "bug", "variants": ["default"]}}
    for integration in ("transformers", "vllm", None):
        assert run.known_entry(unscoped, "Qwen2", "qwen2-mic", "default", "cuda", integration)["issue"] == "bug"


def test_the_schema_rejects_an_empty_integrations_scope():
    pytest.importorskip("jsonschema")
    lock = run.load(CATALOG, "fleet-lock")
    entry = lock["known_failures"]["SmolLM/SmolLM-135M"]
    entry["integrations"] = ["transformers"]
    run.validate(lock, "fleet-lock", "catalog")
    entry["integrations"] = []
    with pytest.raises(SystemExit, match="known_failures"):
        run.validate(lock, "fleet-lock", "catalog")


@pytest.mark.requires_runtime
def test_a_failure_scoped_to_one_integration_leaves_the_others_pass(tmp_path, monkeypatch, capsys):
    import json

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    root = tmp_path / "contract"
    integration = {
        "format": "runnable-markdown",
        "targets": ["fixtures"],
        "variants": {"default": {}},
        "training": False,
    }
    write(
        root / "integration.json",
        {
            "mic": "0.3",
            "name": "Two integrations",
            "usage": "test",
            "checkpoints": {"org/model": {"fixture": "org/model-mic"}},
            "fixtures": {},
            "integrations": {
                "transformers": {**integration, "entry_point": "integrations/transformers/usage.md"},
                "vllm": {**integration, "entry_point": "integrations/vllm/usage.md"},
            },
            "outputs": {"runtime": {"compare": "exact"}},
        },
    )
    for name in ("transformers", "vllm"):
        (root / f"integrations/{name}").mkdir(parents=True)
        (root / f"integrations/{name}/usage.md").write_text("usage")
    lock = tmp_path / "fleet.lock"
    entry = {"issue": "vLLM backend bug", "integrations": ["vllm"]}
    write(
        lock,
        {
            "framework": "transformers",
            "environment": {},
            "contracts": {
                "Two": {
                    "integration": {"repo": "org/model-mic", "revision": "unpublished", "path": str(root)},
                    "fixtures": {},
                    "checkpoints": {"org/model": "a" * 40},
                }
            },
            "known_failures": {"Two/model-mic": entry},
        },
    )
    monkeypatch.setattr(run, "contract_folder", lambda *a: (root, "local"))
    monkeypatch.setattr(run, "targets", lambda *a: iter([("model-mic", "fixture", "/fixture", "a" * 40, None)]))
    template_date = run.template_date if hasattr(run, "template_date") else __import__("template_date")
    monkeypatch.setattr(template_date, "freeze", lambda *a: None)
    passes = {"transformers": True, "vllm": False}  # the same target passes in one integration, fails in the other
    framework = []

    class Executor:
        def __init__(self, *a):
            pass

        def environment(self):
            return {"python": "3.11"}

        def run(self, *a, **kw):
            if passes[framework[0]]:
                return True, json.dumps({"runtime": {"device": "cpu", "dtype": "float32"}}), ""
            return False, "", "RuntimeError: backend bug"

    monkeypatch.setattr(run, "Executor", Executor)

    def check(name):
        framework[:] = [name]
        monkeypatch.setattr(sys, "argv", ["run", "--framework", name, "--lock", str(lock), "--no-baseline"])
        code = run.main()
        (record,) = [
            json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"contract"')
        ]
        return code, record["status"], record.get("failures")

    assert check("vllm") == (0, "xfail", ["RuntimeError: backend bug"])
    assert check("transformers") == (0, "pass", [])
    # unscoped (a plain reason string covers every run of the target), the other integration's pass becomes "remove it"
    write(lock, {**json.loads(lock.read_text()), "known_failures": {"Two/model-mic": entry["issue"]}})
    code, status, failures = check("transformers")
    assert (code, status) == (1, "fail") and "remove it" in failures[0]
