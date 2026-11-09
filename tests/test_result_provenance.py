"""Result records name the fixture and the reviewed baseline as Hub locations (dashboards read them)."""

import importlib.util
import json
import sys

from transformers_mic import resources


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(resources.backend()))
    spec = importlib.util.spec_from_file_location(
        "mic_provenance_run", resources.backend() / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_and_baseline_provenance_is_in_results(tmp_path, monkeypatch, capsys):
    run = runner(monkeypatch)
    root, fixture = tmp_path / "contract", tmp_path / "fixture"
    root.mkdir()
    fixture.mkdir()
    (root / ".mic-repo.json").write_text(json.dumps({"contract_files": []}))
    manifest = {
        "mic": "0.2",
        "name": "Demo",
        "usage": "Regression test",
        "checkpoints": {"test/checkpoint": {"fixture": "test/demo-mic"}},
        "fixtures": {
            "test/demo-mic": {"built_from": "test/checkpoint", "covers": ["test/checkpoint"]}
        },
        "integrations": {
            "transformers": {
                "entry_point": "usage.md",
                "format": "runnable-markdown",
                "requires": [],
                "targets": ["fixtures"],
                "variants": {"default": {}},
                "training": False,
            }
        },
        "outputs": {"runtime": {"compare": "exact"}, "tokens": {"compare": "exact"}},
    }
    (root / "integration.json").write_text(json.dumps(manifest))
    (fixture / "build.json").write_text(
        json.dumps({"built_from": {"repo": "creator/Large-Demo"}, "weights": "random"})
    )
    actual = {"runtime": {"device": "cpu"}, "tokens": [1, 2]}
    baseline = root / "integrations/transformers/expectations/cpu-test/demo-mic.json"
    baseline.parent.mkdir(parents=True)
    identity = {
        "contract": "test/demo-integration",
        "contract_revision": "a" * 40,
        "target": "demo-mic",
        "revision": "b" * 40,
        "framework": "transformers",
        "variant": "default",
        "environment": {},
    }
    baseline.write_text(json.dumps({"recorded_with": identity, "outputs": actual}))
    lock = tmp_path / "fleet.lock"
    lock.write_text(
        json.dumps(
            {
                "framework": "transformers",
                "contracts": {
                    "demo": {
                        "integration": {"repo": "test/demo-integration", "revision": "a" * 40},
                        "fixtures": {"test/demo-mic": {"revision": "b" * 40}},
                        "checkpoints": {},
                    }
                },
                "environment": {},
            }
        )
    )

    class Executor:
        def __init__(self, *a):
            pass

        def environment(self):
            return {"python": "3.11"}

        def run(self, *a, **kw):
            return True, json.dumps(actual), ""

    monkeypatch.setattr(run, "contract_folder", lambda *a: (root, "a" * 40))
    monkeypatch.setattr(
        run, "targets", lambda *a: iter([("demo-mic", "fixture", str(fixture), "b" * 40, None)])
    )
    monkeypatch.setattr(run, "Executor", Executor)
    monkeypatch.setattr(run, "platform_class", lambda *a: "cpu-test")
    monkeypatch.setattr(
        sys, "argv", ["run", "--framework", "transformers", "--lock", str(lock), "--fixtures-only"]
    )
    assert run.main() == 0
    record = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["status"] == "pass"
    assert record["fixture_repo"] == "test/demo-mic"
    assert record["source_model"] == "creator/Large-Demo"
    assert record["weights"] == "random"
    assert record["baseline_repo"] == "test/demo-integration"
    assert record["baseline_revision"] == "a" * 40
    assert (
        record["baseline_path"]
        == "integrations/transformers/expectations/cpu-test/demo-mic.json"
    )
