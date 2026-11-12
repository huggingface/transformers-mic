"""nightly.yml and shadow.yml run the pinned Hub control fleet plus the catalog's rest."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/control_locks.py"
SHA = "a" * 40


@pytest.fixture
def control_locks():
    spec = importlib.util.spec_from_file_location("control_locks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_download(tmp_path, contracts, calls):
    def download(name, revision):
        calls.append((name, revision))
        path = tmp_path / "hub" / f"{name}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "framework": "transformers",
                    "environment": {"python": "3.11", "torch": "2.15.0"},
                    "contracts": {n: {"integration": {"repo": f"huggingface-mic/{n.lower()}-mic"}} for n in contracts},
                }
            )
        )
        return path, revision

    return download


def test_reads_the_pinned_control_fleet_and_names_the_catalogs_rest(tmp_path, capsys, control_locks):
    pin = tmp_path / "contracts.lock"
    pin.write_text(SHA + "\n")
    catalog = tmp_path / "fleet.lock"
    catalog.write_text(json.dumps({"contracts": {"BERT": {}, "Whisper": {}, "WhisperSpeech": {}, "MusicgenMelody": {}}}))
    calls = []
    out = tmp_path / "out"
    download = fake_download(tmp_path, ["BERT", "Whisper", "CLIP"], calls)
    assert control_locks.main(["--out", str(out), "--pin", str(pin), "--catalog", str(catalog)], download=download) == 0
    assert calls == [("control", SHA)]
    lines = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert lines == {
        "CONTROL_LOCK": str((out / "control.lock").resolve()),
        "CONTROL_REVISION": SHA,
        "CATALOG_ONLY": "MusicgenMelody WhisperSpeech",
    }
    assert set(json.loads((out / "control.lock").read_text())["contracts"]) == {"BERT", "Whisper", "CLIP"}


def test_refuses_a_mutable_pin(tmp_path, control_locks):
    pin = tmp_path / "contracts.lock"
    pin.write_text("main\n")
    with pytest.raises(ValueError, match="immutable"):
        control_locks.main(["--out", str(tmp_path), "--pin", str(pin)], download=fake_download(tmp_path, ["BERT"], []))


def test_refuses_an_empty_control_fleet(tmp_path, control_locks):
    pin = tmp_path / "contracts.lock"
    pin.write_text(SHA + "\n")
    with pytest.raises(ValueError, match="not a Transformers lock"):
        control_locks.main(["--out", str(tmp_path / "out"), "--pin", str(pin)], download=fake_download(tmp_path, [], []))


def test_the_repository_pin_is_an_immutable_dataset_revision(control_locks):
    from transformers_mic import fleet

    assert fleet.read_pin(control_locks.PIN)


def test_the_catalog_keeps_its_legacy_contracts_for_tmic_create(control_locks):
    # tmic create copies catalog entries, their example/hub sources and in-tree
    # baselines into a workspace, so the catalog is not repointed at the fleet.
    catalog = json.loads(control_locks.CATALOG.read_text())
    for name, entry in catalog["contracts"].items():
        assert entry["integration"]["repo"] == f"hf-internal-testing/{name}-integration"
        assert "path" in entry["integration"]


def test_installers_take_the_lock_from_the_environment():
    script = (ROOT / ".github/scripts/install_pinned.sh").read_text()
    assert "LOCK=${LOCK:-example/transformers/tests/contracts/fleet.lock}" in script
    for workflow in ("nightly.yml", "shadow.yml"):
        text = (ROOT / ".github/workflows" / workflow).read_text()
        assert ".github/scripts/control_locks.py" in text
        assert '"$CONTROL_LOCK"' in text and "$CATALOG_ONLY" in text
