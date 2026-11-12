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
