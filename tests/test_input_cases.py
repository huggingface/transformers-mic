"""Input cases: every declared case runs for every target and variant, each with its own baseline.

A contract without cases, and the default case of one that declares them, keep today's
baseline names and contents: a baseline recorded before cases existed still passes."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))

import run  # noqa: E402

CASES = {"text": {"default": True}, "image": {}, "text_image": {}}


def test_baseline_names_add_the_case_before_the_variant_except_for_the_default_case():
    assert run.baseline_name("gemma3-mic", "default", None, None) == "gemma3-mic.json"
    assert run.baseline_name("gemma3-mic", "bfloat16", None, None) == "gemma3-mic.bfloat16.json"
    assert run.baseline_name("gemma3-mic", "default", "text", "text") == "gemma3-mic.json"
    assert run.baseline_name("gemma3-mic", "bfloat16", "text", "text") == "gemma3-mic.bfloat16.json"
    assert run.baseline_name("gemma3-mic", "default", "image", "text") == "gemma3-mic.image.json"
    assert run.baseline_name("gemma3-mic", "bfloat16", "image", "text") == "gemma3-mic.image.bfloat16.json"
    # Without a default case every case is named, the default one included.
    assert run.baseline_name("gemma3-mic", "bfloat16", "text", None) == "gemma3-mic.text.bfloat16.json"


def test_input_cases_default_to_one_implicit_case():
    assert run.input_cases({"variants": {"default": {}}}, "x") == ([None], None)
    integration = {"variants": {"default": {}, "eager": {}}, "cases": CASES}
    assert run.input_cases(integration, "x") == (["text", "image", "text_image"], "text")
    assert run.input_cases({**integration, "cases": {"text": {}, "image": {}}}, "x") == (["text", "image"], None)


def test_input_cases_refuse_names_that_read_two_ways():
    with pytest.raises(SystemExit, match="cases named like variants"):
        run.input_cases({"variants": {"default": {}, "image": {}}, "cases": {"text": {}, "image": {}}}, "x")
    with pytest.raises(SystemExit, match="more than one default"):
        run.input_cases({"variants": {"default": {}}, "cases": {"a": {"default": True}, "b": {"default": True}}}, "x")


def test_known_failures_scoped_to_cases():
    known = {"Gemma3/gemma3-mic": {"issue": "image path", "cases": ["image", "text_image"]}}
    assert run.known_entry(known, "Gemma3", "gemma3-mic", "default", "cpu-x86_64", "image")["issue"] == "image path"
    assert run.known_entry(known, "Gemma3", "gemma3-mic", "default", "cpu-x86_64", "text") is None
    # A contract without cases has no case to match.
    assert run.known_entry(known, "Gemma3", "gemma3-mic", "default", "cpu-x86_64", None) is None
    # Unscoped entries still cover every case.
    assert run.known_entry({"Gemma3/gemma3-mic": "bug"}, "Gemma3", "gemma3-mic", case="image") == {"issue": "bug"}


def test_schemas_accept_cases_and_refuse_dotted_names():
    pytest.importorskip("jsonschema")
    manifest = json.loads((ROOT / "example/hub/BERT-integration/integration.json").read_text())
    manifest["integrations"]["transformers"]["cases"] = CASES
    run.validate(manifest, "integration", "cases")
    manifest["integrations"]["transformers"]["cases"] = {"text.image": {}}
    with pytest.raises(SystemExit, match="invalid integration"):
        run.validate(manifest, "integration", "dotted")
    lock = run.load(ROOT / "example/transformers/tests/contracts/fleet.lock", "fleet-lock")
    lock["known_failures"]["BERT/bert-base-uncased"] = {"issue": "x", "cases": ["image"]}
    run.validate(lock, "fleet-lock", "cases")


@pytest.fixture
def contract(tmp_path, monkeypatch):
    """A one-fixture contract whose outputs depend on MIC_CASE, run through run.main()."""
    pytest.importorskip("jsonschema")
    root = tmp_path / "Gemma3-integration"
    manifest = {
        "mic": "0.3",
        "name": "Gemma3",
        "usage": "Next-token prediction",
        "checkpoints": {"google/gemma-3-4b-it": {"fixture": "huggingface-mic/gemma3-mic"}},
        "fixtures": {},
        "integrations": {
            "transformers": {
                "entry_point": "integrations/transformers/usage.md",
                "format": "runnable-markdown",
                "targets": ["fixtures"],
                "variants": {"default": {}, "bfloat16": {"dtype": "bfloat16"}},
                "training": False,
            }
        },
        "outputs": {"runtime": {"compare": "exact"}, "predicted_tokens": {"compare": "exact"}},
    }
    lock = {
        "framework": "transformers",
        "environment": {},
        "contracts": {
            "Gemma3": {
                "integration": {"repo": "huggingface-mic/gemma3-mic", "revision": "unpublished", "path": str(root)},
                "fixtures": {},
                "checkpoints": {"google/gemma-3-4b-it": "a" * 40},
            }
        },
    }
    outputs = {None: ["▁blue"], "text": ["▁blue"], "image": ["▁cat"], "text_image": ["▁dog"]}
    ran = []

    class Executor:
        def __init__(self, *a):
            pass

        def environment(self):
            return {"python": "3.11", "transformers": "test"}

        def run(self, env, fresh_process=False):
            ran.append(env.get("MIC_CASE"))
            dtype = json.loads(env["MIC_MODEL_KWARGS"]).get("dtype", "float32")
            result = {"runtime": {"device": "cuda", "dtype": dtype}, "predicted_tokens": outputs[env.get("MIC_CASE")]}
            return True, json.dumps(result), ""

    monkeypatch.setattr(run, "contract_folder", lambda *a: (root, "local"))
    monkeypatch.setattr(run, "targets", lambda *a: iter([("gemma3-mic", "fixture", "/fixture", "a" * 40, None)]))
    monkeypatch.setattr(run, "Executor", Executor)
    monkeypatch.setattr(
        run.template_date if hasattr(run, "template_date") else __import__("template_date"), "freeze", lambda *a: None
    )

    class Contract:
        expectations = tmp_path / "expectations/Gemma3/cuda"

        @staticmethod
        def write(cases=None, known=None):
            integration = manifest["integrations"]["transformers"]
            integration.pop("cases", None)
            if cases:
                integration["cases"] = cases
            (root / "integrations/transformers").mkdir(parents=True, exist_ok=True)
            (root / "integration.json").write_text(json.dumps(manifest))
            (tmp_path / "fleet.lock").write_text(json.dumps({**lock, **({"known_failures": known} if known else {})}))

        @staticmethod
        def main(*options):
            monkeypatch.setattr(
                sys, "argv", ["run", "--framework", "transformers", "--lock", str(tmp_path / "fleet.lock"), *options]
            )
            ran.clear()
            return run.main()

        @staticmethod
        def approve():
            for candidate in Contract.expectations.glob("*.candidate.json"):
                candidate.rename(candidate.with_name(candidate.name.replace(".candidate", "")))

        outputs_by_case = outputs
        cases_run = ran

    return Contract


def records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"contract"')]


def test_a_contract_without_cases_is_unchanged_but_records_a_null_case(contract, capsys):
    contract.write()
    assert contract.main("--record") == 0
    assert contract.cases_run == [None, None]  # the run and its fresh-process rerun; MIC_CASE never set
    contract.approve()
    assert sorted(p.name for p in contract.expectations.iterdir()) == ["gemma3-mic.json"]
    baseline = json.loads((contract.expectations / "gemma3-mic.json").read_text())
    assert baseline["recorded_with"]["case"] is None
    capsys.readouterr()
    assert contract.main() == 0
    (record,) = records(capsys)
    assert record["case"] is None and record["status"] == "pass"


def test_declaring_cases_keeps_the_old_baseline_for_the_default_case(contract, capsys):
    # A baseline recorded before cases existed: no case in recorded_with.
    contract.write()
    assert contract.main("--record") == 0
    contract.approve()
    old = contract.expectations / "gemma3-mic.json"
    legacy = json.loads(old.read_text())
    del legacy["recorded_with"]["case"]
    old.write_text(json.dumps(legacy))
    before = old.read_bytes()
    capsys.readouterr()

    contract.write(CASES)
    assert contract.main() == 1  # the new cases have no baseline yet
    rows = {r["case"]: r for r in records(capsys)}
    assert contract.cases_run == ["text", "image", "text_image"]
    assert rows["text"]["status"] == "pass" and rows["text"]["baseline"].endswith("gemma3-mic.json")
    assert rows["image"]["failures"][0].startswith("no reviewed baseline at ")
    assert rows["image"]["baseline"].endswith("gemma3-mic.image.json")

    # Recording writes the explicit case into every candidate, the default case's included,
    # next to the old file, which it never touches.
    assert contract.main("--record") == 0
    assert old.read_bytes() == before
    text = json.loads((contract.expectations / "gemma3-mic.candidate.json").read_text())
    assert text["recorded_with"]["case"] == "text"
    (contract.expectations / "gemma3-mic.candidate.json").unlink()  # keep the old baseline
    contract.approve()
    assert sorted(p.name for p in contract.expectations.iterdir()) == [
        "gemma3-mic.image.json",
        "gemma3-mic.json",
        "gemma3-mic.text_image.json",
    ]
    capsys.readouterr()
    assert contract.main() == 0
    assert {r["case"]: r["status"] for r in records(capsys)} == {"text": "pass", "image": "pass", "text_image": "pass"}

    # Variants name the case first.
    assert contract.main("--record", "--variant", "bfloat16") == 0
    assert {p.name for p in contract.expectations.glob("*.candidate.json")} == {
        "gemma3-mic.bfloat16.candidate.json",
        "gemma3-mic.image.bfloat16.candidate.json",
        "gemma3-mic.text_image.bfloat16.candidate.json",
    }


def test_a_baseline_of_another_case_is_refused(contract, capsys):
    contract.write(CASES)
    assert contract.main("--record") == 0
    contract.approve()
    # The image baseline in the text case's place: same outputs would not be enough.
    image = json.loads((contract.expectations / "gemma3-mic.image.json").read_text())
    (contract.expectations / "gemma3-mic.json").write_text(json.dumps(image))
    contract.outputs_by_case["text"] = contract.outputs_by_case["image"]
    capsys.readouterr()
    assert contract.main("--case", "text") == 1
    (record,) = records(capsys)
    assert record["failures"] == ["baseline identity differs: ['case']"]


def test_only_selected_cases_run(contract, capsys):
    contract.write(CASES)
    assert contract.main("--record", "--case", "image") == 0
    assert contract.cases_run == ["image", "image"]
    assert [p.name for p in contract.expectations.iterdir()] == ["gemma3-mic.image.candidate.json"]


def test_a_known_failure_scoped_to_a_case_is_xfail_and_says_when_it_passes(contract, capsys):
    contract.write(CASES)
    assert contract.main("--record") == 0
    contract.approve()
    contract.outputs_by_case["image"] = ["▁dog"]
    known = {"Gemma3/gemma3-mic": {"issue": "image path broken", "cases": ["image"]}}
    contract.write(CASES, known)
    capsys.readouterr()
    assert contract.main() == 0
    assert {r["case"]: r["status"] for r in records(capsys)} == {"text": "pass", "image": "xfail", "text_image": "pass"}
    # Fixed: the entry asks to be removed; the other cases stay ordinary passes.
    contract.outputs_by_case["image"] = ["▁cat"]
    assert contract.main() == 1
    rows = {r["case"]: r for r in records(capsys)}
    assert rows["image"]["status"] == "fail" and "remove it" in rows["image"]["failures"][0]
    assert rows["text"]["status"] == rows["text_image"]["status"] == "pass"
