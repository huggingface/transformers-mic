"""record-mic: PR overlays on the fleet, and the record → publish → merge path."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformers_mic import fleet, pr_record

A, B, C, D = ("a" * 40, "b" * 40, "c" * 40, "d" * 40)


def lock(revision=A):
    return {
        "framework": "transformers",
        "environment": {"torch": "2.15.0"},
        "contracts": {
            "gpt-j-6b": {
                "integration": {"repo": "huggingface-mic/gpt-j-6b-mic", "revision": revision, "files_sha256": "x"},
                "fixtures": {"huggingface-mic/gpt-j-6b-mic": {"revision": revision, "files_sha256": "y"}},
                "checkpoints": {},
            },
            "bert": {
                "integration": {"repo": "huggingface-mic/bert-mic", "revision": C, "files_sha256": "x"},
                "fixtures": {"huggingface-mic/bert-mic": {"revision": C, "files_sha256": "y", "target": "mic-bert"}},
                "checkpoints": {},
            },
        },
    }


OVERLAY = {"contracts": {"control/gpt-j-6b": {"huggingface-mic/gpt-j-6b-mic": {"from": A, "to": B}}}}


def test_overlay_moves_integration_and_fixture_pins():
    got, report = fleet.apply_overlay(lock(), OVERLAY, "control")
    contract = got["contracts"]["gpt-j-6b"]
    assert contract["integration"]["revision"] == contract["fixtures"]["huggingface-mic/gpt-j-6b-mic"]["revision"] == B
    assert got["contracts"]["bert"]["integration"]["revision"] == C
    assert report == {"applied": ["gpt-j-6b:huggingface-mic/gpt-j-6b-mic"], "stale": []}


def test_overlay_is_stale_once_main_moved_and_ignores_other_fleets():
    got, report = fleet.apply_overlay(lock(revision=D), OVERLAY, "control")
    assert got["contracts"]["gpt-j-6b"]["integration"]["revision"] == D
    assert report["stale"] == ["gpt-j-6b:huggingface-mic/gpt-j-6b-mic"]
    assert fleet.apply_overlay(lock(), OVERLAY, "prototypes")[1] == {"applied": [], "stale": []}
    assert fleet.apply_overlay(lock(), None, "control")[1] == {"applied": [], "stale": []}


def test_overlaid_lock_still_round_trips_through_the_tables():
    got, _ = fleet.apply_overlay(lock(), OVERLAY, "control")
    assert fleet.from_tables(fleet.to_tables(got))["contracts"]["gpt-j-6b"]["integration"]["revision"] == B


def test_select_by_model_type_contract_or_target():
    types = {"huggingface-mic/gpt-j-6b-mic": "gptj", "huggingface-mic/bert-mic": "bert"}
    pick = lambda *models: pr_record.select(lock(), models, types=lambda repo, rev: types[repo])  # noqa: E731
    assert pick("gptj") == ["gpt-j-6b"]
    assert pick("GPT-J-6B") == ["gpt-j-6b"]
    assert pick("mic-bert") == ["bert"]
    assert pick("llama") == []


def write_record(root, pr=7, **case):
    file = root / "control/gpt-j-6b/verification/cpu-x86_64-bf16/gpt-j-6b-mic.bfloat16.json"
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps({"recorded_with": {}, "outputs": {"logits": [1.0]}}))
    record = {
        "pr": pr,
        "models": ["gptj"],
        "errors": [],
        "cases": [
            {
                "fleet": "control",
                "contract": "gpt-j-6b",
                "repo": "huggingface-mic/gpt-j-6b-mic",
                "pinned": A,
                "measured_on": A,
                "target": "gpt-j-6b-mic",
                "variant": "bfloat16",
                "platform": "cpu-x86_64-bf16",
                "path_in_repo": "integrations/transformers/expectations/cpu-x86_64-bf16/gpt-j-6b-mic.bfloat16.json",
                "file": str(file.relative_to(root)),
                "verified": True,
                "changes": {"logits": [[0.9], [1.0]]},
                **case,
            }
        ],
    }
    (root / pr_record.RECORD).write_text(json.dumps(record))
    return record


@pytest.mark.parametrize(
    "case, message",
    [
        ({"verified": False}, "verification"),
        ({"path_in_repo": "integration.json"}, "unexpected destination"),
        ({"path_in_repo": "integrations/transformers/expectations/cpu/../../x.json"}, "unexpected destination"),
        ({"file": "../../etc/passwd.json"}, "unexpected source"),
        ({"fleet": "other"}, "unexpected destination"),
    ],
)
def test_validate_refuses_anything_but_verified_expectations(tmp_path, case, message):
    record = write_record(tmp_path, **case)
    with pytest.raises(ValueError, match=message):
        pr_record.validate(tmp_path, record, 7)


def test_validate_refuses_another_prs_record(tmp_path):
    with pytest.raises(ValueError, match="not 8"):
        pr_record.validate(tmp_path, write_record(tmp_path), 8)


class FakeApi:
    def __init__(self):
        self.calls = []

    def list_repo_refs(self, repo):
        return SimpleNamespace(branches=[])

    def create_branch(self, repo, branch, revision):
        self.calls.append(("branch", repo, branch, revision))

    def delete_branch(self, repo, branch):
        self.calls.append(("delete_branch", repo, branch))

    def create_tag(self, repo, tag, revision, exist_ok):
        self.calls.append(("tag", repo, tag, revision))

    def create_commit(self, repo_id, operations, commit_message, repo_type=None, revision=None, parent_commit=None):
        self.calls.append(("commit", repo_id, revision, parent_commit, operations))
        return SimpleNamespace(oid=B if repo_type is None else "e" * 40)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    state = {"overlay": None, "lock": lock()}

    def download(name, revision="main"):
        path = tmp_path / f"{name}.lock"
        path.write_text(json.dumps(state["lock"] if name == "control" else {**lock(), "contracts": {}}))
        return path, "f" * 40

    monkeypatch.setattr(fleet, "resolve", lambda revision="main", repository=None: "f" * 40)
    monkeypatch.setattr(fleet, "download", download)
    monkeypatch.setattr(fleet, "download_overlay", lambda pr, revision: state["overlay"])
    return state


def test_publish_commits_to_a_pr_branch_and_writes_the_overlay(tmp_path, hub):
    root = tmp_path / "rec"
    write_record(root)
    api = FakeApi()
    args = SimpleNamespace(record=root, pr=7, by="maintainer", execute=True)
    assert pr_record.publish(args, api) == 0
    branch, commit, overlay_commit = api.calls
    assert branch == ("branch", "huggingface-mic/gpt-j-6b-mic", "transformers-pr-7", A)
    assert commit[:4] == ("commit", "huggingface-mic/gpt-j-6b-mic", "transformers-pr-7", A)
    assert [op.path_in_repo for op in commit[4]] == [
        "integrations/transformers/expectations/cpu-x86_64-bf16/gpt-j-6b-mic.bfloat16.json"
    ]
    assert (
        overlay_commit[1] == fleet.REPOSITORY and overlay_commit[4][0].path_in_repo == "overlays/transformers-pr-7.json"
    )
    written = json.loads(overlay_commit[4][0].path_or_fileobj)
    assert written["contracts"] == {"control/gpt-j-6b": {"huggingface-mic/gpt-j-6b-mic": {"from": A, "to": B}}}
    assert written["records"][0]["by"] == "maintainer"


def test_publish_refuses_a_record_measured_on_an_older_fleet(tmp_path, hub):
    hub["lock"] = lock(revision=D)  # main moved the contract after the record
    root = tmp_path / "rec"
    write_record(root)
    with pytest.raises(ValueError, match="fleet moved"):
        pr_record.publish(SimpleNamespace(record=root, pr=7, by="m", execute=True), FakeApi())


def test_merge_writes_the_overlay_into_the_tables_and_tags_the_commits(hub):
    hub["overlay"] = OVERLAY
    api = FakeApi()
    assert pr_record.merge(SimpleNamespace(pr=7, execute=True, since="5" * 40), api) == 0
    commit, tag = api.calls
    paths = [getattr(op, "path_in_repo") for op in commit[4]]
    assert "control/contracts.csv" in paths and paths[-1] == "overlays/transformers-pr-7.json"
    assert commit[3] == "f" * 40  # parent: the head the overlay was read at
    contracts = next(op for op in commit[4] if op.path_in_repo == "control/contracts.csv").path_or_fileobj.decode()
    assert B in contracts and A not in contracts
    assert tag == ("tag", "huggingface-mic/gpt-j-6b-mic", "transformers-pr-7", B)


def test_merge_reports_stale_entries(hub):
    hub["overlay"] = OVERLAY
    hub["lock"] = lock(revision=D)
    assert pr_record.merge(SimpleNamespace(pr=7, execute=False, since="5" * 40), FakeApi()) == 1


def test_blocked_cases_reads_only_blocked_decisions(tmp_path):
    report = tmp_path / "gate.json"
    report.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "fleet": "control",
                        "contract": "gpt-j-6b",
                        "variant": "bfloat16",
                        "target": "gpt-j-6b-mic",
                        "verdict": "blocked",
                    },
                    {
                        "fleet": "control",
                        "contract": "bert",
                        "variant": "default",
                        "target": "mic-bert",
                        "verdict": "warn",
                    },
                ]
            }
        )
    )
    assert pr_record.blocked_cases(report) == {("control", "gpt-j-6b", "bfloat16"): {("gpt-j-6b-mic", None)}}


def test_measure_needs_exactly_one_of_models_and_cases(tmp_path):
    for extra in ([], ["--models", "gptj", "--cases", str(tmp_path / "g.json")]):
        with pytest.raises(SystemExit):
            pr_record.main(["measure", "--pr", "7", "--out", str(tmp_path / "rec"), *extra])


def test_measure_cases_records_blocked_targets_without_a_check_pass(tmp_path, monkeypatch, hub):
    report = tmp_path / "gate.json"
    report.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "fleet": "control",
                        "contract": "gpt-j-6b",
                        "variant": "default",
                        "target": "gpt-j-6b-mic",
                        "verdict": "blocked",
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        fleet, "download_for_pr", lambda name, rev, pr: (*fleet.download(name, rev), {"applied": [], "stale": []})
    )
    import huggingface_hub

    snap = tmp_path / "snap"
    (snap / pr_record.EXPECTATIONS).mkdir(parents=True)
    manifest = tmp_path / "integration.json"
    manifest.write_text(json.dumps({"integrations": {"transformers": {"variants": {"default": {}, "bfloat16": {}}}}}))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(manifest))
    calls = []

    def fake_run(root, options, name):
        calls.append(name)
        out = options[options.index("--output") + 1]
        if "--record" in options:
            cand = Path(options[options.index("--expectations") + 1]) / "cpu-x86_64" / "gpt-j-6b-mic.candidate.json"
            cand.parent.mkdir(parents=True, exist_ok=True)
            cand.write_text(json.dumps({"recorded_with": {}, "outputs": {"logits": [1.0]}}))
            Path(out).write_text(
                json.dumps(
                    {"target": "gpt-j-6b-mic", "status": "recorded", "platform": "cpu-x86_64", "baseline": str(cand)}
                )
                + "\n"
            )
        else:
            Path(out).write_text(json.dumps({"target": "gpt-j-6b-mic", "status": "pass"}) + "\n")
        return 0

    monkeypatch.setattr(pr_record, "run", fake_run)
    out = tmp_path / "rec"
    code = pr_record.main(
        [
            "measure",
            "--pr",
            "7",
            "--cases",
            str(report),
            "--out",
            str(out),
            "--transformers-sha",
            "m" * 40,
            "--context",
            '{"runner": "aws-m8i-l-cache"}',
        ]
    )
    assert code == 0
    assert calls == ["control-gpt-j-6b-default-record", "control-gpt-j-6b-default-verify"]  # no check pass
    record = json.loads((out / pr_record.RECORD).read_text())
    assert record["context"] == {"runner": "aws-m8i-l-cache"} and record["transformers_sha"] == "m" * 40
    assert [(c["contract"], c["target"], c["verified"]) for c in record["cases"]] == [
        ("gpt-j-6b", "gpt-j-6b-mic", True)
    ]


@pytest.mark.parametrize("blocked_case", ["image", None, "renamed"])
def test_measure_records_the_blocked_input_cases_under_the_runners_names(tmp_path, monkeypatch, hub, blocked_case):
    """A contract with input cases text (default) and image. The gate names the blocked case, or (older gates)
    only the target, which then means every case of it. A case the contract does not declare (stale or renamed)
    returns no run: an error, not a silent pass."""
    decision = {"fleet": "control", "contract": "gpt-j-6b", "variant": "default", "target": "gpt-j-6b-mic"}
    report = tmp_path / "gate.json"
    case = {"case": blocked_case} if blocked_case else {}
    report.write_text(json.dumps({"decisions": [{**decision, **case, "verdict": "blocked"}]}))
    monkeypatch.setattr(
        fleet, "download_for_pr", lambda name, rev, pr: (*fleet.download(name, rev), {"applied": [], "stale": []})
    )
    import huggingface_hub

    snap = tmp_path / "snap"
    (snap / pr_record.EXPECTATIONS).mkdir(parents=True)
    manifest = tmp_path / "integration.json"
    integration = {"variants": {"default": {}}, "cases": {"text": {"default": True}, "image": {}}}
    manifest.write_text(json.dumps({"integrations": {"transformers": integration}}))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(manifest))
    runs = []

    def fake_run(root, options, name):
        cases = [options[i + 1] for i, o in enumerate(options) if o == "--case"] or ["text", "image"]
        cases = [case for case in cases if case in ("text", "image")]  # like run.py: undeclared cases never run
        runs.append((name, cases))
        out, rows = options[options.index("--output") + 1], []
        for case in cases:
            row = {"target": "gpt-j-6b-mic", "case": case, "platform": "cpu-x86_64"}
            if "--record" in options:
                # Named like run.py: the default case keeps the case-free name.
                name = "gpt-j-6b-mic" + ("" if case == "text" else "." + case) + ".candidate.json"
                cand = Path(options[options.index("--expectations") + 1]) / "cpu-x86_64" / name
                cand.parent.mkdir(parents=True, exist_ok=True)
                cand.write_text(json.dumps({"recorded_with": {"case": case}, "outputs": {"logits": [1.0]}}))
                rows.append({**row, "status": "recorded", "baseline": str(cand)})
            else:
                rows.append({**row, "status": "pass"})
        Path(out).write_text("".join(json.dumps(row) + "\n" for row in rows))
        return 0

    monkeypatch.setattr(pr_record, "run", fake_run)
    out = tmp_path / "rec"
    args = ["measure", "--pr", "7", "--cases", str(report), "--out", str(out), "--transformers-sha", "m" * 40]
    assert pr_record.main(args) == (1 if blocked_case == "renamed" else 0)
    record = json.loads((out / pr_record.RECORD).read_text())
    got = [(c.get("case"), c["path_in_repo"].split("/")[-1], c["verified"]) for c in record["cases"]]
    if blocked_case == "renamed":
        assert got == [] and [e["targets"] for e in record["errors"]] == [["gpt-j-6b-mic [renamed]"]]
        return
    if blocked_case:
        assert runs[0][1] == ["image"]  # only the blocked case is recorded
        assert got == [("image", "gpt-j-6b-mic.image.json", True)]
    else:
        assert runs[0][1] == ["text", "image"]
        assert got == [("text", "gpt-j-6b-mic.json", True), ("image", "gpt-j-6b-mic.image.json", True)]
    assert "gpt-j-6b-mic [image]" in pr_record.summary(record)


def test_publish_refuses_a_record_of_another_commit(tmp_path, hub):
    root = tmp_path / "rec"
    record = write_record(root)
    record["transformers_sha"] = "a" * 40
    (root / pr_record.RECORD).write_text(json.dumps(record))
    args = SimpleNamespace(record=root, pr=7, by="m", execute=True, models=None, merge_sha="b" * 40)
    with pytest.raises(ValueError, match="not the verified merge commit"):
        pr_record.publish(args, FakeApi())


def test_requested_keeps_only_the_named_contracts():
    record = {
        "cases": [{"fleet": "control", "contract": "gpt-j-6b"}, {"fleet": "control", "contract": "bert"}],
        "errors": [],
    }
    types = {"huggingface-mic/gpt-j-6b-mic": "gptj", "huggingface-mic/bert-mic": "bert"}
    got = pr_record.requested(record, ["gptj"], {"control": lock()}, types=lambda repo, rev: types[repo])
    assert [c["contract"] for c in got["cases"]] == ["gpt-j-6b"]
    with pytest.raises(ValueError, match="no recorded case"):
        pr_record.requested(record, ["llama"], {"control": lock()}, types=lambda repo, rev: types[repo])


def test_blocked_cases_reads_an_older_gate_report(tmp_path):
    report = tmp_path / "gate.json"
    decision = {"fleet": "control", "contract": "c", "variant": "v", "target": "t", "verdict": "block"}
    report.write_text(json.dumps({"decisions": [decision]}))
    assert pr_record.blocked_cases(report) == {("control", "c", "v"): {("t", None)}}
