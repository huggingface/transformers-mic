"""Fleet history: a run meets the values its Transformers code should meet (record-mic)."""

import io
import json
from types import SimpleNamespace

from transformers_mic import fleet, pr_record

A, B, C = ("a" * 40, "b" * 40, "c" * 40)
REPO = "huggingface-mic/gpt-j-6b-mic"


def lock(revision):
    return {
        "framework": "transformers",
        "environment": {},
        "contracts": {
            "gpt-j-6b": {
                "integration": {"repo": REPO, "revision": revision, "files_sha256": "x"},
                "fixtures": {REPO: {"revision": revision, "files_sha256": "y"}},
                "checkpoints": {},
            }
        },
    }


def row(frm, to, since):
    return {"contract": "gpt-j-6b", "mic_repo": REPO, "from_revision": frm, "revision": to, "since_transformers": since}


def pinned(lock_):
    c = lock_["contracts"]["gpt-j-6b"]
    assert c["integration"]["revision"] == c["fixtures"][REPO]["revision"]
    return c["integration"]["revision"]


def test_older_code_meets_the_values_before_each_record_it_lacks():
    rows = [row(A, B, "pr1"), row(B, C, "pr2")]
    contains = {"pr1"}  # code with the first record's commit but not the second's
    got, report = fleet.apply_history(lock(C), rows, lambda sha: sha in contains)
    assert pinned(got) == B and report["reverted"] == [f"gpt-j-6b:{REPO}"]
    got, _ = fleet.apply_history(lock(C), rows, lambda sha: False)  # older than both
    assert pinned(got) == A
    got, report = fleet.apply_history(lock(C), rows, lambda sha: True)  # current code
    assert pinned(got) == C and report["reverted"] == []


def test_a_pin_main_moved_another_way_is_left_alone():
    got, report = fleet.apply_history(lock("d" * 40), [row(A, B, "pr1")], lambda sha: False)
    assert pinned(got) == "d" * 40 and report["reverted"] == []


def test_history_keeps_the_newest_rows_per_contract():
    rows = [row(str(i), str(i + 1), f"pr{i}") for i in range(fleet.HISTORY_KEEP + 2)]
    kept = fleet.with_history(rows[:-1], rows[-1:])
    assert len(kept) == fleet.HISTORY_KEEP and kept[-1] == rows[-1] and kept[0] == rows[2]


def test_github_ancestor_reads_compare_and_falls_back_to_latest(monkeypatch):
    answers = {"pr1": "ahead", "pr2": "behind", "pr3": "identical"}

    def urlopen(request, timeout):
        sha = request.full_url.split("/compare/")[1].split("...")[0]
        if sha == "gone":
            raise OSError("404")
        return io.BytesIO(json.dumps({"status": answers[sha]}).encode())

    monkeypatch.setattr(fleet.urllib.request, "urlopen", urlopen)
    is_ancestor = fleet.github_ancestor(C)
    assert [is_ancestor(s) for s in ("pr1", "pr2", "pr3", "gone")] == [True, False, True, True]


def test_download_for_run_matches_history_then_applies_the_overlay(monkeypatch, tmp_path):
    def download(name, revision="main"):
        path = tmp_path / f"{name}.lock"
        path.write_text(json.dumps(lock(C)))
        return path, "f" * 40

    monkeypatch.setattr(fleet, "download", download)
    monkeypatch.setattr(fleet, "download_history", lambda name, rev: [row(B, C, "pr2")])
    overlay = {"contracts": {"control/gpt-j-6b": {REPO: {"from": B, "to": "e" * 40}}}}
    monkeypatch.setattr(fleet, "download_overlay", lambda pr, rev: overlay)
    path, _, report = fleet.download_for_run("control", "main", pr=7, code_revision=A, is_ancestor=lambda s: False)
    assert pinned(json.loads(path.read_text())) == "e" * 40  # back to B for old code, then the PR's overlay on B
    assert report["reverted"] == [f"gpt-j-6b:{REPO}"] and report["applied"] == [f"gpt-j-6b:{REPO}"]
    plain, _, report = fleet.download_for_run("control", "main")
    assert plain == tmp_path / "control.lock" and report == {}


class Api:
    def __init__(self):
        self.commits = []

    def create_commit(self, **kwargs):
        self.commits.append(kwargs)
        return SimpleNamespace(oid="9" * 40)

    def create_tag(self, *a, **k):
        pass


def test_merge_since_records_history_in_the_same_commit(monkeypatch, tmp_path):
    def download(name, revision="main"):
        path = tmp_path / f"{name}.lock"
        path.write_text(json.dumps(lock(A) if name == "control" else {**lock(A), "contracts": {}}))
        return path, "f" * 40

    monkeypatch.setattr(fleet, "resolve", lambda revision="main", repository=None: "f" * 40)
    monkeypatch.setattr(fleet, "download", download)
    monkeypatch.setattr(fleet, "download_history", lambda name, rev: [])
    monkeypatch.setattr(
        fleet, "download_overlay", lambda pr, rev: {"contracts": {"control/gpt-j-6b": {REPO: {"from": A, "to": B}}}}
    )
    api = Api()
    assert pr_record.merge(SimpleNamespace(pr=7, execute=True, since="m" * 40), api) == 0
    ops = {op.path_in_repo: op for op in api.commits[0]["operations"]}
    history = ops["control/history.csv"].path_or_fileobj.decode().splitlines()
    assert history == [",".join(fleet.HISTORY_COLUMNS), f"gpt-j-6b,{REPO},{A},{B},{'m' * 40}"]
    assert "prototypes/history.csv" not in ops  # nothing applied there
