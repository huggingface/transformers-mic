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


def test_a_pin_moved_without_a_row_is_left_alone():
    got, report = fleet.apply_history(lock("d" * 40), [row(A, B, "pr1")], lambda sha: False)
    assert pinned(got) == "d" * 40 and report["unrecorded"] == [f"gpt-j-6b:{REPO}"]


def test_the_newest_applicable_row_wins_whatever_the_row_order():
    # pr2's row was written first, though its commit is newer on main: ancestry decides, not order.
    rows = [row(A, B, "pr1"), row(B, C, "pr2")]
    contains = {"pr2"}  # only answers for pr2: pr1 is not even asked once a newer row applies
    got, _ = fleet.apply_history(lock(C), rows, lambda sha: sha in contains)
    assert pinned(got) == C


def test_an_always_row_applies_to_any_code():
    rows = [row(A, B, "pr1"), row(B, C, fleet.ALWAYS)]  # re-serialized weights after a record
    asked = []
    got, report = fleet.apply_history(lock(C), rows, lambda sha: asked.append(sha) or False)
    assert pinned(got) == C and report["reverted"] == [] and asked == []


def test_a_break_in_the_chain_stops_the_walk():
    # B -> D moved with no row: rows before it describe another chain and are not walked past.
    D = "d" * 40
    rows = [row(A, B, "pr1"), row(D, C, "pr2")]
    got, _ = fleet.apply_history(lock(C), rows, lambda sha: False)
    assert pinned(got) == D


def test_rows_for_a_moved_pin_and_none_for_a_new_contract():
    old, new = lock(A), lock(B)
    new["contracts"]["fresh"] = lock(C)["contracts"]["gpt-j-6b"]
    assert fleet.history_rows(old, new, "s" * 40) == [row(A, B, "s" * 40)]
    assert fleet.history_rows(old, lock(A), fleet.ALWAYS) == []


def test_since_must_be_a_full_sha_or_always():
    assert fleet.check_since(fleet.ALWAYS) == fleet.ALWAYS and fleet.check_since("a" * 40) == "a" * 40
    for bad in ("abc123", "main", None):
        try:
            fleet.check_since(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_merged_overlays_apply_only_to_code_that_contains_the_merge(monkeypatch):
    overlay = {"contracts": {"control/gpt-j-6b": {REPO: {"from": A, "to": B}}}}
    monkeypatch.setattr(fleet, "download_overlay", lambda pr, rev: overlay)
    merged = {7: "m" * 40}.get  # PR 7 merged, PR 8 open
    got, report = fleet.apply_merged_overlays(lock(A), "control", "f" * 40, lambda s: True, merged=merged, prs=[7, 8])
    assert pinned(got) == B and report["merged"] == [f"gpt-j-6b:{REPO}@transformers-pr-7"]
    got, report = fleet.apply_merged_overlays(lock(A), "control", "f" * 40, lambda s: False, merged=merged, prs=[7])
    assert pinned(got) == A and report["merged"] == []
    got, report = fleet.apply_merged_overlays(
        lock(A), "control", "f" * 40, lambda s: True, skip=7, merged=merged, prs=[7]
    )
    assert pinned(got) == A  # the run's own PR goes through --fleet-pr instead


def test_github_merged_reads_the_pull(monkeypatch):
    answers = {7: {"merged": True, "merge_commit_sha": "m" * 40}, 8: {"merged": False, "merge_commit_sha": "x" * 40}}

    def urlopen(request, timeout):
        pr = int(request.full_url.rsplit("/", 1)[1])
        if pr == 9:
            raise OSError("rate limited")
        return io.BytesIO(json.dumps(answers[pr]).encode())

    monkeypatch.setattr(fleet.urllib.request, "urlopen", urlopen)
    assert [fleet.github_merged(pr) for pr in (7, 8, 9)] == ["m" * 40, None, None]


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
    monkeypatch.setattr(fleet, "list_overlays", lambda rev: [7])
    path, _, report = fleet.download_for_run(
        "control", "main", pr=7, code_revision=A, is_ancestor=lambda s: False, merged=lambda pr: None
    )
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
    assert pr_record.merge(SimpleNamespace(pr=7, execute=True, since="5" * 40), api) == 0
    ops = {op.path_in_repo: op for op in api.commits[0]["operations"]}
    history = ops["control/history.csv"].path_or_fileobj.decode().splitlines()
    assert history == [",".join(fleet.HISTORY_COLUMNS), f"gpt-j-6b,{REPO},{A},{B},{'5' * 40}"]
    assert "prototypes/history.csv" not in ops  # nothing applied there
