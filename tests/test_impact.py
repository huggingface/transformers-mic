"""The daily MIC impact observer: classification, outcomes and the blocking replay."""

import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "experiments" / "impact.py"
spec = importlib.util.spec_from_file_location("mic_impact", SOURCE)
impact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impact)

DAY = 20_000  # a UTC day number
T = DAY * 86400


def row(pr="1", contract="huggingface-mic/a-mic", target="a-mic", kind="Baseline assertion", latest="fail", fail_days=(0,), **extra):
    daily = [{"pass": 0, "fail": 0, "skipped": 0} for _ in range(7)]
    for d in fail_days:
        daily[d]["fail"] = 1
    return {
        "pr": pr, "contract": contract, "target": target, "variant": "default", "platform": "cpu-x86_64",
        "failure_kind": kind, "failure": "logits: max delta", "weights": "random", "daily": daily,
        "latest": {"status": latest, "started_at": T + 3600}, "started_at": T + 60, "commit_sha": "abc",
        **extra,
    }


def test_classification_separates_infra_main_and_pr_only():
    main = [row(pr="", contract="huggingface-mic/b-mic", target="b-mic", fail_days=(6,))]
    keys = impact.failing_on_main(main, DAY - 6, DAY)
    assert impact.classify(row(kind="Setup error"), keys) == "infra"
    assert impact.classify(row(contract="huggingface-mic/b-mic", target="b-mic"), keys) == "pre-existing"
    assert impact.classify(row(), keys) == "pr-only"
    # Main failing two days earlier no longer explains a PR failure.
    assert impact.failing_on_main(main, DAY - 6, DAY + 2) == set()


def candidate():
    return {"key": "1#huggingface-mic/a-mic|a-mic|default|cpu-x86_64", "first_fail_at": T + 60, "commits": []}


def info(state="open", merged_at=None, commits=()):
    return {"state": state, "merged_at": merged_at, "commit_times": list(commits), "head": "h"}


def iso(seconds):
    import datetime as dt

    return dt.datetime.fromtimestamp(seconds, tz=dt.UTC).isoformat().replace("+00:00", "Z")


@pytest.mark.parametrize(
    "pushed, expected", [(True, "fixed-by-push"), (False, "recovered-no-push")]
)
def test_recovery_is_split_by_whether_the_author_pushed(pushed, expected):
    commits = [iso(T + 1800)] if pushed else [iso(T)]
    c = impact.update_candidate(candidate(), row(latest="pass"), info(commits=commits), set(), DAY)
    assert c["outcome"] == expected


def test_merged_case_is_watched_on_main():
    merged = info(state="merged", merged_at=iso(T + 7200))
    key = candidate()["key"].split("#", 1)[1]
    c = impact.update_candidate(candidate(), row(), merged, set(), DAY + 1)
    assert c["outcome"] == "merged-watching"
    assert impact.update_candidate(candidate(), row(), merged, {key}, DAY + 1)["outcome"] == "merged-then-main-failed"
    late = DAY + impact.MAIN_WATCH_DAYS
    assert impact.update_candidate(candidate(), None, merged, set(), late)["outcome"] == "merged-main-ok"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def results_dir(tmp_path):
    """Three merged PRs: one blocked rightly, one by a main failure, one by a PR-only flake."""
    base = {"variant": "default", "platform": "cpu-x86_64", "weights": "random", "kind": "fixture",
            "failure_kind": "Baseline assertion", "failure": "x", "latest": "fail"}
    rows = [
        {**base, "pr": "10", "contract": "c/real-mic", "target": "real-mic", "class": "pr-only"},
        {**base, "pr": "11", "contract": "c/main-mic", "target": "main-mic", "class": "pre-existing"},
        {**base, "pr": "12", "contract": "c/flaky-mic", "target": "flaky-mic", "class": "pr-only"},
        {**base, "pr": "13", "contract": "c/open-mic", "target": "open-mic", "class": "pr-only"},
    ]
    write(tmp_path / "days" / "2026-10-01.json", {"rows": rows})
    key = lambda r: f"{r['pr']}#{r['contract']}|{r['target']}|default|cpu-x86_64"
    candidates = {
        key(rows[0]): {"key": key(rows[0]), "pr": "10", "outcome": "merged-then-main-failed", "weights": "random", "kind": "fixture", "first_seen": "2026-10-01", "failure_kind": "Baseline assertion", "failure": "x"},
        key(rows[2]): {"key": key(rows[2]), "pr": "12", "outcome": "merged-main-ok", "weights": "overfit", "kind": "fixture", "first_seen": "2026-10-01", "failure_kind": "Baseline assertion", "failure": "x"},
        key(rows[3]): {"key": key(rows[3]), "pr": "13", "outcome": "open", "weights": "random", "kind": "fixture", "first_seen": "2026-10-01", "failure_kind": "Baseline assertion", "failure": "x"},
    }
    write(tmp_path / "candidates.json", candidates)
    write(tmp_path / "prs.json", {p: {"state": "merged"} for p in ("10", "11", "12")} | {"13": {"state": "open"}})
    write(tmp_path / "merged.json", {"2026-10-01": 200})
    return tmp_path


def test_blocking_replay_counts_right_and_wrong_blocks(tmp_path):
    impact.summarize(results_dir(tmp_path))
    summary = (tmp_path / "summary.md").read_text()
    # A blocks all three merged PRs; B ignores the main failure.
    assert "| A: any red case | 3 | 1 | 2 | 0 |" in summary
    assert "| B: red, not infra, not failing on main | 2 | 1 | 1 | 0 |" in summary
    assert "wrong blocks 1 = 0.50% of 200 merged PRs" in summary
    assert "**Not yet**" in summary  # one day observed, not thirty


def test_a_label_overrides_the_observed_outcome(tmp_path):
    results = results_dir(tmp_path)
    flaky = "12#c/flaky-mic|flaky-mic|default|cpu-x86_64"
    (results / "labels.jsonl").write_text(json.dumps({"key": flaky, "verdict": "real-bug", "by": "t"}) + "\n")
    impact.summarize(results)
    summary = (results / "summary.md").read_text()
    # The maintainer's verdict beats the observed "merged, main stayed green".
    assert "| B: red, not infra, not failing on main | 2 | 2 | 0 | 0 |" in summary
    assert "Labelled `real-bug`: **1**" in summary


def test_a_case_failing_on_several_prs_the_same_day_is_systemic():
    rows = [row(pr=str(n), fail_days=(6,)) for n in range(impact.SYSTEMIC_PRS)] + [
        row(pr="9", contract="huggingface-mic/b-mic", target="b-mic", fail_days=(6,))
    ]
    shared = impact.systemic_cases(rows, DAY - 6)
    today = {case for day, case in shared if day == DAY}
    assert impact.classify(rows[0], set(), today) == "systemic"
    assert impact.classify(rows[-1], set(), today) == "pr-only"


@pytest.mark.parametrize(
    "target, kind",
    [("granite-3.1-2b-instruct-mic", "fixture"), ("mic-mistral3-slim", "fixture"), ("granite-3.1-2b-instruct", "checkpoint")],
)
def test_fixture_names_old_and_new(target, kind):
    assert impact.target_kind({"target": target}) == kind


def test_rule_b_counts_a_systemic_block_as_wrong(tmp_path):
    results = results_dir(tmp_path)
    day = json.loads((results / "days" / "2026-10-01.json").read_text())
    day["rows"][3]["pr"] = "14"
    day["rows"][3]["class"] = "systemic"
    write(results / "days" / "2026-10-01.json", day)
    prs = json.loads((results / "prs.json").read_text()) | {"14": {"state": "merged"}}
    write(results / "prs.json", prs)
    impact.summarize(results)
    assert "| B: red, not infra, not failing on main | 3 | 1 | 2 | 0 |" in (results / "summary.md").read_text()


def nightly(contract, target, kind, status):
    return {"contract": contract, "target": target, "kind": kind, "status": status, "variant": "bfloat16", "platform": "cuda", "failures": ["x"] if status == "fail" else []}


def test_real_weights_only_needs_a_passing_night_and_passing_fixtures():
    runs = {
        "2026-10-01": [nightly("c/a", "a", "checkpoint", "pass"), nightly("c/a", "a-mic", "fixture", "pass"),
                       nightly("c/b", "b", "checkpoint", "fail"), nightly("c/b", "b-mic", "fixture", "pass"),
                       nightly("c/d", "d", "checkpoint", "pass"), nightly("c/d", "d-mic", "fixture", "pass")],
        "2026-10-02": [nightly("c/a", "a", "checkpoint", "fail"), nightly("c/a", "a-mic", "fixture", "pass"),
                       nightly("c/b", "b", "checkpoint", "fail"), nightly("c/b", "b-mic", "fixture", "pass"),
                       nightly("c/d", "d", "checkpoint", "fail"), nightly("c/d", "d-mic", "fixture", "fail")],
    }
    found = impact.real_weights_only(runs)
    assert found["c/a|a|bfloat16|cuda"]["after_pass"] is True  # only real weights saw it
    assert found["c/b|b|bfloat16|cuda"]["since_first_night"] is True  # standing failure, not a catch
    assert "c/d|d|bfloat16|cuda" not in found  # the fixture failed too
