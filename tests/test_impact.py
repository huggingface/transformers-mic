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


def test_gpu_ci_regressions_group_by_pr_and_model_and_skip_flaky(monkeypatch):
    def entry(test, pr=50001, status="git bisect found the bad commit."):
        return {"test": test, "pr_number": pr, "status": status, "bad_commit": "b", "job_link": "j"}

    data = {
        "llama": {"single-gpu": [
            entry("tests/models/llama/test_modeling_llama.py::LlamaIntegrationTest::test_a"),
            entry("tests/models/llama/test_modeling_llama.py::LlamaModelTest::test_b"),
            entry("tests/models/llama/test_modeling_llama.py::LlamaModelTest::test_c", status="flaky: passed on retry"),
            entry("tests/models/llama/test_modeling_llama.py::LlamaModelTest::test_d", pr=None),
        ]},
        "bert": {"multi-gpu": [entry("tests/models/bert/test_modeling_bert.py::BertModelTest::test_e", pr=50002)]},
    }
    monkeypatch.setattr(impact, "get_json", lambda url: data)
    found = impact.gpu_ci_regressions("2026-11-02")
    assert sorted(found) == ["50001#llama", "50002#bert"]
    assert found["50001#llama"]["integration"] is True and len(found["50001#llama"]["tests"]) == 2
    assert found["50002#bert"]["integration"] is False


@pytest.mark.parametrize(
    "files, selected",
    [
        (["src/transformers/models/llama/modeling_llama.py"], True),
        (["src/transformers/cache_utils.py"], True),
        (["src/transformers/models/auto/modeling_auto.py"], True),
        (["src/transformers/models/bert/modeling_bert.py"], False),
        (["tests/models/llama/test_modeling_llama.py", "docs/source/en/x.md"], False),
    ],
)
def test_selection_follows_select_contracts_rules(files, selected):
    assert impact.selects(files, "llama") is selected


@pytest.mark.parametrize(
    "rows, job, files, expected",
    [
        ([{"pr": "7", "failure_kind": "Baseline assertion", "fixture_repo": "m/llama-mic", "failure": "x"}], "failure", [], "caught-on-pr"),
        ([{"pr": "7", "failure_kind": "Setup error", "fixture_repo": "m/llama-mic"}], None, [], "not-run"),
        ([], "success", ["src/transformers/models/bert/x.py"], "not-selected"),
        ([], "success", ["src/transformers/models/llama/x.py"], "missed"),
    ],
)
def test_mic_on_culprit(monkeypatch, rows, job, files, expected):
    monkeypatch.setattr(impact, "mic_job", lambda sha: job)
    item = {"pr": "7", "model": "llama"}
    types = {"m/llama-mic": "llama"}
    assert impact.mic_on_culprit(item, rows, types, {"llama"}, files, "h") == expected
    assert impact.mic_on_culprit({"pr": "7", "model": "gpt2"}, rows, types, {"llama"}, files, "h") == "not-covered"


def test_sticky_onsets_need_a_passing_day_before():
    health = {"rows": [
        {"job": "run_models_gpu", "category": "failure", "sticky": True, "model": "llama", "nodeid": "t/LlamaIntegrationTest::a", "days": [0, 0, 3, 1]},
        {"job": "run_models_gpu", "category": "failure", "sticky": True, "model": "bert", "nodeid": "t/BertModelTest::b", "days": [1, 1, 1, 1]},
        {"job": "tests_torch", "category": "failure", "sticky": True, "model": "gpt2", "nodeid": "t/x", "days": [0, 1, 1, 1]},
    ]}
    found = impact.sticky_gpu_onsets(health, DAY)
    assert list(found) == [f"sticky#llama#{impact.iso_day(DAY + 2)}"]
    item = found[f"sticky#llama#{impact.iso_day(DAY + 2)}"]
    assert item["integration"] is True
    main = [{"failure_kind": "Baseline assertion", "fixture_repo": "m/llama-mic", "daily": [{"fail": 0}, {"fail": 0}, {"fail": 0}, {"fail": 1}]}]
    assert impact.mic_on_main_near(item, main, DAY, {"m/llama-mic": "llama"}, {"llama"}) == "red-on-main"
    assert impact.mic_on_main_near(item, [], DAY, {}, {"llama"}) == "green-on-main"
    assert impact.mic_on_main_near({**item, "model": "bert"}, main, DAY, {}, {"llama"}) == "not-covered"


TEST_FILE = '''
class EomtModelTest(unittest.TestCase):
    def test_tiny(self):
        AutoModel.from_pretrained("hf-internal-testing/tiny-eomt")

class EomtIntegrationTest(unittest.TestCase):
    model_id = "tue-mps/eomt-large"
    def test_a(self):
        Model.from_pretrained("tue-mps/eomt-large")
        load_image("org/images/cat.png")
        Model.from_pretrained("tue-mps/eomt-small")
'''


def test_checkpoints_come_from_the_failing_class():
    files = {"tests/models/eomt/test_modeling_eomt.py": TEST_FILE}
    found = impact.test_checkpoints(["tests/models/eomt/test_modeling_eomt.py::EomtIntegrationTest::test_a"], files)
    assert found == ["tue-mps/eomt-large", "tue-mps/eomt-small"]


def test_suggestions_rank_shrinkable_and_real_checkpoint_models_first():
    files = {"tests/models/eomt/test_modeling_eomt.py": TEST_FILE}
    gpu_ci = {
        "1#eomt": {"model": "eomt", "mic": "not-covered", "integration": True, "date": "d",
                   "tests": ["tests/models/eomt/test_modeling_eomt.py::EomtIntegrationTest::test_a"]},
        "2#nope": {"model": "nope", "mic": "not-covered", "integration": True, "date": "d",
                   "tests": ["tests/models/nope/test_modeling_nope.py::NopeIntegrationTest::test_b"]},
        "3#auto": {"model": "auto", "mic": "not-covered", "integration": False, "date": "d", "tests": ["t"]},
        "sticky#llama#d": {"model": "llama", "mic": "green-on-main", "integration": True, "date": "d",
                           "tests": ["tests/models/llama/test_modeling_llama.py::LlamaIntegrationTest::test_flash_attn_x",
                                     "tests/models/llama/test_modeling_llama.py::LlamaIntegrationTest::test_export"]},
    }
    files["tests/models/nope/test_modeling_nope.py"] = ""
    sweep = {"eomt": {"stage": "ok", "finite": True, "repo": "sweep/eomt"}, "nope": {"stage": "error", "repo": "sweep/nope"}}
    s = impact.build_suggestions(gpu_ci, sweep, files)
    assert [x["model"] for x in s["new"]] == ["eomt", "nope"]  # shrinkable first; auto is not a model
    assert s["new"][0]["checkpoint"] == "tue-mps/eomt-large" and s["new"][0]["checkpoint_source"] == "failing test"
    assert s["new"][1]["checkpoint"] == "sweep/nope" and s["new"][1]["shrinkable"] is False
    assert [t["gpu_only"] for t in s["extend"]["llama"]] == [False, True]


def test_drafts_are_capped_skip_drafted_and_unshrinkable(monkeypatch):
    sent = []
    monkeypatch.setattr(impact, "github_post", lambda path, body: sent.append(body["inputs"]["model"]) or 204)
    suggestions = {"new": [
        {"model": "a", "checkpoint": "o/a", "shrinkable": True},
        {"model": "b", "checkpoint": "o/b", "shrinkable": False},
        {"model": "c", "checkpoint": "o/c", "shrinkable": True},
        {"model": "d", "checkpoint": "o/d", "shrinkable": True},
        {"model": "e", "checkpoint": "o/e", "shrinkable": True},
    ]}
    drafts = {"c": {"checkpoint": "o/c", "conclusion": "success"}}
    assert impact.dispatch_drafts(suggestions, drafts, 2, "2026-11-03T10:00:00Z") == 2
    assert sent == ["o/a", "o/d"]


def test_drafts_find_their_run_by_name(monkeypatch):
    runs = {"workflow_runs": [
        {"id": 9, "display_title": "tmic run o/a", "created_at": "2026-11-03T10:00:05Z", "status": "completed", "conclusion": "failure", "html_url": "u9"},
        {"id": 8, "display_title": "tmic run o/a", "created_at": "2026-11-01T10:00:05Z", "status": "completed", "conclusion": "success", "html_url": "u8"},
    ]}
    monkeypatch.setattr(impact, "github", lambda path: runs)
    drafts = {"a": {"checkpoint": "o/a", "dispatched_at": "2026-11-03T10:00:00Z", "conclusion": None}}
    impact.refresh_drafts(drafts)
    assert drafts["a"]["run_id"] == 9 and drafts["a"]["conclusion"] == "failure"
    lines = "\n".join(impact.draft_lines({"new": [{"model": "a", "integration": 1, "other": 0, "checkpoint": "o/a", "shrinkable": True}], "extend": {}}, drafts))
    assert "[failure](u9)" in lines


def test_a_test_failing_on_many_models_is_one_shared_suggestion():
    gpu_ci = {
        f"sticky#{m}#d": {"model": m, "mic": "green-on-main", "integration": False, "date": "d",
                          "tests": [f"tests/models/{m}/test_modeling_{m}.py::{m}ModelTest::test_torch_export_generate_static"]}
        for m in ("bart", "t5", "marian")
    }
    gpu_ci["sticky#gemma2#d"] = {"model": "gemma2", "mic": "green-on-main", "integration": True, "date": "d",
                                 "tests": ["tests/models/gemma2/test_modeling_gemma2.py::Gemma2IntegrationTest::test_export_static_cache"]}
    s = impact.build_suggestions(gpu_ci, {}, {})
    assert s["shared"] == {"test_torch_export_generate_static": ["bart", "marian", "t5"]}
    assert list(s["extend"]) == ["gemma2"]


def test_overview_has_the_headline_numbers_and_next_targets(tmp_path):
    results = results_dir(tmp_path)
    write(results / "gpu-ci.json", {
        "1#a": {"mic": "caught-on-pr", "integration": True},
        "2#b": {"mic": "missed", "integration": True, "date": "d", "tests": ["t"]},
        "sticky#c#d": {"mic": "not-covered", "integration": True},
        "sticky#e#d": {"mic": "green-on-main", "integration": False},
    })
    write(results / "suggestions.json", {"new": [
        {"model": f"m{i}", "checkpoint": f"o/m{i}", "integration": 1, "other": 2, "shrinkable": True} for i in range(7)
    ], "extend": {}})
    write(results / "drafts.json", {"m0": {"checkpoint": "o/m0", "status": "completed", "conclusion": "success",
                                           "url": "https://github.com/x/runs/9", "shrink": "shrink/9/shrink-m0.html"}})
    impact.summarize(results)
    o = json.loads((results / "overview.json").read_text())
    assert o["window"] == ["2026-10-01", "2026-10-01"]
    assert o["caught"]["real_checkpoint"] == 1 and o["missed"] == {"real_checkpoint": 1, "not_covered": 1, "awaiting_verdict": 1}
    assert o["false_alarms"] == {"prs_with_failure": 4, "prs_not_their_fault": 1}
    assert o["block_merges"]["ready"] is False and any("days observed" in u for u in o["block_merges"]["unmet"])
    assert len(o["next"]) == 7  # every suggestion, up to NEXT_TARGETS
    assert o["next"][0] == {"model": "m0", "checkpoint": "o/m0", "regressions": 3, "real_checkpoint": 1, "other": 2,
                            "shrinkable": True, "next_batch": False, "draft": "success",
                            "run": "https://github.com/x/runs/9", "shrink": "shrink/9/shrink-m0.html"}
    assert o["next"][1]["draft"] == "" and o["next"][1]["shrink"] == ""
    # The next daily run drafts the top DRAFTS_PER_DAY suggestions without a draft.
    assert [t["model"] for t in o["next"] if t["next_batch"]] == ["m1", "m2"]


def test_shrink_pages_are_fetched_once_per_finished_draft(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[3])
        out = Path(cmd[cmd.index("-D") + 1])
        if cmd[3] == "9":
            (out / "shrink-a.html").write_text("<html>")
        return type("Done", (), {"returncode": 0 if cmd[3] == "9" else 1})()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    drafts = {
        "a": {"run_id": 9, "status": "completed"},
        "b": {"run_id": 8, "status": "completed"},  # artifact expired or never uploaded
        "c": {"run_id": 7, "status": "in_progress"},
        "d": {"run_id": 6, "status": "completed", "shrink": "shrink/6/x.html"},
    }
    assert impact.fetch_shrink_pages(drafts, tmp_path) == [9]
    assert calls == ["9", "8"]
    assert drafts["a"]["shrink"] == "shrink/9/shrink-a.html" and (tmp_path / "shrink/9/shrink-a.html").exists()
    assert "shrink" not in drafts["b"]


def test_space_gets_overview_report_and_new_shrink_pages(monkeypatch, tmp_path):
    results = results_dir(tmp_path / "r")
    impact.summarize(results)
    shrink = tmp_path / "s"
    (shrink / "shrink" / "9").mkdir(parents=True)
    (shrink / "shrink" / "9" / "p.html").write_text("<html>")
    uploaded = {}

    class Api:
        def upload_folder(self, folder_path, **kwargs):
            uploaded["files"] = sorted(str(p.relative_to(folder_path)) for p in Path(folder_path).rglob("*") if p.is_file())

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    impact.publish_space(results, "o/space", shrink)
    assert uploaded["files"] == ["README.md", "index.html", "overview.json", "report.html", "shrink/9/p.html", "summary.md", "updated.txt"]


def test_a_model_that_got_a_contract_is_no_longer_suggested():
    gpu_ci = {"1#flint": {"model": "flint", "mic": "not-covered", "integration": True, "date": "d",
                          "tests": ["tests/models/flint/test_modeling_flint.py::FlintIntegrationTest::test_x"]}}
    assert [s["model"] for s in impact.build_suggestions(gpu_ci, {}, {})["new"]] == ["flint"]
    assert impact.build_suggestions(gpu_ci, {}, {}, {"flint"})["new"] == []


def test_a_refresh_drops_suggestions_the_fleet_now_covers(tmp_path, monkeypatch):
    results = results_dir(tmp_path)
    write(results / "suggestions.json", {"new": [{"model": "flex_olmo"}, {"model": "vaultgemma"}], "extend": {"x": []}})
    monkeypatch.setattr(impact, "fixture_model_types", lambda results, download: {"o/vaultgemma-1b-mic": "vaultgemma"})
    assert impact.drop_covered(results, None) == 1
    s = json.loads((results / "suggestions.json").read_text())
    assert s == {"new": [{"model": "flex_olmo"}], "extend": {"x": []}}


def test_a_checkpoint_the_hub_does_not_serve_is_skipped():
    text = 'class MoshiTest:\n    a = "kmhf/hf-moshiko"\n    b = "kmhf/hf-moshiko"\n    c = "kmhf/hf-moshika"\n'
    files = {"tests/models/moshi/test_modeling_moshi.py": text}
    gpu_ci = {"1#moshi": {"model": "moshi", "mic": "not-covered", "integration": False, "date": "d",
                          "tests": ["tests/models/moshi/test_modeling_moshi.py::MoshiTest::test_a"]}}
    sweep = {"moshi": {"stage": "ok", "finite": True, "repo": "gone/sweep"}}
    assert impact.build_suggestions(gpu_ci, sweep, files)["new"][0]["checkpoint"] == "kmhf/hf-moshiko"
    live = {"kmhf/hf-moshika"}
    [s] = impact.build_suggestions(gpu_ci, sweep, files, exists=live.__contains__)["new"]
    assert s["checkpoint"] == "kmhf/hf-moshika" and s["checkpoint_source"] == "failing test"
    [s] = impact.build_suggestions(gpu_ci, sweep, files, exists=lambda repo: False)["new"]
    assert s["checkpoint"] is None and s["checkpoint_source"] is None


def test_a_failed_draft_is_retried_only_on_a_new_checkpoint():
    suggestions = {"new": [{"model": "moshi", "checkpoint": "kmhf/hf-moshika", "shrinkable": True, "integration": 0, "other": 1},
                           {"model": "switch", "checkpoint": "google/switch-base-8", "shrinkable": True, "integration": 0, "other": 1},
                           {"model": "olmo", "checkpoint": "o/olmo", "shrinkable": True, "integration": 0, "other": 1}]}
    drafts = {"moshi": {"checkpoint": "kmhf/hf-moshiko", "conclusion": "failure"},
              "switch": {"checkpoint": "google/switch-base-8", "conclusion": "failure"},
              "olmo": {"checkpoint": "o/older", "conclusion": None, "status": "in_progress"}}
    assert [s["model"] for s in impact.draftable(suggestions, drafts)] == ["moshi"]
    assert impact.live_draft(suggestions["new"][0], drafts) is None
    lines = "\n".join(impact.draft_lines({**suggestions, "extend": {}}, drafts))
    assert "failure" in lines.split("| switch |")[1].split("\n")[0]
    assert "failure" not in lines.split("| moshi |")[1].split("\n")[0]
