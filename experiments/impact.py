"""Daily MIC impact observer: what the contracts job does on Transformers PRs.

Whether MIC should block a PR depends on things replays cannot measure:

1. How many real bugs does it catch on PRs?
2. How often does it send people chasing false alarms?
3. How much work do baseline reviews take?
4. Of the real-checkpoint regressions Transformers' daily GPU CI finds, how many did MIC flag first?
5. Do overfit fixtures catch bugs that random ones miss?

Every day this reads the dashboard's MIC failures (the data behind
/d/mic-failures), classifies each PR failure, follows the PR-only ones until
they resolve, counts the baseline changes published to the fleet, and rewrites
summary.md. It never judges a failure "real": it says what happened (fixed by a
push, merged and then failed on main, recovered without a push, ...) and lists
the cases still waiting for a human verdict in labels.jsonl.

    python experiments/impact.py --results <dir>                # yesterday (UTC)
    python experiments/impact.py --results <dir> --days 30      # seed from the last 30 days

Results layout (the impact-results branch):
    days/<date>.json     the day's PR failures, each with its class
    candidates.json      PR-only failures, followed until they resolve
    gpu-ci.json          the daily GPU CI's bisected regressions, each with what MIC did on the culprit PR
    model-types.json     cache: fixture repo -> model_type
    baselines.jsonl      one line per fleet update: baselines changed/added/removed
    labels.jsonl         human verdicts: {"key", "verdict", "note", "by"}
    summary.md           the five questions, answered with the numbers so far
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DASHBOARD = "https://transformers-ci.huggingface.cool"
HUB = "https://huggingface.co"
GITHUB = "https://api.github.com"
TRANSFORMERS = "huggingface/transformers"
FLEET = "huggingface-mic/transformers-fleet"
ROW_LIMIT = 200
FAILURE_TYPES = (
    "Baseline assertion",
    "Runtime traceback",
    "Setup error",
    "Wrong hardware",
    "Missing baseline",
    "Unexpected pass",
)
# Failures that say nothing about the PR's code.
INFRA = {"Setup error", "Wrong hardware", "Missing baseline"}
# The same case failing on this many PRs in one day is shared breakage (a template date,
# a runner, a stale baseline), not the PRs' code, even when main did not run that case.
SYSTEMIC_PRS = 3
VERDICTS = {"real-bug", "intended", "false-alarm", "flake", "call-not-covered", "needs-real-weights"}
DAILY_CI = "hf-internal-testing/transformers_daily_ci"
MIC_JOB = "pr-ci / fast-integration / run"
# How long a merged PR's case is watched on main before it counts as "main ok".
MAIN_WATCH_DAYS = 3
# Proposed bar for "block merges when MIC is red" after a month of data (rule B below).
DECISION = {
    "days": 30,  # observed days before deciding
    "false_block_rate": 0.01,  # wrongly blocked merged PRs, at most this share of all merged PRs
    "justified_blocks": 1,  # at least this many blocks that stopped a real or unreviewed change
    "unresolved_share": 0.10,  # blocks still without an outcome or verdict, at most this share
    "rebaselines_per_week": 10,  # contracts re-baselined per week, at most
}
JUSTIFIED = {"merged-then-main-failed"}
JUSTIFIED_VERDICTS = {"real-bug", "intended"}
WRONG = {"recovered-no-push", "merged-main-ok"}
WRONG_VERDICTS = {"false-alarm", "flake"}


# ---------------------------------------------------------------- fetching


def get_json(url, headers=None, attempts=4):
    request = urllib.request.Request(url, headers={"User-Agent": "mic-impact", **(headers or {})})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code in (404, 422) or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, json.JSONDecodeError):
            # A body cut off mid-transfer (GitHub's job lists are large) is retried like a timeout.
            if attempt == attempts - 1:
                raise
        time.sleep(5 * (attempt + 1))


def mic_rows(scope, days, group):
    """Every failing case in the window, one request per failure type so no slice hits the row cap."""
    rows, truncated, summary = {}, [], None
    for failure_type in FAILURE_TYPES:
        query = urllib.parse.urlencode(
            {"days": days, "scope": scope, "group": group, "sticky_only": "false", "failure_type": failure_type}
        )
        body = get_json(f"{DASHBOARD}/mic/data?{query}")
        summary = summary or body.get("summary")
        if body.get("total", 0) > len(body.get("rows", [])) and len(body["rows"]) >= ROW_LIMIT:
            truncated.append(failure_type)
        for row in body.get("rows", []):
            rows[(row.get("pr"), case_key(row))] = row
    return list(rows.values()), summary, truncated


def github(path):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return get_json(f"{GITHUB}{path}", headers)


# ---------------------------------------------------------------- classification


def case_key(row):
    return "|".join(str(row.get(k, "")) for k in ("contract", "target", "variant", "platform"))


def target_kind(row):
    """Fixtures are named <name>-mic, older ones mic-<name>-slim; anything else is a real checkpoint."""
    target = str(row.get("target", ""))
    return "fixture" if target.endswith(("-mic", "-slim")) or target.startswith("mic-") else "checkpoint"


def failing_on_main(main_rows, first_day, day):
    """Case keys that failed on main on `day` or the day before (day is a UTC day number)."""
    keys = set()
    for row in main_rows:
        for offset, bucket in enumerate(row.get("daily") or []):
            if bucket.get("fail") and day - 1 <= first_day + offset <= day:
                keys.add(case_key(row))
    return keys


def classify(row, main_keys, systemic=frozenset()):
    """infra | pre-existing | systemic | pr-only. PR-only is the only class that can be a catch."""
    if row.get("failure_kind") in INFRA:
        return "infra"
    if case_key(row) in main_keys:
        return "pre-existing"
    if case_key(row) in systemic:
        return "systemic"
    return "pr-only"


def systemic_cases(pr_rows, first_day):
    """(day, case key) pairs that failed on SYSTEMIC_PRS or more PRs that day."""
    prs = defaultdict(set)
    for row in pr_rows:
        if row.get("failure_kind") not in INFRA:
            for day in failing_days(row, first_day):
                prs[day, case_key(row)].add(row.get("pr"))
    return {key for key, numbers in prs.items() if len(numbers) >= SYSTEMIC_PRS}


def failing_days(row, first_day):
    return [first_day + i for i, bucket in enumerate(row.get("daily") or []) if bucket.get("fail")]


def reduce_row(row):
    keep = (
        "pr", "pr_title", "commit_sha", "contract", "target", "variant", "platform", "weights",
        "failure_kind", "failure", "run_url", "trace_url", "selected_by", "known_failure",
    )
    reduced = {k: row.get(k) for k in keep}
    reduced["failure"] = (reduced["failure"] or "")[:300]
    reduced["kind"] = target_kind(row)
    reduced["latest"] = (row.get("latest") or {}).get("status")
    return reduced


# ---------------------------------------------------------------- following PR-only cases


def pr_info(number, cache):
    if number not in cache:
        pr = github(f"/repos/{TRANSFORMERS}/pulls/{number}")
        commits = github(f"/repos/{TRANSFORMERS}/pulls/{number}/commits?per_page=100")
        cache[number] = {
            "state": "merged" if pr.get("merged_at") else pr.get("state"),
            "merged_at": pr.get("merged_at"),
            "closed_at": pr.get("closed_at"),
            "head": pr["head"]["sha"],
            "commit_times": sorted(c["commit"]["committer"]["date"] for c in commits),
        }
    return cache[number]


def stamp(iso):
    return dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def update_candidate(candidate, row, info, main_keys_after, today):
    """Advance one PR-only case to its outcome. Outcomes are observations, not verdicts."""
    latest = (row.get("latest") or {}) if row else {}
    if row:
        candidate["last_seen"] = today
        if row.get("commit_sha") and row["commit_sha"] not in candidate["commits"]:
            candidate["commits"].append(row["commit_sha"])
        if latest.get("status") == "pass" and candidate["first_fail_at"]:
            # Recovered on a later run of the same PR: by a push, or by itself?
            pushed = any(
                candidate["first_fail_at"] < stamp(t) <= latest.get("started_at", 0)
                for t in info["commit_times"]
            )
            candidate["recovered"] = "after-push" if pushed else "no-push"
    candidate["pr_state"] = info["state"]
    candidate["merged_at"] = info["merged_at"]
    if info["state"] == "merged":
        merged_day = int(stamp(info["merged_at"]) // 86400)
        if candidate["key"].split("#", 1)[1] in main_keys_after:
            candidate["main_after_merge"] = "fail"
        elif today - merged_day >= MAIN_WATCH_DAYS and candidate.get("main_after_merge") is None:
            candidate["main_after_merge"] = "ok"
    candidate["outcome"] = outcome(candidate)
    return candidate


def outcome(c):
    if c.get("recovered") == "after-push":
        return "fixed-by-push"  # the author changed the PR and the case passed: the strongest catch signal
    if c.get("recovered") == "no-push":
        return "recovered-no-push"  # same code passed later: flaky or environment
    if c.get("pr_state") == "merged":
        return {"fail": "merged-then-main-failed", "ok": "merged-main-ok"}.get(c.get("main_after_merge"), "merged-watching")
    if c.get("pr_state") == "closed":
        return "closed-unmerged"
    return "open"


# ---------------------------------------------------------------- baseline review work


def fleet_updates(since):
    """Fleet dataset commits since `since` (ISO date), oldest first, with their parent."""
    commits = get_json(f"{HUB}/api/datasets/{FLEET}/commits/main?limit=500")
    ordered = list(reversed(commits))
    updates = []
    for parent, commit in zip(ordered, ordered[1:]):
        if commit["date"][:10] >= since:
            updates.append({"id": commit["id"], "parent": parent["id"], "date": commit["date"], "title": commit["title"]})
    return updates


def repo_files(repo, revision):
    """Expectation files and weight files (path -> oid) of a model repo at a revision."""
    tree = get_json(f"{HUB}/api/models/{repo}/tree/{revision}?recursive=true")
    files = {f["path"]: (f.get("lfs") or {}).get("oid") or f["oid"] for f in tree if f["type"] == "file"}
    expectations = {p: o for p, o in files.items() if "/expectations/" in p and p.endswith(".json")}
    weights = {p: o for p, o in files.items() if p.endswith((".safetensors", ".bin"))}
    return expectations, weights


def baseline_changes(update, download):
    """What one fleet update did to reviewed baselines: changed, added, removed files per contract."""
    fleet_name = "control" if "control fleet" in update["title"] else "prototypes"
    before = json.loads(Path(download(fleet_name, update["parent"])[0]).read_text())["contracts"]
    after = json.loads(Path(download(fleet_name, update["id"])[0]).read_text())["contracts"]
    records = []
    for name, contract in after.items():
        old = before.get(name)
        new_pin = contract.get("integration", {})
        if old and old.get("integration") == new_pin:
            continue
        repo, revision = new_pin.get("repo"), new_pin.get("revision")
        now, weights_now = repo_files(repo, revision) if repo else ({}, {})
        was, weights_was = {}, {}
        if old and old.get("integration", {}).get("repo"):
            was, weights_was = repo_files(old["integration"]["repo"], old["integration"]["revision"])
        title = ""
        try:
            title = get_json(f"{HUB}/api/models/{repo}/commits/{revision}?limit=1")[0]["title"]
        except (urllib.error.URLError, IndexError, KeyError):
            title = "(commit title unavailable)"
        records.append(
            {
                "date": update["date"][:10],
                "fleet_commit": update["id"],
                "fleet": fleet_name,
                "contract": name,
                "new_contract": old is None,
                "changed": sorted(p for p in now if p in was and was[p] != now[p]),
                "added": sorted(p for p in now if p not in was),
                "removed": sorted(p for p in was if p not in now),
                # Rebuilt fixture weights explain their baselines changing: fleet maintenance,
                # not review work caused by a Transformers change.
                "fixture_rebuilt": bool(weights_was) and weights_was != weights_now,
                "model_commit_title": title[:200],
            }
        )
    return records


# ---------------------------------------------------------------- state


def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def gpu_ci_regressions(date):
    """The daily GPU CI's bisected new failures on `date`, one entry per (culprit PR, model).

    Upstream bisects every new failure of the scheduled GPU run (all @slow tests; the
    *IntegrationTest classes load real checkpoints) to a bad commit and its PR. Flaky ones
    and those without a PR are skipped.
    """
    url = f"{HUB}/datasets/{DAILY_CI}/resolve/main/{date}/ci_results_run_models_gpu/new_failures_with_bad_commit.json"
    try:
        data = get_json(url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {}
        raise
    found = {}
    for entries in (e for by_gpu in data.values() for e in by_gpu.values()):
        for entry in entries:
            if not entry.get("pr_number") or str(entry.get("status", "")).startswith("flaky"):
                continue
            test = entry.get("test", "")
            parts = test.split("/")
            model = parts[2] if test.startswith("tests/models/") and len(parts) > 3 else None
            key = f"{entry['pr_number']}#{model or 'shared'}"
            item = found.setdefault(
                key,
                {
                    "pr": str(entry["pr_number"]),
                    "model": model,
                    "date": date,
                    "bad_commit": entry.get("bad_commit"),
                    "tests": [],
                    "integration": False,
                    "job_link": entry.get("job_link"),
                },
            )
            if test not in item["tests"]:
                item["tests"].append(test)
            item["integration"] = item["integration"] or "IntegrationTest" in test
    return found


def norm(model_type):
    return (model_type or "").replace("-", "_")


def fixture_model_types(results, download):
    """fixture repo -> model_type for every fixture in both fleets (cached: a repo's type does not change)."""
    cache = load(results / "model-types.json", {})
    for name in ("control", "prototypes"):
        lock = json.loads(Path(download(name, "main")[0]).read_text())
        for contract in lock["contracts"].values():
            for repo in contract.get("fixtures", {}):
                if repo not in cache:
                    try:
                        cache[repo] = get_json(f"{HUB}/{repo}/resolve/main/config.json").get("model_type")
                    except urllib.error.HTTPError:
                        cache[repo] = None
    write(results / "model-types.json", cache)
    return cache


def selects(files, model):
    """Whether MIC's changed-file selection runs a model's contracts (example/select_contracts.py rules).

    Approximate: a contract can also depend on another model's directory (its tokenizer's),
    which this treats as not selected.
    """
    for path in files:
        if path.startswith(f"src/transformers/models/{model}/") or path in ("setup.py", "pyproject.toml"):
            return True
        if path.startswith("src/transformers/models/auto/"):
            return True
        if path.startswith("src/transformers/") and not path.startswith("src/transformers/models/"):
            return True
    return False


def mic_job(head_sha):
    """Conclusion of MIC's PR CI job on a commit (success, failure, ...), or None if it never ran."""
    runs = github(f"/repos/{TRANSFORMERS}/actions/runs?head_sha={head_sha}&per_page=30")["workflow_runs"]
    for run in (r for r in runs if r.get("name") == "PR CI"):
        jobs = github(f"/repos/{TRANSFORMERS}/actions/runs/{run['id']}/jobs?per_page=100")["jobs"]
        for job in jobs:
            if job["name"] == MIC_JOB and job.get("conclusion") not in (None, "skipped", "cancelled"):
                return job["conclusion"]
    return None


def sticky_gpu_onsets(health, first_day_number):
    """Sticky GPU failures (the Fix failures page: GPU, failures, sticky) that started inside the
    window, one entry per (model, onset day). A failure already failing on the window's first day
    has no observed onset and is left out.
    """
    found = {}
    for row in health.get("rows", []):
        if row.get("job") != "run_models_gpu" or row.get("category") != "failure" or not row.get("sticky"):
            continue
        days = row.get("days") or []
        # The strip counts failures per day; 0 is a day without one.
        onset = next((i for i in range(1, len(days)) if days[i] > 0 and days[i - 1] == 0), None)
        if onset is None or not row.get("model"):
            continue
        day = first_day_number + onset
        key = f"sticky#{row['model']}#{iso_day(day)}"
        item = found.setdefault(key, {"model": row["model"], "date": iso_day(day), "day": day, "tests": [], "integration": False, "signal": row.get("signal")})
        item["tests"].append(row["nodeid"])
        item["integration"] = item["integration"] or "IntegrationTest" in row["nodeid"]
    return found


def mic_on_main_near(item, main_rows, first_day_main, types, covered):
    """Was MIC red on main for the model within a day of a GPU failure's onset?"""
    model = item["model"]
    if model not in covered:
        return "not-covered"
    for row in main_rows:
        if row.get("failure_kind") in INFRA or norm(types.get(row.get("fixture_repo"))) != model:
            continue
        if any(abs(day - item["day"]) <= 1 for day in failing_days(row, first_day_main)):
            return "red-on-main"
    return "green-on-main"


def mic_on_culprit(item, pr_rows, types, covered, pr_files, head_sha):
    """What MIC did on a GPU CI regression's culprit PR.

    not-covered     no contract for the model
    caught-on-pr    a contract of that model failed (not infra) on the PR
    not-run         MIC's job never ran on the PR's head
    not-selected    it ran, but the PR's files do not select the model's contracts
    missed          selected and ran, and no contract of that model failed
    """
    model = item["model"]
    if not model or model not in covered:
        return "not-covered"
    red = [
        r
        for r in pr_rows
        if str(r.get("pr")) == item["pr"]
        and r.get("failure_kind") not in INFRA
        and norm(types.get(r.get("fixture_repo"))) == model
    ]
    if red:
        item["mic_failures"] = sorted({(r.get("failure") or "").splitlines()[0][:120] for r in red})[:5]
        return "caught-on-pr"
    if mic_job(head_sha) is None:
        return "not-run"
    if not selects(pr_files, model):
        return "not-selected"
    return "missed"


def collect(results, days, today_day, download):
    results = Path(results)
    pr_rows, pr_summary, pr_truncated = mic_rows("prs", days, "pr")
    main_rows, main_summary, main_truncated = mic_rows("main", max(days, 7), "model")
    first_day_prs = today_day - days + 1
    first_day_main = today_day - max(days, 7) + 1

    candidates = load(results / "candidates.json", {})
    gpu_ci = load(results / "gpu-ci.json", {})
    cache = {}

    # Classify every PR failure on the day it first failed, against main on that day.
    by_day = defaultdict(list)
    shared = systemic_cases(pr_rows, first_day_prs)
    for row in pr_rows:
        days_failed = failing_days(row, first_day_prs) or [today_day]
        day = days_failed[0]
        main_keys = failing_on_main(main_rows, first_day_main, day)
        label = classify(row, main_keys, {case for d, case in shared if d == day})
        reduced = {**reduce_row(row), "class": label}
        by_day[day].append(reduced)
        if label == "pr-only" and row.get("pr"):
            key = f"{row['pr']}#{case_key(row)}"
            candidate = candidates.setdefault(
                key,
                {
                    "key": key,
                    "pr": row["pr"],
                    "pr_title": row.get("pr_title"),
                    "case": case_key(row),
                    "kind": target_kind(row),
                    "weights": row.get("weights"),
                    "failure_kind": row.get("failure_kind"),
                    "failure": (row.get("failure") or "")[:300],
                    "first_seen": iso_day(day),
                    "first_fail_at": row.get("started_at"),
                    "commits": [],
                    "run_url": row.get("run_url"),
                },
            )
            candidate["first_fail_at"] = min(candidate["first_fail_at"] or row["started_at"], row["started_at"])

    # Follow every unresolved candidate, seen today or not.
    rows_by_key = {f"{r['pr']}#{case_key(r)}": r for r in pr_rows if r.get("pr")}
    main_today = failing_on_main(main_rows, first_day_main, today_day)
    for key, candidate in candidates.items():
        if candidate.get("outcome") in ("fixed-by-push", "recovered-no-push", "merged-then-main-failed", "merged-main-ok", "closed-unmerged"):
            continue
        try:
            info = pr_info(candidate["pr"], cache)
        except urllib.error.HTTPError:
            continue
        update_candidate(candidate, rows_by_key.get(key), info, main_today, today_day)

    # Every PR that saw a MIC failure: its state, for the blocking replay.
    pr_states = load(results / "prs.json", {})
    for number in {r["pr"] for rows in by_day.values() for r in rows if r.get("pr")}:
        if pr_states.get(number, {}).get("state") in ("merged", "closed"):
            continue
        try:
            info = pr_info(number, cache)
        except urllib.error.HTTPError:
            continue
        pr_states[number] = {k: info[k] for k in ("state", "merged_at", "closed_at")}
    write(results / "prs.json", pr_states)
    merged = load(results / "merged.json", {})
    for day in range(first_day_prs, today_day):  # whole days only
        if iso_day(day) not in merged:
            try:
                merged[iso_day(day)] = merged_count(iso_day(day))
            except urllib.error.HTTPError:
                pass
            time.sleep(2.5)  # search API: 30 requests a minute
    write(results / "merged.json", merged)

    # Real-checkpoint regressions from the daily GPU CI: did MIC flag them on the culprit PR?
    new = {}
    for day in range(first_day_prs - 1, today_day + 1):
        new |= {k: v for k, v in gpu_ci_regressions(iso_day(day)).items() if k not in gpu_ci}
    if new:
        types = fixture_model_types(results, download)
        covered = {norm(t) for t in types.values() if t}
        culprit_rows = pr_rows if days == 30 else mic_rows("prs", 30, "pr")[0]

        for key, item in new.items():
            try:
                info = pr_info(item["pr"], cache)
                files = [f["filename"] for f in github(f"/repos/{TRANSFORMERS}/pulls/{item['pr']}/files?per_page=100")]
                item["mic"] = mic_on_culprit(item, culprit_rows, types, covered, files, info["head"])
            except (urllib.error.URLError, http.client.IncompleteRead, json.JSONDecodeError) as error:
                # Left out of gpu_ci so the next run retries it.
                print(f"MIC lookup failed for {key}: {error}", file=sys.stderr)
                continue
            gpu_ci[key] = item

    # Sticky GPU failures that started in the window (the Fix failures page), bisected or not:
    # was MIC red on main for that model when they started?
    health = get_json(f"{DASHBOARD}/project-health/rows?days=30")
    onsets = {k: v for k, v in sticky_gpu_onsets(health, int(health["first_day"] // 86400)).items() if k not in gpu_ci}
    if onsets:
        types = fixture_model_types(results, download)
        covered = {norm(t) for t in types.values() if t}
        main_30 = main_rows if max(days, 7) == 30 else mic_rows("main", 30, "model")[0]
        for key, item in onsets.items():
            item["mic"] = mic_on_main_near(item, main_30, today_day - 30 + 1, types, covered)
            gpu_ci[key] = item

    for day, rows in by_day.items():
        path = results / "days" / f"{iso_day(day)}.json"
        existing = load(path, {"rows": []})
        merged = {(r["pr"], r["contract"], r["target"], r["variant"], r["platform"]): r for r in existing["rows"] + rows}
        write(path, {"date": iso_day(day), "rows": sorted(merged.values(), key=lambda r: (str(r["pr"]), r["contract"]))})
    write(results / "candidates.json", candidates)
    write(results / "gpu-ci.json", gpu_ci)
    sweep = json.loads(SWEEP.read_text())["rows"] if SWEEP.exists() else {}
    covered_now = {norm(t) for t in fixture_model_types(results, download).values() if t}
    write(results / "suggestions.json", build_suggestions(gpu_ci, sweep, {}, covered_now))

    # Baseline review work: fleet updates since the window opened, each counted once.
    seen = {r["fleet_commit"] for r in read_jsonl(results / "baselines.jsonl")}
    with (results / "baselines.jsonl").open("a") as out:
        for update in fleet_updates(iso_day(first_day_prs)):
            if update["id"] in seen:
                continue
            try:
                records = baseline_changes(update, download)
            except Exception as error:  # a broken historical revision must not stop the day
                records = [{"date": update["date"][:10], "fleet_commit": update["id"], "error": str(error)[:200]}]
            for record in records or [{"date": update["date"][:10], "fleet_commit": update["id"], "contract": None}]:
                out.write(json.dumps(record, sort_keys=True) + "\n")

    write(
        results / "runs" / f"{iso_day(today_day)}.json",
        {
            "date": iso_day(today_day),
            "window_days": days,
            "prs": pr_summary,
            "main": main_summary,
            "truncated": {"prs": pr_truncated, "main": main_truncated},
        },
    )


def iso_day(day):
    return dt.datetime.fromtimestamp(day * 86400, tz=dt.UTC).date().isoformat()


def merged_count(day):
    """PRs merged into Transformers main on `day` (ISO date): the denominator for blocking."""
    query = urllib.parse.quote(f"repo:{TRANSFORMERS} is:pr is:merged base:main merged:{day}")
    return github(f"/search/issues?q={query}&per_page=1")["total_count"]


# ---------------------------------------------------------------- summary


# ---------------------------------------------------------------- suggestions and drafts

SWEEP = Path(__file__).parent / "results" / "sweep.json"
TMIC_RUN = "tmic-run.yml"
MIC_REPO = "huggingface/transformers-mic"
DRAFTS_PER_DAY = 2
NEXT_TARGETS = 5  # rows in the overview's "contracts to add next"
# Test names that exercise hardware or optional backends a CPU fixture contract does not run.
GPU_ONLY = ("flash_attn", "flash_attention", "fa2", "4bit", "8bit", "bnb", "bitsandbytes", "compile", "fp16", "cuda", "multi_gpu", "tensor_parallel")
NOT_MODELS = {"auto", "shared", None}
CHECKPOINT_RE = re.compile(r"""["']([A-Za-z0-9][\w.-]*/[\w.-]+)["']""")
FILE_RE = re.compile(r"\.(json|png|jpe?g|wav|mp3|flac|txt|py|safetensors|bin|pt|gguf|mp4|npy|md)$")


def get_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": "mic-impact"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode()


def test_checkpoints(nodeids, files):
    """Hub checkpoints the failing test classes load (from_pretrained("org/name"), model_id = ...), most used first."""
    found = Counter()
    for nodeid in nodeids:
        path, _, rest = nodeid.partition("::")
        class_name = rest.split("::")[0]
        if path not in files:
            try:
                files[path] = get_text(f"https://raw.githubusercontent.com/{TRANSFORMERS}/main/{path}")
            except (urllib.error.URLError, http.client.IncompleteRead):
                files[path] = ""
        text = files[path]
        start = text.find(f"class {class_name}")
        if start < 0:
            continue
        end = text.find("\nclass ", start + 1)
        for checkpoint in CHECKPOINT_RE.findall(text[start : end if end > 0 else None]):
            if checkpoint.startswith(("hf-internal-testing/", "http")) or FILE_RE.search(checkpoint):
                continue  # tiny test repos, URLs, file paths
            found[checkpoint] += 1
    return [c for c, _ in found.most_common()]


def build_suggestions(gpu_ci, sweep, files, covered=frozenset()):
    """New contracts for models with real-checkpoint regressions and no contract; extensions for the rest.

    covered: model types the fleet covers now. A regression recorded "not-covered" before its model
    got a contract is not a suggestion any more.
    """
    new = defaultdict(lambda: {"integration": 0, "other": 0, "tests": [], "keys": []})
    extend = defaultdict(list)
    for key, item in gpu_ci.items():
        model = item.get("model")
        if model in NOT_MODELS:
            continue
        if item.get("mic") == "not-covered":
            if norm(model or "") in covered:
                continue
            entry = new[model]
            entry["integration" if item.get("integration") else "other"] += 1
            entry["tests"] += [t for t in item["tests"] if t not in entry["tests"]]
            entry["keys"].append(key)
        elif item.get("mic") in ("missed", "green-on-main"):
            for test in item["tests"]:
                gpu_only = any(word in test.lower() for word in GPU_ONLY)
                extend[model].append({"test": test, "gpu_only": gpu_only, "integration": "IntegrationTest" in test, "date": item["date"]})
    suggestions = []
    for model, entry in new.items():
        row = sweep.get(model) or {}
        integration_tests = [t for t in entry["tests"] if "IntegrationTest" in t]
        from_tests = test_checkpoints(integration_tests or entry["tests"], files)
        sweep_repo = row.get("repo") if not str(row.get("repo", "")).startswith("hf-internal-testing/") else None
        checkpoint = (from_tests or [None])[0] or sweep_repo
        suggestions.append(
            {
                "model": model,
                "integration": entry["integration"],
                "other": entry["other"],
                "checkpoint": checkpoint,
                "checkpoint_source": "failing test" if from_tests else ("shrink sweep" if sweep_repo else None),
                "shrinkable": row.get("stage") == "ok" and bool(row.get("finite")),
                "tests": entry["tests"][:5],
            }
        )
    suggestions.sort(key=lambda s: (-s["shrinkable"], -s["integration"], -s["other"], s["model"]))
    # A test that failed on SYSTEMIC_PRS or more models is one shared regression (torch.export, a
    # cache class), not a call for each contract: listed once.
    method_models = defaultdict(set)
    for model, tests in extend.items():
        for t in tests:
            method_models[test_method(t["test"])].add(model)
    shared = {m: sorted(models) for m, models in method_models.items() if len(models) >= SYSTEMIC_PRS}
    extensions = {}
    for model, tests in sorted(extend.items()):
        own = [t for t in tests if test_method(t["test"]) not in shared]
        if own:
            extensions[model] = sorted(own, key=lambda t: (t["gpu_only"], not t["integration"], t["test"]))
    return {"new": suggestions, "extend": extensions, "shared": shared}


def test_method(nodeid):
    """test_torch_export_generate_static from tests/...::Class::test_torch_export_generate_static[...]"""
    return nodeid.split("::")[-1].split("[")[0]


def github_post(path, body):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    request = urllib.request.Request(
        f"{GITHUB}{path}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"User-Agent": "mic-impact", "Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status


def refresh_drafts(drafts):
    """Attach each dispatched draft's tmic-run.yml run (matched by its run name) and its state."""
    if not any(d.get("conclusion") is None for d in drafts.values()):
        return
    runs = github(f"/repos/{MIC_REPO}/actions/workflows/{TMIC_RUN}/runs?per_page=100")["workflow_runs"]
    for draft in drafts.values():
        if draft.get("conclusion") is not None:
            continue
        for run in runs:
            if run.get("display_title") == f"tmic run {draft['checkpoint']}" and run["created_at"] >= draft["dispatched_at"][:19]:
                draft.update(run_id=run["id"], url=run["html_url"], status=run["status"], conclusion=run.get("conclusion"))
                break


def fetch_shrink_pages(drafts, folder):
    """Download each finished draft's tmic-shrink artifact once, into folder/shrink/<run id>/.

    The Space then links the page directly: no artifact download, and the page outlives the
    artifact's 7-day retention. Records the page's Space path in the draft; returns the run ids fetched.
    """
    import subprocess

    fetched = []
    for draft in drafts.values():
        if draft.get("shrink") or draft.get("status") != "completed" or not draft.get("run_id"):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            done = subprocess.run(
                ["gh", "run", "download", str(draft["run_id"]), "--repo", MIC_REPO, "-n", "tmic-shrink", "-D", tmp],
                check=False, capture_output=True, text=True,
            )
            pages = sorted(Path(tmp).glob("*.html")) if done.returncode == 0 else []
            if not pages:
                continue  # no shrink page (the build failed first) or the artifact expired
            target = folder / "shrink" / str(draft["run_id"])
            target.mkdir(parents=True, exist_ok=True)
            for page in pages:
                (target / page.name).write_bytes(page.read_bytes())
            draft["shrink"] = f"shrink/{draft['run_id']}/{pages[0].name}"
            fetched.append(draft["run_id"])
    return fetched


def dispatch_drafts(suggestions, drafts, limit, now):
    """Dispatch tmic-run.yml (draft, cloud CPU/CUDA, publish if it passes) for the top suggestions not drafted yet."""
    sent = 0
    for s in suggestions["new"]:
        if sent >= limit:
            break
        if s["model"] in drafts or not s["checkpoint"] or not s["shrinkable"]:
            continue
        github_post(f"/repos/{MIC_REPO}/actions/workflows/{TMIC_RUN}/dispatches", {"ref": "main", "inputs": {"model": s["checkpoint"]}})
        drafts[s["model"]] = {"checkpoint": s["checkpoint"], "dispatched_at": now, "status": "queued", "conclusion": None}
        sent += 1
    return sent


def draft_lines(suggestions, drafts):
    lines = [
        "## 6. Contracts to add next",
        "",
        "Models with real-checkpoint regressions in the daily GPU CI and no contract, shrinkable models first. "
        f"Each day the top {DRAFTS_PER_DAY} without a draft go to `{TMIC_RUN}`: it drafts the contract and fixture "
        "on a runner, measures them on cloud CPU and CUDA, and publishes a new contract into the prototypes fleet "
        "only if every case passes. Read a published draft's `usage.md` like any other contract.",
        "",
    ]
    if not suggestions["new"]:
        return lines + ["- None yet.", ""]
    lines += ["| model | real-ckpt regressions | other | checkpoint | shrinkable | draft |", "|---|---|---|---|---|---|"]
    for s in suggestions["new"][:25]:
        draft = drafts.get(s["model"])
        if draft:
            state = draft.get("conclusion") or draft.get("status") or "queued"
            draft_cell = f"[{state}]({draft['url']})" if draft.get("url") else state
        else:
            draft_cell = "—" if s["shrinkable"] and s["checkpoint"] else "not draftable"
        checkpoint = f"[{s['checkpoint']}]({HUB}/{s['checkpoint']})" if s["checkpoint"] else "?"
        lines.append(f"| {s['model']} | {s['integration']} | {s['other']} | {checkpoint} | {'yes' if s['shrinkable'] else 'no'} | {draft_cell} |")
    lines += [
        "",
        "### Calls to add to existing contracts",
        "",
        "GPU CI regressions on models whose contract stayed green; **bold** models have a real-checkpoint "
        "(`*IntegrationTest`) one. GPU-only tests (flash-attention, quantization, compile, fp16) are listed apart: "
        "a CPU fixture contract does not run them.",
        "",
    ]
    for method, models in sorted(suggestions.get("shared", {}).items(), key=lambda kv: -len(kv[1])):
        lines.append(
            f"- shared: `{method}` failed on {len(models)} models ({', '.join(models[:6])}{', …' if len(models) > 6 else ''}) — "
            "one regression in shared code; one call in one contract covers it"
        )
    for model, tests in suggestions["extend"].items():
        callable_tests = [t for t in tests if not t["gpu_only"]]
        gpu_tests = [t for t in tests if t["gpu_only"]]
        if callable_tests:
            names = ", ".join(sorted({f"`{t['test'].split('::', 1)[-1]}`" for t in callable_tests})[:4])
            label = f"**{model}**" if any(t["integration"] for t in callable_tests) else model
            lines.append(f"- {label}: {names} — `tmic update '<the call these tests make>'`")
        if gpu_tests:
            names = ", ".join(f"`{t['test'].split('::')[-1]}`" for t in gpu_tests[:3])
            lines.append(f"- {model} (GPU-only): {names}")
    return lines + [""]


def summarize(results):
    results = Path(results)
    candidates = load(results / "candidates.json", {})
    labels = {}
    for label in read_jsonl(results / "labels.jsonl"):
        if label.get("verdict") in VERDICTS:
            labels[label["key"]] = label
    day_rows = [r for path in sorted((results / "days").glob("*.json")) for r in load(path, {"rows": []})["rows"]]
    baselines = [r for r in read_jsonl(results / "baselines.jsonl") if r.get("contract")]
    gpu_ci = load(results / "gpu-ci.json", {})
    runs = sorted((results / "runs").glob("*.json"))
    first = day_rows and min(path.stem for path in (results / "days").glob("*.json"))
    last = day_rows and max(path.stem for path in (results / "days").glob("*.json"))

    classes = Counter(r["class"] for r in day_rows)
    prs_failing = {r["pr"] for r in day_rows}
    prs_noise_only = {
        pr for pr in prs_failing if all(r["class"] != "pr-only" for r in day_rows if r["pr"] == pr)
    }
    outcomes = Counter(c.get("outcome", "open") for c in candidates.values())
    verdicts = Counter(labels[k]["verdict"] for k in candidates if k in labels)
    pending = [c for k, c in candidates.items() if k not in labels and c.get("outcome") not in ("open", "merged-watching")]

    def pct(n, d):
        return f"{n} ({100 * n / d:.0f}%)" if d else str(n)

    decision, checks = replay_blocking(results, day_rows, candidates, labels, baselines, first, last)
    lines = [
        "# MIC impact on Transformers PRs",
        "",
        f"Observed {first} → {last} ({len(runs)} collections). Source: "
        f"[MIC failures]({DASHBOARD}/d/mic-failures/mic-failures). Generated by `experiments/impact.py`.",
        "",
        "Each PR failure is classified on the day it first failed: **infra** (setup error, wrong hardware, "
        "missing baseline), **pre-existing** (the same case failed on main that day or the day before), or "
        f"**systemic** (the same case failed on ≥{SYSTEMIC_PRS} PRs that day), or **PR-only**. PR-only cases are followed to an outcome. Outcomes are observations; verdicts come from "
        "`labels.jsonl`.",
        "",
        *decision,
        "## 1. Real bugs caught on PRs",
        "",
        f"- PR-only failures: **{len(candidates)}** across {len({c['pr'] for c in candidates.values()})} PRs.",
        f"- Fixed by a push, then passed: **{outcomes['fixed-by-push']}** — the strongest catch signal.",
        f"- Merged red, then the same case failed on main: **{outcomes['merged-then-main-failed']}** — a change that landed (a bug, or an intended change needing a re-baseline).",
        f"- Labelled `real-bug`: **{verdicts['real-bug']}**; `intended`: {verdicts['intended']}.",
        "",
        "## 2. False alarms",
        "",
        f"- PR failure groups: {len(day_rows)} — infra {pct(classes['infra'], len(day_rows))}, "
        f"pre-existing {pct(classes['pre-existing'], len(day_rows))}, systemic (same case on ≥{SYSTEMIC_PRS} PRs that day) "
        f"{pct(classes['systemic'], len(day_rows))}, PR-only {pct(classes['pr-only'], len(day_rows))}.",
        f"- PRs that saw a MIC failure: {len(prs_failing)}; of those, only failures they did not cause: "
        f"**{pct(len(prs_noise_only), len(prs_failing))}**.",
        f"- PR-only cases that recovered with no push (flaky/environment): {outcomes['recovered-no-push']}; "
        f"merged and main stayed green for {MAIN_WATCH_DAYS} days: {outcomes['merged-main-ok']}; "
        f"labelled `false-alarm`/`flake`: {verdicts['false-alarm'] + verdicts['flake']}.",
        "",
        "## 3. Baseline review work",
        "",
    ]
    by_week = defaultdict(lambda: Counter())
    for record in baselines:
        week = dt.date.fromisoformat(record["date"]).isocalendar()
        bucket = by_week[f"{week[0]}-W{week[1]:02d}"]
        bucket["updates"] += 1
        bucket["new contracts"] += bool(record.get("new_contract"))
        bucket["changed baselines"] += len(record.get("changed", []))
        bucket["added baselines"] += len(record.get("added", []))
        rebaselined = bool(record.get("changed")) and not record.get("new_contract")
        bucket["re-baselined contracts"] += rebaselined and not record.get("fixture_rebuilt")
        bucket["rebuilt fixtures"] += rebaselined and bool(record.get("fixture_rebuilt"))
    if by_week:
        lines += [
            "Re-baselined = baselines changed with the same fixture weights (the review work a Transformers or "
            "environment change causes). Rebuilt = new fixture weights, so new baselines (fleet maintenance).",
            "",
            "| week | fleet updates | new contracts | re-baselined contracts | rebuilt fixtures | changed files | added files |",
            "|---|---|---|---|---|---|---|",
        ]
        for week, c in sorted(by_week.items()):
            lines.append(
                f"| {week} | {c['updates']} | {c['new contracts']} | {c['re-baselined contracts']} | {c['rebuilt fixtures']} | {c['changed baselines']} | {c['added baselines']} |"
            )
        reasons = [r for r in baselines if r.get("changed") and not r.get("new_contract") and not r.get("fixture_rebuilt")][-8:]
        if reasons:
            lines += ["", "Latest re-baselines (model commit title):", ""]
            lines += [f"- {r['date']} {r['contract']}: {r.get('model_commit_title') or '(no title)'}" for r in reasons]
    else:
        lines.append("- No fleet updates observed yet.")
    lines += [
        "",
        "## 4. Real-checkpoint regressions: did MIC flag them first?",
        "",
        "From Transformers' daily GPU CI (`hf-internal-testing/transformers_daily_ci`): each new failure upstream "
        "bisected to a culprit PR, one entry per (PR, model), flaky ones skipped. `*IntegrationTest` failures load "
        "real checkpoints. For each, what MIC did on the culprit PR:",
        "",
        "| MIC on the culprit PR | integration (real checkpoints) | other GPU tests |",
        "|---|---|---|",
    ]
    bisected = {k: v for k, v in gpu_ci.items() if not k.startswith("sticky#")}
    sticky = {k: v for k, v in gpu_ci.items() if k.startswith("sticky#")}
    for outcome_name, meaning in (
        ("caught-on-pr", "red on the PR: would have blocked a day earlier"),
        ("missed", "selected, ran, green"),
        ("not-selected", "ran, model not selected by the PR's files"),
        ("not-run", "MIC never ran on the PR"),
        ("not-covered", "no contract for the model"),
    ):
        integ = sum(1 for v in bisected.values() if v.get("mic") == outcome_name and v.get("integration"))
        other = sum(1 for v in bisected.values() if v.get("mic") == outcome_name and not v.get("integration"))
        lines.append(f"| {outcome_name}: {meaning} | {integ} | {other} |")
    lines += [
        "",
        f"Sticky GPU failures that started in the window ([Fix failures]({DASHBOARD}/d/pytest-test-health/fix-failures): "
        "GPU, failures, sticky), bisected or not, one entry per (model, first failing day): was MIC red on main for "
        "that model within a day?",
        "",
        "| MIC on main | integration (real checkpoints) | other GPU tests |",
        "|---|---|---|",
    ]
    for outcome_name in ("red-on-main", "green-on-main", "not-covered"):
        integ = sum(1 for v in sticky.values() if v.get("mic") == outcome_name and v.get("integration"))
        other = sum(1 for v in sticky.values() if v.get("mic") == outcome_name and not v.get("integration"))
        lines.append(f"| {outcome_name} | {integ} | {other} |")
    missed = [
        (k, v) for k, v in sorted(gpu_ci.items())
        if v.get("mic") in ("missed", "green-on-main") and v.get("integration") and f"gpu#{k}" not in labels
    ]
    lines += [
        "",
        "Real-checkpoint failures MIC stayed green on, waiting for a verdict. A miss is not proof that real weights "
        "were needed: often the contract does not make the call that broke. "
        'Label each with `{"key": "gpu#<key>", "verdict": "call-not-covered|needs-real-weights"}`.',
        "",
    ]
    for key, v in missed[:20]:
        lines.append(
            f"- `gpu#{key}`: {v['date']}, {len(v['tests'])} test(s), e.g. `{v['tests'][0]}`"
            + (f" ([PR #{v['pr']}](https://github.com/{TRANSFORMERS}/pull/{v['pr']}))" if v.get("pr") else "")
        )
    verdicts_gpu = Counter(labels[f"gpu#{k}"]["verdict"] for k in gpu_ci if f"gpu#{k}" in labels)
    if verdicts_gpu:
        lines.append(
            f"- Labelled: needs-real-weights {verdicts_gpu['needs-real-weights']}, "
            f"call-not-covered {verdicts_gpu['call-not-covered']}."
        )
    lines += [
        "",
        "## 5. Overfit vs random fixtures",
        "",
    ]
    by_weights = defaultdict(Counter)
    for c in candidates.values():
        by_weights[c.get("weights") or "unknown"][c.get("outcome", "open")] += 1
        if c["key"] in labels:
            by_weights[c.get("weights") or "unknown"]["label:" + labels[c["key"]]["verdict"]] += 1
    lines += ["| weights | PR-only | fixed by push | merged→main failed | labelled real-bug |", "|---|---|---|---|---|"]
    for weights, c in sorted(by_weights.items()):
        lines.append(
            f"| {weights} | {sum(v for k, v in c.items() if not k.startswith('label:'))} | {c['fixed-by-push']} | {c['merged-then-main-failed']} | {c['label:real-bug']} |"
        )
    suggestions = load(results / "suggestions.json", {"new": [], "extend": {}})
    drafts = load(results / "drafts.json", {})
    lines += [""] + draft_lines(suggestions, drafts)
    lines += [f"## Waiting for a verdict ({len(pending)})", ""]
    if pending:
        lines.append("Add a line to `labels.jsonl`: `{\"key\": \"<key>\", \"verdict\": \"real-bug|intended|false-alarm|flake\", \"note\": \"...\", \"by\": \"<login>\"}`.")
        lines.append("")
        for c in sorted(pending, key=lambda c: c["first_seen"])[-40:]:
            lines.append(
                f"- `{c['key']}` — {c.get('outcome', 'open')}, {c['weights']} {c['kind']}, {c['failure_kind']}: "
                f"{(c['failure'] or '').splitlines()[0][:100]} — [PR #{c['pr']}](https://github.com/{TRANSFORMERS}/pull/{c['pr']}), "
                f"[run]({DASHBOARD}{c.get('run_url') or ''})"
            )
    else:
        lines.append("- None.")
    (results / "summary.md").write_text("\n".join(lines) + "\n")

    def gpu(kinds):
        return sum(1 for v in gpu_ci.values() if v.get("mic") in kinds and v.get("integration"))

    write(results / "overview.json", {
        "window": [first or None, last or None],
        "caught": {
            "fixed_by_push": outcomes["fixed-by-push"],
            "merged_then_main_failed": outcomes["merged-then-main-failed"],
            "real_checkpoint": gpu(("caught-on-pr", "red-on-main")),
            "labelled_real_bug": verdicts["real-bug"],
        },
        "missed": {
            "real_checkpoint": gpu(("missed", "green-on-main")),
            "not_covered": gpu(("not-covered",)),
            "awaiting_verdict": len(missed),
        },
        "false_alarms": {"prs_with_failure": len(prs_failing), "prs_not_their_fault": len(prs_noise_only)},
        "block_merges": {"ready": all(ok for _, ok in checks), "unmet": [text for text, ok in checks if not ok]},
        "next": [next_target(s, drafts.get(s["model"])) for s in suggestions["new"][:NEXT_TARGETS]],
        "awaiting_verdict": len(pending),
    })


def next_target(suggestion, draft):
    """One row of the overview's "contracts to add next": the model, why, and its draft's state and pages."""
    draft = draft or {}
    return {
        "model": suggestion["model"],
        "checkpoint": suggestion["checkpoint"],
        "regressions": suggestion["integration"] + suggestion["other"],
        "real_checkpoint": suggestion["integration"],
        "draft": draft.get("conclusion") or draft.get("status") or ("" if suggestion["shrinkable"] and suggestion["checkpoint"] else "not draftable"),
        "run": draft.get("url", ""),
        "shrink": draft.get("shrink", ""),
    }


def replay_blocking(results, day_rows, candidates, labels, baselines, first, last):
    """Replay "block the merge while a MIC case is red" on every merged PR, under two rules.

    A: any red case blocks. B: only a red case that is neither infra nor failing on main.
    A block is justified when a red PR-only case went on to fail on main after the merge, or
    was labelled real-bug/intended (an intended change still has to be re-baselined first).
    It is wrong when every red case was infra, pre-existing, recovered with no push, merged
    with main staying green, or labelled false-alarm/flake. Otherwise it is unresolved.
    """
    if not day_rows:
        return ["## Decision: block merges when MIC is red?", "", "- No data yet.", ""]
    prs = load(results / "prs.json", {})
    merged_per_day = load(results / "merged.json", {})
    observed = sorted({r for r in merged_per_day if first <= r <= last})
    merged_total = sum(merged_per_day[d] for d in observed)
    latest = {}
    for row in day_rows:  # day files are in date order; the last row of a case wins
        latest[(row["pr"], row["contract"], row["target"], row["variant"], row["platform"])] = row
    red = defaultdict(list)
    for (pr, *_), row in latest.items():
        if prs.get(pr, {}).get("state") == "merged" and row.get("latest") in ("fail", "error"):
            red[pr].append(row)

    def judge(rows, rule):
        # Rule B cannot tell a systemic failure from the PR's own: it blocks, wrongly.
        rows = rows if rule == "A" else [r for r in rows if r["class"] in ("pr-only", "systemic")]
        if not rows:
            return None
        verdicts = []
        for row in rows:
            key = f"{row['pr']}#{row['contract']}|{row['target']}|{row['variant']}|{row['platform']}"
            candidate, label = candidates.get(key, {}), labels.get(key, {}).get("verdict")
            # A maintainer's verdict wins; observed outcomes only stand in until there is one.
            if label in JUSTIFIED_VERDICTS:
                verdicts.append("justified")
            elif label in WRONG_VERDICTS or row["class"] != "pr-only" or candidate.get("outcome") in WRONG:
                verdicts.append("wrong")
            elif candidate.get("outcome") in JUSTIFIED:
                verdicts.append("justified")
            else:
                verdicts.append("unresolved")
        if "justified" in verdicts:
            return "justified"
        return "unresolved" if "unresolved" in verdicts else "wrong"

    results_by_rule = {rule: Counter(filter(None, (judge(rows, rule) for rows in red.values()))) for rule in "AB"}
    weeks = max(1, len(observed) / 7)
    rebaselined = sum(
        1 for r in baselines
        if r.get("changed") and not r.get("new_contract") and not r.get("fixture_rebuilt") and first <= r["date"] <= last
    )
    b = results_by_rule["B"]
    blocks = sum(b.values())
    checks = [
        (f"{len(observed)} of {DECISION['days']} days observed", len(observed) >= DECISION["days"]),
        (
            f"wrong blocks {b['wrong']} = {100 * b['wrong'] / merged_total:.2f}% of {merged_total} merged PRs (bar ≤ {100 * DECISION['false_block_rate']:.0f}%)"
            if merged_total else "wrong blocks: no merged-PR count yet",
            bool(merged_total) and b["wrong"] / merged_total <= DECISION["false_block_rate"],
        ),
        (f"justified blocks {b['justified']} (bar ≥ {DECISION['justified_blocks']})", b["justified"] >= DECISION["justified_blocks"]),
        (
            f"unresolved blocks {b['unresolved']} of {blocks} (bar ≤ {100 * DECISION['unresolved_share']:.0f}%)",
            not blocks or b["unresolved"] / blocks <= DECISION["unresolved_share"],
        ),
        (
            f"re-baselined contracts {rebaselined / weeks:.1f}/week (bar ≤ {DECISION['rebaselines_per_week']})",
            rebaselined / weeks <= DECISION["rebaselines_per_week"],
        ),
    ]
    lines = [
        "## Decision: block merges when MIC is red?",
        "",
        f"Replayed on the {merged_total} PRs merged {observed[0] if observed else first} → {observed[-1] if observed else last}: "
        "which merges a red MIC case would have stopped, and whether stopping them was right.",
        "",
        "| rule | blocked merges | justified | wrong | unresolved |",
        "|---|---|---|---|---|",
    ]
    for rule, title in (("A", "A: any red case"), ("B", "B: red, not infra, not failing on main")):
        c = results_by_rule[rule]
        lines.append(f"| {title} | {sum(c.values())} | {c['justified']} | {c['wrong']} | {c['unresolved']} |")
    lines += ["", "Proposed bar for rule B (constants `DECISION` in `experiments/impact.py`):", ""]
    lines += [f"- {'✅' if ok else '❌'} {text}" for text, ok in checks]
    lines += [
        "",
        f"**{'Meets the bar' if all(ok for _, ok in checks) else 'Not yet'}** — the call stays with a maintainer; "
        "unresolved blocks shrink as cases get an outcome or a verdict in `labels.jsonl`.",
        "",
    ]
    return lines, checks


def publish_space(results, space, shrink=None):
    """Upload the overview, the full report and new shrink pages to a Hugging Face Space (HF_TOKEN must write to it).

    index.html renders overview.json; report.html renders summary.md. shrink: a folder holding
    shrink/<run id>/*.html pages to add (pages uploaded earlier stay on the Space).
    """
    import shutil

    from huggingface_hub import HfApi

    pages = Path(__file__).parent / "impact_space"
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        for name in ("index.html", "report.html"):
            shutil.copy(pages / name, folder / name)
        for name in ("summary.md", "overview.json"):
            shutil.copy(Path(results) / name, folder / name)
        if shrink and (Path(shrink) / "shrink").is_dir():
            shutil.copytree(Path(shrink) / "shrink", folder / "shrink")
        (folder / "updated.txt").write_text(dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + "\n")
        (folder / "README.md").write_text(
            "---\ntitle: MIC impact\nemoji: 📊\ncolorFrom: blue\ncolorTo: gray\nsdk: static\npinned: false\n---\n\n"
            "Daily evidence for whether a red model integration contract should block a Transformers merge. "
            "Generated by [mic-impact.yml](https://github.com/huggingface/transformers-mic/blob/main/.github/workflows/mic-impact.yml); "
            "data on the [impact-results branch](https://github.com/huggingface/transformers-mic/tree/impact-results).\n"
        )
        HfApi().upload_folder(
            repo_id=space, repo_type="space", folder_path=str(folder),
            commit_message=f"MIC impact {dt.date.today().isoformat()}",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path, help="checkout of the impact-results branch")
    parser.add_argument("--days", type=int, default=1, choices=(1, 7, 14, 30), help="window ending today; 30 to seed")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="only update the drafts' state and shrink pages, then rewrite the summary (a draft run finished)")
    parser.add_argument("--publish-space", metavar="REPO", help="also upload the summary page to this Hugging Face Space")
    parser.add_argument("--draft", type=int, default=0, metavar="N", help=f"dispatch {TMIC_RUN} for the top N suggested contracts not drafted yet")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        drafts = load(args.results / "drafts.json", {})
        refresh_drafts(drafts)
        if not args.refresh:
            from transformers_mic import fleet

            today_day = int(time.time() // 86400)
            collect(args.results, args.days, today_day, fleet.download)
            if args.draft:
                now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                dispatch_drafts(load(args.results / "suggestions.json", {"new": []}), drafts, args.draft, now)
        write(args.results / "drafts.json", drafts)
    with tempfile.TemporaryDirectory() as shrink:
        if args.publish_space:
            drafts = load(args.results / "drafts.json", {})
            if fetch_shrink_pages(drafts, Path(shrink)):
                write(args.results / "drafts.json", drafts)
        summarize(args.results)
        print((args.results / "summary.md").read_text())
        if args.publish_space:
            publish_space(args.results, args.publish_space, shrink)


if __name__ == "__main__":
    main()
