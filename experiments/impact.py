"""Daily MIC impact observer: what the contracts job does on Transformers PRs.

Whether MIC should block a PR depends on things replays cannot measure:

1. How many real bugs does it catch on PRs?
2. How often does it send people chasing false alarms?
3. How much work do baseline reviews take?
4. How many bugs only show with real weights?
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
    real-weights.json    checkpoint failures on main whose fixture passed
    baselines.jsonl      one line per fleet update: baselines changed/added/removed
    labels.jsonl         human verdicts: {"key", "verdict", "note", "by"}
    summary.md           the five questions, answered with the numbers so far
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
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
VERDICTS = {"real-bug", "intended", "false-alarm", "flake"}
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
        except (urllib.error.URLError, TimeoutError):
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


def nightly_runs(ref):
    """{date: [records]} from runs/nightly/ on the shadow-results branch (a git ref)."""
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", ref, "runs/nightly/"], capture_output=True, text=True, check=True).stdout.split()
    runs = defaultdict(list)
    for name in names:
        text = subprocess.run(["git", "show", f"{ref}:{name}"], capture_output=True, text=True, check=True).stdout
        runs[name.split("/")[2]] += [json.loads(line) for line in text.splitlines() if line.strip()]
    return dict(runs)


def real_weights_only(runs):
    """Checkpoint cases that started failing on a night when their contract's fixtures all passed.

    Keyed by case; a case that was already failing on its first observed night is not a
    transition and is reported as `since_first_night` rather than counted as a new catch.
    """
    found, previous = {}, {}
    for night in sorted(runs):
        records = runs[night]
        fixtures = defaultdict(list)
        for r in records:
            if r.get("kind") == "fixture" and r.get("status") in ("pass", "fail", "error"):
                fixtures[r["contract"], r["variant"]].append(r["status"])
        status = {}
        for r in records:
            if r.get("kind") != "checkpoint" or r.get("status") not in ("pass", "fail", "error"):
                continue
            key = "|".join(str(r.get(k, "")) for k in ("contract", "target", "variant", "platform"))
            status[key] = r["status"]
            fixtures_pass = fixtures.get((r["contract"], r["variant"])) and all(s == "pass" for s in fixtures[r["contract"], r["variant"]])
            if r["status"] == "pass" or key in found or not fixtures_pass:
                continue
            found[key] = {
                "first_fail": night,
                "after_pass": previous.get(key) == "pass",
                "since_first_night": key not in previous,
                "failure": "; ".join(r.get("failures") or [])[:300],
                "contract": r["contract"],
            }
        for key, item in found.items():
            if key in status:
                item["latest"] = status[key]
                item["last_night"] = night
        previous.update(status)
    return found


def collect(results, days, today_day, download, shadow_ref=None):
    results = Path(results)
    pr_rows, pr_summary, pr_truncated = mic_rows("prs", days, "pr")
    main_rows, main_summary, main_truncated = mic_rows("main", max(days, 7), "model")
    first_day_prs = today_day - days + 1
    first_day_main = today_day - max(days, 7) + 1

    candidates = load(results / "candidates.json", {})
    real_weights = load(results / "real-weights.json", {})
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

    # Real weights only: from the nightly real-checkpoint runs (shadow-results branch).
    if shadow_ref:
        runs = nightly_runs(shadow_ref)
        real_weights = real_weights_only(runs)
        last = runs[max(runs)] if runs else []
        real_weights["_coverage"] = {
            "nights": len(runs),
            "checkpoint_cases_last_night": sum(1 for r in last if r.get("kind") == "checkpoint" and r.get("status") != "unsupported"),
            "contracts_last_night": len({r["contract"] for r in last if r.get("kind") == "checkpoint"}),
        }

    for day, rows in by_day.items():
        path = results / "days" / f"{iso_day(day)}.json"
        existing = load(path, {"rows": []})
        merged = {(r["pr"], r["contract"], r["target"], r["variant"], r["platform"]): r for r in existing["rows"] + rows}
        write(path, {"date": iso_day(day), "rows": sorted(merged.values(), key=lambda r: (str(r["pr"]), r["contract"]))})
    write(results / "candidates.json", candidates)
    write(results / "real-weights.json", real_weights)

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


def summarize(results):
    results = Path(results)
    candidates = load(results / "candidates.json", {})
    labels = {}
    for label in read_jsonl(results / "labels.jsonl"):
        if label.get("verdict") in VERDICTS:
            labels[label["key"]] = label
    day_rows = [r for path in sorted((results / "days").glob("*.json")) for r in load(path, {"rows": []})["rows"]]
    baselines = [r for r in read_jsonl(results / "baselines.jsonl") if r.get("contract")]
    real_weights = load(results / "real-weights.json", {})
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

    decision = replay_blocking(results, day_rows, candidates, labels, baselines, first, last)
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
    coverage = real_weights.pop("_coverage", {})
    transitions = {k: v for k, v in real_weights.items() if v.get("after_pass")}
    standing = {k: v for k, v in real_weights.items() if v.get("since_first_night")}
    lines += [
        "",
        "## 4. Bugs only real weights show",
        "",
        "From the nightly real-checkpoint runs (`nightly.yml`, A10G, shadow-results `runs/nightly/`): a checkpoint "
        "case that failed on a night when every fixture of the same contract and variant passed.",
        "",
        f"- Started failing after a passing night: **{len(transitions)}**"
        + (": " + "; ".join(f"{k.split('|')[1]} {k.split('|')[2]} ({v['first_fail']}, now {v.get('latest')})" for k, v in sorted(transitions.items())[:10]) if transitions else "."),
        f"- Already failing on the first night observed (not counted as catches): {len(standing)}.",
        f"- Coverage: {coverage.get('nights', 0)} nights; the last one ran {coverage.get('checkpoint_cases_last_night', 0)} checkpoint cases "
        f"from {coverage.get('contracts_last_night', 0)} contracts. Contracts without a nightly checkpoint run cannot answer this question.",
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
    lines += ["", f"## Waiting for a verdict ({len(pending)})", ""]
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
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path, help="checkout of the impact-results branch")
    parser.add_argument("--days", type=int, default=1, choices=(1, 7, 14, 30), help="window ending today; 30 to seed")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--shadow-ref", default="origin/shadow-results", help="git ref of the shadow-results branch (nightly real-checkpoint runs)")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        from transformers_mic import fleet

        today_day = int(time.time() // 86400)
        collect(args.results, args.days, today_day, fleet.download, args.shadow_ref)
    summarize(args.results)
    print((args.results / "summary.md").read_text())


if __name__ == "__main__":
    main()
