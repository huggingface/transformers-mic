"""record-mic: re-record a Transformers PR's expected values, for that PR only.

A PR that changes a model's outputs on purpose turns its contracts red. Transformers PR CI's
contracts job records the cases its gate would block, on the PR's code (no secrets), and keeps
them in its artifact:

    python -m transformers_mic.pr_record measure --pr 50965 --cases contracts-gate.json --out rec/

A maintainer (write access to huggingface/transformers) then asks for them with a
`record-mic: <model> ...` PR comment or `make record-mic`; a job that holds the Hub write token and
runs no PR code validates that run's record and publishes it:

    python -m transformers_mic.pr_record publish --pr 50965 --record rec/ --models gptj \
        --merge-sha <the verified merge commit> --by <login> --execute

`measure --models` (outside CI) checks the selected contracts first and records their red targets.
Recording reruns each target in a fresh process, then checks the candidates in a separate pass.
`publish` commits them to a `transformers-pr-<N>` branch of each contract's mic repo, branched
from the commit the fleet pins, and writes `overlays/transformers-pr-<N>.json` on the fleet
dataset's main: {"<fleet>/<contract>": {"<mic repo>": {"from": pinned, "to": new}}}. Only runs
that pass `tmic check --fleet-pr <N>` apply it, so other PRs and main are unchanged.

When the PR merges, `merge --pr N --execute` writes the overlay into the fleet tables (entries
whose contract moved on main since are reported as stale, not applied) and deletes it; a closed
PR's overlay is removed with `discard`. The maintainer's request plus the diff posted on the PR
are the review: values are never fitted, only re-measured on the PR's code.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import fleet

# What Transformers PR CI runs of each fleet (transformers-ci .github/workflows/pr-ci_contracts.yml).
PR_CI_VARIANTS = {"control": ("default", "bfloat16"), "prototypes": ("bfloat16",)}
EXPECTATIONS = "integrations/transformers/expectations"
PATH = re.compile(r"integrations/transformers/expectations/[a-z0-9_-]+/[A-Za-z0-9_.-]+\.json")
RECORD = "record.json"


def branch(pr):
    return f"transformers-pr-{int(pr)}"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]


def normalise(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def model_type(repo, revision):
    """The fixture's model_type (its config.json at the pinned commit), or None."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        config = read(hf_hub_download(repo, "config.json", revision=revision))
    except (EntryNotFoundError, OSError, ValueError):
        return None
    return config.get("model_type")


def select(lock, models, types=model_type):
    """Contract names of `lock` that `models` name: a model type, a contract or a target name."""
    wanted = {normalise(m) for m in models}
    chosen = []
    for name, contract in sorted(lock["contracts"].items()):
        names = {normalise(name)} | {
            normalise(pin.get("target") or repo.split("/")[-1]) for repo, pin in contract.get("fixtures", {}).items()
        }
        if names & wanted:
            chosen.append(name)
            continue
        pin = contract["integration"]
        kind = types(pin["repo"], pin["revision"])
        if kind and normalise(kind) in wanted:
            chosen.append(name)
    return chosen


def run(root, options, name):
    """transformers-mic's run.py through the packaged backend; its exit code (log in root/logs)."""
    log = Path(root) / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        command = [sys.executable, "-m", "transformers_mic.backend", "run", str(root), *options]
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode


def copy_tolerances(source, destination):
    for override in Path(source).glob("*/tolerances.json"):
        target = Path(destination) / override.parent.name / override.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(override, target)


def changes(old, new):
    """{output: (old, new)} for the outputs whose recorded values differ."""
    old, new = (old or {}).get("outputs", {}), new.get("outputs", {})
    return {k: (old.get(k), new.get(k)) for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)}


def blocked_cases(path):
    """{(fleet, contract, variant): {targets}} of a contracts gate report's `block` decisions."""
    wanted = {}
    for d in read(path).get("decisions", []):
        if d.get("verdict") == "block":
            wanted.setdefault((d["fleet"], d["contract"], d["variant"]), set()).add(d["target"])
    return wanted


def measure(args):
    """Record candidates on this checkout (PR code, no secrets).

    --models: the selected contracts are checked first and their red targets recorded.
    --cases:  a contracts gate report (contracts-gate.json); only its `block` cases are recorded,
              with no check pass: the contracts job that wrote the report already ran it.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    if bool(args.models) == bool(args.cases):
        raise SystemExit("measure needs exactly one of --models and --cases")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    models = [m for m in re.split(r"[,\s]+", args.models or "") if m]
    blocked = blocked_cases(args.cases) if args.cases else {}
    record = {
        "pr": args.pr,
        "models": models,
        "transformers_sha": args.transformers_sha,
        "context": json.loads(args.context or "{}"),
        "cases": [],
        "errors": [],
    }
    for fleet_name, variants in PR_CI_VARIANTS.items():
        path, revision, overlay = fleet.download_for_pr(fleet_name, args.fleet_revision, args.pr)
        base, _ = fleet.download(fleet_name, revision)
        record.setdefault("fleet_revision", revision)
        record.setdefault("overlay", {})[fleet_name] = overlay
        lock, pinned = read(path), read(base)
        lock.pop("known_failures", None)  # candidates must satisfy every output assertion
        if args.cases:
            names = sorted({c for f, c, _ in blocked if f == fleet_name})
            unknown = [n for n in names if n not in lock["contracts"]]
            for name in unknown:
                record["errors"].append(
                    {"fleet": fleet_name, "contract": name, "variant": "", "targets": [], "reason": "not in the fleet"}
                )
            names = [n for n in names if n in lock["contracts"]]
        else:
            names = select(lock, models)
        for name in names:
            contract = lock["contracts"][name]
            pin = contract["integration"]
            folder = root / fleet_name / name
            single = {**lock, "contracts": {name: contract}}
            write(folder / "fleet.lock", single)
            snapshot = Path(
                snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=[f"{EXPECTATIONS}/**"])
            )
            references = snapshot / EXPECTATIONS
            manifest = read(hf_hub_download(pin["repo"], "integration.json", revision=pin["revision"]))
            declared = manifest["integrations"]["transformers"].get("variants", {"default": {}})
            for variant in [v for v in variants if v in declared]:
                options = ["--framework", "transformers", "--lock", str(folder / "fleet.lock"), "--contract", name]
                options += ["--variant", variant, "--isolate", "--fixtures-only"]
                tag = f"{fleet_name}-{name}-{variant}"
                if args.cases:
                    red = sorted(blocked.get((fleet_name, name, variant), ()))
                else:
                    checked = folder / f"{variant}.check.jsonl"
                    run(root, [*options, "--output", str(checked)], tag + "-check")
                    red = sorted({r["target"] for r in rows(checked) if r["status"] == "fail"})
                    crashed = sorted({r["target"] for r in rows(checked) if r["status"] == "error"})
                    if crashed:
                        record["errors"].append(
                            {
                                "fleet": fleet_name,
                                "contract": name,
                                "variant": variant,
                                "targets": crashed,
                                "reason": "crashes: not a value change, fix the code",
                            }
                        )
                if red:
                    record_and_verify(
                        root, record, folder, options, tag, red, references, pinned, fleet_name, name, pin, variant
                    )
    write(root / RECORD, record)
    print(summary(record))
    return 0 if record["cases"] and all(c["verified"] for c in record["cases"]) and not record["errors"] else 1


def record_and_verify(root, record, folder, options, tag, red, references, pinned, fleet_name, name, pin, variant):
    """Record candidates for the `red` targets (each rerun in a fresh process), then check them in a separate pass."""
    candidates, verification = folder / "candidates", folder / "verification"
    copy_tolerances(references, candidates)
    targets = [x for t in red for x in ("--target", t)]
    recorded = folder / f"{variant}.record.jsonl"
    run(
        root,
        [*options, *targets, "--record", "--expectations", str(candidates), "--output", str(recorded)],
        tag + "-record",
    )
    for row in rows(recorded):
        if row["status"] != "recorded":
            # A crash, a non-finite value or a failing assertion is not a value change to record.
            record["errors"].append(
                {
                    "fleet": fleet_name,
                    "contract": name,
                    "variant": variant,
                    "targets": [row["target"]],
                    "reason": "; ".join(row.get("failures", []))[:500] or row["status"],
                }
            )
    for candidate in candidates.glob("*/*.candidate.json"):
        final = verification / candidate.parent.name / candidate.name.replace(".candidate.json", ".json")
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, final)
    copy_tolerances(candidates, verification)
    verified = folder / f"{variant}.verify.jsonl"
    run(root, [*options, *targets, "--expectations", str(verification), "--output", str(verified)], tag + "-verify")
    passed = {r["target"] for r in rows(verified) if r["status"] == "pass"}
    suffix = "" if variant == "default" else f".{variant}"
    for row in rows(recorded):
        if row["status"] != "recorded":
            continue
        platform = row["platform"]
        file = verification / platform / f"{row['target']}{suffix}.json"
        if not file.exists():
            continue
        old = references / platform / file.name
        record["cases"].append(
            {
                "fleet": fleet_name,
                "contract": name,
                "repo": pin["repo"],
                "pinned": pinned["contracts"][name]["integration"]["revision"],
                "measured_on": pin["revision"],
                "target": row["target"],
                "variant": variant,
                "platform": platform,
                "path_in_repo": f"{EXPECTATIONS}/{platform}/{file.name}",
                "file": str(file.relative_to(root)),
                "verified": row["target"] in passed,
                "changes": changes(read(old) if old.exists() else None, read(file)),
            }
        )


def short(value, limit=60):
    text = json.dumps(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def leaves(value, path=""):
    """{path: leaf} of a nested output (lists and dicts)."""
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in leaves(item, f"{path}.{key}" if path else key).items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in leaves(item, f"{path}[{i}]").items()}
    return {path: value}


def describe(old, new):
    """One line for an output that changed: the largest numeric move, and discrete changes."""
    if old is None:
        return "new"
    a, b = leaves(old), leaves(new)
    deltas = [
        abs(b[k] - a[k])
        for k in a.keys() & b.keys()
        if isinstance(a[k], (int, float)) and isinstance(b[k], float) and not isinstance(a[k], bool)
    ]
    discrete = [k for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k) and not isinstance(b.get(k), float)]
    parts = [f"max Δ {max(deltas):.3g}"] if any(deltas) else []
    if len(a) != len(b):
        parts.append(f"{len(a)} → {len(b)} values")
    parts += [f"{k or 'value'}: {short(a.get(k), 30)} → {short(b.get(k), 30)}" for k in discrete[:3]]
    if len(discrete) > 3:
        parts.append(f"+{len(discrete) - 3} more")
    return ", ".join(parts) or "reordered"


def summary(record):
    lines = [f"### record-mic for #{record['pr']}: {', '.join(record['models'])}", ""]
    if record["cases"]:
        lines += [
            "| contract | target | variant | platform | verified | changed outputs |",
            "|---|---|---|---|---|---|",
        ]
        for c in record["cases"]:
            changed = "; ".join(f"`{k}` {describe(o, n)}" for k, (o, n) in c["changes"].items()) or "—"
            lines.append(
                f"| {c['contract']} | {c['target']} | {c['variant']} | {c['platform']} | "
                f"{'✅' if c['verified'] else '❌'} | {changed.replace('|', chr(92) + '|')} |"
            )
    else:
        lines.append("Nothing to re-record: no selected contract is red on this PR.")
    for e in record["errors"]:
        lines.append(f"\n❌ {e['contract']} [{e['variant']}] {', '.join(e['targets'])}: {e['reason']}")
    return "\n".join(lines) + "\n"


def requested(record, models, locks, types=model_type):
    """The record restricted to the contracts `models` name, in fleets {name: lock}."""
    chosen = {(name, c) for name, lock in locks.items() for c in select(lock, models, types)}
    keep = [c for c in record["cases"] if (c["fleet"], c["contract"]) in chosen]
    errors = [e for e in record["errors"] if (e["fleet"], e["contract"]) in chosen]
    if not keep and not errors:
        raise ValueError(f"no recorded case matches {', '.join(models)}")
    return {**record, "cases": keep, "errors": errors}


def validate(root, record, pr, merge_sha=None):
    """Refuse anything but verified expected-value files of this PR (the record comes from PR code)."""
    if record.get("pr") != pr:
        raise ValueError(f"record is for PR {record.get('pr')}, not {pr}")
    if merge_sha and record.get("transformers_sha") != merge_sha:
        raise ValueError(f"record measured {record.get('transformers_sha')}, not the verified merge commit {merge_sha}")
    if record.get("errors") or not record.get("cases"):
        raise ValueError("record has errors or no cases; nothing is published")
    files = {}
    for case in record["cases"]:
        if not case["verified"]:
            raise ValueError(f"{case['contract']}/{case['target']}: candidate failed its verification pass")
        if case["fleet"] not in fleet.FILES or not PATH.fullmatch(case["path_in_repo"]):
            raise ValueError(f"unexpected destination {case['fleet']}:{case['path_in_repo']}")
        source = (root / case["file"]).resolve()
        if root not in source.parents or source.suffix != ".json":
            raise ValueError(f"unexpected source file {case['file']}")
        value = read(source)
        if not isinstance(value, dict) or not {"outputs", "recorded_with"} <= set(value):
            raise ValueError(f"{case['file']} is not a recorded expectation")
        files.setdefault((case["fleet"], case["contract"], case["repo"]), {})[case["path_in_repo"]] = source
    return files


def publish(args, api=None):
    from huggingface_hub import CommitOperationAdd, HfApi

    api = api or HfApi()
    root = Path(args.record).resolve()
    record = read(root / RECORD)
    head = fleet.resolve("main")
    models = [m for m in re.split(r"[,\s]+", getattr(args, "models", None) or "") if m]
    if models:
        record = requested(record, models, {name: read(fleet.download(name, head)[0]) for name in fleet.FILES})
    files = validate(root, record, args.pr, getattr(args, "merge_sha", None))
    overlay = fleet.download_overlay(args.pr, head) or {"pr": args.pr, "contracts": {}, "records": []}
    plan = []
    for (fleet_name, contract, repo), paths in sorted(files.items()):
        lock = read(fleet.download(fleet_name, head)[0])
        pinned = lock["contracts"][contract]["integration"]["revision"]
        move = overlay["contracts"].get(f"{fleet_name}/{contract}", {}).get(repo)
        current = move["to"] if move and move["from"] == pinned else pinned
        measured = {
            c["measured_on"]
            for c in record["cases"]
            if (c["fleet"], c["contract"], c["repo"]) == (fleet_name, contract, repo)
        }
        if measured != {current}:
            raise ValueError(
                f"{contract}: the fleet moved since this record ({sorted(measured)} != {current}); record again"
            )
        plan.append((fleet_name, contract, repo, pinned, current, paths))
    if not args.execute:
        print(
            json.dumps(
                [{"contract": p[1], "repo": p[2], "from": p[3], "parent": p[4], "files": sorted(p[5])} for p in plan],
                indent=2,
            )
        )
        return 0
    for fleet_name, contract, repo, pinned, current, paths in plan:
        name = branch(args.pr)
        refs = api.list_repo_refs(repo)
        existing = next((b.target_commit for b in refs.branches if b.name == name), None)
        if existing and existing != current:
            api.delete_branch(repo, branch=name)
            existing = None
        if not existing:
            api.create_branch(repo, branch=name, revision=current)
        commit = api.create_commit(
            repo_id=repo,
            revision=name,
            parent_commit=current,
            operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(s)) for p, s in sorted(paths.items())],
            commit_message=f"record-mic: expected values for huggingface/transformers#{args.pr}",
        )
        overlay["contracts"].setdefault(f"{fleet_name}/{contract}", {})[repo] = {"from": pinned, "to": commit.oid}
    overlay["records"].append(
        {
            "by": args.by,
            "transformers_sha": record.get("transformers_sha"),
            "cases": [f"{c['contract']}/{c['target']}[{c['variant']}]" for c in record["cases"]],
        }
    )
    commit = api.create_commit(
        repo_id=fleet.REPOSITORY,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(
                path_in_repo=fleet.overlay_path(args.pr),
                path_or_fileobj=(json.dumps(overlay, indent=2) + "\n").encode(),
            )
        ],
        commit_message=f"record-mic: overlay for huggingface/transformers#{args.pr}",
    )
    print(
        json.dumps(
            {"overlay": fleet.overlay_path(args.pr), "fleet_revision": commit.oid, "contracts": overlay["contracts"]}
        )
    )
    return 0


def merge(args, api=None):
    """Write a merged PR's overlay into the fleet tables, then remove it (one commit)."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    since = fleet.check_since(args.since)
    api = api or HfApi()
    head = fleet.resolve("main")
    overlay = fleet.download_overlay(args.pr, head)
    if overlay is None:
        print(json.dumps({"pr": args.pr, "overlay": None}))
        return 0
    operations, reports = [], {}
    for name in fleet.FILES:
        lock, report = fleet.apply_overlay(read(fleet.download(name, head)[0]), overlay, name)
        reports[name] = report
        if report["applied"]:
            operations += [
                CommitOperationAdd(path_in_repo=p, path_or_fileobj=t.encode())
                for p, t in fleet.operations(name, lock).items()
            ]
            if since:
                # Older code (without the PR's commit on main) keeps meeting the values before these.
                added = [
                    {
                        "contract": contract,
                        "mic_repo": repo,
                        "from_revision": move["from"],
                        "revision": move["to"],
                        "since_transformers": since,
                    }
                    for key, moves in sorted(overlay["contracts"].items())
                    for contract in [key.partition("/")[2]]
                    if key.partition("/")[0] == name
                    for repo, move in sorted(moves.items())
                    if f"{contract}:{repo}" in report["applied"]
                ]
                rows = fleet.with_history(fleet.download_history(name, head), added)
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=fleet.table_path(name, fleet.HISTORY),
                        path_or_fileobj=fleet.history_csv(rows).encode(),
                    )
                )
    operations.append(CommitOperationDelete(path_in_repo=fleet.overlay_path(args.pr)))
    result = {"pr": args.pr, "fleet_revision": head, **reports}
    if args.execute:
        commit = api.create_commit(
            repo_id=fleet.REPOSITORY,
            repo_type="dataset",
            parent_commit=head,
            operations=operations,
            commit_message=f"record-mic: merge the expected values of huggingface/transformers#{args.pr}",
        )
        result["fleet_revision"] = commit.oid
        # The fleet now pins commits on transformers-pr-<N> branches: keep them reachable with a tag.
        for moves in overlay["contracts"].values():
            for repo, move in moves.items():
                api.create_tag(repo, tag=branch(args.pr), revision=move["to"], exist_ok=True)
    print(json.dumps(result))
    return 1 if any(r["stale"] for r in reports.values()) else 0


def discard(args, api=None):
    """Remove a closed PR's overlay; its mic-repo branches go too (nothing pins them)."""
    from huggingface_hub import CommitOperationDelete, HfApi

    api = api or HfApi()
    overlay = fleet.download_overlay(args.pr, fleet.resolve("main"))
    if overlay is None or not args.execute:
        print(json.dumps({"pr": args.pr, "overlay": overlay}))
        return 0
    api.create_commit(
        repo_id=fleet.REPOSITORY,
        repo_type="dataset",
        operations=[CommitOperationDelete(path_in_repo=fleet.overlay_path(args.pr))],
        commit_message=f"record-mic: discard the overlay of closed huggingface/transformers#{args.pr}",
    )
    for moves in overlay["contracts"].values():
        for repo in moves:
            api.delete_branch(repo, branch=branch(args.pr))
    return 0


def parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    m = commands.add_parser("measure", help="record candidates on this checkout (PR code, no secrets)")
    m.add_argument("--pr", type=int, required=True)
    m.add_argument("--models", help="model types, contract or target names (comma/space separated)")
    m.add_argument("--cases", type=Path, help="a contracts gate report: record its `block` cases only")
    m.add_argument("--fleet-revision", default="main")
    m.add_argument("--transformers-sha")
    m.add_argument("--context", help="JSON stored with the record (runner, image, run id)")
    m.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("publish", help="publish a measured record for the PR (Hub write token)")
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--by", required=True, help="who asked for it")
    p.add_argument("--models", help="publish only these contracts (model types, contract or target names)")
    p.add_argument(
        "--merge-sha",
        help="refuse a record measured on any other Transformers commit (a consistency check: the record "
        "comes from PR code; the caller ties it to the reviewed code from GitHub's run metadata)",
    )
    p.add_argument("--execute", action="store_true")
    for name, help in (
        ("merge", "write a merged PR's overlay into the fleet"),
        ("discard", "drop a closed PR's overlay"),
    ):
        c = commands.add_parser(name, help=help)
        c.add_argument("--pr", type=int, required=True)
        c.add_argument("--execute", action="store_true")
        if name == "merge":
            c.add_argument(
                "--since",
                required=True,
                help="the PR's commit on Transformers main: runs of code without it keep the previous values "
                "(every pin move writes a history row; a move without one turns history off for that contract)",
            )
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    return {"measure": measure, "publish": publish, "merge": merge, "discard": discard}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
