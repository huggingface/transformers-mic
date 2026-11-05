"""Run the fixture contracts on a slice of past Transformers main commits.

    python .github/scripts/backfill.py --repo /tmp/transformers --since 2026-09-07 --until 2026-10-08 \
        --chunk 3 --chunks 10 --output out/backfill-3.jsonl

The Transformers clone (git clone --filter=blob:none --shallow-since=... -b main) must be installed in editable mode (pip install -e . --no-deps)
with the fleet lock's pinned environment around it, so checking out a commit is
all it takes to test it. Commits are main's first-parent history in the window,
oldest first; chunk i of n takes every n-th commit. For each commit the changed
files are its diff against its first parent, so the selection is the one that
PR would have made. Every result record gets the run metadata under `shadow`,
as in shadow.yml; a commit where the runner produced no record gets one error
record (target "*") so it is counted, not lost.

Older commits can require other dependency versions than the fleet lock pins
(tokenizers<=0.23.0 before 2026-08-17). Before each commit, every pinned
package is checked against that commit's dependency_versions_table.py; one the
commit does not accept is replaced with the newest version it does, and put
back to the pin once a later commit accepts it again. The overrides are
recorded under shadow.pip_overrides, and the runner's environment_differs shows
them too.

With --plan it replays regressions instead (experiments/impact.py --replay-plan),
the way Transformers PR CI runs MIC (transformers-ci's pr-ci_contracts.yml): its
own runner, `run-model-contracts`, in the environment the job installed, each
entry's contracts from its fleet at a fixed revision, in each of the entry's
variants, on its parent and bad refs (`<sha>^` allowed). No pin is adapted: PR CI
does not either. A parent without the model's directory (the bad commit added the
model) gets one `absent` record instead of a run. Records carry `replay`
{key, side, ref, sha, fleet} instead of `shadow`.
"""

import argparse
import importlib.metadata as md
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def dependency_overrides(repo, sha, pinned):
    """Install what this commit's dependency table needs instead of the pins; return {package: requirement}."""
    from packaging.requirements import Requirement

    table = git(repo, "show", f"{sha}:src/transformers/dependency_versions_table.py")
    specs = {normalize(name): spec for name, spec in re.findall(r'"([^"]+)": "([^"]+)"', table)}
    wanted, overrides = [], {}
    for name, version in pinned.items():
        if name == "python" or normalize(name) not in specs:
            continue
        requirement = Requirement(specs[normalize(name)])
        target = f"{name}=={version}"
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            target = overrides[name] = f"{name}{requirement.specifier}"
        try:
            installed = md.version(name).split("+")[0]
        except md.PackageNotFoundError:
            installed = None
        satisfied = installed is not None and (
            Requirement(target).specifier.contains(installed, prereleases=True)
            and (name in overrides or installed == version)
        )
        if not satisfied:
            wanted.append(target)
    if wanted:
        print(f"  installing {wanted}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *wanted], check=True)
    return overrides


def cpu_info():
    model, flags = "", set()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name") and not model:
                model = line.split(":", 1)[1].strip()
            if line.startswith("flags"):
                flags |= set(line.split(":", 1)[1].split()) & {"avx2", "avx512f", "avx512_bf16", "amx_bf16", "amx_tile"}
    except OSError:
        pass
    return model, " ".join(sorted(flags))


def resolve(repo, ref):
    """Full sha of a commit or `<sha>^` in the clone; fetches a commit the clone does not have."""
    try:
        return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        git(repo, "fetch", "--quiet", "origin", ref.rstrip("^"))
        return git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()


RUN_PY = (sys.executable, str(HERE / "example/run.py"), "--framework", "transformers", "--require-all")
PR_CI_RUNNER = ("run-model-contracts",)


def run_commit(repo, sha, pinned, options=(), changed=None, command=RUN_PY):
    """Check out one commit, adapt the pins to it (unless pinned is None) and run the fixture contracts.

    options: more runner options; changed: the changed files, for the selection record.
    Returns (records, overrides).
    """
    git(repo, "checkout", "--quiet", "--force", sha)
    overrides = dependency_overrides(repo, sha, pinned) if pinned is not None else {}
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.jsonl"
        extra = []
        if changed is not None:
            changed_path = Path(tmp) / "changed.txt"
            changed_path.write_text(changed)
            extra = ["--changed-files", str(changed_path)]
        run = subprocess.run(
            [*command, "--fixtures-only", *extra, *options, "--run-all", "--output", str(raw)],
            capture_output=True, text=True, cwd=HERE,
        )
        records = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    if not records:
        records = [{"contract": "*", "framework": "transformers", "variant": "default", "target": "*", "kind": "fixture",
                    "status": "error", "failures": [run.stderr.strip()[-1500:] or f"exit {run.returncode}, no record"]}]
    return records, overrides


def has_model(repo, sha, model):
    try:
        git(repo, "cat-file", "-e", f"{sha}:src/transformers/models/{model}/__init__.py")
        return True
    except subprocess.CalledProcessError:
        return False


def replay(args):
    """--plan: each entry's contracts on its parent and bad refs; records carry `replay` {key, side, ref, sha, fleet}."""
    plan = json.loads(args.plan.read_text())[args.chunk :: args.chunks]
    print(f"{len(plan)} replays in chunk {args.chunk}/{args.chunks}", flush=True)
    for index, entry in enumerate(plan):
        contracts = [a for name in entry["contracts"] for a in ("--contract", name)]
        for side in ("parent", "bad"):
            start = time.perf_counter()
            try:
                sha = resolve(args.repo, entry[side])
                if side == "parent" and entry.get("model") and not has_model(args.repo, sha, entry["model"]):
                    records = [{"contract": None, "target": "*", "status": "absent",
                                "failures": [f"no src/transformers/models/{entry['model']} before this commit"]}]
                else:
                    records = []
                    for variant in entry["variants"]:
                        options = ["--fleet", entry["fleet"], "--fleet-revision", entry["fleet_revision"], "--variant", variant, *contracts]
                        records += run_commit(args.repo, sha, None, options, command=PR_CI_RUNNER)[0]
            except subprocess.CalledProcessError as error:
                sha = None
                records = [{"contract": "*", "target": "*", "status": "error",
                            "failures": [f"{entry[side]}: {(error.stderr or str(error)).strip()[-500:]}"]}]
            meta = {"key": entry["key"], "side": side, "ref": entry[side], "sha": sha, "fleet": entry["fleet"],
                    "run_id": os.environ.get("GITHUB_RUN_ID", "")}
            with args.output.open("a") as f:
                for record in records:
                    f.write(json.dumps({**record, "replay": meta}) + "\n")
            bad = [f"{r['target']}/{r.get('variant', '')}={r['status']}" for r in records if r["status"] not in ("pass", "xfail", "unsupported")]
            print(f"[{index + 1}/{len(plan)}] {entry['key']} {entry['fleet']} {side} {(sha or entry[side])[:10]} "
                  f"{time.perf_counter() - start:.0f}s: {'all pass' if not bad else ' '.join(bad)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--ref", default="main", help="branch whose first-parent history is walked")
    parser.add_argument("--plan", type=Path, help="JSON list of {key, model, fleet, fleet_revision, variants, contracts, parent, bad} to replay instead of a window")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.plan and not (args.since and args.until):
        parser.error("--since and --until, or --plan")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.plan:
        return replay(args)
    pinned = json.loads((HERE / "example/transformers/tests/contracts/fleet.lock").read_text())["environment"]
    shas = git(args.repo, "rev-list", "--first-parent", "--reverse", f"--since={args.since}", f"--until={args.until}", args.ref).split()
    mine = shas[args.chunk :: args.chunks]
    cpu, flags = cpu_info()
    print(f"{len(shas)} commits in {args.since}..{args.until}; chunk {args.chunk}/{args.chunks}: {len(mine)}", flush=True)
    for index, sha in enumerate(mine):
        subject = git(args.repo, "log", "-1", "--format=%s", sha).strip()
        date = git(args.repo, "log", "-1", "--format=%cI", sha).strip()
        changed = git(args.repo, "diff", "--name-only", f"{sha}^", sha)
        start = time.perf_counter()
        records, overrides = run_commit(args.repo, sha, pinned, changed=changed)
        meta = {
            "transformers_ref": "main", "transformers_sha": sha, "commit_subject": subject, "commit_date": date,
            "event": "backfill", "run_id": os.environ.get("GITHUB_RUN_ID", ""), "runner": os.environ.get("RUNNER_NAME", ""),
            "arch": platform.machine(), "cpu": cpu, "cpu_flags": flags,
            "changed_files_source": f"diff {sha[:10]}^..{sha[:10]}", "changed_files": len(changed.split()),
            "changed_files_truncated": False, "position": shas.index(sha), "pip_overrides": overrides,
        }
        with args.output.open("a") as f:
            for record in records:
                f.write(json.dumps({**record, "shadow": meta}) + "\n")
        bad = [f"{r['target']}={r['status']}" for r in records if r["status"] not in ("pass", "xfail")]
        print(f"[{index + 1}/{len(mine)}] {sha[:10]} {time.perf_counter() - start:.0f}s {subject[:60]!r}: "
              f"{'all pass' if not bad else ' '.join(bad)}", flush=True)


if __name__ == "__main__":
    main()
