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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--until", required=True)
    parser.add_argument("--ref", default="main", help="branch whose first-parent history is walked")
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shas = git(args.repo, "rev-list", "--first-parent", "--reverse", f"--since={args.since}", f"--until={args.until}", args.ref).split()
    mine = shas[args.chunk :: args.chunks]
    cpu, flags = cpu_info()
    pinned = json.loads((HERE / "example/transformers/tests/contracts/fleet.lock").read_text())["environment"]
    print(f"{len(shas)} commits in {args.since}..{args.until}; chunk {args.chunk}/{args.chunks}: {len(mine)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, sha in enumerate(mine):
        git(args.repo, "checkout", "--quiet", "--force", sha)
        subject = git(args.repo, "log", "-1", "--format=%s", sha).strip()
        date = git(args.repo, "log", "-1", "--format=%cI", sha).strip()
        changed = git(args.repo, "diff", "--name-only", f"{sha}^", sha)
        overrides = dependency_overrides(args.repo, sha, pinned)
        meta = {
            "transformers_ref": "main", "transformers_sha": sha, "commit_subject": subject, "commit_date": date,
            "event": "backfill", "run_id": os.environ.get("GITHUB_RUN_ID", ""), "runner": os.environ.get("RUNNER_NAME", ""),
            "arch": platform.machine(), "cpu": cpu, "cpu_flags": flags,
            "changed_files_source": f"diff {sha[:10]}^..{sha[:10]}", "changed_files": len(changed.split()),
            "changed_files_truncated": False, "position": shas.index(sha), "pip_overrides": overrides,
        }
        with tempfile.TemporaryDirectory() as tmp:
            changed_path, raw = Path(tmp) / "changed.txt", Path(tmp) / "raw.jsonl"
            changed_path.write_text(changed)
            start = time.perf_counter()
            run = subprocess.run(
                [sys.executable, str(HERE / "example/run.py"), "--framework", "transformers", "--fixtures-only",
                 "--changed-files", str(changed_path), "--run-all", "--output", str(raw)],
                capture_output=True, text=True, cwd=HERE,
            )
            records = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
        if not records:
            records = [{"contract": "*", "framework": "transformers", "variant": "default", "target": "*", "kind": "fixture",
                        "status": "error", "failures": [run.stderr.strip()[-1500:] or f"exit {run.returncode}, no record"]}]
        with args.output.open("a") as f:
            for record in records:
                f.write(json.dumps({**record, "shadow": meta}) + "\n")
        bad = [f"{r['target']}={r['status']}" for r in records if r["status"] not in ("pass", "xfail")]
        print(f"[{index + 1}/{len(mine)}] {sha[:10]} {time.perf_counter() - start:.0f}s {subject[:60]!r}: "
              f"{'all pass' if not bad else ' '.join(bad)}", flush=True)


if __name__ == "__main__":
    main()
