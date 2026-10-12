"""Replay a historical Transformers regression through the contract runner.

    python example/replay.py d7ba937671 --contract T5                 # UMT5 non-causal SDPA (#49135)
    python example/replay.py <fix-sha> [--bad <sha>] [run.py options]

Extracts the bad build (default <fix>^) and the fixed build from
$TRANSFORMERS_REPO with git archive, never checking anything out, into
$REPLAY_WORKDIR/builds, then runs example/run.py unchanged on each build with
its src first on PYTHONPATH (followed by $OVERLAY for older dependency pins).
Fixtures, contracts, and baselines are the reviewed ones in this repository;
nothing is rebuilt or re-recorded. A target counts as caught when the bad
build fails and the fixed build passes.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
REPO = Path(os.environ.get("TRANSFORMERS_REPO", Path.home() / "Dev/transformers"))
WORKDIR = Path(os.environ.get("REPLAY_WORKDIR", Path(tempfile.gettempdir()) / "model-integration-replay"))


def sha(ref):
    return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short=10", ref], text=True).strip()


def build(ref):
    folder = WORKDIR / "builds" / ref
    if not (folder / "src").exists():
        folder.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "-C", REPO, "archive", ref, "src"], capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", folder], input=archive.stdout, check=True)
    return folder / "src"


def run(ref, options):
    path = [str(build(ref)), os.environ.get("OVERLAY", ""), os.environ.get("PYTHONPATH", "")]
    env = {**os.environ, "PYTHONPATH": ":".join(p for p in path if p)}
    out = subprocess.run(
        [sys.executable, str(HERE / "run.py"), "--framework", "transformers", *options],
        capture_output=True, text=True, env=env,
    )
    records = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    if not records:
        raise SystemExit(f"run.py produced no records on {ref}:\n{out.stderr[-1500:]}")
    return {r["target"]: r for r in records}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fix")
    parser.add_argument("--bad", help="defaults to <fix>^")
    args, options = parser.parse_known_args()
    fix, bad = sha(args.fix), sha(args.bad or f"{args.fix}^")
    results = {"bad": run(bad, options), "fix": run(fix, options)}
    print(json.dumps({"fix": fix, "bad": bad, "options": options}))
    for target, on_bad in results["bad"].items():
        on_fix = results["fix"].get(target, {})
        verdict = (
            "caught" if on_bad["status"] == "fail" and on_fix.get("status") == "pass"
            else "unsupported" if on_bad["status"] == "unsupported"
            else "clean" if on_bad["status"] == on_fix.get("status") == "pass"
            else "inconclusive"
        )
        print(json.dumps({
            "target": target,
            "verdict": verdict,
            "bad": on_bad["status"],
            "fix": on_fix.get("status"),
            "bad_failures": [f[:200] for f in on_bad.get("failures", [])],
            "fix_failures": [f[:200] for f in on_fix.get("failures", [])],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
