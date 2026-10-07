"""List bug fixes in a contract's code that none of its results noticed: candidate usage paths to add.

    python example/find_misses.py --repo ~/Dev/transformers results/*.jsonl
    git -C shadow-results show ... | python example/find_misses.py --repo /tmp/transformers -

Reads shadow or backfill result records (JSON lines, with run metadata under
`shadow`) and a Transformers clone holding those commits. A commit is a miss
candidate for a contract when

  - its subject says it fixes something (fix, bug, regression, broken, fall
    back, restore, correct, prevent),
  - it changes the contract's code: a model directory the contract's fixtures
    use (as select_contracts.py finds them; tier "model"), or with --shared
    also user-facing shared code (tokenization, processing, image processing,
    feature extraction, generation, pipelines, model loading; tier "shared",
    much noisier since it matches every contract), and
  - no result of that contract changed at that commit (same status and
    failures as on the commit before it, in first-parent order).

A fix that changes nothing the contract checks means the bug it fixes was
invisible to the contract while it existed: either the contract does not make
the affected call, or the fixture cannot reach the path. Each candidate is a
question for the contract owner (see docs/improving-contracts.md), not a
verdict: many fixes touch code the contract has no reason to call.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import run
import select_contracts

HERE = Path(__file__).parent
FIX = re.compile(r"\b(fix(es|ed)?|bug|regression|broken|fall back|restore[sd]?|correct(s|ed)?|prevent(s|ed)?)\b", re.I)
SHARED = re.compile(
    r"^src/transformers/(tokenization_utils\w*|processing_utils|image_processing\w*|image_utils|feature_extraction\w*"
    r"|audio_utils|generation/|pipelines/|modeling_utils|core_model_loading|conversion_mapping|models/auto/)"
)


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="JSON lines files, or - for stdin")
    parser.add_argument("--repo", required=True, help="Transformers clone containing the commits")
    parser.add_argument("--framework", default="transformers")
    parser.add_argument("--shared", action="store_true", help="also list fixes that only touch shared code")
    args = parser.parse_args()

    records = {}
    for name in args.results:
        lines = sys.stdin if name == "-" else open(name)
        for line in lines:
            r = json.loads(line)
            records[(r["shadow"]["transformers_sha"], r["contract"], r["target"])] = r
    order = git(args.repo, "rev-list", "--first-parent", "--reverse", "HEAD").split()
    seen = {sha for sha, _, _ in records}
    order = [sha for sha in order if sha in seen]

    lock_dir = HERE / args.framework / "tests/contracts"
    lock = run.load(lock_dir / "fleet.lock", "fleet-lock")
    dirs = {
        contract["integration"]["repo"]: {
            d for repo, pin in contract.get("fixtures", {}).items()
            for d in select_contracts.code_dirs(run.fixture_folder(lock_dir, repo, pin, False))
        }
        for contract in lock["contracts"].values()
    }
    outcome = defaultdict(dict)  # contract -> sha -> {target: (status, failures)}
    for (sha, contract, target), r in records.items():
        outcome[contract].setdefault(sha, {})[target] = (r["status"], tuple(f.split(":")[0] for f in r.get("failures", [])))

    found = {}
    for previous, sha in zip(order, order[1:]):
        subject = git(args.repo, "log", "-1", "--format=%s", sha).strip()
        if not FIX.search(subject):
            continue
        files = git(args.repo, "diff", "--name-only", f"{sha}^", sha).split()
        for contract, code in dirs.items():
            model = [f for f in files if f.startswith("src/transformers/models/") and f.split("/")[3] in code]
            shared = [f for f in files if SHARED.match(f)]
            tier = "model" if model else "shared" if shared and args.shared else None
            if tier is None or contract not in outcome:
                continue
            if outcome[contract].get(sha) == outcome[contract].get(previous):
                found[tier] = found.get(tier, 0) + 1
                print(json.dumps({"tier": tier, "contract": contract.split("/")[-1], "commit": sha[:10], "subject": subject, "files": (model or shared)[:6]}))
    print(json.dumps({"miss_candidates": found, "commits": len(order)}), file=sys.stderr)


if __name__ == "__main__":
    main()
