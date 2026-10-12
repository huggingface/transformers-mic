"""Review candidate baselines against the approved ones, and approve them.

    python example/review_candidates.py --framework transformers                 # report only
    python example/review_candidates.py --framework transformers --platform cpu-arm64
    python example/review_candidates.py --framework transformers --approve       # rename the clean ones

For each <target>[.<variant>].candidate.json under expectations/, compare it
with the approved baseline next to it, using the contract's own comparison
rules (manifest plus the platform's tolerances.json):

  unchanged  every output the approved baseline has still matches; outputs
             only the candidate has are listed as added (a contract that
             gained a check), outputs only the baseline has as removed
  changed    an existing output no longer matches: needs a reason before
             approval (a framework change, an environment change, a bug)
  new        no approved baseline yet

--approve renames unchanged and new candidates to approved baselines and
leaves changed ones for a person to decide. It never edits outputs.
"""

import argparse
import json
from pathlib import Path

import run

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", default="transformers")
    parser.add_argument("--platform", help="only this platform's candidates")
    parser.add_argument("--approve", action="store_true", help="approve unchanged and new candidates")
    args = parser.parse_args()
    root = HERE / args.framework / "tests/contracts/expectations"
    counts = {}
    for candidate in sorted(root.glob("*/*/*.candidate.json")):
        contract, platform = candidate.parent.parent.name, candidate.parent.name
        if args.platform and platform != args.platform:
            continue
        approved = candidate.with_name(candidate.name.replace(".candidate.json", ".json"))
        manifest = run.load(HERE / "hub" / f"{contract}-integration/integration.json", "integration")
        new = run.load(candidate, "baseline")["outputs"]
        if not approved.exists():
            verdict, detail = "new", sorted(k for k in manifest["outputs"] if run.lookup(new, k) is not None)
        else:
            old = run.load(approved, "baseline")["outputs"]
            rules = run.rules_for(manifest, candidate.parent)
            shared = {k: r for k, r in rules.items() if run.lookup(old, k) is not None}
            failures = run.compare(shared, new, old)
            added = sorted(k for k in rules if run.lookup(old, k) is None and run.lookup(new, k) is not None)
            removed = sorted(k for k in rules if run.lookup(old, k) is not None and run.lookup(new, k) is None)
            verdict = "changed" if failures or removed else "unchanged"
            detail = failures + [f"added: {k} = {json.dumps(run.lookup(new, k))[:120]}" for k in added] + [f"removed: {k}" for k in removed]
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"{verdict:9} {contract}/{platform}/{candidate.name.removesuffix('.candidate.json')}")
        for line in detail:
            print(f"          {line}")
        if args.approve and verdict in ("unchanged", "new"):
            candidate.rename(approved)
    print(json.dumps({"summary": counts, "approved": args.approve}))
    return 1 if counts.get("changed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
