"""Round-trip published fleet contracts through usage_template, numerically.

For each contract, records the approved usage.md and the page rendered from its
extracted slots on this machine, on its slim fixtures, and compares the two
recorded outputs. Baselines are not needed: both pages run here, so any difference
is the template's.

    python tools/usage_template_fleet_roundtrip.py --out work/roundtrip [--contract X ...] [--sample N]
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from huggingface_hub import snapshot_download  # noqa: E402

from transformers_mic import fleet, usage_template  # noqa: E402

USAGE = "integrations/transformers/usage.md"


def local_copy(repo, revision, destination):
    if not destination.exists():
        shutil.copytree(snapshot_download(repo, revision=revision), destination, symlinks=False)
    return destination


def record(lock_path, name, expectations, variant):
    out = expectations / "out.jsonl"
    subprocess.run(
        [sys.executable, str(ROOT / "example/run.py"), "--framework", "transformers", "--lock", str(lock_path),
         "--contract", name, "--fixtures-only", "--local-contracts", "--record", "--expectations", str(expectations),
         "--variant", variant, "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines() if '"target"' in line] if out.exists() else []
    candidates = {
        path.name.removesuffix(".candidate.json"): json.loads(path.read_text()).get("outputs")
        for path in expectations.rglob("*.candidate.json")
    }
    return {row["target"]: row["status"] for row in rows}, candidates, [row.get("contract_revision") for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", action="append")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--variant", default="default")
    args = parser.parse_args()
    contracts = {}
    for name in fleet.FILES:
        contracts.update(json.loads(Path(fleet.download(name)[0]).read_text())["contracts"])
    names = args.contract or sorted(contracts)
    if args.sample:
        names = sorted(random.Random(17).sample(names, args.sample))
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        entry = contracts[name]
        work = args.out / name
        try:
            pages = {}
            for kind in ("approved", "rendered"):
                lock = {"framework": "transformers", "environment": {}, "contracts": {name: json.loads(json.dumps(entry))}}
                pin = lock["contracts"][name]["integration"]
                for repo, fixture in lock["contracts"][name]["fixtures"].items():
                    fixture["path"] = str(local_copy(repo, fixture["revision"], work / kind / repo.replace("/", "--")))
                pin["path"] = str(local_copy(pin["repo"], pin["revision"], work / kind / pin["repo"].replace("/", "--")))
                usage = Path(pin["path"]) / USAGE
                if kind == "rendered":
                    approved = usage.read_text()
                    checkpoint = next(iter(json.loads((Path(pin["path"]) / "integration.json").read_text())["checkpoints"]))
                    usage.write_text(usage_template.render(usage_template.extract(approved), checkpoint))
                lock_path = work / f"{kind}.lock"
                lock_path.write_text(json.dumps(lock, indent=1))
                expectations = work / f"{kind}-expectations"
                shutil.rmtree(expectations, ignore_errors=True)
                pages[kind] = record(lock_path, name, expectations, args.variant)
            (a_status, a_out, _), (r_status, r_out, r_labels) = pages["approved"], pages["rendered"]
            if not a_out:
                verdict = f"approved page records nothing here ({a_status})"
            elif a_out == r_out and a_status == r_status:
                verdict = "same"
            else:
                verdict = "DIFFERENT"
            results[name] = {"verdict": verdict, "approved": a_status, "rendered": r_status, "labels": r_labels}
        except Exception as error:  # noqa: BLE001 - report and continue
            results[name] = {"verdict": f"error: {type(error).__name__}: {str(error)[:200]}"}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        print(name, json.dumps(results[name]), flush=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    verdicts = [r["verdict"] for r in results.values()]
    print(json.dumps({v: verdicts.count(v) for v in sorted(set(verdicts))}, indent=1))


if __name__ == "__main__":
    main()
