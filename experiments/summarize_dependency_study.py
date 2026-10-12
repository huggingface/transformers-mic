"""Summarize GPU dependency checks and repeats without approving baselines."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))
import run


def summarize(folder):
    meta = json.loads((folder / "study.json").read_text())
    report = {"transformers_sha": meta["transformers_sha"], "harness_sha": meta["harness_sha"],
              "run_url": f"https://github.com/huggingface/model-integration-contracts/actions/runs/{meta['run_id']}",
              "requested_profiles": meta["profiles"], "frozen_hashes": meta["frozen_hashes"], "profiles": {}}
    for profile in meta["profiles"]:
        root = folder / profile
        versions = {}
        for line in (root / "pip-freeze.txt").read_text().splitlines():
            if "==" in line:
                name, version = line.split("==", 1)
                versions[name.lower()] = version
        info = {"packages": versions, "pip_check": (root / "pip-check.txt").read_text().strip(),
                "counts": Counter(), "repeats": Counter(), "max_absolute_delta": 0,
                "max_tolerance_fraction": 0, "checks": [], "timings": {}}
        failed = [p for p in meta["passes"] if p["profile"] == profile and p["harness_exit"]]
        if failed:
            info["harness_failures"] = failed
            info["failure_log"] = (root / f"{failed[0]['variant']}-{failed[0]['repeat']}.log").read_text()[-3000:]
            report["profiles"][profile] = info
            continue
        for variant in ("default", "eager", "bfloat16"):
            samples = [json.loads((root / f"{variant}-{n}.json").read_text()) for n in (1, 2)]
            times = [p["seconds"] for p in meta["passes"] if p["profile"] == profile and p["variant"] == variant]
            assert len(times) == 2, (profile, variant, times)
            info["timings"][variant] = {"fleet_seconds": times, "median": statistics.median(times)}
            records = [{(r["contract"], r["target"]): r for r in s["records"]} for s in samples]
            assert records[0].keys() == records[1].keys(), (profile, variant, "target set differs")
            observations = [{(o["contract"], o["target"]): o for o in s["observations"]} for s in samples]
            for sample in samples:
                assert all(o["outputs"] is None or o["outputs"]["runtime"]["device"] == "cuda"
                           for o in sample["observations"]), "GPU study fell back to CPU"
                info["counts"].update(r["status"] for r in sample["records"])
            for key, first in records[0].items():
                pair = [r[key] for r in records]
                row = {"contract": key[0], "target": key[1], "kind": first["kind"], "variant": variant,
                       "statuses": [r["status"] for r in pair], "failures": [r.get("failures", []) for r in pair]}
                if first["status"] == "unsupported":
                    assert all(r.get("skip") == "declared" for r in pair), row
                    row["repeat"] = "declared_exclusion"
                    info["checks"].append(row)
                    continue
                obs = [o.get(key, {}) for o in observations]
                outputs = [o.get("outputs") for o in obs]
                drifts = [v for o in obs for v in o.get("numeric_drift", {}).values()]
                row["max_absolute_delta"] = max((d.get("max_absolute_delta", 0) for d in drifts), default=0)
                row["max_tolerance_fraction"] = max((d.get("max_tolerance_fraction", 0) for d in drifts), default=0)
                info["max_absolute_delta"] = max(info["max_absolute_delta"], row["max_absolute_delta"])
                info["max_tolerance_fraction"] = max(info["max_tolerance_fraction"], row["max_tolerance_fraction"])
                if any(o is None for o in outputs):
                    row["repeat"] = "unavailable"
                else:
                    name = key[0].rsplit("/", 1)[-1].removesuffix("-integration")
                    manifest = run.load(ROOT / "example/hub" / f"{name}-integration/integration.json", "integration")
                    rules = run.rules_for(manifest, ROOT / "example/transformers/tests/contracts/expectations" / name / "cuda")
                    repeat_rules = {k: ({"compare": "exact"} if r["compare"] in ("true", "all_true") else r)
                                    for k, r in rules.items()}
                    differences = [] if outputs[0] == outputs[1] else run.compare(repeat_rules, outputs[1], outputs[0])
                    row["repeat_failures"] = differences
                    row["repeat"] = "changed" if differences else "exact" if outputs[0] == outputs[1] else "within_tolerance"
                info["repeats"][row["repeat"]] += 1
                info["checks"].append(row)
        report["profiles"][profile] = info
    report["notes"] = [
        "Approved baselines, fleet pins and tolerances were kept fixed; hashes checked after each environment.",
        "Each fleet pass starts a fresh process and batches its targets; two passes per variant and environment.",
        "Timings include process startup, cached Hub lookup and all targets; these are not cold/warm cost measurements.",
        "PyTorch profile upgrades torch/torchvision and CUDA libraries together; torchaudio is absent because no matching wheel exists.",
        "The no-torchaudio control identifies coverage lost before the PyTorch/CUDA upgrade.",
        "Absolute drift mixes output scales; tolerance fraction measures distance to the actual per-output comparison limit.",
    ]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.folder)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, profile in report["profiles"].items():
        print(name, dict(profile["counts"]), dict(profile["repeats"]),
              "max tolerance fraction", profile["max_tolerance_fraction"])
