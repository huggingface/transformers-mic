"""Run one MIC from a fleet lock and compare it with its reviewed baseline.

    python example/run.py --framework transformers --target slim
    python example/run.py --framework transformers --target real --variant eager
    python example/run.py --framework transformers.js --target real
    python example/run.py ... --record   # write a candidate baseline for review

A sketch of the shared runner: it resolves pinned inputs from the lock, runs the
integration as a separate process, reads the last stdout line, applies the
manifest's comparison rules, and prints a result record. --record never touches
an approved baseline; it writes <target>.candidate.json next to it, and a
reviewer renames it.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent


def resolve(lock, manifest, framework, target):
    """Return (MODEL_ID, MODEL_REVISION) for a target, from pinned inputs only."""
    contract = lock["contracts"][0]
    integration = manifest["integrations"][framework]
    if target in integration.get("unsupported", {}):
        raise SystemExit(f"unsupported: {framework} {target}: {integration['unsupported'][target]}")
    artifact = integration["artifacts"][target]
    if artifact == "fixture":
        return str((HERE / contract["fixture"]["path"]).resolve()), ""
    if artifact == "model":
        return contract["model"]["repo"], contract["model"]["revision"]
    return artifact["repo"], contract["artifacts"][artifact["repo"]]


def environment(language, entry):
    if language == "javascript":
        node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
        package = entry.parent / "node_modules/@huggingface/transformers/package.json"
        return {"node": node, "@huggingface/transformers": json.loads(package.read_text())["version"]}
    # Ask the libraries, not package metadata: an editable install keeps the
    # version it had when it was installed.
    probe = (
        "import importlib, json, platform\n"
        "versions = {'python': platform.python_version()}\n"
        "for name in ('transformers', 'torch', 'tokenizers', 'numpy'):\n"
        "    try:\n"
        "        versions[name] = importlib.import_module(name).__version__\n"
        "    except ImportError:\n"
        "        versions[name] = None\n"
        "print(json.dumps(versions))\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout
    return json.loads(out)


def lookup(outputs, dotted):
    value = outputs
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def flatten(value):
    if isinstance(value, list):
        return [x for item in value for x in flatten(item)]
    return [value]


def compare(rules, actual, expected):
    failures = []
    for name, rule in rules.items():
        got, want = lookup(actual, name), lookup(expected, name)
        if want is None and got is None:
            continue  # not produced by this framework (e.g. no training in Transformers.js)
        if got is None:
            failures.append(f"{name}: missing")
            continue
        kind = rule["compare"]
        if kind == "exact" and got != want:
            failures.append(f"{name}: {got!r} != {want!r}")
        elif kind == "allclose":
            a, b = flatten(got), flatten(want)
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in a):
                failures.append(f"{name}: non-finite values")
            elif len(a) != len(b):
                failures.append(f"{name}: shape {len(a)} != {len(b)}")
            else:
                worst = max(abs(x - y) - rule["atol"] - rule["rtol"] * abs(y) for x, y in zip(a, b))
                if worst > 0:
                    delta = max(abs(x - y) for x, y in zip(a, b))
                    failures.append(f"{name}: max delta {delta:.3g} over tolerance")
        elif kind == "all_true" and not all(got.values()):
            failures.append(f"{name}: {[k for k, v in got.items() if not v]} false")
        elif kind == "true" and got is not True:
            failures.append(f"{name}: {got!r}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lock", type=Path, default=HERE / "fleet.lock")
    parser.add_argument("--framework", required=True)
    parser.add_argument("--target", choices=("slim", "real"), required=True)
    parser.add_argument("--variant", default="default")
    parser.add_argument("--record", action="store_true", help="write a candidate baseline for review")
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text())
    contract = lock["contracts"][0]
    root = HERE / contract["integration"]["path"]
    manifest = json.loads((root / "integration.json").read_text())
    integration = manifest["integrations"][args.framework]
    model_id, revision = resolve(lock, manifest, args.framework, args.target)
    variant = integration["variants"][args.variant]
    entry = root / integration["entry_point"]
    command = [sys.executable, entry.name] if integration["language"] == "python" else ["node", entry.name]

    start = time.perf_counter()
    run = subprocess.run(
        command,
        cwd=entry.parent,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MODEL_ID": model_id,
            "MODEL_REVISION": revision,
            "MODEL_KWARGS": json.dumps(variant),
        },
    )
    seconds = time.perf_counter() - start
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    record = {
        "contract": contract["integration"]["repo"],
        "contract_revision": contract["integration"]["revision"],
        "framework": args.framework,
        "target": args.target,
        "variant": args.variant,
        "model": {"id": model_id if args.target == "real" else contract["fixture"]["repo"], "revision": revision or contract["fixture"]["revision"]},
        "environment": environment(integration["language"], entry),
        "seconds": round(seconds, 2),
    }
    if run.returncode or not lines:
        record.update(status="error", failures=[run.stderr.strip()[-1000:]])
        print(json.dumps(record, indent=1))
        return 1
    actual = json.loads(lines[-1])

    # Each variant has its own reviewed baseline: eager and SDPA need not agree
    # within tolerance outside float32.
    template = manifest["expectations"]["default" if args.variant == "default" else "variant"]
    baseline = root / template.format(framework=args.framework, target=args.target, variant=args.variant)
    if args.record:
        candidate = baseline.with_name(baseline.name.removesuffix(".json") + ".candidate.json")
        candidate.write_text(json.dumps({"recorded_with": record, "outputs": actual}, indent=1) + "\n")
        print(f"candidate baseline written to {candidate.relative_to(HERE)}; review it, then rename it to {baseline.name}")
        return 0
    if integration["training"] and "training" not in actual:
        record.update(status="fail", failures=["training: missing"])
    else:
        expected = json.loads(baseline.read_text())["outputs"]
        failures = compare(manifest["outputs"], actual, expected)
        record.update(status="fail" if failures else "pass", failures=failures)
    print(json.dumps(record, indent=1))
    return record["status"] != "pass"


if __name__ == "__main__":
    sys.exit(main())
