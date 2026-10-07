"""Run a framework's MICs from its fleet lock and compare with its own baselines.

    python example/run.py --framework transformers                  # every target, default variant
    python example/run.py --framework transformers --variant eager
    python example/run.py --framework transformers --target SmolLM-slim
    python example/run.py --framework transformers.js
    python example/run.py ... --record                              # candidate baselines for review

A sketch of the shared runner. For each contract in
<framework>/tests/contracts/fleet.lock it reads the manifest from the pinned
integration repo, expands the targets (each slim fixture once, then each
checkpoint), runs the contract in its own process, and compares the result with
<framework>/tests/contracts/expectations/<contract>/<platform>/<target>[.<variant>].json.

Runnable-markdown contracts are collected with doc-builder's own block
collector, so the runner executes exactly what the docs build tests and renders
(requires doc-builder from main). --record never touches an approved baseline;
it writes <name>.candidate.json next to it, and a reviewer renames it.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
_DECORATOR = re.compile(r"^\s*#\s*pytest-decorator:.*$", re.MULTILINE)


def short(repo):
    return repo.split("/")[-1]


def targets(lock_dir, contract, manifest, framework):
    """Yield (target, model, revision) for every target, or (target, None, reason) if unsupported."""
    integration = manifest["integrations"][framework]
    unsupported = integration.get("unsupported", {})
    if "fixtures" in integration["targets"]:
        for repo, pin in contract.get("fixtures", {}).items():
            yield short(repo), str((lock_dir / pin["path"]).resolve()), pin["revision"]
    elif "fixtures" in unsupported:
        for repo in manifest["fixtures"]:
            yield short(repo), None, unsupported["fixtures"]
    for repo in manifest["checkpoints"]:
        if repo in unsupported:
            yield short(repo), None, unsupported[repo]
        elif repo not in contract["checkpoints"]:
            continue  # not in this framework's lock
        elif framework == "transformers.js":
            artifact = integration["artifacts"][repo]["repo"]
            yield short(repo), artifact, contract["artifacts"][artifact]
        else:
            # Pin by path: the runner resolves the locked revision, the contract
            # just loads what it is given.
            from huggingface_hub import snapshot_download

            revision = contract["checkpoints"][repo]
            files = ["*.json", "*.txt", "*.safetensors"]
            yield short(repo), snapshot_download(repo, revision=revision, allow_patterns=files), revision


def command_for(entry, fmt, workdir):
    if fmt == "script":
        return ["node", entry.name], entry.parent
    from doc_builder.testing import DocIntegrationTest

    blocks = DocIntegrationTest._collect_runnable_blocks_from_text(entry.read_text())
    script = workdir / f"{entry.stem}.py"
    script.write_text("\n\n".join(_DECORATOR.sub("", block.code) for block in blocks))
    return [sys.executable, str(script)], entry.parent


def environment(fmt, cwd):
    if fmt == "script":
        node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
        package = cwd / "node_modules/@huggingface/transformers/package.json"
        return {"node": node, "@huggingface/transformers": json.loads(package.read_text())["version"]}
    probe = (
        "import importlib, json, platform\n"
        "versions = {'python': platform.python_version()}\n"
        "for name in ('transformers', 'torch', 'tokenizers', 'numpy'):\n"
        "    versions[name] = importlib.import_module(name).__version__\n"
        "print(json.dumps(versions))\n"
    )
    return json.loads(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout)


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
                failures.append(f"{name}: {len(a)} values, baseline has {len(b)}")
            elif any(abs(x - y) > rule["atol"] + rule["rtol"] * abs(y) for x, y in zip(a, b)):
                failures.append(f"{name}: max delta {max(abs(x - y) for x, y in zip(a, b)):.3g} over tolerance")
        elif kind == "all_true" and not all(got.values()):
            failures.append(f"{name}: {[k for k, v in got.items() if not v]} false")
        elif kind == "true" and got is not True:
            failures.append(f"{name}: {got!r}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", choices=("transformers", "transformers.js"), required=True)
    parser.add_argument("--target", help="run only this target (fixture or checkpoint name)")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--record", action="store_true", help="write candidate baselines for review")
    args = parser.parse_args()

    lock_dir = HERE / args.framework / "tests/contracts"
    lock = json.loads((lock_dir / "fleet.lock").read_text())
    status = 0
    for name, contract in lock["contracts"].items():
        root = (lock_dir / contract["integration"]["path"]).resolve()
        manifest = json.loads((root / "integration.json").read_text())
        integration = manifest["integrations"][args.framework]
        variant = integration["variants"][args.variant]
        entry = root / integration["entry_point"]
        with tempfile.TemporaryDirectory() as tmp:
            command, cwd = command_for(entry, integration["format"], Path(tmp))
            env_info = environment(integration["format"], cwd)
            for target, model, revision in targets(lock_dir, contract, manifest, args.framework):
                if args.target and target != args.target:
                    continue
                record = {"contract": contract["integration"]["repo"], "target": target, "variant": args.variant}
                if model is None:
                    print(json.dumps({**record, "status": "unsupported", "reason": revision}))
                    continue
                start = time.perf_counter()
                run = subprocess.run(
                    command,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "MIC_RUNNER": "1",
                        "MIC_MODEL": model,
                        "MIC_MODEL_REVISION": "" if os.path.isdir(model) else revision,
                        "MIC_MODEL_KWARGS": json.dumps(variant),
                    },
                )
                record.update(
                    revision=revision,
                    framework=args.framework,
                    contract_revision=contract["integration"]["revision"],
                    environment=env_info,
                    seconds=round(time.perf_counter() - start, 2),
                )
                lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
                if run.returncode or not lines:
                    print(json.dumps({**record, "status": "error", "failures": [run.stderr.strip()[-800:]]}))
                    status = 1
                    continue
                actual = json.loads(lines[-1])
                platform = actual["runtime"]["device"]
                suffix = "" if args.variant == "default" else f".{args.variant}"
                baseline = lock_dir / "expectations" / name / platform / f"{target}{suffix}.json"
                if args.record:
                    candidate = baseline.with_name(f"{target}{suffix}.candidate.json")
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    candidate.write_text(json.dumps({"recorded_with": record, "outputs": actual}, indent=1) + "\n")
                    print(json.dumps({**record, "status": "recorded", "candidate": str(candidate.relative_to(HERE))}))
                    continue
                if integration["training"] and "training" not in actual:
                    failures = ["training: missing"]
                elif not baseline.exists():
                    failures = [f"no reviewed baseline at {baseline.relative_to(HERE)}"]
                else:
                    failures = compare(manifest["outputs"], actual, json.loads(baseline.read_text())["outputs"])
                print(json.dumps({**record, "status": "fail" if failures else "pass", "failures": failures}))
                status |= bool(failures)
    return status


if __name__ == "__main__":
    sys.exit(main())
