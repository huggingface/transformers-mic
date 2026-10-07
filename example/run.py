"""Run a framework's MICs from its fleet lock and compare with its own baselines.

    python example/run.py --framework transformers                  # every contract, target, default variant
    python example/run.py --framework transformers --contract T5 --variant eager
    python example/run.py --framework transformers --target t5-small-slim
    python example/run.py --framework transformers --fixtures-only  # what a framework PR runs
    python example/run.py --framework transformers --isolate        # one process per target
    python example/run.py --framework transformers.js
    python example/run.py ... --record                              # candidate baselines for review
    python example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu
    python example/run.py ... --output results.jsonl                # one result record per line

For each contract in <framework>/tests/contracts/fleet.lock it reads the
manifest from the pinned integration repo, expands the targets (each slim
fixture once, then each checkpoint), runs the contract, and compares the result
with <framework>/tests/contracts/expectations/<contract>/<platform>/<target>[.<variant>].json.
The manifest, fleet lock, fixture builds, baselines, tolerances, and result
records are validated against the JSON schemas in schemas/.

Runnable-markdown contracts are collected with doc-builder's own block
collector (doc-builder from main), so the runner executes exactly what the docs
build tests and renders. They run in this process, one after the other, so
imports are paid once for the whole run; --isolate runs each target in its own
process instead. Script contracts (Transformers.js) always run in their own process.

Fixtures are checked against the file hashes in their build.json, which the
fleet lock pins; a fixture that is missing or differs is an error, not a skip
(build it with build_fixtures.py). Comparison tolerances come from the
manifest, overridden per platform by expectations/<contract>/<platform>/tolerances.json,
since GPU and lower-precision runs drift more than CPU float32.

Non-finite numeric outputs and failed assertions (all_true, true) fail on
their own, and are never recorded;
--record also reruns each target in a fresh process and refuses outputs that
differ. A
framework can list a target under known_failures in its fleet lock (with the
issue): a failure is then reported as xfail, and a pass as a failure, so the
entry is removed once the bug is fixed. local_patches in the fleet lock name
workarounds from patches.py, applied before every contract (in this process and
in isolated ones) and recorded in each result's environment.
--record never touches an approved baseline; it writes <name>.candidate.json
next to it, and a reviewer renames it. --golden <platform> checks a run against
another platform's reviewed baseline: discrete outputs must match exactly;
logits are reported as max delta, not compared.
"""

import argparse
import contextlib
import gc
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from functools import cache
from pathlib import Path

HERE = Path(__file__).parent
SCHEMAS = HERE / "schemas"
_DECORATOR = re.compile(r"^\s*#\s*pytest-decorator:.*$", re.MULTILINE)
WEIGHT_PATTERNS = ["*.msgpack", "*.h5", "*.ot", "*.onnx", "onnx/*", "*.tflite", "flax_model*", "tf_model*", "rust_model*"]


@cache
def schema(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text())


def validate(value, name, where):
    import jsonschema

    try:
        jsonschema.validate(value, schema(name))
    except jsonschema.ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
        raise SystemExit(f"{where}: invalid {name} at {path}: {exc.message}") from None
    return value


def load(path, name):
    return validate(json.loads(Path(path).read_text()), name, path)


def short(repo):
    return repo.split("/")[-1]


def digest(files):
    """One hash for a fixture's {file: sha256} map, as pinned in the fleet lock."""
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def file_hashes(folder):
    skip = ("README.md", "build.json")
    return {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(folder.iterdir())
        if f.is_file() and f.name not in skip
    }


def fixture_problem(folder, pin):
    """Why this fixture folder cannot be used for the pinned build, or None."""
    build_file = folder / "build.json"
    hint = f"build it with: python example/build_fixtures.py --fixture {short(str(folder))}"
    if not build_file.exists():
        return f"fixture not built ({hint})"
    build = load(build_file, "fixture-build")
    if pin.get("files_sha256") and digest(build["sha256"]) != pin["files_sha256"]:
        return f"build.json does not match the fleet lock pin ({hint})"
    actual = file_hashes(folder)
    if actual != build["sha256"]:
        changed = sorted(set(actual) ^ set(build["sha256"]) | {k for k in actual if actual[k] != build["sha256"].get(k)})
        return f"fixture files differ from build.json: {changed} ({hint})"
    return None


def checkpoint_path(repo, revision):
    """Local snapshot of a pinned checkpoint with its weights (downloaded if needed)."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    cached = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    if cached.is_dir() and any(f.suffix in (".safetensors", ".bin") for f in cached.iterdir()):
        return str(cached)
    files = HfApi().list_repo_files(repo, revision=revision)
    ignore = WEIGHT_PATTERNS + (["*.bin", "*.pt"] if any(f.endswith(".safetensors") for f in files) else [])
    return snapshot_download(repo, revision=revision, ignore_patterns=ignore)


def targets(lock_dir, contract, manifest, framework, fixtures_only=False):
    """Yield (target, kind, model, revision, problem) for every target of a contract."""
    integration = manifest["integrations"][framework]
    unsupported = integration.get("unsupported", {})
    if "fixtures" in integration["targets"]:
        for repo, pin in contract.get("fixtures", {}).items():
            folder = (lock_dir / pin["path"]).resolve()
            yield short(repo), "fixture", str(folder), pin["revision"], fixture_problem(folder, pin)
    elif "fixtures" in unsupported:
        for repo in manifest["fixtures"]:
            yield short(repo), "fixture", None, None, unsupported["fixtures"]
    if fixtures_only:
        return
    for repo in manifest["checkpoints"]:
        if repo in unsupported:
            yield short(repo), "checkpoint", None, None, unsupported[repo]
        elif repo not in contract["checkpoints"]:
            continue  # not in this framework's lock
        elif framework == "transformers.js":
            artifact = integration["artifacts"][repo]["repo"]
            yield short(repo), "checkpoint", artifact, contract["artifacts"][artifact], None
        else:
            # Pin by path: the runner resolves the locked revision, the contract
            # just loads what it is given.
            revision = contract["checkpoints"][repo]
            yield short(repo), "checkpoint", lambda r=repo, v=revision: checkpoint_path(r, v), revision, None


def contract_code(entry):
    from doc_builder.testing import DocIntegrationTest

    blocks = DocIntegrationTest._collect_runnable_blocks_from_text(entry.read_text())
    return "\n\n".join(_DECORATOR.sub("", block.code) for block in blocks)


@contextlib.contextmanager
def environ(values):
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def chdir(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def execute_inline(code, filename, env):
    """Run contract code in this process; return (ok, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    namespace = {"__name__": "__mic_contract__", "__file__": filename}
    ok = True
    with environ(env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, filename, "exec"), namespace)
        except BaseException:  # noqa: BLE001 - a contract failure is a result, not a runner crash
            traceback.print_exc(file=err)
            ok = False
    namespace.clear()
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return ok, out.getvalue(), err.getvalue()


def patch_prelude(names):
    """Code that applies the fleet lock's local patches in a contract's own process."""
    if not names:
        return ""
    return f"import sys  # mic: local patches\nsys.path.insert(0, {str(HERE)!r})\nimport patches\npatches.apply({sorted(names)!r})\n\n"


class Executor:
    """Runs one contract's code for many targets: in this process, or one process each."""

    def __init__(self, entry, fmt, isolate, workdir, patch_names=()):
        self.entry, self.fmt = entry, fmt
        self.inline = fmt == "runnable-markdown" and not isolate
        if fmt == "script":
            self.command, self.cwd = ["node", entry.name], entry.parent
        else:
            self.code = contract_code(entry)
            script = workdir / f"{entry.parent.parent.parent.name}-{entry.stem}.py"
            script.write_text(patch_prelude(patch_names) + self.code)
            self.command, self.cwd = [sys.executable, str(script)], entry.parent

    def run(self, env, fresh_process=False):
        if self.inline and not fresh_process:
            with chdir(self.cwd):
                return execute_inline(self.code, str(self.entry), env)
        run = subprocess.run(self.command, cwd=self.cwd, capture_output=True, text=True, env={**os.environ, **env})
        return run.returncode == 0, run.stdout, run.stderr

    def environment(self):
        if self.fmt == "script":
            node = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
            package = self.cwd / "node_modules/@huggingface/transformers/package.json"
            return {"node": node, "@huggingface/transformers": json.loads(package.read_text())["version"]}
        probe = (
            "import importlib, json, platform\n"
            "versions = {'python': platform.python_version()}\n"
            "for name in ('transformers', 'torch', 'tokenizers', 'numpy'):\n"
            "    versions[name] = importlib.import_module(name).__version__\n"
            "print(json.dumps(versions))\n"
        )
        if self.inline:
            ok, out, err = execute_inline(probe, "<environment>", {})
            return json.loads(out)
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


def violated(rules, actual):
    """The contract's own assertions (all_true, true) that fail, with or without a baseline."""
    return compare({k: r for k, r in rules.items() if r["compare"] in ("all_true", "true")}, actual, actual)


def non_finite(rules, actual):
    """allclose outputs holding NaN or inf: a failure with or without a baseline."""
    return [
        f"{name}: non-finite values"
        for name, rule in rules.items()
        if rule["compare"] == "allclose"
        and lookup(actual, name) is not None
        and not all(isinstance(x, (int, float)) and math.isfinite(x) for x in flatten(lookup(actual, name)))
    ]


def rules_for(manifest, platform_dir):
    """The manifest's comparison rules with this platform's tolerance overrides."""
    rules = {name: dict(rule) for name, rule in manifest["outputs"].items()}
    overrides = platform_dir / "tolerances.json"
    if overrides.exists():
        for name, tolerance in load(overrides, "tolerances")["outputs"].items():
            if rules.get(name, {}).get("compare") != "allclose":
                raise SystemExit(f"{overrides}: {name} is not an allclose output of this contract")
            rules[name].update(tolerance)
    return rules


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", choices=("transformers", "transformers.js"), required=True)
    parser.add_argument("--contract", action="append", help="run only this contract (repeatable)")
    parser.add_argument("--target", action="append", help="run only this target, fixture or checkpoint name (repeatable)")
    parser.add_argument("--fixtures-only", action="store_true", help="skip real checkpoints")
    parser.add_argument("--variant", default="default")
    parser.add_argument("--isolate", action="store_true", help="run each target in its own process")
    parser.add_argument("--record", action="store_true", help="write candidate baselines for review")
    parser.add_argument("--golden", metavar="PLATFORM", help="check discrete outputs against this platform's baseline")
    parser.add_argument("--output", type=Path, help="append result records to this JSON lines file")
    args = parser.parse_args()

    lock_dir = HERE / args.framework / "tests/contracts"
    lock = load(lock_dir / "fleet.lock", "fleet-lock")
    started = time.perf_counter()
    counts = {}
    status = 0

    known = lock.get("known_failures", {})
    local_patches = sorted(lock.get("local_patches", {}))
    if local_patches and args.framework == "transformers":
        import patches

        patches.apply(local_patches)

    def emit(record):
        nonlocal status
        reason = known.get(f"{record['contract'].split('/')[-1].removesuffix('-integration')}/{record['target']}")
        if reason and not args.record and not args.golden:
            if record["status"] in ("fail", "error"):
                record = {**record, "status": "xfail", "known_failure": reason}
            elif record["status"] == "pass":
                record = {**record, "status": "fail", "failures": [f"listed in known_failures but passes; remove it: {reason}"]}
        validate(record, "result", f"result for {record['contract']}/{record['target']}")
        print(json.dumps(record), flush=True)
        if args.output:
            with args.output.open("a") as f:
                f.write(json.dumps(record) + "\n")
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        status |= record["status"] in ("fail", "error")

    with tempfile.TemporaryDirectory() as tmp:
        for name, contract in lock["contracts"].items():
            if args.contract and name not in args.contract:
                continue
            root = (lock_dir / contract["integration"]["path"]).resolve()
            manifest = load(root / "integration.json", "integration")
            integration = manifest["integrations"][args.framework]
            base = {
                "contract": contract["integration"]["repo"],
                "contract_revision": contract["integration"]["revision"],
                "framework": args.framework,
                "variant": args.variant,
            }
            variant = integration["variants"].get(args.variant)
            executor = Executor(root / integration["entry_point"], integration["format"], args.isolate, Path(tmp), local_patches)
            env_info = None
            for target, kind, model, revision, problem in targets(lock_dir, contract, manifest, args.framework, args.fixtures_only):
                if args.target and target not in args.target:
                    continue
                record = {**base, "target": target, "kind": kind, "revision": revision}
                if variant is None:
                    emit({**record, "status": "unsupported", "failures": [f"variant {args.variant!r} not defined"]})
                    continue
                if problem:
                    emit({**record, "status": "unsupported" if model is None else "error", "failures": [problem]})
                    continue
                if callable(model):
                    model = model()
                env_info = env_info or executor.environment()
                start = time.perf_counter()
                env = {
                    "MIC_RUNNER": "1",
                    "MIC_MODEL": model,
                    "MIC_MODEL_REVISION": "" if os.path.isdir(model) else revision,
                    "MIC_MODEL_KWARGS": json.dumps(variant),
                }
                ok, stdout, stderr = executor.run(env)
                environment = {**env_info, "local_patches": local_patches} if local_patches else env_info
                record.update(environment=environment, seconds=round(time.perf_counter() - start, 2))
                lines = [line for line in stdout.splitlines() if line.startswith("{")]
                if not ok or not lines:
                    emit({**record, "status": "error", "failures": [stderr.strip()[-1500:] or "no result record"]})
                    continue
                actual = json.loads(lines[-1])
                platform = actual["runtime"]["device"]
                record["platform"] = platform
                suffix = "" if args.variant == "default" or args.golden else f".{args.variant}"
                platform_dir = lock_dir / "expectations" / name / (args.golden or platform)
                baseline = platform_dir / f"{target}{suffix}.json"
                record["baseline"] = str(baseline.relative_to(HERE))
                if args.golden:
                    if not baseline.exists():
                        emit({**record, "status": "error", "failures": [f"no golden baseline at {record['baseline']}"]})
                        continue
                    expected = load(baseline, "baseline")["outputs"]
                    exact = {k: r for k, r in manifest["outputs"].items() if r["compare"] == "exact" and k != "runtime"}
                    failures = compare(exact, actual, expected)
                    deltas = {
                        k: max(abs(x - y) for x, y in zip(flatten(lookup(actual, k)), flatten(lookup(expected, k))))
                        for k, r in manifest["outputs"].items()
                        if r["compare"] == "allclose" and lookup(actual, k) is not None
                    }
                    emit({**record, "golden": args.golden, "status": "fail" if failures else "pass", "failures": failures, "max_delta": deltas})
                    continue
                broken = non_finite(manifest["outputs"], actual) + violated(manifest["outputs"], actual)
                if broken:
                    # Never record non-finite outputs or failed assertions: a baseline would approve the bug.
                    emit({**record, "status": "fail", "failures": broken})
                    continue
                if args.record:
                    ok2, stdout2, _ = executor.run(env, fresh_process=True)  # must reproduce outside this process
                    again = [line for line in stdout2.splitlines() if line.startswith("{")]
                    # Same rules as a check: exact outputs equal, numeric ones within this platform's tolerance.
                    unstable = compare(rules_for(manifest, platform_dir), json.loads(again[-1]), actual) if again else ["no result record"]
                    if not ok2 or unstable:
                        emit({**record, "status": "fail", "failures": [f"not recorded: a fresh-process rerun differs: {unstable}"]})
                        continue
                    candidate = baseline.with_name(f"{target}{suffix}.candidate.json")
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    recorded = {k: v for k, v in record.items() if k != "baseline"}
                    candidate.write_text(json.dumps(validate({"recorded_with": recorded, "outputs": actual}, "baseline", candidate), indent=1) + "\n")
                    emit({**record, "status": "recorded", "baseline": str(candidate.relative_to(HERE))})
                    continue
                if integration["training"] and "training" not in actual:
                    failures = ["training: missing"]
                elif not baseline.exists():
                    failures = [f"no reviewed baseline at {record['baseline']}"]
                else:
                    reviewed = load(baseline, "baseline")
                    identity = reviewed["recorded_with"]
                    mismatch = [k for k in ("target", "variant", "framework") if identity.get(k) != record[k]]
                    if identity.get("platform", platform) != platform:
                        mismatch.append("platform")
                    failures = [f"baseline identity differs: {mismatch}"] if mismatch else []
                    failures += compare(rules_for(manifest, platform_dir), actual, reviewed["outputs"])
                emit({**record, "status": "fail" if failures else "pass", "failures": failures})
    summary = {"summary": counts, "seconds": round(time.perf_counter() - started, 1), "isolate": args.isolate}
    print(json.dumps(summary), file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
