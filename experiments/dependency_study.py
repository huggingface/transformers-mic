"""Check frozen contracts under dependency upgrades; never record baselines.

Run on the GPU worker after install_pinned.sh. Each fleet pass starts a new
process; the observer saves outputs and numeric drift without changing the
contract execution or comparison. Package versions are explicit, not latest.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "example"))
import run

PROFILES = {
    "pinned": [],
    "tokenizers": ["tokenizers==0.23.2"],
    "numpy": ["numpy==2.2.6"],
    "pillow": ["pillow==12.3.0"],
    "combined": ["tokenizers==0.23.2", "numpy==2.2.6", "pillow==12.3.0"],
    "no-torchaudio": [],
    "torch-cu130": ["torch==2.14.1", "torchvision==0.29.1"],
}


def frozen_hashes():
    folder = ROOT / "example/transformers/tests/contracts"
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def observe(variant, out, contract=None):
    import torch

    if contract is None and not torch.cuda.is_available():
        raise RuntimeError("GPU study cannot fall back to CPU")
    observations = []
    execute, compare = run.Executor.run, run.compare

    def execution(executor, env, fresh_process=False):
        result = execute(executor, env, fresh_process)
        rows = [json.loads(line) for line in result[1].splitlines() if line.startswith("{")]
        observations.append({"entry": str(executor.entry), "model": env["MIC_MODEL"],
                             "outputs": rows[-1] if rows else None, "numeric_drift": {}})
        return result

    def comparison(rules, actual, expected, required=()):
        drift = {}
        for name, rule in rules.items():
            a, b = run.lookup(actual, name), run.lookup(expected, name)
            if rule["compare"] != "allclose" or a is None or b is None:
                continue
            if run.structure(a) != run.structure(b):
                drift[name] = {"shape_changed": True}
                continue
            pairs = list(zip(run.flatten(a), run.flatten(b)))
            drift[name] = {
                "max_absolute_delta": max((abs(x - y) for x, y in pairs), default=0),
                "max_tolerance_fraction": max((abs(x - y) / (rule["atol"] + rule["rtol"] * abs(y))
                    if rule["atol"] + rule["rtol"] * abs(y) else (0 if x == y else float("inf"))
                    for x, y in pairs), default=0),
            }
        observations[-1]["numeric_drift"] = drift
        return compare(rules, actual, expected, required)

    run.Executor.run, run.compare = execution, comparison
    sys.argv = ["run.py", "--framework", "transformers", "--variant", variant, "--require-all"]
    if contract:
        sys.argv += ["--contract", contract]
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            code = run.main()
    finally:
        run.Executor.run, run.compare = execute, compare
    records = [json.loads(line) for line in stdout.getvalue().splitlines() if line.startswith("{")]
    executed = [r for r in records if r["status"] != "unsupported"]
    if len(executed) != len(observations):
        raise RuntimeError(f"execution/record mismatch: {len(observations)} / {len(executed)}")
    for record, observation in zip(executed, observations):
        observation.update(contract=record["contract"], target=record["target"], kind=record["kind"])
    out.write_text(json.dumps({"exit_code": code, "records": records, "observations": observations}, indent=2) + "\n")
    return 0  # A completed experiment may find contract failures.


def study(out, profiles=None):
    out.mkdir(parents=True, exist_ok=True)
    hashes = frozen_hashes()
    lock = json.loads((ROOT / "example/transformers/tests/contracts/fleet.lock").read_text())
    pins = [f"{p}=={lock['environment'][p]}" for p in ("tokenizers", "numpy", "pillow")]
    meta = {"transformers_sha": subprocess.check_output(["git", "-C", "/tmp/transformers", "rev-parse", "HEAD"], text=True).strip(),
            "harness_sha": os.environ.get("GITHUB_SHA") or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "run_id": os.environ.get("GITHUB_RUN_ID"), "profiles": {k: v for k, v in PROFILES.items() if profiles is None or k in profiles},
            "frozen_hashes": hashes, "passes": []}
    pip = [sys.executable, "-m", "pip"]
    for profile, packages in meta["profiles"].items():
        folder = out / profile
        folder.mkdir(exist_ok=True)
        with (folder / "install.log").open("w") as log:
            subprocess.run([*pip, "install", "--quiet", "--no-deps", *pins], stdout=log, stderr=log, check=True)
            if profile in ("no-torchaudio", "torch-cu130"):
                subprocess.run([*pip, "uninstall", "-y", "torchaudio"], stdout=log, stderr=log, check=True)
            if packages:
                options = ["--index-url", "https://download.pytorch.org/whl/cu130", "--upgrade"] if profile == "torch-cu130" else ["--no-deps"]
                subprocess.run([*pip, "install", "--quiet", *options, *packages], stdout=log, stderr=log, check=True)
        (folder / "pip-freeze.txt").write_text(subprocess.check_output([*pip, "freeze"], text=True))
        with (folder / "pip-check.txt").open("w") as log:
            subprocess.run([*pip, "check"], stdout=log, stderr=log)
        study_env = dict(os.environ)
        if profile == "torch-cu130":
            import importlib.metadata as md
            # cu12/cu13 distributions can own the same library path. The image
            # already has cu13 metadata, so pip may leave files overwritten by
            # installing the pinned cu12 control. Restore the actual wheel bytes.
            cuda_requirements = [r for r in md.distribution("torch").requires if r.startswith("nvidia-")]
            with (folder / "restore-cuda-wheels.log").open("w") as log:
                subprocess.run([*pip, "install", "--quiet", "--force-reinstall", "--no-deps", *cuda_requirements], stdout=log, stderr=log, check=True)
            from packaging.requirements import Requirement
            library_dirs = sorted({str(md.distribution(name).locate_file(path).parent)
                                  for name in (Requirement(r).name for r in cuda_requirements)
                                  for path in md.distribution(name).files if ".so" in str(path) and "/stubs/" not in str(path)})
            study_env["LD_LIBRARY_PATH"] = ":".join([*library_dirs, os.environ.get("LD_LIBRARY_PATH", "")])
            (folder / "library-path.txt").write_text(study_env["LD_LIBRARY_PATH"] + "\n")
            with (folder / "backend-probe.log").open("w") as log:
                subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"], env=study_env, stdout=log, stderr=log, check=True)
        for variant in ("default", "eager", "bfloat16"):
            for repeat in (1, 2):
                name = f"{variant}-{repeat}"
                started = time.monotonic()
                with (folder / f"{name}.log").open("w") as log:
                    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--observe", variant,
                                             "--out", str(folder / f"{name}.json")], stdout=log, stderr=log, cwd=ROOT, env=study_env)
                meta["passes"].append({"profile": profile, "variant": variant, "repeat": repeat,
                                       "seconds": round(time.monotonic() - started, 2), "harness_exit": result.returncode})
                (out / "study.json").write_text(json.dumps(meta, indent=2) + "\n")
                print(meta["passes"][-1], flush=True)
        assert frozen_hashes() == hashes, "fleet inputs, expectations or tolerances changed"
    return int(any(p["harness_exit"] for p in meta["passes"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observe", choices=("default", "eager", "bfloat16"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, help="run only these environments")
    parser.add_argument("--contract", help="one contract for observer validation")
    args = parser.parse_args()
    raise SystemExit(observe(args.observe, args.out, args.contract) if args.observe else study(args.out.resolve(), args.profiles))
