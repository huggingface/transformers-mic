"""Record and independently check every declared case on one cloud profile."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from huggingface_hub import hf_hub_download, snapshot_download

from . import fleet, cloud_inputs
from .cloud_progress import Progress
from .baselines import sha256
from .workspace import read, write


def safe(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError("Unsafe contract, target or variant name")
    return value


def expected_targets(contract, manifest, fixtures_only):
    integration = manifest["integrations"]["transformers"]
    unsupported = integration.get("unsupported", {})
    result = []
    if "fixtures" in integration["targets"]:
        result += [
            pin.get("target", repo.split("/")[-1])
            for repo, pin in contract.get("fixtures", {}).items()
        ]
    elif "fixtures" in unsupported:
        result += [
            contract.get("fixtures", {})
            .get(repo, {})
            .get("target", repo.split("/")[-1])
            for repo in manifest["fixtures"]
        ]
    if not fixtures_only and "checkpoints" in integration["targets"]:
        result += [
            repo.split("/")[-1]
            for repo in manifest["checkpoints"]
            if repo in contract.get("checkpoints", {}) or repo in unsupported
        ]
    if len(result) != len(set(result)):
        raise ValueError("Target short names collide in this contract")
    return [safe(target) for target in result]


def read_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def profile_exclusions(contract, manifest, variant, profile):
    resources = manifest["integrations"]["transformers"].get("resources", {})
    specs = [
        (pin.get("target", repo.split("/")[-1]), resources.get("fixture", {}))
        for repo, pin in contract.get("fixtures", {}).items()
    ]
    specs += [
        (repo.split("/")[-1], resources.get(repo, {}))
        for repo in manifest["checkpoints"]
    ]
    result = set()
    for target, spec in specs:
        spec = spec or {}
        spec = {**spec, **spec.get("variants", {}).get(variant, {})}
        devices = spec.get("devices", ["cpu", "cuda"])
        if profile not in devices or (profile == "cpu" and spec.get("gpus", 0)):
            result.add(target)
    return result


def assess(targets, variant, rows, verification, profile, declared_profile_skips=()):
    """An omitted, duplicate or hardware-skipped case is never a complete run."""
    by_target = {}
    for row in rows:
        if row.get("variant") != variant or row["target"] in by_target:
            raise ValueError("Unexpected variant or duplicate runner case")
        by_target[row["target"]] = row
    if set(by_target) != set(targets):
        raise ValueError("Runner omitted or added declared targets")
    checked = {row["target"]: row for row in verification}
    if (
        len(checked) != len(verification)
        or any(row.get("variant") != variant for row in verification)
        or set(checked) - set(targets)
    ):
        raise ValueError("Unexpected or duplicate verification case")
    result = []
    for target in targets:
        row = by_target[target]
        if row["status"] == "unsupported" and (
            row.get("skip") == "declared"
            or (row.get("skip") == "hardware" and target in declared_profile_skips)
        ):
            result.append(
                {
                    "target": target,
                    "variant": variant,
                    "status": "declared-unsupported",
                    "reason": row.get("failures", []),
                }
            )
            continue
        platform_name = row.get("platform", "")
        valid_platform = (
            platform_name == "cuda"
            if profile == "cuda"
            else platform_name.startswith("cpu-")
        )
        verification_row = checked.get(target, {})
        okay = (
            row["status"] == "recorded"
            and valid_platform
            and verification_row.get("status") == "pass"
            and verification_row.get("platform") == platform_name
        )
        result.append(
            {
                "target": target,
                "variant": variant,
                "platform": platform_name,
                "status": "validated" if okay else "failed",
                "record": row,
                "check": verification_row,
            }
        )
    return result


def runtime(profile):
    import torch

    if torch.get_num_threads() != 4:
        raise ValueError("Baseline profiles require OMP_NUM_THREADS=4")
    if profile == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA baseline profile has no GPU")
    if profile == "cpu" and torch.cuda.is_available():
        raise ValueError("CPU profile must hide CUDA devices")
    driver = None
    if profile == "cuda":
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    cpu = None
    try:
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            None,
        )
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "cpu": cpu,
        "driver": driver,
        "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if profile == "cuda" else None,
        "cuda": torch.version.cuda,
        "image": os.environ.get("MIC_BASELINE_IMAGE", ""),
        "packages": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines(),
    }


def execute(root, command, name):
    log = root / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "transformers_mic.backend",
                "run",
                str(root),
                *command,
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return process.returncode


def measure(args):
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = {
        "id": args.job_id,
        "fleet_revision": args.fleet_revision,
        "transformers_sha": args.transformers_sha,
        "mic_sha": args.mic_sha,
        "fleets": args.fleets.split(","),
        "fixtures_only": args.fixtures_only,
    }
    # Contracts left out on purpose (e.g. an open upstream bug makes them
    # unmeasurable); their published baselines are not touched.
    excluded = sorted({n for n in getattr(args, "exclude", "").split(",") if n})
    if excluded:
        request["exclude"] = excluded
    for key in ("fleet_revision", "transformers_sha", "mic_sha"):
        if not re.fullmatch(r"[0-9a-f]{40}", request[key]):
            raise ValueError("Cloud revisions must be immutable commit SHAs")
    workspace = cloud_inputs.from_env()
    if workspace:
        request["workspace"] = workspace

    def load_lock(name):
        if workspace:
            return cloud_inputs.lock(workspace)
        source, _ = fleet.download(name, args.fleet_revision)
        return read(source)

    report = {
        "request": request,
        "profile": args.profile,
        "complete": False,
        "cases": [],
        "expected_cases": [],
        "models": [],
        "contracts": [],
    }
    write(root / "collection.json", report)
    report["runtime"] = runtime(args.profile)
    import transformers

    installed = Path(transformers.__file__).resolve()
    checkout = Path(os.environ["MIC_TRANSFORMERS_CHECKOUT"]).resolve()
    if not installed.is_relative_to(checkout):
        raise ValueError(
            "Measured Transformers did not come from the requested checkout"
        )
    actual_sha = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_sha != args.transformers_sha:
        raise ValueError("Installed framework revision does not match the request")
    progress = Progress(args.job_id, args.profile)
    progress.emit("planning", 0, 0)
    for fleet_name in request["fleets"]:
        if fleet_name not in fleet.FILES:
            raise ValueError("Unknown fleet")
        for name, contract in load_lock(fleet_name)["contracts"].items():
            safe(name)
            if name in excluded:
                continue
            pin = contract["integration"]
            manifest = read(
                hf_hub_download(
                    pin["repo"], "integration.json", revision=pin["revision"]
                )
            )
            # Runs only see sources at these pins: run.py refuses files that differ
            # from the lock's files_sha256, so the artifact names them instead of
            # carrying copies (the collection checks them against the fleet).
            for kind, repo, source in [
                ("contracts", pin["repo"], pin),
                *(
                    ("models", repo, fixture)
                    for repo, fixture in contract.get("fixtures", {}).items()
                ),
            ]:
                entry = {
                    "repo": repo,
                    "revision": source["revision"],
                    "files_sha256": source.get("files_sha256"),
                }
                if entry not in report[kind]:
                    report[kind].append(entry)
            planned_variants = manifest["integrations"]["transformers"].get(
                "variants", {"default": {}}
            )
            if not isinstance(planned_variants, dict) or not planned_variants:
                raise ValueError("Contract declares no variants")
            planned_targets = expected_targets(contract, manifest, args.fixtures_only)
            if not planned_targets and not args.fixtures_only:
                raise ValueError("Contract declares no targets present in this fleet")
            for variant in planned_variants:
                safe(variant)
                report["expected_cases"].extend(
                    f"{fleet_name}/{name}/{target}/{variant}"
                    for target in planned_targets
                )
    total = len(report["expected_cases"])
    completed = 0
    write(root / "collection.json", report)
    for fleet_name in request["fleets"]:
        if fleet_name not in fleet.FILES:
            raise ValueError("Unknown fleet")
        lock = load_lock(fleet_name)
        if lock.get("local_patches"):
            raise ValueError("Cloud baselines require an unpatched fleet")
        lock.pop(
            "known_failures", None
        )  # candidates must satisfy every output assertion
        for name, contract in lock["contracts"].items():
            safe(name)
            if name in excluded:
                continue
            pin = contract["integration"]
            manifest = read(
                hf_hub_download(
                    pin["repo"], "integration.json", revision=pin["revision"]
                )
            )
            targets = expected_targets(contract, manifest, args.fixtures_only)
            variants = manifest["integrations"]["transformers"].get(
                "variants", {"default": {}}
            )
            folder = root / fleet_name / name
            folder.mkdir(parents=True, exist_ok=True)
            write(folder / "integration.json", manifest)
            reference = (
                Path(
                    snapshot_download(
                        pin["repo"],
                        revision=pin["revision"],
                        allow_patterns=["integrations/transformers/expectations/**"],
                    )
                )
                / "integrations/transformers/expectations"
            )
            if reference.exists():
                shutil.copytree(reference, folder / "references", dirs_exist_ok=True)
            single = {**lock, "contracts": {name: contract}}
            write(folder / "fleet.lock", single)
            for variant in variants:
                safe(variant)
                print(f"{args.profile}: {fleet_name}/{name}/{variant}", flush=True)
                expected = [
                    f"{fleet_name}/{name}/{target}/{variant}" for target in targets
                ]
                assert all(key in report["expected_cases"] for key in expected)
                # Store tolerance overrides next to candidates too; never loosen them.
                candidate_root = folder / "candidates"
                for override in (folder / "references").glob("*/tolerances.json"):
                    target = candidate_root / override.parent.name / override.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(override, target)
                output = folder / (variant + ".record.jsonl")
                options = [
                    "--framework",
                    "transformers",
                    "--lock",
                    str(folder / "fleet.lock"),
                    "--contract",
                    name,
                    "--variant",
                    variant,
                    "--isolate",
                    "--require-all",
                ]
                if args.fixtures_only:
                    options.append("--fixtures-only")
                progress.emit(
                    "recording", completed, total, f"{fleet_name}/{name}/{variant}"
                )
                record_code = execute(
                    root,
                    [
                        *options,
                        "--record",
                        "--expectations",
                        str(candidate_root),
                        "--output",
                        str(output),
                    ],
                    f"{fleet_name}-{name}-{variant}-record",
                )
                verification_root = folder / "verification"
                for candidate in candidate_root.glob("*/*.candidate.json"):
                    target = (
                        verification_root
                        / candidate.parent.name
                        / candidate.name.replace(".candidate.json", ".json")
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate, target)
                for override in candidate_root.glob("*/tolerances.json"):
                    target = verification_root / override.parent.name / override.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(override, target)
                progress.emit(
                    "validating", completed, total, f"{fleet_name}/{name}/{variant}"
                )
                checked = folder / (variant + ".check.jsonl")
                check_code = execute(
                    root,
                    [
                        *options,
                        "--expectations",
                        str(verification_root),
                        "--output",
                        str(checked),
                    ],
                    f"{fleet_name}-{name}-{variant}-check",
                )
                try:
                    cases = assess(
                        targets,
                        variant,
                        read_rows(output),
                        read_rows(checked),
                        args.profile,
                        profile_exclusions(contract, manifest, variant, args.profile),
                    )
                    declared_skip = any(
                        case["status"] == "declared-unsupported" for case in cases
                    )
                    if any(
                        code not in (0, 1) for code in (record_code, check_code)
                    ) or ((record_code or check_code) and not declared_skip):
                        raise ValueError(
                            f"Runner exited unsuccessfully: record={record_code}, check={check_code}"
                        )
                except ValueError as error:
                    cases = [
                        {
                            "target": target,
                            "variant": variant,
                            "status": "failed",
                            "error": str(error),
                        }
                        for target in targets
                    ]
                for case in cases:
                    case.update(
                        fleet=fleet_name,
                        contract=name,
                        repo=pin["repo"],
                        revision=pin["revision"],
                    )
                    case["key"] = f"{fleet_name}/{name}/{case['target']}/{variant}"
                    case["files"] = {}
                    if case["status"] == "validated":
                        suffix = "" if variant == "default" else "." + variant
                        candidate = (
                            candidate_root
                            / case["platform"]
                            / (case["target"] + suffix + ".candidate.json")
                        )
                        paths = [candidate, folder / "integration.json"]
                        paths += list(
                            (folder / "references").glob(f"{case['platform']}/*.json")
                        )
                        for path in paths:
                            case["files"][path.relative_to(root).as_posix()] = sha256(
                                path
                            )
                        case["candidate"] = candidate.relative_to(root).as_posix()
                    report["cases"].append(case)
                completed += len(cases)
                progress.emit(
                    "validating", completed, total, f"{fleet_name}/{name}/{variant}"
                )
                write(root / "collection.json", report)
    keys = [case["key"] for case in report["cases"]]
    report["complete"] = (
        bool(keys)
        and len(keys) == len(set(keys))
        and set(keys) == set(report["expected_cases"])
        and all(
            case["status"] in ("validated", "declared-unsupported")
            for case in report["cases"]
        )
    )
    write(root / "collection.json", report)
    progress.emit(
        "complete",
        completed,
        total,
        "Validated collection"
        if report["complete"]
        else "Incomplete collection; inspect failed cases",
    )
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "job-id",
        "fleet-revision",
        "transformers-sha",
        "mic-sha",
        "fleets",
        "out",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--profile", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--exclude", default="")
    return measure(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
