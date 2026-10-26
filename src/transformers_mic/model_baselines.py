"""Stage one reviewed model, import cloud candidates, and bind validation evidence."""

import copy
import hashlib
import importlib
from pathlib import Path
import shutil
import sys
import tempfile

from huggingface_hub import HfApi

from . import resources
from .workspace import digest, read, write


def backend():
    sys.path.insert(0, str(resources.backend()))
    return importlib.import_module("run")


def inputs(workspace):
    runner = backend()
    lock = workspace.lock
    if lock.get("local_patches") or len(lock["contracts"]) != 1:
        raise ValueError("Cloud model baselines require one unpatched contract")
    contract = lock["contracts"][workspace.state["contract"]]
    if set(contract.get("fixtures", {})) != {
        workspace.fixture
    } or not workspace.fixture.endswith("-mic"):
        raise ValueError(
            "Migrate this model to its single <name>-mic repository before cloud measurement"
        )
    fixture = runner.source_folder(
        workspace.lock_path.parent,
        workspace.fixture,
        contract["fixtures"][workspace.fixture],
    )
    if problem := runner.fixture_problem(
        fixture, contract["fixtures"][workspace.fixture]
    ):
        raise ValueError(problem)
    files = runner.contract_files(workspace.contract_root)
    build = read(fixture / "build.json")
    source = {
        "contract": workspace.state["contract"],
        "repo": workspace.fixture,
        "files": files,
        "fixture": build["sha256"],
        "build": hashlib.sha256((fixture / "build.json").read_bytes()).hexdigest(),
        "checkpoints": contract["checkpoints"],
        "environment": lock["environment"],
        "targets": {
            k: v.get("target", k.split("/")[-1])
            for k, v in contract["fixtures"].items()
        },
    }
    return digest(source), fixture, files


def stage(workspace, root):
    """Upload only measurement inputs to a branch, preserving model main/fleet."""
    source, fixture, files = inputs(workspace)
    metadata = root / "model-input.json"
    if metadata.exists():
        previous = read(metadata)
        if previous["source"] != source:
            raise ValueError(
                "Model inputs changed; use a new baseline collection directory"
            )
        return previous["workspace"]
    repo = workspace.fixture
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    branch = "tmic-baselines/" + source
    api.create_branch(repo, branch=branch, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mic-cloud-input-") as directory:
        folder = Path(directory)
        for name in [*read(fixture / "build.json")["sha256"], "build.json"]:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe fixture path")
            target = folder / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fixture / path, target)
        for name in files:
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = (workspace.contract_root / name).read_bytes()
            if target.exists() and target.read_bytes() != content:
                raise ValueError("Fixture and contract have conflicting files")
            target.write_bytes(content)
        write(
            folder / ".mic-repo.json",
            {"format": "mic-umbrella-v1", "contract_files": sorted(files)},
        )
        if workspace.expectations.exists():
            shutil.copytree(
                workspace.expectations,
                folder / "integrations/transformers/expectations",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.candidate.json"),
            )
        lock = copy.deepcopy(workspace.lock)
        lock.pop("known_failures", None)
        contract = lock["contracts"][workspace.state["contract"]]
        contract["integration"].update(
            repo=repo, revision="unpublished", files_sha256=digest(files)
        )
        pin = contract["fixtures"][repo]
        pin.update(
            revision="unpublished",
            files_sha256=digest(read(fixture / "build.json")["sha256"]),
        )
        for pin in [contract["integration"], *contract["fixtures"].values()]:
            pin.pop("path", None)
        write(folder / ".mic/baseline-lock.json", lock)
        # Start each content-addressed staging branch from a clean tree. Extra
        # files from an earlier model build must not affect fixture verification.
        receipt = api.upload_folder(
            repo_id=repo,
            revision=branch,
            folder_path=folder,
            delete_patterns=["*"],
            commit_message="Stage reviewed MIC input for cloud baselines",
        )
    result = {"repo": repo, "revision": receipt.oid}
    write(metadata, {"source": source, "workspace": result})
    return result


def require_cpu_coverage(manifest, reports):
    """Transformers CI uses native low-precision x86 CPU baselines."""
    variants = manifest["integrations"]["transformers"]["variants"]
    for _, report in reports:
        if report["profile"] != "cpu":
            continue
        for case in report["cases"]:
            if case["status"] != "validated":
                continue
            dtype = variants[case["variant"]].get("dtype", "float32")
            required = {
                "bfloat16": "cpu-x86_64-bf16",
                "float16": "cpu-x86_64-fp16",
            }.get(dtype, "cpu-x86_64")
            if case["platform"] != required:
                raise ValueError(
                    f"Missing Transformers CI hardware coverage for {case.get('target', case.get('key', 'case'))}/{case['variant']}: "
                    f"measured {case['platform']}, requires {required}. "
                    "Measure on a CPU runner with the required native instructions; "
                    "do not copy or rename baselines from another hardware class."
                )


def import_candidates(workspace, root):
    from .baselines import collection

    source, _, _ = inputs(workspace)
    if read(root / "model-input.json")["source"] != source:
        raise ValueError("Model changed while measuring cloud baselines")
    state, reports = collection(root)
    require_cpu_coverage(workspace.manifest, reports)
    if state["job"].get("workspace") != read(root / "model-input.json")["workspace"]:
        raise ValueError("Cloud job measured a different model snapshot")
    candidates = {}
    for folder, report in reports:
        for case in report["cases"]:
            if case["contract"] != workspace.state["contract"]:
                raise ValueError("Cloud collection contains an unrelated contract")
            if case["status"] != "validated":
                continue
            candidate = folder / case["candidate"]
            relative = Path(candidate.parent.name) / candidate.name
            content = candidate.read_bytes()
            if str(relative) in candidates:
                raise ValueError("Duplicate cloud candidate")
            target = workspace.expectations / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            candidates[str(relative).replace(".candidate.json", ".json")] = (
                hashlib.sha256(content).hexdigest()
            )
    write(root / "model-candidates.json", {"source": source, "files": candidates})
    workspace.save_state(validated=False, validation=None, validations={})


def verify(workspace, root, *, save=True):
    from .baselines import collection

    state, reports = collection(root)
    require_cpu_coverage(workspace.manifest, reports)
    metadata = read(root / "model-input.json")
    if (
        state["request"]["profiles"] != ["cpu", "cuda"]
        or state["job"].get("workspace") != metadata["workspace"]
    ):
        raise ValueError(
            "Model validation needs both CPU/CUDA profiles for the staged input"
        )
    proof = read(root / "model-candidates.json")
    if proof["source"] != inputs(workspace)[0] or not proof["files"]:
        raise ValueError("Cloud evidence does not match the current model inputs")
    if list(workspace.expectations.rglob("*.candidate.json")):
        raise ValueError("Review every cloud candidate before validation")
    for name, expected in proof["files"].items():
        path = workspace.expectations / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Reviewed expectations differ from the cloud measurements")
    # Re-derive the receipt from the verified artifact, rather than trusting an
    # editable receipt to enlarge or reduce the required hardware matrix.
    expected_files = {}
    variants = set()
    for folder, report in reports:
        for case in report["cases"]:
            variants.add(case["variant"])
            if case["status"] != "validated":
                continue
            candidate = folder / case["candidate"]
            expected_files[
                str(
                    Path(candidate.parent.name)
                    / candidate.name.replace(".candidate.json", ".json")
                )
            ] = hashlib.sha256(candidate.read_bytes()).hexdigest()
            variants.add(case["variant"])
    if expected_files != proof["files"]:
        raise ValueError("Cloud candidate receipt does not cover the measured matrix")
    if save:
        manifest = workspace.manifest
        evidence = {
            "lock_sha256": hashlib.sha256(workspace.lock_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                (workspace.contract_root / "integration.json").read_bytes()
            ).hexdigest(),
            "usage_sha256": hashlib.sha256(
                (
                    workspace.contract_root
                    / manifest["integrations"]["transformers"]["entry_point"]
                ).read_bytes()
            ).hexdigest(),
            "cloud_collection": str(root),
        }
        declared = set(manifest["integrations"]["transformers"]["variants"])
        if variants != declared:
            raise ValueError("Cloud evidence is missing a declared variant")
        workspace.save_state(
            validated=True,
            validation=evidence,
            validations={
                variant: {**evidence, "fingerprint": workspace.validation_fingerprint()}
                for variant in variants
            },
        )
    return 0
