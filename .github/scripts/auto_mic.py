"""Automatic MIC creation for tmic-run.yml: draft with read credentials, publish with the trusted CLI only.

  ready    (run job, read-only token) after `tmic run --mode agent --local-only`: decide whether
           the draft can go to the cloud. Ready = validated, or stopped at `record` only because
           this CPU runner lacks hardware a declared target needs (the cloud measures it).
  publish  (publish job, mic-publish environment) measure every declared CPU/CUDA case in the
           fleet images, approve NEW candidates only, verify, publish the model, re-verify and add
           it to the prototypes fleet. It installs no runtime and runs no model code: the model
           code runs on the cloud runners, with a read token.

Automatic publication is for new contracts only: a model already in a fleet, or a Hub repo that
already has content, needs the reviewed update flow. Changed baselines are never approved here.
When only the real checkpoint fails in the cloud (jetmoe-8b ran out of the 22 GB GPU in its
training step) and every fixture case recorded, the contract is published for its fixtures only.
Each phase writes <phase>-result.json, uploaded as an artifact.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Measured cases that do not stop a draft: the cloud runners have the hardware.
HARDWARE_SKIPS = {"hardware", "declared", "wrong-hardware"}


def tmic(*args):
    subprocess.run(
        [sys.executable, "-m", "transformers_mic", *map(str, args)], check=True
    )


def records(log):
    """The JSON result lines of a tmic record/check log."""
    out = []
    for line in Path(log).read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def hardware_gap_only(log):
    """True when every case recorded or was skipped for hardware, and at least one recorded."""
    cases = [r for r in records(log) if "status" in r and "summary" not in r]
    recorded = [r for r in cases if r["status"] == "recorded"]
    others = [r for r in cases if r["status"] != "recorded"]
    return bool(recorded) and all(
        r["status"] == "unsupported" and r.get("skip") in HARDWARE_SKIPS for r in others
    )


def checkpoint_only_failure(collection):
    """True when a cloud collection failed only on real checkpoints, and some fixture case recorded."""
    cases = [
        r
        for log in sorted(Path(collection).glob("artifacts/*/logs/*-record.log"))
        for r in records(log)
        if "status" in r and "summary" not in r
    ]
    failed = [r for r in cases if r["status"] not in ("recorded", "unsupported")]
    return (
        bool(failed)
        and all(r.get("kind") == "checkpoint" for r in failed)
        and any(r.get("kind") == "fixture" and r["status"] == "recorded" for r in cases)
    )


def fixtures_only(workspace):
    """Declare the fixtures as the contract's only target, as tmic create --fixtures-only does."""
    from transformers_mic.workspace import read, write

    path = workspace.contract_root / "integration.json"
    manifest = read(path)
    manifest["integrations"]["transformers"]["targets"] = ["fixtures"]
    write(path, manifest)


def measure(args, collection):
    tmic(
        "baselines",
        "run",
        "--model-workspace",
        args.workspace,
        "--fleet",
        "prototypes",
        "--out",
        collection,
        "--transport",
        "github",
        "--transformers-ref",
        args.transformers_ref,
    )


def readiness(result, workspace_root):
    """(ready, reason) for a `tmic run --mode agent --local-only` result.json."""
    status = result.get("status")
    if status == "validated":
        return True, "validated on this runner"
    question = result.get("pending_question") or {}
    context = question.get("context") or {}
    if (
        status == "needs-input"
        and question.get("kind") == "retry"
        and context.get("failed_stage") == "record"
    ):
        log = Path(workspace_root) / ".mic/automation/logs/record.log"
        if log.is_file() and hardware_gap_only(log):
            return (
                True,
                "every CPU case recorded; the rest need hardware this runner lacks",
            )
        return False, "record failed on this runner"
    stage = (
        context.get("failed_stage")
        or context.get("stage")
        or question.get("kind")
        or "?"
    )
    return False, f"tmic run stopped: {status} at {stage}"


def ready(args):
    from transformers_mic.workspace import Workspace, write

    result = json.loads(Path(args.result).read_text())
    ok, reason = readiness(result, args.workspace)
    if not ok:
        raise SystemExit(f"Not ready for cloud measurement: {reason}")
    workspace = Workspace(args.workspace)
    if workspace.state["model"] != args.model:
        raise SystemExit("The workspace belongs to another checkpoint")
    write(
        Path(args.workspace) / ".mic/automatic-source-review.json",
        {
            "policy": "Operator enabled automatic MIC creation and publication",
            "contract_sha256": workspace.contract_digest(),
            "run": os.environ.get("GITHUB_RUN_ID"),
        },
    )
    return {"status": "ready", "reason": reason}


def require_new(workspace, contract_name=None, overwrite_unexported=False):
    """Refuse a contract that is in a fleet, or a Hub repo that already has content.

    contract_name: the fleet name to export under (default: the workspace's contract).
    overwrite_unexported: accept a Hub repo with content that no fleet references (published once,
    never exported). Returns True when that repo will be overwritten.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    from transformers_mic import fleet
    from transformers_mic.workspace import read

    name = contract_name or workspace.state["contract"]
    for fleet_name in ("control", "prototypes"):
        path, _ = fleet.download(fleet_name)
        for key, entry in read(path)["contracts"].items():
            if (
                key.casefold() == name.casefold()
                or entry["integration"]["repo"] == workspace.fixture
            ):
                raise ValueError(
                    f"Already in the {fleet_name} fleet as {key} ({entry['integration']['repo']}); "
                    "pick another contract_name or use the reviewed update flow"
                )
    try:
        files = HfApi().list_repo_files(workspace.fixture, revision="main")
    except RepositoryNotFoundError:
        files = []
    if any(f not in ("README.md", ".gitattributes") for f in files):
        if overwrite_unexported:
            return True
        raise ValueError(
            f"{workspace.fixture} already has published content that no fleet references. If an earlier "
            "run of this workflow published it and then failed to export, download its "
            "tmic-cloud-workspace artifact and run `tmic --workspace work/run export --fleet prototypes "
            "--execute` from it; to replace it, dispatch again with overwrite_unexported."
        )
    return False


def clear_expectations(repo):
    """Delete an unexported repo's old expectations before overwriting it.

    publish removes only the contract and fixture files it owned; a baseline the new measurement
    does not produce would otherwise stay in the snapshot, and the runner reads every one.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    folders = sorted(
        {
            f.split("/expectations/")[0] + "/expectations"
            for f in api.list_repo_files(repo)
            if "/expectations/" in f
        }
    )
    for folder in folders:
        api.delete_folder(
            folder,
            repo_id=repo,
            commit_message=f"Clear {folder} before republishing (never exported)",
        )
    return folders


def export_new(workspace, contract_name=None, attempts=8):
    """Add the contract to prototypes; retry only when another export landed first (409 / changed)."""
    from huggingface_hub.errors import HfHubHTTPError

    from transformers_mic import commands, fleet
    from transformers_mic.workspace import read

    pin = workspace.lock["contracts"][workspace.state["contract"]]["integration"]
    name = contract_name or workspace.state["contract"]
    for attempt in range(attempts):
        path, revision = fleet.download("prototypes")
        existing = read(path)["contracts"].get(name)
        if existing:
            if (
                existing["integration"]["repo"] == pin["repo"]
                and existing["integration"]["revision"] == pin["revision"]
            ):
                return revision  # an earlier attempt landed
            raise ValueError(
                f"{name} appeared in the fleet during publication; refusing to overwrite it"
            )
        options = SimpleNamespace(
            workspace=str(workspace.root),
            fleet="prototypes",
            contract_name=contract_name,
            expected_revision=revision,
            execute=True,
            transformers_checkout=None,
        )
        try:
            # export revalidates every receipt and keeps the other entries (parent_commit).
            commands.export(options)
            return fleet.resolve("main")
        except HfHubHTTPError as error:
            if (
                error.response is None
                or error.response.status_code != 409
                or attempt == attempts - 1
            ):
                raise
        except ValueError as error:
            if (
                "Dataset changed since preview" not in str(error)
                or attempt == attempts - 1
            ):
                raise
        time.sleep(min(2**attempt, 30))
    raise RuntimeError("unreachable")


def publish(args):
    from transformers_mic.workspace import Workspace

    workspace = Workspace(args.workspace)
    overwrite = require_new(workspace, args.contract_name, args.overwrite_unexported)
    # The draft recorded on a laptop-like CPU runner; the fleet's references come from the CI
    # images. Keep the draft's candidates for inspection, measure afresh.
    if workspace.expectations.exists():
        shutil.move(
            str(workspace.expectations), workspace.root / ".mic/draft-expectations"
        )
    collection, scope = args.collection, "all"
    try:
        measure(args, collection)
    except subprocess.CalledProcessError:
        if not checkpoint_only_failure(collection):
            raise
        print("auto_mic: only the real checkpoint failed; publishing for the fixtures only", file=sys.stderr)
        fixtures_only(workspace)
        collection, scope = f"{args.collection}-fixtures", "fixtures"
        measure(args, collection)
    tmic("--workspace", args.workspace, "review", "--approve")  # NEW and unchanged only
    tmic(
        "baselines",
        "verify-model",
        "--model-workspace",
        args.workspace,
        "--out",
        collection,
    )
    # Again: the collection can take hours.
    overwrite = require_new(workspace, args.contract_name, args.overwrite_unexported)
    if overwrite:
        clear_expectations(workspace.fixture)
    tmic("--workspace", args.workspace, "publish", "--execute")
    tmic(
        "baselines",
        "verify-model",
        "--model-workspace",
        args.workspace,
        "--out",
        collection,
    )
    workspace = Workspace(args.workspace)  # publish rewrote the lock
    revision = export_new(workspace, args.contract_name)
    pin = workspace.lock["contracts"][workspace.state["contract"]]["integration"]
    return {
        "status": "published",
        "repo": workspace.fixture,
        "revision": pin["revision"],
        "fleet_revision": revision,
        "scope": scope,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("ready", "publish"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--workspace", default="work/run")
    parser.add_argument(
        "--result", default="result.json", help="tmic run's agent-mode result (ready)"
    )
    parser.add_argument("--collection", default="work/cloud")
    parser.add_argument("--transformers-ref", default="main")
    parser.add_argument(
        "--contract-name", default=None, help="fleet name to export under (publish)"
    )
    parser.add_argument(
        "--overwrite-unexported",
        action="store_true",
        help="overwrite a published repo that no fleet references (publish)",
    )
    args = parser.parse_args()
    outcome = {
        "status": "failed",
        "model": args.model,
        "phase": args.phase,
        "run": os.environ.get("GITHUB_RUN_ID"),
    }
    try:
        outcome.update(ready(args) if args.phase == "ready" else publish(args))
    except BaseException as error:
        outcome["error"] = str(error)
        raise
    finally:
        Path(f"{args.phase}-result.json").write_text(
            json.dumps(outcome, indent=2) + "\n"
        )
    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
