"""Temporary GitHub-authenticated ARC test while dashboard OAuth is deferred.

Run with the Python environment in which MIC is installed. This deliberately
tests the workflow directly; it does not exercise dashboard/Serge authorization.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from filelock import FileLock
from huggingface_hub import hf_hub_download

from transformers_mic import baselines, fleet, model_baselines, cloud_progress
from transformers_mic.workspace import Workspace, read, write

REPO = "huggingface/transformers-mic"
WORKFLOW = "cloud-baselines.yml"


def failure_summary(root):
    details = []
    for profile in ("cpu", "cuda"):
        path = root / "artifacts" / profile / "collection.json"
        if not path.is_file():
            continue
        for case in read(path).get("cases", []):
            if case.get("status") != "failed":
                continue
            failures = case.get("record", {}).get("failures", []) or case.get(
                "check", {}
            ).get("failures", [])
            reason = (
                str(failures[0]).strip().splitlines()[-1]
                if failures
                else "case failed; see worker logs"
            )
            details.append(
                f"{profile}/{case.get('variant')}/{case.get('target')}: {reason}"
            )
    return "; ".join(details[:4]) or "inspect the Actions URL and partial artifacts"


def restore_hidden_sources(root):
    """Recover source dotfiles omitted by older GitHub artifact uploads.

    Only immutable Hub source files are eligible, never measured candidates.
    The worker's recorded checksum must match before installing any file.
    """
    state = read(root / "cloud.json")
    for profile in state["request"]["profiles"]:
        folder = root / "artifacts" / profile
        report = read(folder / "collection.json")
        for kind in ("models", "contracts"):
            for source in report.get(kind, []):
                repo, revision = source["repo"], source["revision"]
                if not re.fullmatch(r"[a-f0-9]{40}", revision):
                    raise ValueError(
                        "Source recovery requires an immutable Hub revision"
                    )
                prefix = PurePosixPath(kind, repo.replace("/", "--"), revision)
                for name, expected in source["files"].items():
                    path = PurePosixPath(name)
                    if path.is_absolute() or ".." in path.parts or "\\" in name:
                        raise ValueError("Unsafe source artifact path")
                    relative = path.relative_to(prefix)
                    target = folder / path
                    if target.exists() or not any(
                        part.startswith(".") for part in relative.parts
                    ):
                        continue
                    recovered = Path(
                        hf_hub_download(repo, relative.as_posix(), revision=revision)
                    )
                    if baselines.sha256(recovered) != expected:
                        raise ValueError(
                            "Recovered source differs from the measured artifact"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(recovered, target)


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True).strip()


def api(path):
    return json.loads(gh("api", path))


def sha(repository, ref):
    result = api(f"repos/{repository}/commits/{ref}")["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", result):
        raise ValueError("Expected immutable GitHub commit")
    return result


def find_run(job_id):
    runs = api(
        f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs"
        "?event=workflow_dispatch&branch=main&per_page=100"
    )["workflow_runs"]
    matches = [r for r in runs if r["display_title"] == "MIC baselines · " + job_id]
    if len(matches) > 1:
        raise ValueError("Multiple matching runs; inspect Actions before continuing")
    return matches[0] if matches else None


def watch(run_id):
    previous = None
    while True:
        remote = api(f"repos/{REPO}/actions/runs/{run_id}")
        jobs = api(f"repos/{REPO}/actions/runs/{run_id}/jobs?per_page=100").get(
            "jobs", []
        )
        view = {
            "run_url": remote.get("html_url"),
            "status": remote["conclusion"]
            if remote["status"] == "completed"
            else remote["status"],
            "workflow_jobs": [
                {
                    "name": job["name"],
                    "status": "failed"
                    if job.get("conclusion") == "failure"
                    else job["status"],
                    "queued_at": (job.get("created_at") or job.get("started_at"))
                    if job["status"] == "queued"
                    else None,
                    "step": next(
                        (
                            step["name"]
                            for step in job.get("steps", [])
                            if step["status"] == "in_progress"
                        ),
                        "",
                    ),
                    "completed_steps": sum(
                        step["status"] == "completed" for step in job.get("steps", [])
                    ),
                    "total_steps": len(job.get("steps", [])),
                }
                for job in jobs
            ],
        }
        previous = cloud_progress.display(view, previous)
        if remote["status"] == "completed":
            return remote
        time.sleep(10)


def run(args):
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(root / ".cloud.lock", timeout=0):
            return run_locked(args, root)
    except KeyboardInterrupt:
        interrupt(root)
        return 130


def interrupt(root, *, interactive=True):
    """Offer remote cancellation after the local watcher has stopped."""
    from . import terminal

    path = Path(root) / "cloud.json"
    if not path.exists():
        return
    try:
        state = read(path)
        if state.get("transport") != "github-test":
            return
        job = state["job"]
        remote = (
            api(f"repos/{REPO}/actions/runs/{job['run_id']}")
            if job.get("run_id")
            else find_run(job["id"])
        )
        if not remote:
            print(
                "\nRemote dispatch unresolved; resume to locate the same run.",
                file=sys.stderr,
            )
            return
        if remote["status"] == "completed":
            print("\nRemote workflow has already finished.", file=sys.stderr)
            return
        url = remote["html_url"]
        print(f"\n  Remote workflow: {url}", file=sys.stderr)
        answer = ""
        if interactive and sys.stdin.isatty():
            print(
                terminal.color(
                    "  Cancel the remote CPU/CUDA workflow? [y/N] ",
                    "question",
                    sys.stderr,
                ),
                end="",
                file=sys.stderr,
                flush=True,
            )
            try:
                answer = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                pass
        if answer not in {"y", "yes"}:
            print(
                "  Remote run continues. Repeat the command to reconnect.",
                file=sys.stderr,
            )
            return
        gh(
            "api",
            "--method",
            "POST",
            f"repos/{REPO}/actions/runs/{remote['id']}/cancel",
        )
        state["cancel_requested"] = True
        write(path, state)
        print(
            "  Cancellation requested. GitHub may take a moment to stop the jobs.",
            file=sys.stderr,
        )
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(
            f"\n  Could not confirm remote cancellation: {error}. Check GitHub Actions.",
            file=sys.stderr,
        )


def run_locked(args, root):
    state_path = root / "cloud.json"
    options = {
        "fleets": ["control", "prototypes"] if args.fleet == "all" else [args.fleet],
        "fixtures_only": not args.include_checkpoints,
        "upload_repo": args.upload_repo,
        "transformers_ref": args.transformers_ref,
        "profiles": ["cpu", "cuda"],
    }
    model = None
    if getattr(args, "model_workspace", None):
        # Check GitHub access before staging inputs on the Hub.
        api(f"repos/{REPO}")
        model = Workspace(args.model_workspace)
        if args.fleet == "all" or not args.include_checkpoints:
            raise ValueError(
                "Model measurement requires one fleet and all declared targets"
            )
        options["workspace"] = model_baselines.stage(model, root)
    if state_path.exists():
        state = read(state_path)
        if state.get("transport") != "github-test" or any(
            state["request"].get(k) != v for k, v in options.items()
        ):
            raise ValueError("Different collection/options; use another --out")
    else:
        variables = api(f"repos/{REPO}/actions/variables")["variables"]
        configured = {v["name"]: v["value"] for v in variables}
        for name in ("MIC_CPU_BASELINE_IMAGE", "MIC_GPU_BASELINE_IMAGE"):
            if not re.fullmatch(
                r"[a-zA-Z0-9./:_-]+@sha256:[a-f0-9]{64}", configured.get(name, "")
            ):
                raise ValueError(f"Configure repository variable {name} first")
        job_id = uuid.uuid4().hex
        request = {**options, "fleet_revision": fleet.resolve()}
        job = {
            "id": job_id,
            "fleet_revision": request["fleet_revision"],
            "mic_sha": sha(REPO, "main"),
            "transformers_sha": sha("huggingface/transformers", args.transformers_ref),
            "fleets": request["fleets"],
            "fixtures_only": request["fixtures_only"],
        }
        if model:
            job["workspace"] = options["workspace"]
        state = {"transport": "github-test", "request": request, "job": job}
        write(state_path, state)
    job = state["job"]
    remote = (
        api(f"repos/{REPO}/actions/runs/{job['run_id']}")
        if job.get("run_id")
        else find_run(job["id"])
    )
    ended = (
        remote.get("conclusion") if remote and remote["status"] == "completed" else None
    )
    # A retry after a cancelled or failed run must measure again, not re-read the
    # old run; its artifacts were already downloaded and are kept in the archive.
    if ended in ("cancelled", "failure", "timed_out"):
        kept = "cancelled" if ended == "cancelled" else "failed"
        archive = root / kept / job["id"]
        archive.mkdir(parents=True, exist_ok=True)
        for item in root.iterdir():
            if item.name not in {".cloud.lock", "cancelled", "failed"}:
                shutil.move(str(item), archive / item.name)
        print(f"Previous run {ended}; starting a fresh collection.", flush=True)
        return run_locked(args, root)
    if not remote and not state.get("dispatch_attempted"):
        # Persist before submission. An ambiguous timeout must never allocate
        # another GPU runner automatically on the next invocation.
        state["dispatch_attempted"] = True
        write(state_path, state)
        inputs = {
            key: job[key] for key in ("fleet_revision", "mic_sha", "transformers_sha")
        }
        inputs.update(
            job_id=job["id"],
            fleets=",".join(job["fleets"]),
            fixtures_only=str(job["fixtures_only"]).lower(),
            upload_repo=args.upload_repo,
        )
        if "workspace" in job:
            inputs.update(
                workspace_repo=job["workspace"]["repo"],
                workspace_revision=job["workspace"]["revision"],
            )
        subprocess.run(
            [
                "gh",
                "workflow",
                "run",
                WORKFLOW,
                "--repo",
                REPO,
                "--ref",
                "main",
                "--json",
            ],
            input=json.dumps(inputs),
            text=True,
            check=True,
        )
    for _ in range(12):
        if remote:
            break
        time.sleep(5)
        remote = find_run(job["id"])
    if not remote:
        raise ValueError(
            "Dispatch is unresolved. Repeat this command to look for the same job; "
            "it will not dispatch again. Check Actions before choosing a new --out."
        )
    job.update(run_id=remote["id"], run_url=remote["html_url"])
    write(state_path, state)
    print(job["run_url"], flush=True)
    remote = watch(remote["id"])
    if remote["status"] != "completed":
        raise ValueError("Run is still active; repeat this command to resume")
    job["status"] = "completed" if remote["conclusion"] == "success" else "failed"
    write(state_path, state)
    # gh downloads only this run's artifacts; MIC then verifies the recorded
    # provenance, complete case matrix and content hashes before review/upload.
    cloud_progress.display({"status": "downloading results", "run_url": job["run_url"]})
    with tempfile.TemporaryDirectory(dir=root) as directory:
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(remote["id"]),
                "--repo",
                REPO,
                "--dir",
                directory,
            ],
            check=True,
        )
        for profile in ("cpu", "cuda"):
            source = Path(directory) / ("mic-baselines-" + profile)
            if source.exists():
                target = root / "artifacts" / profile
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), target)
        publication = Path(directory) / "mic-baselines-publication" / "publication.json"
        if publication.exists():
            shutil.copy2(publication, root / "publication.json")
    if job["status"] != "completed":
        cloud_progress.display({"status": "failed", "run_url": job["run_url"]})
        raise ValueError("Workflow failed: " + failure_summary(root))
    restore_hidden_sources(root)
    baselines.collection(root)
    if model:
        model_baselines.import_candidates(model, root)
    cloud_progress.display({"status": "completed", "run_url": job["run_url"]})
    print(f"Validated collection: {root}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-workspace")
    parser.add_argument(
        "--fleet", choices=("control", "prototypes", "all"), default="control"
    )
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--transformers-ref", default="main")
    parser.add_argument("--upload-repo", default="")
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Remote run continues. Repeat the same command to resume.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
