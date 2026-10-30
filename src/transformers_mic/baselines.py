"""Cloud baseline collections: dashboard submission, retrieval, review and Hub upload."""

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import uuid
import zipfile
from io import BytesIO

from filelock import FileLock
from huggingface_hub import HfApi

from . import cloud_inputs, dashboard_auth, fleet, cloud_progress
from .workspace import read, write

TERMINAL = {"completed", "failed", "cancelled", "unknown"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_archive(payload, destination):
    """Extract only ordinary, bounded files into a fresh artifact directory."""
    destination = Path(destination)
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        files = archive.infolist()
        if (
            len(files) > 100_000
            or sum(item.file_size for item in files) > 2_000_000_000
        ):
            raise ValueError("Baseline artifact is too large")
        seen = set()
        for item in files:
            path = PurePosixPath(item.filename)
            normalized = path.as_posix()
            if (
                not path.parts
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or stat.S_ISLNK(item.external_attr >> 16)
                or normalized in seen
            ):
                raise ValueError("Unsafe baseline artifact path")
            seen.add(normalized)
        for item in files:
            target = destination / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))


def collection(root):
    """Validate profile completeness and the checksums used for review/upload."""
    root = Path(root)
    state = read(root / "cloud.json")
    reports = []
    pins = None
    for profile in state["request"]["profiles"]:
        folder = root / "artifacts" / profile
        report = read(folder / "collection.json")
        request = report["request"]
        if (
            any(
                request.get(key) != state["job"][key]
                for key in (
                    "id",
                    "fleet_revision",
                    "transformers_sha",
                    "mic_sha",
                    "fleets",
                    "fixtures_only",
                )
            )
            or report["profile"] != profile
        ):
            raise ValueError("Artifact provenance does not match the requested job")
        if request.get("exclude", []) != state["job"].get("exclude", []):
            raise ValueError("Artifact exclusions do not match the requested job")
        if request.get("workspace") != state["job"].get("workspace"):
            raise ValueError("Artifact model input does not match the requested job")
        if not report.get("complete") or not report.get("cases"):
            raise ValueError(
                f"{profile}: baseline collection is incomplete; inspect collection.json"
            )
        keys = [case["key"] for case in report["cases"]]
        if len(keys) != len(set(keys)) or set(keys) != set(report["expected_cases"]):
            raise ValueError("Collection does not cover its declared case matrix")
        for model in report.get("models", []) + report.get("contracts", []):
            if "files" not in model:
                if pins is None:
                    pins = pinned_sources(state["job"])
                if (
                    model["repo"],
                    model["revision"],
                    model["files_sha256"],
                ) not in pins:
                    raise ValueError(
                        f"{profile}: measured {model['repo']}@{model['revision'][:10]}, which the requested fleet does not pin"
                    )
                continue
            # Older artifacts carried copies of the sources.
            for relative, expected in model["files"].items():
                path = PurePosixPath(relative)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in relative
                    or sha256(folder / relative) != expected
                ):
                    raise ValueError("Model artifact changed after execution")
        for case in report["cases"]:
            if case["status"] == "declared-unsupported":
                continue
            if case["status"] != "validated":
                raise ValueError("Collection contains an unvalidated case")
            if case.get("candidate") not in case["files"]:
                raise ValueError("Validated case is missing its measured candidate")
            for relative, expected in case["files"].items():
                path = PurePosixPath(relative)
                if path.is_absolute() or ".." in path.parts or "\\" in relative:
                    raise ValueError("Unsafe collection path")
                if sha256(folder / relative) != expected:
                    raise ValueError(
                        f"Measured baseline changed after execution: {relative}"
                    )
        reports.append((folder, report))
    if not any(
        case["status"] == "validated"
        for _, report in reports
        for case in report["cases"]
    ):
        raise ValueError("Collection contains no validated measurements")
    return state, reports


def pinned_sources(job):
    """(repo, revision, files_sha256) of every contract and fixture the job's fleet pins."""
    if job.get("workspace"):
        locks = [cloud_inputs.lock(job["workspace"])]
    else:
        locks = [
            read(fleet.download(name, job["fleet_revision"])[0])
            for name in job["fleets"]
        ]
    return {
        (repo, pin["revision"], pin.get("files_sha256"))
        for lock in locks
        for contract in lock["contracts"].values()
        for repo, pin in [
            (contract["integration"]["repo"], contract["integration"]),
            *contract.get("fixtures", {}).items(),
        ]
    }


def fingerprint(reports):
    return hashlib.sha256(
        json.dumps([report for _, report in reports], sort_keys=True).encode()
    ).hexdigest()


def candidates(reports):
    for folder, report in reports:
        for case in report["cases"]:
            if case["status"] == "validated":
                candidate = folder / case["candidate"]
                contract = candidate.parents[2]
                reference = (
                    contract
                    / "references"
                    / candidate.parent.name
                    / candidate.name.replace(".candidate.json", ".json")
                )
                yield (
                    report["profile"],
                    case,
                    candidate,
                    reference,
                    contract / "integration.json",
                )


def review(root, approve=False, approve_changed=False, reason=None):
    """Approve exact immutable measurements; changed values need an explicit reason."""
    import difflib
    import importlib
    import sys
    from . import resources

    sys.path.insert(0, str(resources.backend()))
    runner = importlib.import_module("run")
    _, reports = collection(root)
    current = fingerprint(reports)
    receipts_path = Path(root) / "reviews.json"
    receipts = (
        read(receipts_path)
        if receipts_path.exists()
        else {"collection": current, "approved": {}}
    )
    if receipts["collection"] != current:
        receipts = {"collection": current, "approved": {}}
    pending = 0
    for profile, case, candidate, reference, manifest_path in candidates(reports):
        measured = read(candidate)["outputs"]
        verdict = "new"
        if reference.exists():
            old = read(reference)["outputs"]
            rules = runner.rules_for(read(manifest_path), candidate.parent)
            shared = {
                key: rule
                for key, rule in rules.items()
                if runner.lookup(old, key) is not None
            }
            removed = [key for key in shared if runner.lookup(measured, key) is None]
            verdict = (
                "changed"
                if removed or runner.compare(shared, measured, old)
                else "unchanged"
            )
            if verdict == "changed":
                print(
                    "\n".join(
                        difflib.unified_diff(
                            json.dumps(old, indent=2).splitlines(),
                            json.dumps(measured, indent=2).splitlines(),
                            fromfile="approved",
                            tofile="measured",
                        )
                    )
                )
        key = profile + "/" + case["key"]
        print(verdict + " " + key)
        accepted = approve_changed if verdict == "changed" else approve
        if accepted:
            if verdict == "changed" and not (reason or "").strip():
                raise ValueError("Changed baselines require --reason")
            receipts["approved"][key] = {
                "sha256": sha256(candidate),
                "verdict": verdict,
                "reason": reason
                if verdict == "changed"
                else "New or unchanged measured reference",
            }
        if receipts["approved"].get(key, {}).get("sha256") != sha256(candidate):
            pending += 1
    if approve or approve_changed:
        write(receipts_path, receipts)
    print(json.dumps({"pending": pending, "collection": current}))
    return 3 if pending else 0


def publish(root, execute=False):
    """Open Hub model PRs from reviewed measurements; never bypass framework review."""
    from huggingface_hub import CommitOperationAdd

    state, reports = collection(root)
    receipt = read(Path(root) / "reviews.json")
    if receipt["collection"] != fingerprint(reports):
        raise ValueError("Collection changed since review")
    groups = {}
    for profile, case, candidate, _, _ in candidates(reports):
        key = profile + "/" + case["key"]
        if receipt["approved"].get(key, {}).get("sha256") != sha256(candidate):
            raise ValueError("Unreviewed baseline: " + key)
        if not case["repo"].endswith("-mic"):
            raise ValueError(
                "Legacy split repositories must be migrated before cloud baseline publication"
            )
        group = groups.setdefault(
            case["repo"], {"revision": case["revision"], "files": {}}
        )
        if group["revision"] != case["revision"]:
            raise ValueError(
                "Selected fleets pin different revisions of the same MIC repository"
            )
        destination = (
            f"integrations/transformers/expectations/{case['platform']}/"
            + candidate.name.replace(".candidate.json", ".json")
        )
        if (
            destination in group["files"]
            and read(group["files"][destination])["outputs"]
            != read(candidate)["outputs"]
        ):
            raise ValueError("Overlapping fleets produced conflicting references")
        group["files"][destination] = candidate
    publications = state.setdefault("publications", {})
    if any(
        item.get("collection") != fingerprint(reports) for item in publications.values()
    ):
        raise ValueError(
            "Publication belongs to an earlier collection; use a new --out"
        )
    for repo, group in sorted(groups.items()):
        print(
            json.dumps(
                {
                    "repository": repo,
                    "parent_revision": group["revision"],
                    "files": sorted(group["files"]),
                    "create_pr": True,
                    "execute": execute,
                }
            )
        )
        if not execute or repo in publications:
            continue
        title = "Review cloud MIC baselines " + state["job"]["id"]
        # A call that stalled after the Hub opened the PR left no receipt: adopt that PR.
        if existing := open_pull_request(repo, title):
            publications[repo] = {**existing, "collection": fingerprint(reports)}
            write(Path(root) / "cloud.json", state)
            print(json.dumps(publications[repo]))
            continue
        result = HfApi().create_commit(
            repo_id=repo,
            repo_type="model",
            parent_commit=group["revision"],
            create_pr=True,
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=path.read_bytes())
                for name, path in sorted(group["files"].items())
            ],
            commit_message=title,
            commit_description="Measured and independently validated on CPU and CUDA. Review reasons: "
            + json.dumps(receipt["approved"], sort_keys=True),
        )
        publications[repo] = {
            "revision": result.oid,
            "url": result.pr_url,
            "collection": fingerprint(reports),
        }
        write(Path(root) / "cloud.json", state)
        print(json.dumps(publications[repo]))
    return 0


def open_pull_request(repo, title):
    api = HfApi()
    for discussion in api.get_repo_discussions(repo, discussion_type="pull_request"):
        if discussion.status == "open" and discussion.title == title:
            details = api.get_discussion_details(repo, discussion.num)
            commits = [e for e in details.events if e.type == "commit"]
            return {"revision": commits[-1].oid, "url": discussion.url}
    return None


def upload(root, repository, execute):
    state, reports = collection(root)
    preview = {
        "repository": repository,
        "repo_type": "dataset",
        "job_id": state["job"]["id"],
        "profiles": [report["profile"] for _, report in reports],
        "execute": execute,
    }
    if not execute:
        print(json.dumps(preview))
        return 0
    existing = state.get("upload", {})
    if existing.get("repository") == repository and existing.get(
        "collection"
    ) == fingerprint(reports):
        print(json.dumps(existing))
        return 0
    api = HfApi()
    api.create_repo(repository, repo_type="dataset", exist_ok=True)
    allowed = set()
    for _, report in reports:
        prefix = report["profile"] + "/"
        allowed.add(prefix + "collection.json")
        for case in report["cases"]:
            allowed.update(prefix + relative for relative in case.get("files", {}))
        for model in report.get("models", []) + report.get("contracts", []):
            allowed.update(prefix + relative for relative in model.get("files", {}))
    result = api.upload_folder(
        repo_id=repository,
        repo_type="dataset",
        folder_path=str(Path(root) / "artifacts"),
        path_in_repo="collections/" + state["job"]["id"],
        allow_patterns=sorted(allowed),
        commit_message="Add measured MIC baseline candidates " + state["job"]["id"],
    )
    state["upload"] = {
        **preview,
        "revision": result.oid,
        "collection": fingerprint(reports),
        "url": f"https://huggingface.co/datasets/{repository}/tree/{result.oid}/collections/{state['job']['id']}",
    }
    write(Path(root) / "cloud.json", state)
    print(json.dumps(state["upload"]))
    return 0


def run(args):
    if (
        args.baselines_command == "run"
        and getattr(args, "transport", "dashboard") == "github"
    ):
        from . import github_baselines
        from types import SimpleNamespace

        if args.no_wait or args.no_download or args.fleet_revision:
            raise ValueError(
                "GitHub test transport watches and downloads a collection resolved from fleet main"
            )
        return github_baselines.run(
            SimpleNamespace(
                out=args.out,
                fleet=args.fleet,
                include_checkpoints=not args.fixtures_only,
                upload_repo=args.upload_repo or "",
                transformers_ref=args.transformers_ref,
                model_workspace=args.model_workspace,
                exclude=args.exclude,
            )
        )
    if getattr(args, "exclude", None):
        raise ValueError("--exclude requires --transport github")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(root / ".cloud.lock", timeout=0):
            return run_locked(args, root)
    except KeyboardInterrupt:
        import sys

        print(
            "Stopped watching. The remote job continues; repeat the command to resume.",
            file=sys.stderr,
        )
        return 130


def run_locked(args, root):
    if args.baselines_command == "verify-model":
        from .model_baselines import verify
        from .workspace import Workspace

        return verify(Workspace(args.model_workspace), root)
    if args.baselines_command == "upload":
        return upload(root, args.repo, args.execute)
    if args.baselines_command == "review":
        return review(root, args.approve, args.approve_changed, args.reason)
    if args.baselines_command == "publish":
        return publish(root, args.execute)
    dashboard = dashboard_auth.dashboard_url(args.dashboard)
    if (
        args.baselines_command == "run"
        and args.fleet_revision
        and not re.fullmatch(r"[a-f0-9]{40}", args.fleet_revision)
    ):
        raise ValueError(
            "--fleet-revision must be an immutable 40-character dataset SHA"
        )
    if args.baselines_command == "run" and args.no_download and not args.upload_repo:
        raise ValueError("--no-download requires --upload-repo")
    token = dashboard_auth.access_token(dashboard)
    model = None
    model_input = None
    if args.baselines_command == "run" and getattr(args, "model_workspace", None):
        from . import model_baselines
        from .workspace import Workspace

        model = Workspace(args.model_workspace)
        if args.fleet == "all" or args.fixtures_only or args.no_download:
            raise ValueError(
                "Model measurement requires one --fleet and all declared targets, with artifact download"
            )
        model_input = model_baselines.stage(model, root)
    path = root / "cloud.json"
    if path.exists():
        state = read(path)
        if state["dashboard"] != dashboard:
            raise ValueError("Workspace belongs to a different dashboard")
    elif args.baselines_command != "run":
        raise ValueError("No cloud collection here; run tmic baselines run first")
    else:
        revision = fleet.resolve(args.fleet_revision or "main")
        request = {
            "request_id": uuid.uuid4().hex,
            "fleet_revision": revision,
            "fleets": ["control", "prototypes"]
            if args.fleet == "all"
            else [args.fleet],
            "profiles": ["cpu", "cuda"],
            "transformers_ref": args.transformers_ref,
            "fixtures_only": args.fixtures_only,
            "upload_repo": args.upload_repo,
        }
        if model_input:
            request["workspace"] = model_input
        state = {"dashboard": dashboard, "request": request}
        write(path, state)  # persist the idempotency key before the network call
    if args.baselines_command == "run":
        expected_fleets = (
            ["control", "prototypes"] if args.fleet == "all" else [args.fleet]
        )
        if (
            state["request"].get("workspace") != model_input
            or state["request"]["fleets"] != expected_fleets
            or state["request"]["fixtures_only"] != args.fixtures_only
            or state["request"]["transformers_ref"] != args.transformers_ref
            or state["request"]["upload_repo"] != args.upload_repo
            or (
                args.fleet_revision
                and args.fleet_revision != state["request"]["fleet_revision"]
            )
        ):
            raise ValueError("Collection options changed; use a different --out")
        if "job" not in state:
            state["job"] = dashboard_auth.request(
                dashboard, "/mic-jobs", body=state["request"], token=token
            )
            write(path, state)
    if "job" not in state:
        raise ValueError(
            "Submission is incomplete; repeat tmic baselines run with the same options"
        )
    endpoint = "/mic-jobs/" + state["job"]["id"]
    last_progress = None
    while True:
        try:
            state["job"] = dashboard_auth.request(dashboard, endpoint, token=token)
        except dashboard_auth.DashboardError as error:
            if (
                args.baselines_command == "run"
                and not args.no_wait
                and (
                    error.error == "mic_service_unavailable"
                    or error.error.startswith("Could not reach")
                )
            ):
                import sys

                print(
                    "[cloud] Connection interrupted; reconnecting to the same job…",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(10)
                continue
            raise
        write(path, state)
        last_progress = cloud_progress.display(state["job"], last_progress)
        if (
            state["job"]["status"] in TERMINAL
            or args.baselines_command == "status"
            or args.no_wait
        ):
            break
        time.sleep(10)
    if args.baselines_command == "status" or args.no_wait:
        print(json.dumps(state["job"]), flush=True)
        return 0
    if state["job"]["status"] not in ("completed", "failed"):
        raise ValueError("Cloud job did not complete: " + state["job"]["status"])
    if publication := state["job"].get("publication_artifact"):
        payload = dashboard_auth.request_bytes(
            dashboard, endpoint + "/artifacts/" + str(publication["id"]), token=token
        )
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            if archive.getinfo("publication.json").file_size > 16384:
                raise ValueError("Invalid publication receipt")
            receipt = json.loads(archive.read("publication.json"))
        if (
            receipt.get("job_id") != state["job"]["id"]
            or receipt.get("repository") != state["request"]["upload_repo"]
            or not re.fullmatch(r"[a-f0-9]{40}", receipt.get("revision", ""))
        ):
            raise ValueError("Publication receipt does not match the job")
        state["remote_upload"] = receipt
        write(path, state)
        print(json.dumps(receipt), flush=True)
    if args.no_download:
        if state["job"]["status"] != "completed" or "remote_upload" not in state:
            raise ValueError("Remote publication did not complete")
        return 0
    for artifact in state["job"].get("artifacts", []):
        profile = artifact["profile"]
        if profile not in state["request"]["profiles"]:
            raise ValueError("Unexpected artifact profile")
        target = root / "artifacts" / profile
        # Fetch through the authenticated dashboard; the CLI needs no GitHub token.
        payload = dashboard_auth.request_bytes(
            dashboard, endpoint + "/artifacts/" + str(artifact["id"]), token=token
        )
        staging = root / (".artifact-" + uuid.uuid4().hex)
        try:
            extract_archive(payload, staging)
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    if state["job"]["status"] == "failed":
        raise ValueError(
            "Cloud job failed; available partial artifacts were downloaded. Inspect "
            + state["job"].get("run_url", "cloud.json")
        )
    collection(
        root
    )  # partial/failed runs stay downloadable but cannot be uploaded as complete
    if model:
        model_baselines.import_candidates(model, root)
    print(
        json.dumps(
            {
                "status": "measured",
                "out": str(root),
                "next": "Review candidates, or tmic baselines upload --out PATH --repo OWNER/DATASET --execute",
            }
        )
    )
    return 0
