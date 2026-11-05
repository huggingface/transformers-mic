"""Check the fixtures of every ST MIC: the Sentence Transformers fleet, plus the
Transformers fleets' contracts that advertise a sentence_transformers usage."""

import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers_mic import backend, fleet, resources


def preview_lock(repo, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Preview MIC revision must be an immutable commit SHA")
    sys.path.insert(0, str(resources.backend()))
    import run

    root = Path(snapshot_download(repo, revision=revision))
    manifest = run.load(root / "integration.json", "integration")
    if "sentence_transformers" not in manifest["integrations"]:
        raise ValueError("Preview MIC has no sentence_transformers integration")
    build = run.load(root / "build.json", "fixture-build")
    pin = {"revision": revision, "files_sha256": run.digest(build["sha256"])}
    if problem := run.fixture_problem(root, pin):
        raise ValueError(problem)
    return {
        "framework": "sentence_transformers",
        "environment": {},
        "contracts": {
            "EmbeddingGemma": {
                "integration": {
                    "repo": repo,
                    "revision": revision,
                    "files_sha256": run.digest(run.contract_files(root)),
                },
                "fixtures": {repo: pin},
                "checkpoints": build["covers"],
                "integrations": ["sentence_transformers"],
            }
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/sentence-transformers"))
    parser.add_argument("--mode", choices=("check", "record"), default="check")
    parser.add_argument("--fleet-revision", default="main")
    parser.add_argument("--st-fleet-revision", default="main")
    parser.add_argument("--model-repo", default="")
    parser.add_argument("--model-revision", default="")
    args = parser.parse_args()
    if bool(args.model_repo) != bool(args.model_revision):
        parser.error("--model-repo and --model-revision must be supplied together")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.model_repo:
        inputs = [("preview", preview_lock(args.model_repo, args.model_revision))]
        (args.out / "model-input.json").write_text(
            json.dumps({"repo": args.model_repo, "revision": args.model_revision})
        )
    else:
        revision = fleet.resolve(args.fleet_revision)
        (args.out / "fleet-revision.txt").write_text(revision + "\n")
        path, st_revision = fleet.download_st(args.st_fleet_revision)
        (args.out / "st-fleet-revision.txt").write_text(st_revision + "\n")
        inputs = [(fleet.ST_FLEET, json.loads(path.read_text()))]
        for name in fleet.FILES:
            path, _ = fleet.download(name, revision)
            inputs.append((name, json.loads(path.read_text())))
    count, status, seen = 0, 0, set()
    for name, lock in inputs:
        # Every ST fleet contract is an ST MIC; the others only if they say so.
        selected = {
            key: value
            for key, value in lock["contracts"].items()
            if (
                lock.get("framework") == fleet.ST_FRAMEWORK
                or "sentence_transformers" in value.get("integrations", [])
            )
            and value["integration"]["repo"] not in seen
        }
        seen.update(value["integration"]["repo"] for value in selected.values())
        if not selected:
            continue
        count += len(selected)
        lock["contracts"] = selected
        # These projects are tested at freshly resolved upstream mains, never
        # against the dependency versions recorded in a fleet baseline environment.
        lock["environment"] = {
            package: version
            for package, version in lock.get("environment", {}).items()
            if package.lower().replace("_", "-")
            not in {"transformers", "sentence-transformers"}
        }
        # Transformers known failures do not apply to this integration.
        lock.pop("known_failures", None)
        lock.pop("local_patches", None)
        local = args.out / f"{name}.lock"
        local.write_text(json.dumps(lock, indent=2) + "\n")
        for contract in selected:
            options = [
                "--framework",
                "sentence_transformers",
                "--lock",
                str(local.resolve()),
                "--contract",
                contract,
                "--fixtures-only",
                "--isolate",
                "--require-all",
                "--output",
                str((args.out / f"{name}-{contract}.jsonl").resolve()),
            ]
            if args.mode == "record":
                options += [
                    "--record",
                    "--expectations",
                    str((args.out / "candidates" / name / contract).resolve()),
                ]
            status = max(status, backend.invoke("run", Path.cwd(), options).returncode)
    if not count:
        raise SystemExit("No fleet MICs advertise sentence_transformers")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
