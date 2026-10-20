"""Publish one umbrella snapshot: slim weights, contract, and reviewed expectations."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import HfApi, hf_hub_download

import run
from mic_card import merged_card

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    lock_path = HERE / "transformers/tests/contracts/fleet.lock"
    lock = run.load(lock_path, "fleet-lock")
    if len(lock["contracts"]) != 1:
        raise ValueError("Publish a single MIC workspace at a time")
    name, contract = next(iter(lock["contracts"].items()))
    integration_repo = contract["integration"]["repo"]
    root = run.source_folder(
        lock_path.parent, integration_repo, contract["integration"]
    )
    state = HERE.parent / ".mic/project.json"
    if state.exists():
        model = json.loads(state.read_text())["model"]
        repo = json.loads((root / "integration.json").read_text())["checkpoints"][
            model
        ]["fixture"]
    elif len(contract["fixtures"]) == 1:
        repo = next(iter(contract["fixtures"]))
    else:
        raise ValueError(
            "Select one model workspace before publishing a shared contract"
        )
    if repo not in contract["fixtures"] or not repo.endswith("-mic"):
        raise ValueError(
            "Regenerate this legacy workspace to publish a single <name>-mic repository"
        )
    pin = contract["fixtures"][repo]
    fixture = run.fixture_folder(lock_path.parent, repo, pin, False)
    problem = run.fixture_problem(fixture, pin)
    if problem:
        raise ValueError(problem)
    files = run.contract_files(root)
    contract_hash = run.digest(files)
    with tempfile.TemporaryDirectory(prefix="mic-publish-") as tmp:
        staging = Path(tmp)
        build = run.load(fixture / "build.json", "fixture-build")
        for path in [*build["sha256"], "build.json", "README.md", ".gitattributes"]:
            source = fixture / path
            if source.exists():
                target = staging / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for path in files:
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            source = root / path
            if (
                path in build["sha256"]
                and target.exists()
                and target.read_bytes() != source.read_bytes()
            ):
                raise ValueError(f"Contract/fixture path collision: {path}")
            shutil.copy2(source, target)
        expectations = root / "integrations/transformers/expectations"
        if not expectations.exists():
            raise ValueError("No reviewed expectations to publish")
        shutil.copytree(
            expectations,
            staging / "integrations/transformers/expectations",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.candidate.json"),
        )
        layout = {"format": "mic-umbrella-v1", "contract_files": sorted(files)}
        (staging / ".mic-repo.json").write_text(json.dumps(layout, indent=2) + "\n")
        integration_card = (
            (root / "README.md").read_text() if (root / "README.md").exists() else ""
        )
        fixture_card = (
            (fixture / "README.md").read_text()
            if (fixture / "README.md").exists()
            else ""
        )
        checkpoint = build.get("built_from", {}).get("repo", "")
        (staging / "README.md").write_text(
            merged_card(repo, integration_card, fixture_card, checkpoint)
        )
        api = HfApi()
        stale = []
        if api.repo_exists(repo):
            previous_layout = json.loads(
                Path(hf_hub_download(repo, ".mic-repo.json")).read_text()
            )
            previous_build = json.loads(
                Path(hf_hub_download(repo, "build.json")).read_text()
            )
            owned = set(previous_layout["contract_files"]) | set(
                previous_build["sha256"]
            )
            stale = sorted(name for name in owned if not (staging / name).exists())
        api.create_repo(repo, private=args.private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo,
            folder_path=staging,
            delete_patterns=stale,
            ignore_patterns=["**/*.candidate.json", ".cache/*", "**/__pycache__/*"],
            commit_message="Publish MIC usage, slim fixture and reviewed Transformers expectations",
        )
    contract["integration"].update(
        repo=repo, revision=commit.oid, files_sha256=contract_hash
    )
    pin["revision"] = commit.oid
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"published": repo, "revision": commit.oid, "contract": name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
