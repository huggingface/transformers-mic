"""Publish the locally built fixtures to the Hub and pin their revisions in the fleet lock.

    python example/publish_fixtures.py                      # every fixture whose local files match its pin
    python example/publish_fixtures.py --fixture proto-whisper-tiny-slim

A fixture is published only if its local files match the file hashes the fleet
lock pins (the bytes its baselines were recorded with). The whole folder is
uploaded, README.md (tagged `test-artifact`) and build.json included, and the
commit it creates becomes the fixture's revision in the lock. Rebuilding a
fixture changes its pin, so its baselines must be re-recorded before publishing.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

import run

HERE = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", default="transformers")
    parser.add_argument("--fixture", action="append", help="publish only this fixture (short name, repeatable)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    lock_path = HERE / args.framework / "tests/contracts/fleet.lock"
    lock = run.load(lock_path, "fleet-lock")
    for contract in lock["contracts"].values():
        for repo, pin in contract.get("fixtures", {}).items():
            if args.fixture and run.short(repo) not in args.fixture:
                continue
            folder = (lock_path.parent / pin["path"]).resolve()
            problem = run.fixture_problem(folder, {**pin, "revision": "unpublished"})
            if problem:
                raise SystemExit(f"{repo}: not published: {problem}")
            api.create_repo(repo, private=args.private, exist_ok=True)
            commit = api.upload_folder(
                repo_id=repo,
                folder_path=folder,
                commit_message=f"Slim fixture {run.short(repo)} (files_sha256 {pin['files_sha256'][:12]})",
            )
            pin["revision"] = commit.oid
            lock_path.write_text(json.dumps(lock, indent=2) + "\n")
            print(json.dumps({"published": repo, "revision": commit.oid, "url": f"https://huggingface.co/{repo}"}), flush=True)


if __name__ == "__main__":
    main()
