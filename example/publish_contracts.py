"""Publish the contracts in example/hub to the Hub and pin their revisions in every fleet lock that uses them.

    python example/publish_contracts.py                     # every contract whose files changed since its pin
    python example/publish_contracts.py --contract Whisper  # one contract (name as in the fleet lock)

The contract folder (manifest, framework entry points, README.md tagged
`test-artifact`) is uploaded as is, without installed packages, and the commit
it creates becomes the contract's revision, with the hash of its files, in each
framework's fleet lock that lists the repository. A contract whose files match
its pin is skipped. Publishing a contract that changes outputs means
re-recording the baselines that depend on it.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

import run

HERE = Path(__file__).parent
FRAMEWORKS = ("transformers", "transformers.js")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contract", action="append", help="publish only this contract (repeatable)")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    locks = {fw: HERE / fw / "tests/contracts/fleet.lock" for fw in FRAMEWORKS}
    loaded = {fw: run.load(path, "fleet-lock") for fw, path in locks.items()}
    # repo -> (source folder, [(framework, contract name)])
    repos = {}
    for fw, lock in loaded.items():
        for name, contract in lock["contracts"].items():
            if args.contract and name not in args.contract:
                continue
            pin = contract["integration"]
            folder = (locks[fw].parent / pin["path"]).resolve()
            source, users = repos.setdefault(pin["repo"], (folder, []))
            if source != folder:
                raise SystemExit(f"{pin['repo']}: fleet locks point at different folders ({source}, {folder})")
            users.append((fw, name))
    for repo, (folder, users) in repos.items():
        files = run.contract_files(folder)
        files_sha256 = run.digest(files)
        pins = [loaded[fw]["contracts"][name]["integration"] for fw, name in users]
        if all(p["revision"] != "unpublished" and p.get("files_sha256") == files_sha256 for p in pins):
            print(json.dumps({"unchanged": repo, "revision": pins[0]["revision"]}), flush=True)
            continue
        api.create_repo(repo, private=args.private, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo,
            folder_path=folder,
            ignore_patterns=["node_modules/*", "**/node_modules/*", ".cache/*"],
            commit_message=f"MIC {run.short(repo)} (files_sha256 {files_sha256[:12]})",
        )
        for pin in pins:
            pin.update(revision=commit.oid, files_sha256=files_sha256)
        for fw in {fw for fw, _ in users}:
            locks[fw].write_text(json.dumps(loaded[fw], indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"published": repo, "revision": commit.oid, "files": len(files), "url": f"https://huggingface.co/{repo}"}), flush=True)


if __name__ == "__main__":
    main()
