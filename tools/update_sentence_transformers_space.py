"""Build the static ST results Space; --execute publishes the prepared files."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

SPACE = "huggingface-mic/sentence-transformers-ci"
ASSETS = Path(__file__).resolve().parents[1] / "spaces/sentence-transformers"


def result_run(folder, *, run_id, attempt, conclusion, mode, repository, now=None):
    rows = []
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "status" in row:
                rows.append(
                    {
                        key: row[key]
                        for key in (
                            "contract_name",
                            "contract",
                            "target",
                            "kind",
                            "framework",
                            "variant",
                            "status",
                            "failures",
                            "skip",
                            "seconds",
                            "platform",
                            "environment",
                            "revision",
                            "contract_revision",
                            "baseline",
                        )
                        if key in row
                    }
                )
    counts = {
        name: sum(row["status"] == name for row in rows)
        for name in ("pass", "fail", "error", "unsupported", "recorded")
    }
    if counts["fail"]:
        status = "fail"
    elif (
        conclusion != "success"
        or counts["error"]
        or not (counts["pass"] + counts["recorded"])
    ):
        status = "error"
    elif mode == "record":
        status = "recorded"
    else:
        status = "pass"
    pin = folder / "fleet-revision.txt"
    freeze = folder / "pip-freeze.txt"
    # Keep only relevant public package versions/direct references, not arbitrary pip URLs.
    dependencies = (
        [
            line
            for line in freeze.read_text().splitlines()
            if line.startswith(
                (
                    "sentence-transformers==",
                    "transformers==",
                    "torch==",
                    "numpy==",
                    "sentence-transformers @ git+https://github.com/huggingface/sentence-transformers.git@",
                    "transformers @ git+https://github.com/huggingface/transformers.git@",
                )
            )
        ]
        if freeze.exists()
        else []
    )
    started = folder / "started-at.txt"
    model_input = folder / "model-input.json"
    return {
        "model_input": json.loads(model_input.read_text())
        if model_input.exists()
        else None,
        "id": f"{run_id}-{attempt}",
        "run_id": str(run_id),
        "attempt": str(attempt),
        "time": now
        or (
            started.read_text().strip()
            if started.exists()
            else datetime.now(timezone.utc).isoformat()
        ),
        "mode": mode,
        "status": status,
        "conclusion": conclusion,
        "counts": counts,
        "results": rows,
        "url": f"https://github.com/{repository}/actions/runs/{run_id}/attempts/{attempt}",
        "fleet_revision": pin.read_text().strip() if pin.exists() else None,
        "dependencies": dependencies,
    }


def history(previous, latest):
    runs = [
        latest,
        *(row for row in previous.get("runs", []) if row["id"] != latest["id"]),
    ]
    runs.sort(key=lambda row: row["time"], reverse=True)
    return {"schema_version": 1, "runs": runs[:90]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("out/sentence-transformers-space")
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Existing data.json; otherwise fetch published history",
    )
    parser.add_argument("--space", default=SPACE)
    parser.add_argument(
        "--run-id",
        default=os.getenv("GITHUB_RUN_ID"),
        required=not os.getenv("GITHUB_RUN_ID"),
    )
    parser.add_argument("--attempt", default=os.getenv("GITHUB_RUN_ATTEMPT", "1"))
    parser.add_argument(
        "--repository",
        default=os.getenv("GITHUB_REPOSITORY", "huggingface/transformers-mic"),
    )
    parser.add_argument(
        "--conclusion",
        choices=("success", "failure", "cancelled", "skipped"),
        required=True,
    )
    parser.add_argument("--mode", choices=("check", "record"), default="check")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    previous = {"runs": []}
    if args.history:
        previous = json.loads(args.history.read_text())
    else:
        try:
            previous = json.loads(
                Path(
                    hf_hub_download(args.space, "data.json", repo_type="space")
                ).read_text()
            )
        except (EntryNotFoundError, RepositoryNotFoundError):
            pass
    latest = result_run(
        args.results,
        run_id=args.run_id,
        attempt=args.attempt,
        conclusion=args.conclusion,
        mode=args.mode,
        repository=args.repository,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "index.html"):
        shutil.copyfile(ASSETS / name, args.out / name)
    (args.out / "data.json").write_text(
        json.dumps(history(previous, latest), indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "space": args.space,
                "out": str(args.out),
                "status": latest["status"],
                "execute": args.execute,
            }
        )
    )
    if args.execute:
        api = HfApi()
        api.create_repo(
            args.space, repo_type="space", space_sdk="static", exist_ok=True
        )
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=args.out,
            allow_patterns=["README.md", "index.html", "data.json"],
            commit_message=f"Sentence Transformers MIC run {latest['id']}: {latest['status']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
