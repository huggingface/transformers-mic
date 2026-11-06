"""Measure a local workspace without Hub writes; verify every declared variant."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from transformers_mic.workspace import Workspace, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--profile", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    workspace = Workspace(args.workspace)
    if cache := os.environ.get("HF_HOME"):
        Path(cache).mkdir(parents=True, exist_ok=True)
    import torch
    if torch.cuda.is_available() != (args.profile == "cuda"):
        raise ValueError("Runner hardware does not match the requested profile")
    command = [sys.executable, "-m", "transformers_mic.cli", "--workspace", str(workspace.root)]
    subprocess.run([*command, "record", "--all-variants"], check=True)
    # Preserve candidates as measurements. Approval below is only a temporary
    # local comparison target for this run, never publication or fleet approval.
    shutil.copytree(workspace.expectations, workspace.root / ".mic/candidates", dirs_exist_ok=True)
    subprocess.run([*command, "review", "--approve"], check=True)
    subprocess.run([*command, "check", "--all-variants"], check=True)
    cases = []
    for variant in workspace.manifest["integrations"]["transformers"]["variants"]:
        for action, status in (("record", "recorded"), ("check", "pass")):
            path = workspace.root / ".mic" / f"{action}.{variant}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            if len(rows) != len(workspace.manifest["fixtures"]):
                raise ValueError(f"{action}/{variant}: incomplete fixture coverage")
            for row in rows:
                platform = row.get("platform", "")
                if row["status"] != status or row.get("baseline_missing"):
                    raise ValueError(f"{action}/{variant}: {row}")
                if not (platform == "cuda" if args.profile == "cuda" else platform.startswith("cpu-")):
                    raise ValueError(f"Unexpected platform: {platform}")
                cases.append({"action": action, "variant": variant, "platform": platform, "status": row["status"]})
    write(workspace.root / ".mic/runner-validation.json", {"profile": args.profile, "cases": cases,
          "scope": "candidate recording and independent rechecks only; no Hub publication"})
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
