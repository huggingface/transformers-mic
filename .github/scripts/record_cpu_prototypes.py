"""Record native CPU references, then check and export without framework patches."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(sys.argv[1]).resolve()
spec = json.loads((root / "models.json").read_text())
selected = set(sys.argv[2:])
models = [item for item in spec["models"] if not selected or item["name"] in selected]


def run(item):
    name = item["name"]
    workspace = root / "projects" / name
    lock_path = workspace / "example/transformers/tests/contracts/fleet.lock"
    lock = json.loads(lock_path.read_text())
    patches = lock.pop("local_patches", {})
    # Wav2Vec2's missing SpecAugment parameter still needs a controlled reference.
    # Only reference recording uses this patch; the final check is unpatched and
    # explicitly excludes the unresolved loss/gradient outputs, as the old proof does.
    if name == "wav2vec2" and "masked-spec-embed-init" in patches:
        lock["local_patches"] = {
            "masked-spec-embed-init": patches["masked-spec-embed-init"]
        }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    stages = [
        ("record", ["record", "--all-variants"]),
        ("review", ["review", "--approve"]),
    ]
    row = {}
    for stage, args in stages:
        row[stage] = invoke(name, workspace, stage, args)
        if row[stage]:
            return name, row
    lock.pop("local_patches", None)
    if name == "wav2vec2":
        lock["known_failures"] = {
            "mic-wav2vec2/mic-wav2vec2-slim": {
                "issue": "https://github.com/huggingface/transformers/issues/49385",
                "skip_outputs": ["training.loss", "training.gradients"],
            }
        }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    row["unpatched-check"] = invoke(
        name, workspace, "unpatched-check", ["check", "--all-variants"]
    )
    if not row["unpatched-check"]:
        # Each worker exports independently; combine reviewed locks afterwards.
        row["export"] = invoke(
            name,
            workspace,
            "export",
            [
                "export",
                "--contract-name",
                "mic-" + name,
                "--transformers-checkout",
                str(root / "exports" / name),
            ],
        )
    return name, row


def invoke(name, workspace, stage, args):
    print(f"{name}: {stage}", flush=True)
    log = root / "logs" / f"{name}-{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        result = subprocess.run(
            ["tmic", "--workspace", str(workspace), *args],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        print("\n".join(log.read_text().splitlines()[-20:]), flush=True)
    return result.returncode


report = {}
with ThreadPoolExecutor(max_workers=4) as pool:
    for name, row in pool.map(run, models):
        report[name] = row
        (root / "cpu-report.json").write_text(json.dumps(report, indent=2) + "\n")
sys.exit(int(not report or any(any(row.values()) for row in report.values())))
