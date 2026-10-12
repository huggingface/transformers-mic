"""Check published pins, record CPU expectations, and export passing contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(sys.argv[1]).resolve()
spec = json.loads((root / "models.json").read_text())
report = {}
for item in spec["models"]:
    name = item["name"]
    if len(sys.argv) > 2 and name not in sys.argv[2:]:
        continue
    workspace = root / "projects" / name
    stages = [
        ("published-gpu", ["check", "--all-variants"], False),
        ("cpu-record", ["record", "--all-variants"], True),
        ("cpu-review", ["review"], True),
        ("cpu-approve", ["review", "--approve"], True),
        ("cpu-check", ["check", "--all-variants"], True),
        ("gpu-recheck", ["check", "--all-variants"], False),
        ("export", ["export", "--transformers-checkout", str(root / "export")], False),
    ]
    if os.environ.get("MIC_GPU_ONLY") == "true":
        # Native CPU references are recorded separately on the actual CI runner.
        stages = stages[:1]
    row = report[name] = {}
    for stage, args, cpu in stages:
        print(f"{name}: {stage}", flush=True)
        env = dict(os.environ)
        if cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        log = root / "logs" / f"{name}-{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            result = subprocess.run(
                ["tmic", "--workspace", str(workspace), *args],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        row[stage] = result.returncode
        (root / "validation-report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if result.returncode:
            print("\n".join(log.read_text().splitlines()[-30:]), flush=True)
            break
sys.exit(int(any(any(row.values()) for row in report.values())))
