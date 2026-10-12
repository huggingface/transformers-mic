"""Can CPU golden outputs check WebGPU kernels? Runs the SmolLM Transformers.js contract.

    python experiments/webgpu_golden.py [--runs 3]

For each pinned ONNX conversion, runs example/.../transformers.js/integration.js
in a fresh process per run on CPU and WebGPU, in fp32, fp16, and q4f16, and
compares every run with the CPU fp32 golden (first run): generated tokens,
next-token logits, run-to-run drift within a configuration, and timings.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "example/hub/SmolLM-integration/integrations/transformers.js"
LOCK = json.loads((ROOT / "example/transformers.js/tests/contracts/fleet.lock").read_text())
ARTIFACTS = LOCK["contracts"]["SmolLM"]["artifacts"]
CONFIGS = [
    ("cpu", "fp32"), ("cpu", "fp16"), ("cpu", "q4f16"),
    ("webgpu", "fp32"), ("webgpu", "fp16"), ("webgpu", "q4f16"),
]


def run_contract(artifact, revision, device, dtype):
    env = {
        **os.environ,
        "MIC_MODEL": artifact,
        "MIC_MODEL_REVISION": revision,
        "MIC_MODEL_KWARGS": json.dumps({"dtype": dtype, "device": device}),
    }
    proc = subprocess.run(
        ["node", "integration.js"], cwd=CONTRACT, env=env, capture_output=True, text=True
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if proc.returncode or not lines:
        return {"error": proc.stderr.strip()[-500:]}
    return json.loads(lines[-1])


def max_delta(a, b):
    flat = lambda rows: [x for row in rows for x in row]
    return max(abs(x - y) for x, y in zip(flat(a), flat(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/results/webgpu_golden.json"
    )
    args = parser.parse_args()
    package = json.loads((CONTRACT / "node_modules/@huggingface/transformers/package.json").read_text())
    ort = json.loads((CONTRACT / "node_modules/onnxruntime-node/package.json").read_text())
    report = {
        "environment": {
            "machine": platform.platform(),
            "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip(),
            "@huggingface/transformers": package["version"],
            "onnxruntime-node": ort["version"],
        },
        "rows": [],
    }
    for artifact, revision in ARTIFACTS.items():
        golden = None
        for device, dtype in CONFIGS:
            runs = [run_contract(artifact, revision, device, dtype) for _ in range(args.runs)]
            ok = [r for r in runs if "error" not in r]
            row = {"artifact": artifact, "device": device, "dtype": dtype, "runs": len(runs)}
            if not ok:
                row["error"] = runs[0]["error"]
            else:
                if golden is None and (device, dtype) == ("cpu", "fp32"):
                    golden = ok[0]
                row.update(
                    tokens_equal_golden=[r["generated_tokens"] == golden["generated_tokens"] for r in ok],
                    max_logit_delta_golden=max(
                        max_delta(r["next_token_logits"], golden["next_token_logits"]) for r in ok
                    ),
                    max_logit_drift_run_to_run=max(
                        max_delta(r["next_token_logits"], ok[0]["next_token_logits"]) for r in ok
                    ),
                    generated_text=ok[0]["generated_text"],
                    timing_median_ms={
                        key: round(statistics.median(r["timing"][key] for r in ok), 1)
                        for key in ok[0]["timing"]
                    },
                )
            report["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "generated_text"}), flush=True)
            args.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
