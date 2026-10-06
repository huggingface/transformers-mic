"""Measure what one contract check costs, phase by phase, slim fixture vs real checkpoint.

    python cost.py --fixtures <frozen dir from replay.py> --output results/cost.json

Each (family, target) runs in a fresh Python process so imports and first calls
are measured cold. The OS file cache is not purged, and downloads are not
included; artifact sizes are reported so download time can be estimated.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
WEIGHTS = (".safetensors", ".bin")


def child(family, target, fixtures, attn):
    """Runs inside the fresh process; prints one JSON line."""
    t0 = time.perf_counter()
    import resource

    import torch
    import transformers

    import experiment as exp

    transformers.logging.set_verbosity_error()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    phases = {"import": time.perf_counter() - t0}
    manifest = json.loads((fixtures / "baselines.json").read_text())
    if target == "real":
        frozen = next(f for f in manifest["fixtures"] if f["family"] == family and f["weights"] == "real")
        path = exp.snapshot(tuple(frozen["source"]))
    else:
        frozen = next(
            f for f in manifest["fixtures"]
            if f["family"] == family and f["weights"] == target and f["seed"] == 17
        )
        path = fixtures / frozen["path"]
    task_cls = exp.TASKS[family]

    def timed(name, fn):
        start = time.perf_counter()
        value = fn()
        phases[name] = time.perf_counter() - start
        return value

    task = timed("processor_and_inputs", lambda: task_cls(path))
    model = timed("model_load", lambda: task.load_model(path, attn=attn))
    actual = timed("first_inference", lambda: exp.observe(task, model))
    warm = []
    for _ in range(3):
        start = time.perf_counter()
        exp.observe(task, model)
        warm.append(time.perf_counter() - start)
    phases["warm_inference_median"] = statistics.median(warm)
    timed("compare", lambda: exp.compare(actual, frozen["baseline"]))
    if "training" in frozen:
        timed("training_probe", lambda: exp.training_probe(task, model))
    files = [f for f in Path(path).iterdir() if f.is_file()]
    print(json.dumps({
        "family": family,
        "target": target,
        "attn": attn,
        "parameters": sum(p.numel() for p in model.parameters()),
        "weight_bytes": sum(f.stat().st_size for f in files if f.name.endswith(WEIGHTS)),
        "other_bytes": sum(f.stat().st_size for f in files if not f.name.endswith(WEIGHTS)),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "phases": phases,
        "check_seconds": sum(v for k, v in phases.items() if k != "warm_inference_median"),
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", default=["random", "overfit", "real"])
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        family, target, fixtures = args.child
        return child(family, target, Path(fixtures), args.attn)
    rows = []
    for fixtures in args.fixtures:
        manifest = json.loads((fixtures / "baselines.json").read_text())
        for family in dict.fromkeys(f["family"] for f in manifest["fixtures"]):
            for target in args.targets:
                runs = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    out = subprocess.run(
                        [sys.executable, __file__, "--attn", args.attn, "--fixtures", str(fixtures),
                         "--output", "-", "--child", family, target, str(fixtures)],
                        capture_output=True, text=True, cwd=HERE, env={**os.environ, "HF_HUB_OFFLINE": "1"},
                    )
                    wall = time.perf_counter() - start
                    line = [l for l in out.stdout.splitlines() if l.startswith("{")]
                    if out.returncode or not line:
                        runs.append({"error": out.stderr[-400:]})
                        continue
                    runs.append({**json.loads(line[-1]), "process_wall_seconds": wall})
                ok = [r for r in runs if "error" not in r]
                row = {"family": family, "target": target, "runs": runs}
                if ok:
                    row["median_process_wall_seconds"] = statistics.median(r["process_wall_seconds"] for r in ok)
                    row["median_check_seconds"] = statistics.median(r["check_seconds"] for r in ok)
                rows.append(row)
                print(json.dumps({k: row.get(k) for k in ("family", "target", "median_process_wall_seconds",
                                                          "median_check_seconds")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"attn": args.attn, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
