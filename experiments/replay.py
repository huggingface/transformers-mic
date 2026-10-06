"""Freeze slim fixtures on a known-good build, then check them on another build.

    # On the known-good build (fixtures, baselines, and real-checkpoint records):
    PYTHONPATH=<good>/src python replay.py freeze \\
        --families text t5 --out <dir> --build <sha>
    # On the build under test, with nothing regenerated:
    PYTHONPATH=<bad>/src python replay.py check \\
        --fixtures <dir> --build <sha> --output <result.json>

Reuses the experiment's tasks, assertions, and tolerances. Fixtures are frozen
before the bad build is run; check never trains or rewrites them.
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import transformers

import experiment as exp


# Real checkpoints whose weights are compared alongside the slim fixtures.
REAL = {
    "text": exp.SOURCES["text"],
    "clip": exp.CLIP_REFERENCE,
    "bert": exp.SOURCES["bert"],
    "t5": exp.SOURCES["t5"],
    "whisper": exp.SOURCES["whisper"],
    "wav2vec2": exp.SOURCES["wav2vec2"],
    "umt5": exp.SOURCES["umt5"],
    "olmo": exp.SOURCES["olmo"],
}


def record(task, model, gradients):
    row = {"baseline": exp.observe(task, model)}
    if gradients:
        row["training"] = exp.training_probe(task, model)
    return row


def freeze(args):
    if exp.DTYPE != torch.float32:
        raise SystemExit("freeze in float32; CONTRACT_DTYPE only applies to check")
    manifest = {"build": args.build, "transformers": transformers.__version__}
    manifest["fixtures"] = []
    for family in args.families:
        task_cls = exp.TASKS[family]
        task = task_cls(exp.snapshot(exp.SOURCES[family]))
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = task.make_model()
            for weights in ("random", "overfit"):
                if weights == "overfit":
                    exp.train(task, model, args.max_steps)
                path = args.out / family / str(seed) / weights
                exp.save_fixture(task, model, path)
                # Record from the saved artifact, exactly as check will load it.
                loaded = task.load_model(path)
                loaded_task = task_cls(path)
                manifest["fixtures"].append(
                    {
                        "family": family,
                        "seed": seed,
                        "weights": weights,
                        "path": str(path.relative_to(args.out)),
                        **record(loaded_task, loaded, gradients=True),
                    }
                )
                write(args.out, manifest)
                exp.log("frozen", family=family, seed=seed, weights=weights)
                del loaded, loaded_task
            del model
            gc.collect()
        if args.real:
            source = REAL[family]
            real_path = exp.snapshot(source)
            model = task.load_model(real_path)
            manifest["fixtures"].append(
                {
                    "family": family,
                    "seed": None,
                    "weights": "real",
                    "source": source,
                    **record(task_cls(real_path), model, gradients=False),
                }
            )
            write(args.out, manifest)
            exp.log("frozen", family=family, weights="real")
            del model
            gc.collect()
    write(args.out, manifest)


def write(out, manifest):
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(manifest, indent=1) + "\n")


def check(args):
    manifest = json.loads((args.fixtures / "baselines.json").read_text())
    result = {
        "frozen_build": manifest["build"],
        "build": args.build,
        "attn": args.attn,
        "device": str(exp.DEVICE),
        "dtype": str(exp.DTYPE).removeprefix("torch."),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "transformers_file": transformers.__file__,
        "rows": [],
    }
    for frozen in manifest["fixtures"]:
        if args.families and frozen["family"] not in args.families:
            continue
        task_cls = exp.TASKS[frozen["family"]]
        row = {k: frozen[k] for k in ("family", "seed", "weights")}
        start = time.perf_counter()
        try:
            if frozen["weights"] == "real":
                path = exp.snapshot(tuple(frozen["source"]))
            else:
                path = args.fixtures / frozen["path"]
            task = task_cls(path)
            # Baselines are recorded with eager; eager/SDPA agree within tolerance.
            # Choose at load time: set_attn_implementation() after loading does not
            # reach the T5-family encoder/decoder stacks.
            model = task.load_model(path, attn=args.attn)
            actual = exp.observe(task, model)
            row.update(smoke_detected=False, **exp.compare(actual, frozen["baseline"]))
            row["outputs"] = actual["outputs"]
            if "training" in frozen:
                probe = exp.training_probe(task, model)
                row["training_detected"] = not probe["pass"]
                row["loss_detected"] = probe["loss"] is None or not math.isclose(
                    probe["loss"], frozen["training"]["loss"],
                    rel_tol=exp.RTOL, abs_tol=exp.ATOL,
                )
                row["loss"] = probe["loss"]
            del model, task
        except Exception as exc:
            row.update(
                smoke_detected=True,
                numerical_detected=True,
                recorded_output_detected=True,
                training_detected=True,
                loss_detected=True,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        row["baseline_outputs"] = frozen["baseline"]["outputs"]
        row["seconds"] = time.perf_counter() - start
        result["rows"].append(row)
        exp.log(
            "checked",
            **{k: row.get(k) for k in ("family", "seed", "weights", "smoke_detected",
                                       "numerical_detected", "recorded_output_detected",
                                       "max_absolute_delta")},
        )
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--families", nargs="+", choices=list(exp.TASKS), required=True)
    f.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    f.add_argument("--max-steps", type=int, default=500)
    f.add_argument("--real", action="store_true", help="also record real checkpoints")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--build", required=True)
    c = sub.add_parser("check")
    c.add_argument("--fixtures", type=Path, required=True)
    c.add_argument("--families", nargs="+", choices=list(exp.TASKS))
    c.add_argument("--build", required=True)
    c.add_argument("--attn", default="eager", help="attention implementation to check")
    c.add_argument("--output", type=Path, required=True)
    for p in (f, c):
        p.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    # On CUDA, an op without a deterministic kernel must not read as a detection.
    torch.use_deterministic_algorithms(True, warn_only=exp.DEVICE.type == "cuda")
    transformers.logging.set_verbosity_error()
    freeze(args) if args.command == "freeze" else check(args)


if __name__ == "__main__":
    main()
