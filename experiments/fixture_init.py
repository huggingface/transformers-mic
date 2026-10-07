"""Compare four ways to fill a slim fixture's weights on the same faults and checks.

    python fixture_init.py --families text clip bert t5 whisper wav2vec2

  random   stage 2: the shrunk real config, seeded random init
  overfit  stage 3: the random fixture trained on the task's few examples
  pruned   the real checkpoint at full width, keeping only its first layers
  sliced   the stage 2 architecture, each tensor a leading slice of the real one

Each fixture is saved, then evaluated with experiment.py's synthetic faults and
controls (float32, on CONTRACT_DEVICE), and loaded again in float32, bfloat16,
and float16 to record whether outputs, loss, and gradients stay finite.
Random and overfit depend on the seed; pruned and sliced do not and are built once.
"""

import argparse
import contextlib
import gc
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import torch
import transformers

import experiment as exp


# Real checkpoints whose weights feed the pruned and sliced fixtures.
REAL = {
    "text": exp.SOURCES["text"],
    "clip": exp.CLIP_REFERENCE,
    "bert": exp.SOURCES["bert"],
    "t5": exp.SOURCES["t5"],
    "whisper": exp.SOURCES["whisper"],
    "wav2vec2": exp.SOURCES["wav2vec2"],
}
# Faults that only change the inputs: their logit deltas measure input sensitivity.
INPUT_FAULTS = {
    "text": ("wrong_country_token", "ignore_padding"),
    "clip": ("swap_red_blue", "repeat_first_image", "repeat_first_text", "scale_pixels"),
    "bert": ("wrong_country_token", "ignore_padding", "wrong_token_type"),
    "t5": ("wrong_country_token", "ignore_padding"),
    "whisper": ("repeat_first_audio", "reverse_audio", "scale_features"),
    "wav2vec2": ("repeat_first_audio", "reverse_audio"),
}
LAYER_KEYS = (
    "num_hidden_layers",
    "num_layers",
    "num_decoder_layers",
    "encoder_layers",
    "decoder_layers",
)
DTYPES = ("float32", "bfloat16", "float16")


@contextlib.contextmanager
def placement(device, dtype):
    """Load tasks and models with experiment.py's globals pointed elsewhere."""
    with patch.object(exp, "DEVICE", torch.device(device)), patch.object(
        exp, "DTYPE", getattr(torch, dtype)
    ):
        yield


def real_model(task, family):
    """The full real checkpoint in float32 on CPU, the source for pruned and sliced."""
    with placement("cpu", "float32"):
        return task.load_model(exp.snapshot(REAL[family])).eval()


def shrink_layers(config, layers):
    for sub in (config, *(getattr(config, k, None) for k in ("text_config", "vision_config"))):
        for key in LAYER_KEYS:
            if sub is not None and isinstance(getattr(sub, key, None), int):
                setattr(sub, key, min(layers, getattr(sub, key)))
    return config


def build_pruned(task, real, layers):
    config = exp.no_dropout(shrink_layers(real.config.__class__.from_dict(real.config.to_dict()), layers))
    if getattr(task.processor, "pad_token_id", None) is not None and task.name == "text":
        config.pad_token_id = task.processor.pad_token_id
    model = type(real)._from_config(config, attn_implementation="eager", dtype=torch.float32)
    missing, _ = model.load_state_dict(real.state_dict(), strict=False)
    if missing:
        raise ValueError(f"pruned fixture has weights the real checkpoint lacks: {missing}")
    return model.to(exp.DEVICE), {"layers": layers}


def build_sliced(task, real):
    model = task.make_model()
    source = real.state_dict()
    copied, kept_random = 0, []
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            original = source.get(name)
            if original is None or original.dim() != tensor.dim() or any(
                o < t for o, t in zip(original.shape, tensor.shape)
            ):
                kept_random.append(name)
                continue
            tensor.copy_(original[tuple(slice(0, n) for n in tensor.shape)])
            copied += 1
    return model, {"tensors_copied": copied, "tensors_kept_random": kept_random}


def weight_stats(model):
    values = torch.cat([p.detach().float().flatten().cpu() for p in model.parameters()])
    return {"std": float(values.std()), "max_abs": float(values.abs().max())}


def dtype_health(task_cls, path, dtype, reference=None):
    """Load a saved fixture in dtype and report finiteness of outputs, loss, gradients."""
    row = {"dtype": dtype}
    try:
        with placement(str(exp.DEVICE), dtype):
            task = task_cls(path)
            model = task.load_model(path, attn="sdpa").eval()
            with exp.mutation(task, model, "none") as (train, infer), torch.inference_mode():
                result = task.observe(model, train, infer)
            numerical = torch.tensor(result["numerical"])
            row.update(
                outputs_finite=bool(torch.isfinite(numerical).all()),
                max_abs_logit=float(numerical.abs().max()),
                outputs=result["outputs"],
            )
            if reference is not None:
                row["max_delta_vs_float32"] = float(
                    (numerical - torch.tensor(reference["numerical"])).abs().max()
                )
                row["outputs_match_float32"] = result["outputs"] == reference["outputs"]
            probe = exp.training_probe(task, model)
            row.update(
                loss=probe["loss"],
                loss_and_gradients_finite=probe["finite"],
                training_probe_pass=probe["pass"],
            )
            row["numerical"] = result["numerical"] if dtype == "float32" else None
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        gc.collect()
    return row


def sensitivity(family, evaluation):
    deltas = {
        f["name"]: f.get("max_absolute_delta")
        for f in evaluation["faults"]
        if f["name"] in INPUT_FAULTS[family]
    }
    known = [d for d in deltas.values() if d is not None]
    return {"input_fault_deltas": deltas, "min_input_delta": min(known) if known else None}


def record(args, report, task, model, family, weights, seed, build):
    exp.log("fixture_start", family=family, seed=seed, weights=weights)
    folder = args.artifacts / family / weights / str(seed)
    row = {
        "family": family,
        "weights": weights,
        "seed": seed,
        "build": build,
        "parameters": sum(p.numel() for p in model.parameters()),
        "weight_stats": weight_stats(model),
        "config": model.config.to_dict(),
    }
    row["artifact"] = exp.save_fixture(task, model, folder)
    row["weights_bytes"] = sum(
        f["bytes"] for name, f in row["artifact"]["files"].items() if name.endswith(".safetensors")
    )
    try:
        row["evaluation"] = exp.evaluate(task, model, folder)
        row.update(sensitivity(family, row["evaluation"]))
    except Exception as exc:
        row["evaluation_error"] = f"{type(exc).__name__}: {exc}"[:500]
    float32 = dtype_health(type(task), folder, "float32")
    row["dtypes"] = [float32] + [
        dtype_health(type(task), folder, d, reference=float32 if "numerical" in float32 else None)
        for d in DTYPES[1:]
    ]
    float32.pop("numerical", None)
    for health in row["dtypes"]:
        health.pop("numerical", None)
    report["fixtures"].append(row)
    report["summary"] = summarize(report["fixtures"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    exp.log(
        "fixture_done",
        family=family,
        weights=weights,
        seed=seed,
        min_input_delta=row.get("min_input_delta"),
        error=row.get("evaluation_error"),
        finite={h["dtype"]: h.get("outputs_finite") and h.get("loss_and_gradients_finite") for h in row["dtypes"]},
    )


def summarize(fixtures):
    keys = {
        "numerical": lambda f: f["numerical_detected"],
        "outputs": lambda f: f["recorded_output_detected"],
        "all_checks": lambda f: f["numerical_detected"]
        or f.get("training_detected")
        or f.get("loss_detected"),
    }
    summary = {}
    for row in fixtures:
        entry = summary.setdefault(row["family"], {}).setdefault(
            row["weights"],
            {"fixtures": 0, "fault_cases": 0, "control_false_alarms": 0, "errors": 0,
             **{k: 0 for k in keys}, "min_input_delta": None, "non_finite": [],
             "weights_bytes": row["weights_bytes"], "parameters": row["parameters"],
             "build_seconds": []},
        )
        entry["fixtures"] += 1
        entry["build_seconds"].append(round(row["build"]["seconds"], 3))
        for health in row["dtypes"]:
            if not (health.get("outputs_finite") and health.get("loss_and_gradients_finite")):
                entry["non_finite"].append(
                    {"seed": row["seed"], "dtype": health["dtype"], "error": health.get("error")}
                )
        if "evaluation" not in row:
            entry["errors"] += 1
            continue
        faults = row["evaluation"]["faults"]
        entry["fault_cases"] += len(faults)
        for key, hit in keys.items():
            entry[key] += sum(bool(hit(f)) for f in faults)
        entry["control_false_alarms"] += sum(
            c["numerical_detected"] or c["recorded_output_detected"]
            or c.get("training_detected", False) or c.get("loss_detected", False)
            for c in row["evaluation"]["controls"]
        )
        delta = row.get("min_input_delta")
        if delta is not None:
            current = entry["min_input_delta"]
            entry["min_input_delta"] = delta if current is None else min(current, delta)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", choices=list(REAL), default=list(REAL))
    parser.add_argument("--weights", nargs="+", default=["random", "overfit", "pruned", "sliced"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--layers", type=int, default=2, help="layers kept by the pruned fixture")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results" / "fixture_init.json"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path(tempfile.gettempdir()) / "model-integration-fixture-init",
    )
    args = parser.parse_args()
    if exp.DTYPE != torch.float32:
        raise SystemExit("build in float32; this script loads each dtype itself")
    torch.set_num_threads(args.threads)
    # CTC loss (Wav2Vec2) has no deterministic CUDA backward.
    torch.use_deterministic_algorithms(True, warn_only=exp.DEVICE.type == "cuda")
    transformers.logging.set_verbosity_error()
    report = {
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(exp.DEVICE),
            "device_name": torch.cuda.get_device_name(0) if exp.DEVICE.type == "cuda" else None,
            "threads": args.threads,
            "atol": exp.ATOL,
            "rtol": exp.RTOL,
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "fixtures": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for family in args.families:
        task_cls = exp.TASKS[family]
        task = task_cls(exp.snapshot(exp.SOURCES[family]))
        for seed in args.seeds:
            if not {"random", "overfit"} & set(args.weights):
                break
            torch.manual_seed(seed)
            start = time.perf_counter()
            model = task.make_model()
            build = {"seconds": time.perf_counter() - start}
            if "random" in args.weights:
                record(args, report, task, model, family, "random", seed, build)
            if "overfit" in args.weights:
                training = exp.train(task, model, args.max_steps)
                build = {"seconds": build["seconds"] + training["seconds"], "training": training}
                record(args, report, task, model, family, "overfit", seed, build)
            del model
            gc.collect()
        if not {"pruned", "sliced"} & set(args.weights):
            continue
        start = time.perf_counter()
        real = real_model(task, family)
        load_seconds = time.perf_counter() - start
        for weights in ("pruned", "sliced"):
            if weights not in args.weights:
                continue
            torch.manual_seed(args.seeds[0])
            start = time.perf_counter()
            if weights == "pruned":
                model, details = build_pruned(task, real, args.layers)
            else:
                model, details = build_sliced(task, real)
            build = {
                "seconds": time.perf_counter() - start,
                "real_load_seconds": load_seconds,
                **details,
            }
            record(args, report, task, model, family, weights, args.seeds[0], build)
            del model
            gc.collect()
        del real
        gc.collect()
    exp.log("complete", output=str(args.output))


if __name__ == "__main__":
    main()
