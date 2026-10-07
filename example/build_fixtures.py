"""Build slim fixtures for the contracts in a fleet lock, with the same recipe for every model.

    python example/build_fixtures.py                                   # every fixture with a build recipe
    python example/build_fixtures.py --fixture proto-whisper-tiny-slim
    python example/build_fixtures.py --keys-only                       # check fixture keys, build nothing

For each fixture in a contract's manifest that has a `build` recipe:

1. Fixture key. Every checkpoint the fixture covers must have the same key:
   its processor files byte for byte, its config after shrinking, and the
   parameters its weights lack. The build fails on a mismatch instead of
   testing a checkpoint through the wrong fixture.
2. Shrink. The real config goes through the generic rule from
   experiments/sweep.py (widths by field name in every sub-config, depth by
   layer pattern, MoE to 4 experts); nothing else changes, so dropout and
   SpecAugment stay as configured (contracts seed them). Seeded random weights.
3. Copy. Every non-weight file of the source checkpoint (tokenizer,
   processor, generation config) is copied unchanged; only config.json and the
   weights are new. The weights mirror the real checkpoint's tensor names:
   parameters the real checkpoint lacks are dropped from the fixture too, so
   loading initializes them as it does for users, and the build checks that
   the fixture reports the same missing parameters.
4. Capture. The contract's own runnable markdown runs against the random
   fixture, and the runner records the arguments of each top-level model call
   by name. The training step's call is the batch used below.
5. Input-sensitivity gate (perturb.py). Perturbations are defined by input
   name, not per model. Random weights are kept if every perturbation moves
   the logits by at least 10x the strictest tolerance the fixture must hold.
   Otherwise the fixture is overfit on the contract's training batch and must
   then pass the gate; a fixture that does neither is a build failure.
6. Numeric health. The saved fixture is reloaded in float32, bfloat16, and
   float16; outputs, loss, and gradients must be finite.

build.json records the key, the gate results for random weights (and for the
overfit weights when training was needed), the training run, the health
checks, the copied files, and every file's sha256. The fleet lock pins that
hash map; the runner refuses fixtures whose files differ from it. Fixtures
without a recipe (the SmolLM ones, built by an earlier script) are left alone.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys

import time
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Fixtures are built on the CPU, including the contract run that captures their
# inputs: some GPU kernels have no deterministic backward (CTC loss).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import transformers
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoConfig, PreTrainedModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import run  # noqa: E402
from perturb import logits_of, sensitivity  # noqa: E402
from sweep import shrink  # noqa: E402

WEIGHTS = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".ot", ".onnx", ".tflite", ".ckpt")
NOT_COPIED = {"config.json", "README.md", ".gitattributes"}
# Config fields that name or date the source rather than describe the model.
VOLATILE = {"_name_or_path", "transformers_version", "dtype", "torch_dtype", "_commit_hash"}
GATE_FACTOR = 10
DTYPES = ("float32", "bfloat16", "float16")


def source_snapshot(repo, revision):
    """Config and processor files of a pinned checkpoint, without its weights."""
    try:
        ignore = [f"*{suffix}" for suffix in WEIGHTS] + run.WEIGHT_PATTERNS
        return Path(snapshot_download(repo, revision=revision, ignore_patterns=ignore))
    except Exception:
        cached = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
        if (cached / "config.json").exists():
            return cached
        raise


def copied_files(source):
    return sorted(
        f.name
        for f in source.iterdir()
        if f.is_file()
        and f.name not in NOT_COPIED
        and not f.name.endswith(WEIGHTS)
        and not f.name.endswith(".index.json")
    )


def model_class(source, recipe):
    if "model_class" in recipe:
        return getattr(transformers, recipe["model_class"])
    return getattr(transformers, AutoConfig.from_pretrained(source).architectures[0])


def shrunk_config(source, cls):
    return shrink(cls.config_class.from_pretrained(source))


def missing_parameters(cls, path):
    """Parameters a checkpoint's weights lack, in the model's own names, as loading reports them."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        model, info = cls.from_pretrained(path, output_loading_info=True)
    del model
    return sorted(info["missing_keys"])


def fixture_key(source, cls, missing):
    config = {k: v for k, v in shrunk_config(source, cls).to_dict().items() if k not in VOLATILE}
    processor = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in copied_files(source)}
    blob = json.dumps({"config": config, "processor": processor, "missing": missing}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest(), config


def save(model, out, source, missing):
    """Write the fixture: new config and weights without the parameters the real
    checkpoint lacks, every processor file copied unchanged."""
    from safetensors.torch import load_file, save_file

    out.mkdir(parents=True, exist_ok=True)
    for file in out.iterdir():
        if file.is_file() and file.name != "README.md":
            file.unlink()
    model.save_pretrained(out)
    weights = out / "model.safetensors"
    tensors = load_file(weights)
    absent = [name for name in missing if name not in tensors]
    if absent:
        raise SystemExit(f"cannot mirror the real checkpoint: the fixture has no tensor named {absent}")
    save_file({k: v for k, v in tensors.items() if k not in missing}, weights, metadata={"format": "pt"})
    copied = copied_files(source)
    for name in copied:
        shutil.copyfile(source / name, out / name)  # overwrites the generation config save_pretrained wrote
    return copied


@contextlib.contextmanager
def captured_calls():
    """Record the keyword arguments of every top-level model call made inside."""
    calls, depth = [], [0]
    call = torch.nn.Module.__call__

    def recording(self, *args, **kwargs):
        if depth[0] == 0:
            calls.append(
                {
                    "training": self.training,
                    "class": type(self).__name__,
                    "kwargs": {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()},
                }
            )
        depth[0] += 1
        try:
            return call(self, *args, **kwargs)
        finally:
            depth[0] -= 1

    with patch.object(PreTrainedModel, "__call__", recording):
        yield calls


def contract_batch(entry, fixture_dir):
    """Run the contract on the fixture; return the training step's model inputs by name."""
    code = run.contract_code(entry)
    env = {"MIC_RUNNER": "1", "MIC_MODEL": str(fixture_dir), "MIC_MODEL_REVISION": "", "MIC_MODEL_KWARGS": "{}"}
    with captured_calls() as calls, run.chdir(entry.parent):
        ok, _, stderr = run.execute_inline(code, str(entry), env)
    if not ok:
        raise SystemExit(f"the contract fails on the random fixture:\n{stderr[-1500:]}")
    training = [c for c in calls if c["training"]]
    chosen = training[0] if training else calls[0]
    # Fixtures are built on the CPU; the contract may have put its inputs on a GPU.
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in chosen["kwargs"].items() if isinstance(v, (torch.Tensor, bool))}
    return batch, {"call": "training step" if training else "first inference call", "inputs": sorted(batch)}


def overfit(model, batch, max_steps):
    start = time.perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    stable = 0
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0:
            stable = stable + 1 if loss.item() < 0.05 else 0
            if stable == 2:
                break
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"seconds": round(time.perf_counter() - start, 2), "steps": step, "loss": loss.item(), "converged": stable == 2}


def numeric_health(cls, folder, batch):
    """Reload the saved fixture per dtype; forward and backward must stay finite."""
    rows = {}
    for name in DTYPES:
        dtype = getattr(torch, name)
        row = {}
        try:
            model = cls.from_pretrained(folder, dtype=dtype).train()
            inputs = {k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v for k, v in batch.items()}
            output = model(**inputs)
            logits = logits_of(output)
            row["outputs_finite"] = all(bool(torch.isfinite(t).all()) for t in logits.values())
            if output.get("loss") is not None:
                output.loss.backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                row["loss_finite"] = bool(torch.isfinite(output.loss))
                row["gradients_finite"] = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
            row["finite"] = all(v for k, v in row.items() if k.endswith("finite"))
        except Exception as exc:  # noqa: BLE001 - recorded, then fails the build below
            row = {"finite": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows[name] = row
    return rows


def depth(config, prefix=""):
    """Layer counts of a shrunk config dict and its sub-configs, for the README."""
    keys = ("num_hidden_layers", "num_layers", "num_decoder_layers", "encoder_layers", "decoder_layers")
    parts = [f"{prefix}{k} {config[k]}" for k in keys if isinstance(config.get(k), int)]
    for key, value in config.items():
        if isinstance(value, dict) and key.endswith("_config"):
            parts.append(depth(value, f"{key.removesuffix('_config')} "))
    return ", ".join(p for p in parts if p)


def readme(name, contract, build, cls, config):
    lacks = build.get("missing_parameters", [])
    mirror = (
        f"The real checkpoint lacks {', '.join(f'`{n}`' for n in lacks)}, so the fixture omits it too, and loading initializes it as it does for users."
        if lacks
        else "The real checkpoint has every parameter, and so does the fixture."
    )
    covers = ", ".join(f"[`{r}`](https://huggingface.co/{r})" for r in build["covers"])
    gate = build["gate"]["random"]
    if build["weights"] == "random":
        weights = (
            f"**Random weights (seed {build['seed']}).** They passed the input-sensitivity gate: every perturbation "
            f"of the contract's inputs moved the logits by at least {gate['threshold']:g} (weakest: `{gate['weakest']}`, "
            f"{gate['deltas'][gate['weakest']]:.3g}), so no training was needed."
        )
    else:
        weights = (
            f"**Overfit to the contract's training step** ({build['training']['steps']} steps). Random weights failed "
            f"the input-sensitivity gate: `{gate['weakest']}` moved the logits by only {gate['deltas'][gate['weakest']]:.2g} "
            f"(threshold {gate['threshold']:g})."
        )
    return f"""---
tags:
  - mic
  - test-artifact
base_model: {build['built_from']['repo']}
---

# Slim fixture: {short(name)}

A {build['parameters'] / 1e6:.1f}M-parameter fixture for the [{contract} MIC](../{contract}-integration), covering {covers}. This unpublished example is a test artifact, not a usable model, and should be excluded from candidate-model search when published.

- **Fixture key `{build['key'][:12]}`.** It covers every checkpoint whose processor files are byte-identical and whose config matches after shrinking.
- **Same architecture, smaller config:** `{cls.__name__}`, shrunk by the generic rule in [`sweep.py`](../../../experiments/sweep.py) ({depth(config)}).
- **Processor files copied byte for byte** from `{build['built_from']['repo']}`: {', '.join(f'`{f}`' for f in build['copied_unchanged'])}.
- {weights}
- **Same tensor names as the real checkpoint.** {mirror}
- **Finite in float32, bfloat16, and float16** (forward, loss, gradients).

Built by [`build_fixtures.py`](../../build_fixtures.py); [`build.json`](build.json) records the pinned sources, gate results, and file hashes. Only `README.md` and `build.json` are committed; rebuild the other files with `python example/build_fixtures.py --fixture {short(name)}`.
"""


def short(repo):
    return repo.split("/")[-1]


def environment():
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


def build_fixture(name, fixture, contract_name, manifest, root, revisions, out):
    recipe = fixture["build"]
    keys, missing = {}, {}
    for repo in fixture["covers"]:
        source = source_snapshot(repo, revisions[repo])
        cls = model_class(source, recipe)
        missing[repo] = missing_parameters(cls, run.checkpoint_path(repo, revisions[repo]))
        keys[repo] = fixture_key(source, cls, missing[repo])
    if len({key for key, _ in keys.values()}) != 1:
        raise SystemExit(f"{name}: covered checkpoints have different fixture keys: { {r: k[:12] for r, (k, _) in keys.items()} }")
    key, config = next(iter(keys.values()))
    source_repo = fixture["built_from"]
    lacks = missing[source_repo]
    source = source_snapshot(source_repo, revisions[source_repo])
    cls = model_class(source, recipe)
    tolerance = recipe.get("gate_tolerance") or max(
        r["atol"] for r in manifest["outputs"].values() if r["compare"] == "allclose"
    )
    threshold = GATE_FACTOR * tolerance

    torch.manual_seed(recipe["seed"])
    model = cls._from_config(shrunk_config(source, cls), dtype=torch.float32).eval()
    copied = save(model, out, source, lacks)
    entry = root / manifest["integrations"]["transformers"]["entry_point"]
    batch, captured = contract_batch(entry, out)
    accepted = recipe.get("accept_insensitive", {})
    gate = {"random": sensitivity(model, batch, threshold, accepted)}
    weights, training = "random", None
    if not gate["random"]["pass"]:
        torch.manual_seed(recipe["seed"])  # dropout and SpecAugment stay on while training
        training = overfit(model, batch, recipe.get("max_steps", 500))
        gate["overfit"] = sensitivity(model, batch, threshold, accepted)
        if not training["converged"] or not gate["overfit"]["pass"]:
            raise SystemExit(f"{name}: random weights fail the gate and overfitting did not fix it: {training}, {gate}")
        weights = "overfit"
        save(model, out, source, lacks)
    mirrored = missing_parameters(cls, out)
    if mirrored != lacks:
        raise SystemExit(f"{name}: fixture lacks {mirrored}, the real checkpoint lacks {lacks}")
    health = numeric_health(cls, out, batch)
    if not all(row["finite"] for row in health.values()):
        raise SystemExit(f"{name}: numeric health check failed: {health}")
    build = {
        "fixture": name,
        "key": key,
        "key_fields": config,
        "covers": {repo: revisions[repo] for repo in fixture["covers"]},
        "built_from": {"repo": source_repo, "revision": revisions[source_repo]},
        "seed": recipe["seed"],
        "builder": "example/build_fixtures.py",
        "model_class": cls.__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "missing_parameters": lacks,
        "captured": captured,
        "weights": weights,
        "gate": gate,
        **({"training": training} if training else {}),
        "numeric_health": health,
        "copied_unchanged": copied,
        "environment": environment(),
        "sha256": run.file_hashes(out),
    }
    run.validate(build, "fixture-build", f"{name} build.json")
    (out / "build.json").write_text(json.dumps(build, indent=1, default=str) + "\n")
    (out / "README.md").write_text(readme(name, contract_name, build, cls, config))
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", default="transformers", help="whose fleet lock pins the sources")
    parser.add_argument("--fixture", action="append", help="build only this fixture (short name, repeatable)")
    parser.add_argument("--keys-only", action="store_true", help="check that covered checkpoints share a key")
    parser.add_argument("--out", type=Path, help="build into this directory instead, leaving the fixtures and lock untouched (reproducibility checks)")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    transformers.logging.set_verbosity_error()

    lock_dir = HERE / args.framework / "tests/contracts"
    lock_path = lock_dir / "fleet.lock"
    lock = run.load(lock_path, "fleet-lock")
    if lock.get("local_patches"):
        import patches

        patches.apply(sorted(lock["local_patches"]))
    for contract_name, contract in lock["contracts"].items():
        root = (lock_dir / contract["integration"]["path"]).resolve()
        manifest = run.load(root / "integration.json", "integration")
        for name, fixture in manifest["fixtures"].items():
            if "build" not in fixture or (args.fixture and short(name) not in args.fixture):
                continue
            pin = contract["fixtures"][name]
            if args.keys_only:
                keys = {}
                for repo in fixture["covers"]:
                    source = source_snapshot(repo, contract["checkpoints"][repo])
                    cls = model_class(source, fixture["build"])
                    lacks = missing_parameters(cls, run.checkpoint_path(repo, contract["checkpoints"][repo]))
                    keys[repo] = fixture_key(source, cls, lacks)[0][:12]
                print(json.dumps({"fixture": name, "keys": keys, "shared": len(set(keys.values())) == 1}))
                continue
            start = time.perf_counter()
            out = args.out / short(name) if args.out else (lock_dir / pin["path"]).resolve()
            build = build_fixture(name, fixture, contract_name, manifest, root, contract["checkpoints"], out)
            if args.out:
                same = run.digest(build["sha256"]) == pin.get("files_sha256")
                print(json.dumps({"fixture": name, "matches_pin": same, "differs": [] if same else sorted(
                    f for f, h in build["sha256"].items()
                    if h != json.loads(((lock_dir / pin["path"]) / "build.json").read_text())["sha256"].get(f))}), flush=True)
                continue
            pin["files_sha256"] = run.digest(build["sha256"])
            lock_path.write_text(json.dumps(lock, indent=2) + "\n")
            gate = build["gate"]
            print(
                json.dumps(
                    {
                        "built": name,
                        "key": build["key"][:12],
                        "weights": build["weights"],
                        "parameters": build["parameters"],
                        "random_weakest": [gate["random"]["weakest"], gate["random"]["deltas"].get(gate["random"]["weakest"])],
                        "overfit_weakest": [gate["overfit"]["weakest"], gate["overfit"]["deltas"].get(gate["overfit"]["weakest"])] if "overfit" in gate else None,
                        "steps": build.get("training", {}).get("steps"),
                        "seconds": round(time.perf_counter() - start, 1),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
