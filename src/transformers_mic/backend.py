"""Run the existing backend in a fresh process with a workspace as its data root."""

import importlib
import os
from pathlib import Path
import subprocess
import sys

from . import resources


def invoke(tool, workspace, args=(), *, capture=False, env=None):
    command = [
        sys.executable,
        "-m",
        "transformers_mic.backend",
        tool,
        str(Path(workspace).resolve()),
        *map(str, args),
    ]
    return subprocess.run(command, text=True, capture_output=capture, env=env)


def dispatch(tool, workspace, args):
    source = resources.backend()
    sys.path.insert(0, str(source))
    if source.name == "example":
        sys.path.insert(1, str(source.parent / "experiments"))
    runner = importlib.import_module("run")
    runner.HERE = workspace / "example"
    runner.SCHEMAS = source / "schemas"
    runner.schema.cache_clear()
    if tool == "build-stage":
        builder = importlib.import_module("build_fixtures")
        builder.HERE = runner.HERE
        name, fixture, out, *selection = args
        lock = runner.load(
            runner.HERE / "transformers/tests/contracts/fleet.lock", "fleet-lock"
        )
        if lock.get("local_patches"):
            importlib.import_module("patches").apply(sorted(lock["local_patches"]))
        entry = lock["contracts"][name]
        root = (
            runner.HERE / "transformers/tests/contracts" / entry["integration"]["path"]
        ).resolve()
        manifest = runner.load(root / "integration.json", "integration")
        import torch

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        builder.build_fixture(
            fixture,
            manifest["fixtures"][fixture],
            name,
            manifest,
            root,
            entry["checkpoints"],
            Path(out),
            keep_random=bool(selection),
        )
        return 0
    module = runner if tool == "run" else importlib.import_module(tool)
    if hasattr(module, "HERE"):
        module.HERE = runner.HERE
    if tool == "publish_contracts":
        module.FRAMEWORKS = ("transformers",)
    if tool == "replay":

        def replay_run(ref, options):
            import json

            path = [
                str(module.build(ref)),
                os.environ.get("OVERLAY", ""),
                os.environ.get("PYTHONPATH", ""),
            ]
            env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in path if p)}
            out = invoke(
                "run",
                workspace,
                ["--framework", "transformers", *options],
                capture=True,
                env=env,
            )
            records = [
                json.loads(line)
                for line in out.stdout.splitlines()
                if line.startswith("{")
            ]
            if not records:
                raise ValueError(f"No replay records on {ref}: {out.stderr[-1500:]}")
            return {r["target"]: r for r in records}

        module.run = replay_run
    sys.argv = [tool, *args]
    return module.main() or 0


if __name__ == "__main__":
    try:
        raise SystemExit(dispatch(sys.argv[1], Path(sys.argv[2]), sys.argv[3:]))
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing runtime dependency: {exc}. Install transformers-mic[runtime] and doc-builder from its Git repository."
        ) from None
