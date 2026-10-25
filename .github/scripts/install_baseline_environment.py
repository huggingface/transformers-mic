"""Install the pinned fleet dependencies and exact framework commit in a job venv."""

import importlib.metadata
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from transformers_mic import fleet
from transformers_mic.workspace import read


def main():
    revision = os.environ["FLEET_REVISION"]
    sha = os.environ["TRANSFORMERS_SHA"]
    if not all(re.fullmatch(r"[a-f0-9]{40}", value) for value in (revision, sha)):
        raise ValueError("Immutable revisions required")
    environments = []
    for name in os.environ["FLEETS"].split(","):
        path, _ = fleet.download(name, revision)
        environments.append(read(path)["environment"])
    if not environments or any(
        environment != environments[0] for environment in environments
    ):
        raise ValueError(
            "Selected fleets have different dependency environments; measure separately"
        )
    environment = environments[0]
    if platform.python_version() != environment[
        "python"
    ] and not platform.python_version().startswith(environment["python"] + "."):
        raise ValueError("Approved image Python does not match the fleet environment")
    checkout = Path(os.environ["MIC_TRANSFORMERS_CHECKOUT"])
    subprocess.run(["git", "init", str(checkout)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "fetch",
            "--depth",
            "1",
            "https://github.com/huggingface/transformers",
            sha,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], check=True
    )
    pip = [sys.executable, "-m", "pip", "install"]
    # Install framework dependencies first, then enforce the recorded versions.
    subprocess.run(
        [
            *pip,
            "-e",
            str(checkout),
            "timm",
            "soundfile",
            "librosa",
            "scipy",
            "pillow",
            "hf-doc-builder @ git+https://github.com/huggingface/doc-builder@c69e96c087165ba49b1788e0c3464654dd60253c",
        ],
        check=True,
    )
    torch = {"torch", "torchvision", "torchaudio"}
    torch_pins = [
        f"{key}=={value}" for key, value in environment.items() if key in torch
    ]
    other_pins = [
        f"{key}=={value}"
        for key, value in environment.items()
        if key not in torch | {"python", "transformers"}
    ]
    index = "cpu" if not os.environ.get("CUDA_VISIBLE_DEVICES") else "cu128"
    if torch_pins:
        subprocess.run(
            [
                *pip,
                *torch_pins,
                "--index-url",
                "https://download.pytorch.org/whl/" + index,
            ],
            check=True,
        )
    if other_pins:
        subprocess.run([*pip, *other_pins], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    for key, expected in environment.items():
        if key in ("python", "transformers"):
            continue  # Python belongs to the pinned image; framework to the requested SHA.
        if (
            importlib.metadata.version(key).split("+", 1)[0]
            != expected.split("+", 1)[0]
        ):
            raise ValueError(f"Dependency drift: {key}")


if __name__ == "__main__":
    main()
