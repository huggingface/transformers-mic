"""Install the exact framework commit into Transformers' CI image for baseline measurement."""

import importlib.metadata
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from transformers_mic import fleet, cloud_inputs
from transformers_mic.workspace import read


def main():
    revision = os.environ["FLEET_REVISION"]
    sha = os.environ["TRANSFORMERS_SHA"]
    if not all(re.fullmatch(r"[a-f0-9]{40}", value) for value in (revision, sha)):
        raise ValueError("Immutable revisions required")
    environments = []
    workspace = cloud_inputs.from_env()
    if workspace:
        environments.append(cloud_inputs.lock(workspace)["environment"])
    else:
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
    # Constrain every install to the image's own versions: missing tools are added,
    # nothing the image ships (numpy, torch, ...) is upgraded or downgraded.
    frozen = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    constraints = checkout.parent / "image-constraints.txt"
    constraints.write_text(
        "\n".join(
            line
            for line in frozen
            # Packages installed by name below (the framework, the pinned doc-builder,
            # as Transformers CI installs them) are not held to the image's version.
            if "==" in line
            and not line.lower().startswith(
                ("transformers==", "hf-doc-builder==", "-e")
            )
        )
        + "\n"
    )
    pip = [sys.executable, "-m", "pip", "install", "-c", str(constraints)]
    # Measure in Transformers' own CI image: its torch, numpy and system libraries are
    # what fast-integration and the GPU nightly run, so nothing here overrides them.
    # Install the framework commit and the contract tools the way that CI does.
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
            "jsonschema",
            "hf-doc-builder @ git+https://github.com/huggingface/doc-builder@c69e96c087165ba49b1788e0c3464654dd60253c",
        ],
        check=True,
    )
    try:
        importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        # An image without torch (an operator override): follow the Dockerfiles.
        torch_pins, index = transformers_torch_stack(checkout)
        subprocess.run(
            [
                *pip,
                *torch_pins,
                "--index-url",
                "https://download.pytorch.org/whl/" + index,
            ],
            check=True,
        )
    for key in sorted(environment):
        try:
            print(f"{key}: {importlib.metadata.version(key)}", flush=True)
        except importlib.metadata.PackageNotFoundError:
            print(f"{key}: not installed", flush=True)


def transformers_torch_stack(checkout):
    """Torch requirements and wheel index from Transformers' CI Dockerfiles."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        # docker/torch-light.dockerfile: uv pip install 'torch<=X' 'torchaudio' ... --index-url .../cpu
        text = (checkout / "docker/torch-light.dockerfile").read_text()
        line = next(
            line
            for line in text.splitlines()
            if "pip install" in line and re.search(r"[\s']torch[<=>' ]", line)
        )
        specs = re.findall(r"'(torch[a-z]*[^']*)'", line)
        if not any(re.split(r"[<>=!~ ]", spec)[0] == "torch" for spec in specs):
            raise ValueError(f"No torch requirement in torch-light.dockerfile: {line}")
        return specs, "cpu"
    # docker/transformers-all-latest-gpu/Dockerfile: ARG PYTORCH='X', ARG CUDA='cuNNN'
    text = (checkout / "docker/transformers-all-latest-gpu/Dockerfile").read_text()
    args = dict(re.findall(r"^ARG (PYTORCH|CUDA)='([^']+)'", text, re.M))
    if set(args) != {"PYTORCH", "CUDA"}:
        raise ValueError(
            "No PYTORCH/CUDA args in transformers-all-latest-gpu/Dockerfile"
        )
    return [f"torch=={args['PYTORCH']}", "torchvision", "torchaudio"], args["CUDA"]


if __name__ == "__main__":
    main()
