"""Shared fleet snapshots and the framework's single revision pin."""

import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPOSITORY = "huggingface-mic/transformers-fleet"
FILES = {"control": "fleet-lock.json", "prototypes": "prototype-fleet.lock"}


def resolve(revision="main"):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    return HfApi().dataset_info(REPOSITORY, revision=revision).sha


def read_pin(path):
    revision = Path(path).read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{path} must contain one immutable dataset commit SHA")
    return revision


def download(name="control", revision="main"):
    revision = resolve(revision)
    return Path(
        hf_hub_download(REPOSITORY, FILES[name], repo_type="dataset", revision=revision)
    ), revision


def pin(checkout, revision):
    path = Path(checkout).resolve() / "tests/contracts.lock"
    # Resolve named refs before writing; a framework pin must always be immutable.
    revision = resolve(revision)
    for name in FILES:
        download(name, revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(revision + "\n")
    return path
