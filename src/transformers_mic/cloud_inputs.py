"""Immutable, model-scoped baseline input on a separate Hub branch."""

import os
import re

from huggingface_hub import hf_hub_download

from .workspace import read


def from_env():
    repo = os.getenv("MIC_WORKSPACE_REPO", "")
    revision = os.getenv("MIC_WORKSPACE_REVISION", "")
    if not repo and not revision:
        return None
    allowed = os.getenv("MIC_BASELINE_INPUT_NAMESPACES", "huggingface-mic").split(",")
    if (
        not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*-mic", repo
        )
        or repo.split("/")[0] not in allowed
        or not re.fullmatch(r"[a-f0-9]{40}", revision)
    ):
        raise ValueError(
            "Model input must be an approved MIC repository at an immutable revision"
        )
    return {"repo": repo, "revision": revision}


def lock(workspace):
    result = read(
        hf_hub_download(
            workspace["repo"], ".mic/baseline-lock.json", revision=workspace["revision"]
        )
    )
    if result.get("local_patches") or len(result["contracts"]) != 1:
        raise ValueError("Cloud model input requires one unpatched contract")
    contract = next(iter(result["contracts"].values()))
    if set(contract.get("fixtures", {})) != {workspace["repo"]}:
        raise ValueError("Model snapshot must contain its single slim fixture")
    for pin in [contract["integration"], *contract["fixtures"].values()]:
        if pin.get("repo", workspace["repo"]) != workspace["repo"] or not re.fullmatch(
            r"[a-f0-9]{64}", pin.get("files_sha256", "")
        ):
            raise ValueError("Model snapshot has an invalid source pin")
        pin.pop("path", None)
        pin["revision"] = workspace["revision"]
    return result
