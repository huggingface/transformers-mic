"""Integrations a publish carries over from the Hub instead of re-checking.

A MIC can declare integrations that another fleet owns: Sentence Transformers is
checked by its own fleet and nightly. Re-measuring the Transformers integration
should not demand that every other integration is checked again when this
workspace did not change it, so publication compares each one with the Hub:

- the primary integration (Transformers when declared) always needs its checks;
- another integration is carried over only when everything that decides its
  result is identical to the published revision the workspace is based on (the
  lock's pin): every local file under ``integrations/<name>/`` (entry point and
  reviewed expectations), its manifest section, the comparison rules it
  inherits, the checkpoints and fixtures it runs, and the fixture's weights
  (``build.json``). The Hub's ``main``, which the publish commits onto, must
  still carry the same manifest section and entry point, since the publish
  uploads the workspace's ``integration.json`` and the lock pins the contract
  files' digest (expectations other fleets added there since are kept);
- anything else (a difference, a first publish, the Hub unreachable) needs its
  checks, as before.

``publish_mic.py --carry NAME`` then neither uploads nor deletes that
integration's files, so the Hub keeps them byte for byte.
"""

import hashlib
import json
from pathlib import Path

from .hub import canonical_repo
from .workspace import digest, primary_integration

SKIPPED_PARTS = {".cache", "__pycache__", "node_modules", ".git"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def integration_files(root, manifest, framework):
    """{path: sha256} of what decides one integration: its folder and entry point."""
    folder = Path(root) / "integrations" / framework
    paths = {
        p
        for p in folder.rglob("*")
        if p.is_file()
        and not p.name.endswith(".candidate.json")
        and not SKIPPED_PARTS & set(p.relative_to(root).parts)
    }
    entry = manifest["integrations"][framework].get("entry_point")
    if entry and (Path(root) / entry).is_file():
        paths.add(Path(root) / entry)
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths}


def manifest_view(manifest, framework):
    """The parts of integration.json one integration's result depends on."""
    section = manifest.get("integrations", {}).get(framework)
    if section is None:
        return None
    return {
        "integration": section,
        "outputs": section.get("outputs", manifest.get("outputs")),
        "checkpoints": manifest.get("checkpoints"),
        "fixtures": manifest.get("fixtures"),
    }


def published(repo, revision, frameworks):
    """The published integration.json, build.json and integration folders at one revision."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            canonical_repo(repo),
            revision=revision,
            allow_patterns=[
                "integration.json",
                "build.json",
                *(f"integrations/{name}/**" for name in frameworks),
            ],
        )
    )


def difference(
    workspace, framework, published_root, fixture_sha256, expectations=True
):
    """Why one integration differs from a published snapshot, or None when it does not.

    ``expectations=False`` compares only what the publish uploads or pins for it:
    the manifest and the files outside ``expectations/``."""
    root = workspace.contract_root
    manifest = workspace.manifest
    manifest_path = published_root / "integration.json"
    if not manifest_path.is_file():
        return "no integration.json"
    other = json.loads(manifest_path.read_text())
    if manifest_view(other, framework) is None:
        return "not declared"
    if manifest_view(manifest, framework) != manifest_view(other, framework):
        return "integration.json differs"
    build = published_root / "build.json"
    if fixture_sha256 is None or not build.is_file():
        return "no fixture pin to compare"
    if digest(json.loads(build.read_text())["sha256"]) != fixture_sha256:
        return "fixture weights differ"
    entry = manifest["integrations"][framework].get("entry_point")
    if not entry or not (root / entry).is_file():
        return f"entry point {entry} is missing"
    changed = sorted(
        path
        for path, value in integration_files(root, manifest, framework).items()
        if (expectations or "expectations" not in Path(path).parts)
        and (
            not (published_root / path).is_file()
            or sha256(published_root / path) != value
        )
    )
    if changed:
        return "files differ: " + ", ".join(changed)
    return None


def plan(workspace, *, fetch=None):
    """{integration: None when carried over unchanged, else why it needs its checks}.

    The primary integration is always checked (reason "primary")."""
    manifest = workspace.manifest
    primary = primary_integration(manifest)
    others = [name for name in manifest["integrations"] if name != primary]
    result = {primary: "primary"}
    if not others:
        return result
    contract = workspace.lock["contracts"][workspace.state["contract"]]
    pin = contract["integration"]
    revision = pin.get("revision")
    if not revision or revision == "unpublished":
        return {**result, **{name: "not published yet" for name in others}}
    fixture = contract.get("fixtures", {}).get(workspace.fixture, {})
    fixture_sha256 = fixture.get("files_sha256")
    try:
        fetch = fetch or published
        base = fetch(pin["repo"], revision, others)
        main = fetch(pin["repo"], "main", others)
    except Exception as error:  # offline or unknown revision: check everything
        reason = f"published revision unavailable ({type(error).__name__})"
        return {**result, **{name: reason for name in others}}
    for name in others:
        if why := difference(workspace, name, base, fixture_sha256):
            result[name] = f"{why} vs {revision[:12]}"
        elif why := difference(
            workspace, name, main, fixture_sha256, expectations=False
        ):
            result[name] = (
                f"{why} on the Hub's main since {revision[:12]}; base the workspace on main"
            )
        else:
            result[name] = None
    return result


def describe(name, reason, revision):
    if reason == "primary":
        return "checked"
    if reason is None:
        owner = (
            "checked by its own fleet"
            if name == "sentence_transformers"
            else "not re-checked"
        )
        return f"unchanged since {revision[:12]}, carried over; {owner}"
    return f"changed ({reason}); checked"
