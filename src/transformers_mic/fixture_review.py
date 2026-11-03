"""tmic fixture: the tiny model as a local folder to review before anything runs on it.

`fixture/` in the workspace holds the shrunk config.json and weights in the
serialized format the source checkpoint uses. Before the build it is a preview
(seeded random weights, no contract code run); after the build it is a copy of
the built fixture. Editing `fixture/config.json` is how a reviewer changes the
shrink: the edited values become the recipe's `config_overrides`, and the next
`tmic fixture` (or `tmic run`) shrinks again with them.
"""

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from . import backend
from .workspace import Workspace, digest, read, write

FOLDER = "fixture"
RECORD = ".mic/fixture-review.json"
PAGE = ".mic/fixture-review.html"
# Fields that change with the environment or loader, never with the shrink.
VOLATILE = {"transformers_version", "_name_or_path", "_commit_hash"}
EDIT_REASON = "Set by the reviewer in fixture/config.json during tmic fixture review"


def folder(workspace):
    return Path(workspace.root) / FOLDER


def recipe_key(workspace):
    """What decides the shrunk config and tensor names: recipe, sources and their pins."""
    fixture = workspace.manifest["fixtures"][workspace.fixture]
    contract = workspace.lock["contracts"][workspace.state["contract"]]
    return digest(
        {
            "fixture": workspace.fixture,
            "build": fixture.get("build"),
            "built_from": fixture.get("built_from"),
            "pins": {
                repo: contract["checkpoints"].get(repo)
                for repo in fixture.get("covers", [])
            },
        }
    )


def needed(workspace):
    """A fixture this run will build: declared, with a recipe, not already on the Hub."""
    manifest = workspace.manifest
    if "fixtures" not in manifest["integrations"]["transformers"]["targets"]:
        return False
    if not (manifest["fixtures"].get(workspace.fixture) or {}).get("build"):
        return False
    pin = workspace.lock["contracts"][workspace.state["contract"]]["fixtures"][
        workspace.fixture
    ]
    return pin.get("revision") == "unpublished"


def record(workspace):
    path = Path(workspace.root) / RECORD
    return read(path) if path.exists() else None


def comparable(config):
    return {k: v for k, v in config.items() if k not in VOLATILE}


def review_sha256(data):
    """What the reviewer approves: config, serialized names and file format, not weight values."""
    return digest(
        {
            "config": comparable(data["config"]),
            "fixture_keys": data["fixture_keys"],
            "weights_file": data["weights_file"],
            "source": data["source"],
        }
    )


def fresh(workspace):
    data = record(workspace)
    return bool(
        data
        and data.get("recipe_key") == recipe_key(workspace)
        and (folder(workspace) / "config.json").exists()
    )


def flatten(value, prefix=""):
    if isinstance(value, dict) and value:
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


def edits(workspace):
    """Leaf values the reviewer changed or added in fixture/config.json, as dotted paths."""
    data = record(workspace)
    path = folder(workspace) / "config.json"
    if not data or not path.exists():
        return {}
    try:
        edited = json.loads(path.read_text())
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    before = flatten(comparable(data["config"]))
    after = flatten(comparable(edited))
    return {
        key: value for key, value in after.items() if before.get(key, object()) != value
    }


def apply_edits(workspace, changes):
    """Turn reviewer edits into the recipe's config_overrides; returns the fields set."""
    manifest = workspace.manifest
    recipe = manifest["fixtures"][workspace.fixture]["build"]
    overrides = recipe.setdefault(
        "config_overrides", {"fields": {}, "reason": EDIT_REASON}
    )
    overrides.setdefault("fields", {}).update(changes)
    if EDIT_REASON not in overrides.get("reason", ""):
        overrides["reason"] = (
            overrides.get("reason", "").rstrip(". ") + ". " + EDIT_REASON
        ).lstrip(". ")
    write(workspace.contract_root / "integration.json", manifest)
    workspace.update_layout()
    return sorted(changes)


def artifact(workspace):
    """The built fixture for the current recipe, if the build already made one."""
    pin = workspace.lock["contracts"][workspace.state["contract"]]["fixtures"][
        workspace.fixture
    ]
    if not pin.get("path") or not pin.get("files_sha256"):
        return None
    path = (workspace.lock_path.parent / pin["path"]).resolve()
    if not (path / "build.json").exists():
        return None
    built = read(path / "build.json")
    fixture = workspace.manifest["fixtures"][workspace.fixture]

    def shape(recipe):
        # `tmic build --weights overfit` adds "overfit" to the recipe it records;
        # it changes the weight values, never the config or the names.
        return {k: v for k, v in (recipe or {}).items() if k != "overfit"}

    if shape(built.get("recipe")) != shape(fixture.get("build")) or built.get(
        "built_from", {}
    ).get("repo") != fixture.get("built_from"):
        return None
    return path


def stale(workspace):
    """fixture/ no longer shows what the build will use: edited, re-recipied or rebuilt."""
    if not fresh(workspace) or edits(workspace):
        return True
    built = artifact(workspace)
    return record(workspace).get("artifact") != (str(built) if built else None)


def refresh(workspace, force=False):
    """Write fixture/ and its review record; reuse them while the recipe is unchanged."""
    if not force and not stale(workspace):
        return record(workspace), False
    built = artifact(workspace)
    root = Path(workspace.root)
    with tempfile.TemporaryDirectory(prefix="fixture-", dir=root / ".mic") as temp:
        out = Path(temp) / FOLDER
        result = Path(temp) / "review.json"
        if built:
            shutil.copytree(built, out)
        options = [
            workspace.state["contract"],
            workspace.fixture,
            str(out),
            str(result),
        ]
        if built:
            options.append(str(built))
        process = backend.invoke("fixture-stage", root, options)
        if process.returncode:
            raise SystemExit(process.returncode)
        data = read(result)
        target = folder(workspace)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(out), target)
    (root / PAGE).write_text(data.pop("page"))
    data.update(
        recipe_key=recipe_key(workspace),
        artifact=str(built) if built else None,
        folder=str(folder(workspace)),
    )
    data["review_sha256"] = review_sha256(data)
    write(root / RECORD, data)
    return data, True


NUMBER = re.compile(r"(?<=\.)\d+(?=\.|$)")


def patterns(keys):
    """Tensor names with layer/expert indices collapsed: 88 layers read as one line."""
    return Counter(NUMBER.sub("{n}", key) for key in keys)


def weight_names(data):
    source, fixture = patterns(data["source_keys"]), patterns(data["fixture_keys"])
    # The fixture omits what the real checkpoint lacks (tied aliases): not a difference.
    lacking = set(patterns(data.get("missing_parameters") or []))
    return {
        "matching": sorted(set(source) & set(fixture)),
        "only_source": sorted(set(source) - set(fixture) - lacking),
        "only_fixture": sorted(set(fixture) - set(source)),
        "tensors": {
            "source": len(data["source_keys"]),
            "fixture": len(data["fixture_keys"]),
        },
    }


def human(n):
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def headline(data):
    """Four short lines for the prompt; details() has everything."""
    names = weight_names(data)
    p = data.get("parameters") or {}
    size = (
        f"{human(p['original'])} → {human(p['fixture'])} parameters"
        if p.get("original")
        else ""
    )
    changes = data["config_changes"]
    shown = ", ".join(
        f"{c['field']} {c['original']}→{c['fixture']}" for c in changes[:3]
    )
    more = f", +{len(changes) - 3} more" if len(changes) > 3 else ""
    if names["only_source"] and not names["only_fixture"]:
        weights = (
            f"Weight names: {len(names['matching'])} match the source; "
            f"{len(names['only_source'])} only in the source (unused heads or quantization scales)"
        )
    elif names["only_fixture"]:
        weights = (
            f"Weight names: {len(names['only_source'])} only in the source, "
            f"{len(names['only_fixture'])} only in the tiny model, {len(names['matching'])} match"
        )
    else:
        weights = (
            f"Weight names: all {len(names['matching'])} patterns match the source"
        )
    findings = data["findings"]
    from .terminal import short_path

    return [
        f"Folder: {short_path(data['folder'])}  (config.json, {data['weights_file']})",
        " · ".join(x for x in (data["model_class"], size) if x),
        f"Config: {len(changes)} values differ from the source"
        + (f" ({shown}{more})" if changes else ""),
        weights,
        "Findings: " + ("; ".join(findings) if findings else "none"),
    ]


def details(data, page=None):
    names = weight_names(data)
    source = f"{data['source']['repo']}@{data['source']['revision'][:12]}"
    p = data.get("parameters") or {}
    lines = [
        f"Tiny model folder: {data['folder']}",
        f"  {', '.join(data['files'])}",
        "  To change the config, edit config.json there and continue: tmic shrinks again with your values.",
        f"Source: {source} · {data['model_class']}",
    ]
    if p.get("original"):
        lines.append(f"Parameters: {human(p['original'])} → {human(p['fixture'])}")
    lines.append(f"Weights: {data['weights']} · file {data['weights_file']}")
    lines += [
        "",
        f"Config: {len(data['config_changes'])} values differ from the source ({data['config_unchanged']} unchanged)",
    ]
    width = max((len(c["field"]) for c in data["config_changes"]), default=0)
    lines += [
        f"  {c['field']:<{width}}  {c['original']} → {c['fixture']}"
        for c in data["config_changes"]
    ]
    lines += [
        "",
        (
            f"Serialized weight names ({names['tensors']['source']} tensors in the source, "
            f"{names['tensors']['fixture']} in the tiny model; layer/expert numbers shown as {{n}}):"
        ),
    ]
    unused = (
        f"Only in the source: the tiny {data['model_class']} has no such weights "
        "(heads the class does not use, or a quantized source's scales: the tiny model is dense)"
        if not names["only_fixture"]
        else "Only in the source"
    )
    for title, key in (
        (unused, "only_source"),
        ("Only in the tiny model", "only_fixture"),
    ):
        if names[key]:
            lines.append(f"  {title} ({len(names[key])}):")
            lines += [f"    {name}" for name in names[key]]
    lines.append(f"  Matching the source ({len(names['matching'])}):")
    lines += [f"    {name}" for name in names["matching"]]
    if data.get("missing_parameters"):
        lines.append(
            f"  Not serialized, as in the source: {', '.join(data['missing_parameters'])}"
        )
    lines += ["", "Findings: " + ("none" if not data["findings"] else "")]
    lines += [f"  - {f}" for f in data["findings"]]
    if data.get("tester_notes"):
        lines.append("Model-tester notes (advisory):")
        lines += [f"  - {f}" for f in data["tester_notes"]]
    if page:
        lines += ["", f"Layer-by-layer comparison: {page}"]
    return "\n".join(lines)


def summary(data):
    """The machine-readable part of the question context."""
    names = weight_names(data)
    return {
        "folder": data["folder"],
        "model_class": data["model_class"],
        "weights_file": data["weights_file"],
        "parameters": data.get("parameters"),
        "config_changes": len(data["config_changes"]),
        "weight_names": {
            "matching": len(names["matching"]),
            "only_source": names["only_source"],
            "only_fixture": names["only_fixture"],
        },
        "findings": data["findings"],
    }


def run(args):
    workspace = Workspace(args.workspace)
    if not (workspace.manifest["fixtures"].get(workspace.fixture) or {}).get("build"):
        raise ValueError(
            "This fixture has no build recipe, so there is nothing to shrink"
        )
    changes = edits(workspace)
    if changes:
        print(
            "Applying your config.json edits as config_overrides: "
            + ", ".join(apply_edits(workspace, changes))
        )
    data, _ = refresh(workspace, force=args.force)
    page = Path(workspace.root) / PAGE
    if args.json:
        print(
            json.dumps(
                {
                    **summary(data),
                    "review_sha256": data["review_sha256"],
                    "page": str(page),
                },
                indent=2,
            )
        )
    else:
        print(details(data, page))
    if args.open:
        import webbrowser

        webbrowser.open(page.resolve().as_uri())
    return 0


def snapshot(path):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
