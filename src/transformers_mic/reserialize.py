"""Store a published fixture's pytorch_model.bin as model.safetensors.

Fixtures built before transformers-mic #177 kept the source checkpoint's
container. The weights are rewritten, never rebuilt: the same tensor names
(tied aliases included) and bit-identical values, so the approved baselines
still describe the fixture. Only the file and its hash in build.json change,
which gives the fixture a new files_sha256 and therefore a new fleet pin.

  tmic reserialize huggingface-mic/opt-125m-mic            # convert and verify, upload nothing
  tmic reserialize huggingface-mic/opt-125m-mic --execute  # commit to the repo, repin every fleet
"""

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.errors import HfHubHTTPError

from . import fleet, resources
from .workspace import read

BINARY = "pytorch_model.bin"
SAFETENSORS = "model.safetensors"
COMMIT_MESSAGE = "Store the fixture weights as model.safetensors (tensor names and values unchanged)"


def runner():
    sys.path.insert(0, str(resources.backend()))
    return importlib.import_module("run")


def _bytes(tensor):
    import torch

    return tensor.detach().reshape(-1).contiguous().view(torch.uint8)


def differences(before, after):
    """Why two {name: tensor} maps are not the same tensors bit for bit; [] if they are."""
    import torch

    problems = []
    if set(before) != set(after):
        problems.append(
            f"tensor names differ: only before {sorted(set(before) - set(after))}, "
            f"only after {sorted(set(after) - set(before))}"
        )
    for name in sorted(set(before) & set(after)):
        a, b = before[name], after[name]
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            problems.append(f"{name}: {a.dtype}{tuple(a.shape)} became {b.dtype}{tuple(b.shape)}")
        elif not torch.equal(_bytes(a), _bytes(b)):
            problems.append(f"{name}: values differ")
    return problems


def _dump(build, original):
    """build.json in the indentation the file already uses (builders wrote 1, migrations 2)."""
    for indent in (1, 2):
        if json.dumps(json.loads(original), indent=indent, default=str) + "\n" == original:
            return json.dumps(build, indent=indent, default=str) + "\n"
    return json.dumps(build, indent=1, default=str) + "\n"


def convert(folder, source_file=None):
    """Rewrite folder/pytorch_model.bin as model.safetensors in place and update build.json.

    Every tensor keeps its name, dtype, shape and bytes. Tensors sharing storage
    (tied aliases) are cloned, since safetensors refuses shared memory, and both
    names are kept. Returns {"tensors": n, "sha256": new file hash}.
    """
    import torch
    from safetensors.torch import load_file, save_file

    folder = Path(folder)
    original = (folder / "build.json").read_text()
    build = json.loads(original)
    if not (folder / BINARY).exists():
        raise ValueError(f"{folder}: no {BINARY} to convert")
    if (folder / SAFETENSORS).exists():
        raise ValueError(f"{folder}: already has {SAFETENSORS}")
    weights = [f for f in build["sha256"] if f.endswith((".bin", ".safetensors")) or f.endswith(".bin.index.json")]
    if weights != [BINARY]:
        raise ValueError(f"{folder}: expected exactly one weights file, {BINARY}; build.json lists {weights}")
    state = torch.load(folder / BINARY, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{folder}: {BINARY} is not a flat {{name: tensor}} state dict")
    save_file(
        {name: tensor.detach().clone().contiguous() for name, tensor in state.items()},
        folder / SAFETENSORS,
        metadata={"format": "pt"},
    )
    problems = differences(state, load_file(folder / SAFETENSORS))
    if problems:
        (folder / SAFETENSORS).unlink()
        raise ValueError(f"{folder}: safetensors copy differs: {problems[:5]}")
    (folder / BINARY).unlink()
    digest = hashlib.sha256((folder / SAFETENSORS).read_bytes()).hexdigest()
    build["sha256"] = dict(sorted(
        (SAFETENSORS if name == BINARY else name, digest if name == BINARY else value)
        for name, value in build["sha256"].items()
    ))
    serialization = dict(build.get("serialization") or {})
    serialization.update(
        file=SAFETENSORS,
        source_file=serialization.get("source_file") or source_file or BINARY,
        migrated_from=BINARY,
    )
    build["serialization"] = serialization
    (folder / "build.json").write_text(_dump(build, original))
    return {"tensors": len(state), "sha256": digest}


def loaded_differences(before, after, model_class):
    """Load both folders with from_pretrained; why the loaded models differ, [] if they do not.

    Also refuses unexpected or mismatched keys, and missing keys beyond the ones
    build.json says the real checkpoint lacks."""
    import transformers

    cls = getattr(transformers, model_class)
    models = {}
    for label, folder in (("before", before), ("after", after)):
        model, info = cls.from_pretrained(folder, output_loading_info=True)
        models[label] = (model.state_dict(), {k: sorted(map(str, v)) for k, v in info.items()})
    problems = differences(models["before"][0], models["after"][0])
    if models["before"][1] != models["after"][1]:
        problems.append(f"loading info differs: {models['before'][1]} vs {models['after'][1]}")
    info = models["after"][1]
    lacks = set(read(Path(after) / "build.json").get("missing_parameters", []))
    if info.get("unexpected_keys") or info.get("mismatched_keys"):
        problems.append(f"unexpected {info.get('unexpected_keys')}, mismatched {info.get('mismatched_keys')}")
    if extra := sorted(set(info.get("missing_keys", [])) - lacks):
        problems.append(f"missing keys the real checkpoint has: {extra}")
    return problems


def source_weights_file(build):
    """The file the source checkpoint loads from (recorded as serialization.source_file)."""
    source = build["built_from"]
    files = HfApi().list_repo_files(source["repo"], revision=source["revision"])
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if name in files:
            return name
    return SAFETENSORS if SAFETENSORS in files else BINARY


def users(repo):
    """{fleet: [contract names]} of every fleet contract that pins this repository."""
    found = {}
    for name in fleet.FILES:
        lock = read(fleet.download(name)[0])
        found[name] = sorted(c for c, contract in lock["contracts"].items() if repo in contract.get("fixtures", {}))
    return {name: contracts for name, contracts in found.items() if contracts}


def repin(lock, repo, old_sha256, revision, files_sha256):
    """Point every pin of `repo` in a fleet lock at its new revision; returns the contract names.

    The contract a repository carries keeps its files_sha256: only the weights changed."""
    changed = []
    for name, contract in lock["contracts"].items():
        pin = contract.get("fixtures", {}).get(repo)
        if pin is None:
            continue
        if pin["files_sha256"] != old_sha256:
            raise ValueError(f"{name}: the fleet pins another build of {repo}; export that first")
        pin.update(revision=revision, files_sha256=files_sha256)
        if contract["integration"]["repo"] == repo:
            contract["integration"]["revision"] = revision
        changed.append(name)
    return changed


def export(repo, old_sha256, revision, files_sha256, fleets, attempts=5):
    """Commit the new pins to each fleet, retrying a commit another export raced."""
    revisions = {}
    for name in fleets:
        for attempt in range(attempts):
            path, parent = fleet.download(name)
            lock = read(path)
            changed = repin(lock, repo, old_sha256, revision, files_sha256)
            try:
                commit = HfApi().create_commit(
                    repo_id=fleet.REPOSITORY,
                    repo_type="dataset",
                    parent_commit=parent,
                    operations=[
                        CommitOperationAdd(path_in_repo=p, path_or_fileobj=text.encode())
                        for p, text in fleet.operations(name, lock).items()
                    ],
                    commit_message=f"Repin {', '.join(changed)} in the {name} fleet: {repo.split('/')[-1]} weights as {SAFETENSORS}",
                )
            except HfHubHTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status in (409, 412) and attempt + 1 < attempts:
                    continue  # the dataset moved since download: repin on the new head
                raise
            revisions[name] = {"revision": commit.oid, "contracts": changed}
            break
    return revisions


def stage(repo, revision, folder):
    """Download the fixture and contract files of `repo@revision` one by one (never a snapshot copy)."""
    build = read(hf_hub_download(repo, "build.json", revision=revision))
    names = ["build.json", *build["sha256"]]
    files = HfApi().list_repo_files(repo, revision=revision)
    for extra in (".mic-repo.json", "README.md"):
        if extra in files:
            names.append(extra)
    for name in names:
        target = Path(folder) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(repo, name, revision=revision), target)
    return build


def resume(repo, head, run_):
    """Repin the fleets after a reserialize commit whose export did not finish."""
    api = HfApi()
    commits = api.list_repo_commits(repo)
    if commits[0].commit_id != head or commits[0].title != COMMIT_MESSAGE or len(commits) < 2:
        raise ValueError(f"{repo}@{head[:10]}: no {BINARY}, and the head is not a reserialize commit")
    new_sha256 = run_.digest(read(hf_hub_download(repo, "build.json", revision=head))["sha256"])
    old_sha256 = run_.digest(read(hf_hub_download(repo, "build.json", revision=commits[1].commit_id))["sha256"])
    lagging = []
    for name in fleet.FILES:
        lock = read(fleet.download(name)[0])
        pins = [c["fixtures"][repo] for c in lock["contracts"].values() if repo in c.get("fixtures", {})]
        if any(p["files_sha256"] == old_sha256 for p in pins):
            lagging.append(name)
        elif any(p["files_sha256"] != new_sha256 for p in pins):
            raise ValueError(f"{name}: the fleet pins another build of {repo}")
    return old_sha256, new_sha256, lagging


def run(args):
    repo = args.repo
    run_ = runner()
    api = HfApi()
    head = api.model_info(repo).sha
    if BINARY not in api.list_repo_files(repo, revision=head):
        old_sha256, new_sha256, lagging = resume(repo, head, run_)
        result = {"repo": repo, "head": head, "converted": "already", "lagging_fleets": lagging, "execute": args.execute}
        if args.execute and lagging:
            result["fleet_revisions"] = export(repo, old_sha256, head, new_sha256, lagging)
        print(json.dumps(result, indent=2))
        return 0
    with tempfile.TemporaryDirectory(prefix="tmic-reserialize-") as tmp:
        before, after = Path(tmp) / "before", Path(tmp) / "after"
        build = stage(repo, head, before)
        old_sha256 = run_.digest(build["sha256"])
        if problem := run_.fixture_problem(before, {"files_sha256": old_sha256}):
            raise ValueError(f"{repo}@{head[:10]}: {problem}")
        pinned = users(repo)
        locks = {name: read(fleet.download(name)[0]) for name in pinned}
        for name, contracts in pinned.items():
            for contract in contracts:
                entry = locks[name]["contracts"][contract]
                if entry["fixtures"][repo]["files_sha256"] != old_sha256:
                    raise ValueError(f"{name}/{contract}: the fleet pins another build of {repo} than its head")
                if entry["integration"]["repo"] == repo:
                    files = json.loads((before / ".mic-repo.json").read_text())["contract_files"]
                    contract_files = {
                        f: hashlib.sha256(Path(hf_hub_download(repo, f, revision=head)).read_bytes()).hexdigest()
                        for f in sorted(files)
                    }
                    if run_.digest(contract_files) != entry["integration"]["files_sha256"]:
                        raise ValueError(
                            f"{name}/{contract}: the contract at {repo}@{head[:10]} differs from its fleet pin; "
                            "export that change first"
                        )
        shutil.copytree(before, after)
        converted = convert(after, source_weights_file(build))
        new_sha256 = run_.digest(read(after / "build.json")["sha256"])
        if problem := run_.fixture_problem(after, {"files_sha256": new_sha256}):
            raise ValueError(f"{repo}: converted fixture is inconsistent: {problem}")
        problems = loaded_differences(before, after, build["model_class"])
        if problems:
            raise ValueError(f"{repo}: the loaded model changed: {problems[:5]}")
        result = {
            "repo": repo,
            "head": head,
            "tensors": converted["tensors"],
            "fixture_sha256": {"before": old_sha256, "after": new_sha256},
            "fleets": pinned,
            "execute": args.execute,
        }
        if args.out:
            shutil.copytree(after, args.out, dirs_exist_ok=True)
            result["out"] = str(args.out)
        if args.execute:
            commit = api.create_commit(
                repo_id=repo,
                parent_commit=head,
                operations=[
                    CommitOperationDelete(path_in_repo=BINARY),
                    CommitOperationAdd(path_in_repo=SAFETENSORS, path_or_fileobj=str(after / SAFETENSORS)),
                    CommitOperationAdd(path_in_repo="build.json", path_or_fileobj=str(after / "build.json")),
                ],
                commit_message=COMMIT_MESSAGE,
            )
            result["revision"] = commit.oid
            result["fleet_revisions"] = export(repo, old_sha256, commit.oid, new_sha256, list(pinned))
    result["next"] = "Done." if args.execute else "Review, then repeat with --execute to commit and repin."
    print(json.dumps(result, indent=2))
    return 0
