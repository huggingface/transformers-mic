"""User workflows; expected values are produced only by the execution backend."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from huggingface_hub import HfApi, hf_hub_download
import jsonschema

from . import backend, history, planner, policy, resources
from .workspace import Workspace, digest, read, write


def manifest_schema():
    return read(resources.backend() / "schemas/integration.schema.json")


def validate_plan(plan, model, fixture, old=None):
    jsonschema.validate(plan, planner.PLAN_SCHEMA)
    manifest = plan["manifest"]
    jsonschema.validate(manifest, manifest_schema())
    if set(manifest["checkpoints"]) != {model} or set(manifest["fixtures"]) != {
        fixture
    }:
        raise ValueError(
            "Generated plan must contain exactly the requested checkpoint and fixture"
        )
    integration = manifest["integrations"]["transformers"]
    if (
        integration["entry_point"] != "integrations/transformers/usage.md"
        or integration["format"] != "runnable-markdown"
    ):
        raise ValueError(
            "Generated entry point must be integrations/transformers/usage.md, runnable-markdown"
        )
    recipe = manifest["fixtures"][fixture]["build"]
    previous_recipe = old["fixtures"][fixture].get("build", {}) if old else {}
    for key in ("overfit", "accept_insensitive"):
        if recipe.get(key) != previous_recipe.get(key):
            raise ValueError(
                "The LLM cannot select overfit or exempt insensitive inputs"
            )
    if (
        manifest["checkpoints"][model]["fixture"] != fixture
        or manifest["fixtures"][fixture]["built_from"] != model
    ):
        raise ValueError(
            "Generated fixture must be built from and assigned to the requested checkpoint"
        )
    for rule in manifest["outputs"].values():
        if rule["compare"] == "allclose" and (
            rule["atol"] != 1e-4 or rule["rtol"] != 1e-4
        ):
            raise ValueError("Generated numeric tolerances must be atol=rtol=1e-4")
    if old and not set(old["outputs"]).issubset(manifest["outputs"]):
        raise ValueError("Update removes existing output assertions")
    # Syntax is checked without importing or running generated code.
    blocks = re.findall(r"```py runnable:[^\n]*\n(.*?)```", plan["usage"], re.S)
    if not blocks:
        raise ValueError("Generated usage contains no runnable Python blocks")
    ast.parse("\n".join(blocks))
    return manifest


def model_context(model, revision=None):
    api = HfApi()
    info = api.model_info(model, revision=revision)
    config = read(hf_hub_download(model, "config.json", revision=info.sha))
    return info.sha, config


def create(args):
    workspace = Workspace(args.out)
    workspace.ensure_new()
    catalog = read(resources.catalog_lock())
    names = [
        n for n, c in catalog["contracts"].items() if args.model in c["checkpoints"]
    ]
    if args.contract:
        names = [n for n in names if n == args.contract]
    if names and not args.llm:
        name = names[0]
        workspace.from_catalog(args.model, name, args.weights, args.readable_output)
        write(
            workspace.root / ".mic/plan.json",
            {
                "origin": "catalog",
                "contract": name,
                "checkpoint_pins": workspace.lock["contracts"][name]["checkpoints"],
                "decision": "Reuse published usage, fixtures and expectations; no LLM needed.",
            },
        )
        if not args.offline and not args.no_history:
            metadata = read(
                workspace.root
                / "example/hub"
                / workspace.fixture.split("/")[-1]
                / "build.json"
            )
            write(
                workspace.root / ".mic/history.json",
                history.collect(
                    args.model,
                    metadata.get("key_fields", {}).get("model_type"),
                    args.history_repo,
                    args.limit,
                    args.history_file,
                ),
            )
    else:
        if args.offline:
            raise ValueError(
                "No catalog contract matches this model; new models need Hub metadata and an LLM usage plan"
            )
        revision, config = model_context(args.model, args.revision)
        namespace = args.namespace or HfApi().whoami()["name"]
        slug = re.sub(r"[^A-Za-z0-9_.-]", "-", args.model.split("/")[-1])
        fixture = f"{namespace}/proto-{slug}-slim"
        evidence = (
            {"items": [], "complete": False, "source": "not requested"}
            if args.no_history
            else history.collect(
                args.model,
                config.get("model_type"),
                args.history_repo,
                args.limit,
                args.history_file,
            )
        )
        context = {
            "model": args.model,
            "revision": revision,
            "config": config,
            "fixture_repo": fixture,
            "manifest_schema": manifest_schema(),
            "failure_candidates": evidence["items"][:30],
            "request": "Design a small useful integration path; include inference and a training step where supported.",
        }
        plan, llm = planner.propose(
            context, model=args.llm_model, provider=args.provider
        )
        manifest = validate_plan(plan, args.model, fixture)
        workspace.from_plan(
            args.model,
            revision,
            namespace,
            plan["name"],
            manifest,
            plan["usage"],
            args.weights,
            args.readable_output,
        )
        write(workspace.root / ".mic/plan.json", {**plan, "llm": llm})
        write(workspace.root / ".mic/history.json", evidence)
    print(
        json.dumps(
            {
                "workspace": str(workspace.root),
                "contract": workspace.state["contract"],
                "origin": workspace.state["origin"],
                "next": [
                    "mic build",
                    "mic record",
                    "mic review",
                    "mic check",
                    "mic publish",
                ],
            }
        )
    )
    if args.build:
        args.workspace = str(workspace.root)
        args.rebuild = False
        build(args)
    if args.record:
        args.workspace = str(workspace.root)
        args.variant, args.isolate, args.target = "default", False, None
        return execute(args, record=True)
    return 0


def research(args):
    workspace = Workspace(args.workspace)
    result = history.collect(
        workspace.state["model"],
        args.model_type,
        args.history_repo,
        args.limit,
        args.history_file,
    )
    write(workspace.root / ".mic/history.json", result)
    print(
        json.dumps(
            {
                "candidates": len(result["items"]),
                "complete": False,
                "path": str(workspace.root / ".mic/history.json"),
            }
        )
    )
    return 0


def build(args):
    workspace = Workspace(args.workspace)
    state, lock, manifest = workspace.state, workspace.lock, workspace.manifest
    fixture = workspace.fixture
    pin = lock["contracts"][state["contract"]]["fixtures"][fixture]
    requested = args.weights if args.weights else state["weights"]
    readable = args.readable_output or state["readable_output"]
    if "fixtures" not in manifest["integrations"]["transformers"]["targets"]:
        choice = {
            "weights": None,
            "accepted": True,
            "reason": "Checkpoint-only path; the manifest explicitly excludes slim execution.",
        }
        workspace.save_state(weight_decision=choice)
        print(json.dumps(choice))
        return 0
    if pin["revision"] != "unpublished" and not args.rebuild:
        metadata = read(
            workspace.root / "example/hub" / fixture.split("/")[-1] / "build.json"
        )
        choice = policy.decision(metadata, requested, readable, reused=True)
        if choice["accepted"]:
            workspace.save_state(weight_decision=choice)
            print(json.dumps(choice))
            return 0
    original_manifest = workspace.manifest
    recipe = manifest["fixtures"][fixture].get("build")
    if not recipe:
        raise ValueError(
            "This legacy fixture has no build recipe; add one before rebuilding"
        )
    if requested == "overfit" or readable:
        recipe["overfit"] = (
            "Explicit CLI overfit/readable-output requirement; historical detection still needs replay validation."
        )
    elif requested == "random":
        recipe.pop("overfit", None)
    write(workspace.contract_root / "integration.json", manifest)
    with tempfile.TemporaryDirectory(
        prefix="build-", dir=workspace.root / ".mic"
    ) as temp:
        out = Path(temp) / "fixture"
        # Explicit random builds must not silently fall back to overfit.
        stage_args = [state["contract"], fixture, str(out)]
        if requested == "random":
            stage_args.append("random")
        try:
            result = backend.invoke("build-stage", workspace.root, stage_args)
            if result.returncode:
                return result.returncode
            metadata = read(out / "build.json")
            choice = policy.decision(metadata, requested, readable)
            if not choice["accepted"]:
                print(json.dumps(choice))
                return 1
            artifact_hash = digest(metadata["sha256"])
            target = (
                workspace.root
                / ".mic/artifacts"
                / fixture.split("/")[-1]
                / artifact_hash
            )
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(out, target)
            pin.update(
                revision="unpublished",
                files_sha256=artifact_hash,
                path=os.path.relpath(target, workspace.lock_path.parent),
            )
            write(workspace.lock_path, lock)
            workspace.save_state(
                weight_decision=choice,
                validated=False,
                validation=None,
                weights=requested,
                readable_output=readable,
            )
            print(
                json.dumps(
                    {
                        **choice,
                        "artifact": str(target),
                        "next": "mic record; review the new candidates before publishing",
                    }
                )
            )
        finally:
            write(workspace.contract_root / "integration.json", original_manifest)
    return 0


def execute_one(args, record=False):
    workspace = Workspace(args.workspace)
    options = [
        "--framework",
        "transformers",
        "--contract",
        workspace.state["contract"],
        "--local-contracts",
        "--variant",
        args.variant,
        "--require-all",
    ]
    if args.isolate:
        options.append("--isolate")
    if args.target:
        for target in args.target:
            options += ["--target", target]
    if record:
        options.append("--record")
    output = (
        workspace.root
        / ".mic"
        / f"{'record' if record else 'check'}.{args.variant}.jsonl"
    )
    output.unlink(missing_ok=True)
    options += ["--output", str(output)]
    result = backend.invoke("run", workspace.root, options)
    if not record:
        # Publication requires every declared target, not a focused partial check.
        rows = (
            [json.loads(line) for line in output.read_text().splitlines()]
            if output.exists()
            else []
        )
        evidence = {
            "variant": args.variant,
            "records": rows,
            "lock_sha256": hashlib.sha256(workspace.lock_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                (workspace.contract_root / "integration.json").read_bytes()
            ).hexdigest(),
            "usage_sha256": hashlib.sha256(
                (
                    workspace.contract_root
                    / workspace.manifest["integrations"]["transformers"]["entry_point"]
                ).read_bytes()
            ).hexdigest(),
        }
        key = args.variant
        validations = workspace.state.get("validations", {})
        if not args.target and result.returncode == 0 and bool(rows):
            validations[key] = {
                **evidence,
                "fingerprint": workspace.validation_fingerprint(),
            }
        else:
            validations.pop(key, None)
        workspace.save_state(
            validated=not args.target and result.returncode == 0 and bool(rows),
            validation=evidence,
            validations=validations,
        )
    return result.returncode


def execute(args, record=False):
    workspace = Workspace(args.workspace)
    variants = (
        list(
            workspace.manifest["integrations"]["transformers"].get(
                "variants", {"default": {}}
            )
        )
        if args.all_variants
        else [args.variant]
    )
    status = 0
    for variant in variants:
        args.variant = variant
        status = max(status, execute_one(args, record))
    return status


def review(args):
    workspace = Workspace(args.workspace)
    options = ["--framework", "transformers"]
    if args.platform:
        options += ["--platform", args.platform]
    if args.approve:
        options.append("--approve")
        workspace.save_state(validated=False)
    return backend.invoke("review_candidates", workspace.root, options).returncode


def update(args):
    workspace = Workspace(args.workspace)
    old = workspace.manifest
    if len(old["checkpoints"]) != 1 or len(old["fixtures"]) != 1:
        raise ValueError(
            "LLM updates currently require a single-checkpoint contract; edit grouped contracts manually and use record/review/check"
        )
    entry = old["integrations"]["transformers"]["entry_point"]
    evidence_path = workspace.root / ".mic/history.json"
    context = {
        "model": workspace.state["model"],
        "fixture_repo": workspace.fixture,
        "manifest_schema": manifest_schema(),
        "existing_manifest": old,
        "existing_usage": (workspace.contract_root / entry).read_text(),
        "request": args.request,
        "failure_candidates": read(evidence_path)["items"][:30]
        if evidence_path.exists()
        else [],
    }
    plan, llm = planner.propose(context, model=args.llm_model, provider=args.provider)
    validate_plan(plan, workspace.state["model"], workspace.fixture, old)
    backup = workspace.snapshot()
    write(workspace.contract_root / "integration.json", plan["manifest"])
    (workspace.contract_root / entry).write_text(plan["usage"])
    write(
        workspace.root / ".mic/update-plan.json",
        {**plan, "llm": llm, "previous_source": str(backup)},
    )
    workspace.save_state(validated=False, validation=None)
    print(
        json.dumps(
            {
                "updated": workspace.state["contract"],
                "previous_source": str(backup),
                "weights": "retained",
                "next": "record and replay using the existing fixture first",
            }
        )
    )
    if args.retrain:
        args.rebuild = True
        args.weights, args.readable_output = None, False
        return build(args)
    return 0


def replay(args):
    workspace = Workspace(args.workspace)
    options = [
        args.fix,
        "--contract",
        workspace.state["contract"],
        "--local-contracts",
        "--variant",
        args.variant,
    ]
    if args.bad:
        options += ["--bad", args.bad]
    env = {**os.environ, "TRANSFORMERS_REPO": str(Path(args.repo).resolve())}
    result = backend.invoke("replay", workspace.root, options, capture=True, env=env)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, end="")
        return result.returncode
    rows = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    write(workspace.root / ".mic/replay.json", rows)
    caught = [r for r in rows if r.get("verdict") == "caught"]
    if args.match_output:
        caught = [
            r
            for r in caught
            if any(f.startswith(args.match_output + ":") for f in r["bad_failures"])
        ]
    return 0 if caught else 1


def publication_ready(workspace):
    state = workspace.state
    proof = state.get("validation") or {}
    paths = {
        "lock_sha256": workspace.lock_path,
        "manifest_sha256": workspace.contract_root / "integration.json",
        "usage_sha256": workspace.contract_root
        / workspace.manifest["integrations"]["transformers"]["entry_point"],
    }
    if not state.get("validated") or any(
        proof.get(key) != hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
    ):
        raise ValueError(
            "Run mic check on all declared targets after reviewing candidates; source or lock changes invalidate validation"
        )
    if list(workspace.lock_path.parent.glob("expectations/**/*.candidate.json")):
        raise ValueError(
            "Unreviewed candidate baselines remain; run mic review before publishing"
        )
    fingerprint = workspace.validation_fingerprint()
    variants = workspace.manifest["integrations"]["transformers"].get(
        "variants", {"default": {}}
    )
    if any(
        state.get("validations", {}).get(v, {}).get("fingerprint") != fingerprint
        for v in variants
    ):
        raise ValueError(
            "Every declared variant needs a successful check on the current sources and approved expectations; use mic check --all-variants"
        )


def publish(args):
    workspace = Workspace(args.workspace)
    publication_ready(workspace)
    print(
        json.dumps(
            {
                "contract": workspace.state["contract"],
                "fixture": workspace.fixture,
                "execute": args.execute,
                "next": "mic publish --execute, then mic export --transformers-checkout PATH",
            }
        )
    )
    if not args.execute:
        return 0
    name = workspace.state["contract"]
    pin = workspace.lock["contracts"][name]["fixtures"][workspace.fixture]
    if pin["revision"] == "unpublished":
        options = ["--fixture", workspace.fixture.split("/")[-1]] + (
            ["--private"] if args.private else []
        )
        result = backend.invoke("publish_fixtures", workspace.root, options)
        if result.returncode:
            return result.returncode
    result = backend.invoke(
        "publish_contracts",
        workspace.root,
        ["--contract", name] + (["--private"] if args.private else []),
    )
    workspace.save_state(validated=False)
    print("Run mic check --all-variants on the published pins before export.")
    return result.returncode


def export(args):
    workspace = Workspace(args.workspace)
    publication_ready(workspace)
    state = workspace.state
    source = workspace.lock
    name = state["contract"]
    contract = source["contracts"][name]
    pins = [contract["integration"], *contract.get("fixtures", {}).values()]
    if any(p["revision"] == "unpublished" or not p.get("files_sha256") for p in pins):
        raise ValueError(
            "Publish immutable contract and fixture revisions before exporting to Transformers"
        )
    target = Path(args.transformers_checkout).resolve() / "tests/contracts"
    path = target / "fleet.lock"
    old = (
        read(path)
        if path.exists()
        else {
            "framework": "transformers",
            "environment": source["environment"],
            "contracts": {},
        }
    )
    if old["environment"] != source["environment"]:
        raise ValueError(
            "Destination environment differs; review the environment change separately"
        )
    # Hub-only paths keep the exported lock independent of this workspace.
    clean = json.loads(json.dumps(contract))
    for pin in [clean["integration"], *clean.get("fixtures", {}).values()]:
        pin.pop("path", None)
    old["contracts"][name] = clean
    old.setdefault("local_patches", {}).update(source.get("local_patches", {}))
    write(path, old)
    shutil.copytree(
        workspace.lock_path.parent / "expectations" / name,
        target / "expectations" / name,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.candidate.json"),
    )
    print(
        json.dumps(
            {
                "exported": str(target),
                "contract": name,
                "next": "Review git diff, install transformers-mic, and invoke mic check --lock tests/contracts/fleet.lock in CI.",
            }
        )
    )
    return 0
