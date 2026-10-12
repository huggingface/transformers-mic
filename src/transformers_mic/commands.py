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

from . import backend, history, planner, policy, references, resources
from .workspace import Workspace, catalog_patches, digest, read, write


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
    recipe = manifest["fixtures"][fixture].get("build")
    if not isinstance(recipe, dict):
        raise ValueError("Generated fixture needs a build recipe")
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
    if old and any(
        manifest["outputs"][key] != rule for key, rule in old["outputs"].items()
    ):
        raise ValueError(
            "Update changes existing comparison rules; review those changes manually"
        )
    if not {"default", "eager", "bfloat16"}.issubset(integration.get("variants", {})):
        raise ValueError(
            "Generated contract must declare default, eager and bfloat16 variants"
        )
    if old:
        previous = old["integrations"]["transformers"]
        if previous.get("training") and not integration.get("training"):
            raise ValueError("Update disables existing training assertions")
        if not set(previous["targets"]).issubset(integration["targets"]):
            raise ValueError("Update removes existing target types")
        if any(
            integration["variants"].get(key) != value
            for key, value in previous["variants"].items()
        ):
            raise ValueError("Update changes existing variants")
    if integration.get("training"):
        required = {
            "training.loss": "allclose",
            "training.gradients": "all_true",
            "training.updated": "true",
        }
        if any(
            manifest["outputs"].get(key, {}).get("compare") != rule
            for key, rule in required.items()
        ):
            raise ValueError(
                "Training contracts require training.loss (allclose), training.gradients (all_true nonempty named booleans), and training.updated (true)"
            )
    if integration.get("variants", {}).get("bfloat16") != {"dtype": "bfloat16"}:
        raise ValueError("Use bfloat16 variant kwargs {dtype: bfloat16}")
    # Syntax is checked without importing or running generated code.
    blocks = re.findall(
        r"```(?:py|python) runnable:[^\n]*\n(.*?)```", plan["usage"], re.S
    )
    if not blocks:
        raise ValueError("Generated usage contains no runnable Python blocks")
    ast.parse("\n".join(blocks))
    return manifest


def propose_valid(context, args, model, fixture, old=None):
    # A provider can emit schema-valid JSON that violates runner conventions.
    # Return validation feedback for one repair; never execute an invalid draft.
    for attempt in range(2):
        plan = None
        try:
            plan, llm = planner.propose(
                context, model=args.llm_model, provider=args.provider
            )
            validate_plan(plan, model, fixture, old)
            return plan, llm
        except (ValueError, jsonschema.ValidationError) as exc:
            if attempt:
                raise
            context = {
                **context,
                "previous_draft": plan,
                "validation_error": str(exc)[:3000],
                "repair_request": "Fix the validation error and return the complete corrected draft.",
            }


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
    if names and args.regenerate and not args.llm:
        name = names[0]
        original = read(
            resources.catalog() / "hub" / f"{name}-integration/integration.json"
        )
        old_fixture = original["checkpoints"][args.model]["fixture"]
        namespace = args.namespace or HfApi().whoami()["name"]
        slug = args.name or re.sub(r"[^A-Za-z0-9_-]", "-", args.model.split("/")[-1])
        fixture = f"{namespace}/mic-{slug}-slim"
        manifest = json.loads(json.dumps(original).replace(old_fixture, fixture))
        manifest["checkpoints"] = {args.model: {"fixture": fixture}}
        manifest["fixtures"] = {fixture: manifest["fixtures"][fixture]}
        manifest["fixtures"][fixture].update(built_from=args.model, covers=[args.model])
        if args.fixtures_only:
            manifest["integrations"]["transformers"]["targets"] = ["fixtures"]
        pin = catalog["contracts"][name]["checkpoints"][args.model]
        source = resources.catalog() / "hub" / f"{name}-integration"
        usage = (
            source / manifest["integrations"]["transformers"]["entry_point"]
        ).read_text()
        workspace.from_plan(
            args.model,
            pin,
            namespace,
            slug,
            manifest,
            usage,
            args.weights,
            args.readable_output,
        )
        # Preserve frozen assets used by the catalog usage (e.g. speech).
        for item in source.iterdir():
            if item.is_dir() and item.name != "integrations":
                shutil.copytree(
                    item, workspace.contract_root / item.name, dirs_exist_ok=True
                )
        workspace.save_state(origin="catalog-regenerated")
        lock = workspace.lock
        lock["local_patches"] = catalog_patches(catalog, name)
        write(workspace.lock_path, lock)
        write(
            workspace.root / ".mic/plan.json",
            {
                "origin": "catalog-regenerated",
                "source_contract": name,
                "scope": "fixtures" if args.fixtures_only else "all",
                "manifest": manifest,
            },
        )
    elif names and not args.llm:
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
        fixture = f"{namespace}/mic-{args.name or slug}-slim"
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
        context.update(references.collect(args.model, revision, config))
        sample = resources.catalog() / "hub/BERT-integration"
        context["example_plan"] = {
            "name": "BERT",
            "manifest": read(sample / "integration.json"),
            "usage": (sample / "integrations/transformers/usage.md").read_text(),
            "rationale": "Use the model's public API and measure expectations by execution.",
            "historical_candidates": [],
        }
        plan, llm = propose_valid(context, args, args.model, fixture)
        manifest = plan["manifest"]
        if args.name:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.name):
                raise ValueError("Contract name must be a directory slug")
            plan["name"] = args.name
        if args.fixtures_only:
            manifest["integrations"]["transformers"]["targets"] = ["fixtures"]
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
                    "tmic build",
                    "tmic record",
                    "tmic review",
                    "tmic check",
                    "tmic publish",
                ],
            }
        )
    )
    if args.build or args.record:
        args.workspace = str(workspace.root)
        args.rebuild = False
        status = build(args)
        if status:
            return status
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
    local_metadata = (workspace.lock_path.parent / pin["path"]).resolve() / "build.json"
    if not args.rebuild and (
        pin["revision"] != "unpublished"
        or (pin.get("files_sha256") and local_metadata.exists())
    ):
        metadata = read(local_metadata)
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
                        "next": "tmic record; review the new candidates before publishing",
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
    plan, llm = propose_valid(
        context, args, workspace.state["model"], workspace.fixture, old
    )
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
            "Run tmic check on all declared targets after reviewing candidates; source or lock changes invalidate validation"
        )
    if list(workspace.lock_path.parent.glob("expectations/**/*.candidate.json")):
        raise ValueError(
            "Unreviewed candidate baselines remain; run tmic review before publishing"
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
            "Every declared variant needs a successful check on the current sources and approved expectations; use tmic check --all-variants"
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
                "next": "tmic publish --execute, then tmic export --transformers-checkout PATH",
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
    print("Run tmic check --all-variants on the published pins before export.")
    return result.returncode


def export(args):
    workspace = Workspace(args.workspace)
    publication_ready(workspace)
    state = workspace.state
    source = workspace.lock
    if source.get("local_patches"):
        raise ValueError(
            "Validate without local patches before exporting to Transformers; use narrow known_failures for unresolved framework bugs"
        )
    name = state["contract"]
    contract = source["contracts"][name]
    pins = [contract["integration"], *contract.get("fixtures", {}).values()]
    if any(p["revision"] == "unpublished" or not p.get("files_sha256") for p in pins):
        raise ValueError(
            "Publish immutable contract and fixture revisions before exporting to Transformers"
        )
    if workspace.contract_digest() != contract["integration"]["files_sha256"]:
        raise ValueError(
            "Local contract differs from its published pin; publish the reviewed source before export"
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
    destination_name = getattr(args, "contract_name", None) or name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", destination_name):
        raise ValueError("Destination contract name must be a safe folder name")
    existing_names = set(old["contracts"])
    expectations = target / "expectations"
    if expectations.exists():
        existing_names.update(p.name for p in expectations.iterdir() if p.is_dir())
    if any(
        n != destination_name and n.casefold() == destination_name.casefold()
        for n in existing_names
    ):
        raise ValueError(
            "Destination name differs only by case from an existing contract; use --contract-name"
        )
    if old["environment"] != source["environment"]:
        raise ValueError(
            "Destination environment differs; review the environment change separately"
        )
    # Hub-only paths keep the exported lock independent of this workspace.
    clean = json.loads(json.dumps(contract))
    for pin in [clean["integration"], *clean.get("fixtures", {}).values()]:
        pin.pop("path", None)
    old["contracts"][destination_name] = clean
    if source.get("known_failures"):
        old.setdefault("known_failures", {}).update(source["known_failures"])
    write(path, old)
    shutil.copytree(
        workspace.lock_path.parent / "expectations" / name,
        target / "expectations" / destination_name,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*.candidate.json"),
    )
    print(
        json.dumps(
            {
                "exported": str(target),
                "contract": destination_name,
                "next": "Review git diff, install transformers-mic, and invoke tmic check --lock tests/contracts/fleet.lock in CI.",
            }
        )
    )
    return 0
