"""One-model workflow with resumable stages and explicit human decisions."""

import difflib
import hashlib
import json
from pathlib import Path
import shlex
import sys

from filelock import FileLock, Timeout
from huggingface_hub.utils import validate_repo_id

from . import batch, fleet, model_baselines, contract_repair
from .interaction import Interaction, Pending, PROTOCOL_VERSION, load_answer
from .workspace import Workspace, read, write

NEEDS_INPUT = 3
STAGES = (
    "source",
    "build",
    "auto",
    "record",
    "review",
    "check",
    "publish-preview",
    "publish",
    "published-check",
    "export-preview",
    "export",
    "pin",
)


def ask(args, question, details, **kwargs):
    interaction = getattr(args, "_interaction", None)
    if interaction is None:
        mode = "agent" if args.non_interactive else "human"
        interaction = Interaction(mode, {}, lambda: None, lambda: {})
    return interaction.ask(question, details, **kwargs)


def file_snapshot(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def last_json(path):
    for line in reversed(Path(path).read_text().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No result JSON in {path}")


def review_changed(args, workspace):
    """Changed baselines need an explicit decision and a recorded reason."""
    reviewed_source = workspace.contract_digest()
    for candidate in sorted(workspace.expectations.rglob("*.candidate.json")):
        approved = candidate.with_name(
            candidate.name.replace(".candidate.json", ".json")
        )
        before = approved.read_text() if approved.exists() else ""
        after = candidate.read_text()
        detail = "\n".join(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=str(approved),
                tofile=str(candidate),
                lineterm="",
            )
        )
        reason = ask(
            args,
            f"Approve changed baseline {candidate.name}?",
            detail,
            reason=True,
            kind="baseline-change",
            context={
                "candidate": file_snapshot(candidate),
                "approved": file_snapshot(approved) if approved.exists() else None,
                "source_sha256": workspace.contract_digest(),
            },
        )
        if (
            workspace.contract_digest() != reviewed_source
            or candidate.read_text() != after
            or (approved.read_text() if approved.exists() else "") != before
        ):
            raise ValueError(
                "Baseline changed during review; rerun to review the current files"
            )
        audit = workspace.root / ".mic/automation-reviews.json"
        receipts = read(audit) if audit.exists() else []
        receipts.append(
            {
                "path": str(approved.relative_to(workspace.root)),
                "old_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "new_sha256": hashlib.sha256(after.encode()).hexdigest(),
                "source_sha256": workspace.contract_digest(),
                "reason": reason,
            }
        )
        write(audit, receipts)
        candidate.replace(approved)
        workspace.save_state(validated=False, validation=None, validations={})


def run(args):
    args.mode = "agent" if args.non_interactive else args.mode
    if args.local_only:
        args.baseline_execution = "local"
    try:
        return locked_run(args)
    except (ValueError, OSError) as error:
        if args.mode != "agent":
            raise
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": "error",
                    "error": str(error),
                }
            )
        )
        return 2


def locked_run(args):
    validate_repo_id(args.model)
    if args.timeout <= 0 or args.baseline_timeout <= 0:
        raise ValueError("Stage timeout must be positive")
    if args.transformers_checkout:
        args.transformers_checkout = str(Path(args.transformers_checkout).resolve())
    model_slug = args.model.replace("/", "--")
    root = Path(
        args.out
        or (
            args.workspace
            if args.workspace != "." or Path(".mic/project.json").exists()
            else Path("work/mic") / model_slug
        )
    ).resolve()
    if args.answer and args.mode != "agent":
        raise ValueError("--answer requires --mode agent")
    answer = load_answer(args.answer)
    lock = FileLock(root / ".mic/automation.lock")
    try:
        with lock.acquire(timeout=0):
            return run_locked(args, root, answer)
    except Timeout:
        raise ValueError(
            f"Another tmic run is using {root}; wait for it to finish"
        ) from None


def run_locked(args, root, answer):
    workspace = Workspace(root)
    folder = root / ".mic/automation"
    state_path = root / ".mic/automation.json"
    config = {
        key: getattr(args, key)
        for key in (
            "model",
            "namespace",
            "revision",
            "name",
            "contract",
            "regenerate",
            "llm",
            "weights",
            "readable_output",
            "fixtures_only",
            "no_history",
            "provider",
            "llm_model",
        )
    }
    state = (
        read(state_path)
        if state_path.exists()
        else {
            "model": args.model,
            "config": config,
            "next_stage": "source",
            "stages": {},
        }
    )
    if answer is not None and not state.get("pending_question"):
        raise ValueError(
            "There is no pending question; resume without --answer to obtain one"
        )
    if state["config"] != config:
        raise ValueError(
            "Automation options differ from this workspace; reuse the original options or choose another --out"
        )
    if workspace.state_path.exists() and workspace.state["model"] != args.model:
        raise ValueError("Workspace belongs to a different model; choose another --out")
    execution = {
        "mode": args.baseline_execution,
        "dashboard": args.dashboard,
        "transport": args.baseline_transport,
    }
    if (
        state.get(
            "execution",
            {
                "mode": "local",
                "dashboard": args.dashboard,
                "transport": args.baseline_transport,
            },
        )
        != execution
    ):
        state["next_stage"] = "source"
        state.pop("dataset_revision", None)
    state["execution"] = execution
    resume = shlex.join(
        ["tmic", "run", args.model, "--out", str(root), *resume_options(args)]
    )
    state["resume"] = resume

    def fingerprint():
        return (
            workspace.validation_fingerprint()
            if workspace.state_path.exists()
            else None
        )

    def save():
        state["fingerprint"] = fingerprint()
        write(state_path, state)

    args._interaction = Interaction(
        args.mode,
        state,
        lambda: write(state_path, state),
        lambda: {
            "workspace": str(root),
            "model": args.model,
            "stage": state["next_stage"],
            "workspace_sha256": fingerprint(),
            "fleet": args.fleet,
            "contract_name": args.contract_name,
            "transformers_checkout": args.transformers_checkout,
        },
        answer,
    )
    output = sys.stderr if args.mode == "agent" else sys.stdout

    def stage(name, options, *, create=False):
        print(
            f"tmic: {name} (log: {folder / 'logs' / (name + '.log')})",
            file=output,
            flush=True,
        )
        command = options if create else ["--workspace", str(root), *options]
        timeout = args.baseline_timeout if options[0] == "baselines" else args.timeout
        result = batch.invoke(folder, name, command, timeout)
        state["stages"][name] = result
        state["stages"][name]["command"] = command
        save()
        log = folder / result["log"]
        text = log.read_text()
        if result["exit_code"]:
            print("\n".join(text.splitlines()[-60:]), file=output, flush=True)
        return result["exit_code"], log

    def review_repair(attempt):
        if not contract_repair.current(workspace, attempt):
            attempt["status"] = "stale"
            state["next_stage"] = "source"
            save()
            return False
        proposal = read(attempt["proposal"]["path"])
        ask(
            args,
            "Apply this proposed contract repair and retry validation?",
            proposal["plan"]["rationale"]
            + "\n\n"
            + Path(attempt["diff"]["path"]).read_text(),
            kind="contract-repair",
            context={
                key: attempt[key]
                for key in (
                    "stage",
                    "source",
                    "usage",
                    "manifest",
                    "log",
                    "proposal",
                    "diff",
                    "evidence",
                )
            },
        )
        if not contract_repair.current(workspace, attempt):
            attempt["status"] = "stale"
            state["next_stage"] = "source"
            save()
            return False
        contract_repair.apply(workspace, attempt)
        # Approval covers the exact diff against the already reviewed source.
        # Rerun the fixture gates as well as measurements for changed inputs.
        state["reviewed_source_sha256"] = workspace.contract_digest()
        state["rebuild_after_repair"] = True
        state["next_stage"] = "build"
        save()
        return True

    try:
        for attempt in state.get("repairs", []):
            if attempt["status"] == "generating":
                attempt["status"] = "interrupted"
            if attempt["status"] == "proposed":
                review_repair(attempt)
        pending = state.get("pending_question")
        if (
            args.mode == "agent"
            and pending
            and pending["kind"] == "retry"
            and pending["context"].get("workspace_sha256") == fingerprint()
        ):
            context = {
                key: pending["context"][key]
                for key in ("failed_stage", "exit_code", "log")
            }
            context["log"] = file_snapshot(Path(context["log"]["path"]))
            ask(
                args,
                pending["question"],
                pending["details"],
                kind="retry",
                context=context,
            )
        if (
            args.mode == "agent"
            and pending
            and pending["kind"] == "baseline-change"
            and pending["context"].get("workspace_sha256") == fingerprint()
        ):
            review_changed(args, workspace)
            save()
        while not workspace.state_path.exists():
            options = [
                "create",
                args.model,
                "--out",
                str(root),
                "--namespace",
                args.namespace,
                "--weights",
                args.weights,
                "--provider",
                args.provider,
                "--llm-model",
                args.llm_model,
            ]
            for name in ("revision", "name", "contract"):
                if value := getattr(args, name):
                    options += ["--" + name, value]
            for name in (
                "regenerate",
                "llm",
                "fixtures_only",
                "readable_output",
                "no_history",
            ):
                if getattr(args, name):
                    options.append("--" + name.replace("_", "-"))
            code, log = stage("create", options, create=True)
            if code:
                ask(
                    args,
                    "Workspace creation needs attention; fix the reported problem and retry?",
                    str(log),
                    kind="retry",
                    context={
                        "failed_stage": "create",
                        "exit_code": code,
                        "log": file_snapshot(log),
                    },
                )
                continue
            if workspace.state["model"] != args.model:
                raise ValueError("Created workspace does not match the requested model")
        if state.get("fingerprint") != fingerprint():
            # Manual source/reference/lock edits invalidate previous checks and decisions.
            state["next_stage"] = "source"
            state.pop("dataset_revision", None)
        destination = {"fleet": args.fleet, "contract_name": args.contract_name}
        if state.get("destination", destination) != destination:
            if state["next_stage"] == "done" or STAGES.index(
                state["next_stage"]
            ) >= STAGES.index("export-preview"):
                state["next_stage"] = "export-preview"
                state.pop("dataset_revision", None)
        state["destination"] = destination
        if (
            state["next_stage"] == "done"
            and args.transformers_checkout
            and state.get("pin_checkout")
            != str(Path(args.transformers_checkout).resolve())
        ):
            state["next_stage"] = "pin"
        save()
        while state["next_stage"] != "done":
            if (
                state["next_stage"] != "source"
                and state.get("reviewed_source_sha256") != workspace.contract_digest()
            ):
                state["next_stage"] = "source"
                save()
            name = state["next_stage"]
            if args.local_only and STAGES.index(name) >= STAGES.index(
                "publish-preview"
            ):
                state["status"] = "validated"
                save()
                print(
                    json.dumps(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "status": "validated",
                            "workspace": str(root),
                            "resume": resume,
                        }
                    )
                )
                return 0
            before = fingerprint()
            code, log = 0, None
            if name == "source":
                manifest = workspace.manifest
                usage = (
                    workspace.contract_root
                    / manifest["integrations"]["transformers"]["entry_point"]
                )
                ask(
                    args,
                    "Review the executable contract before running it.",
                    f"Manifest: {workspace.contract_root / 'integration.json'}\nUsage: {usage}\n"
                    f"Targets: {manifest['integrations']['transformers']['targets']}\n"
                    f"Variants: {list(manifest['integrations']['transformers']['variants'])}\n"
                    "All declared targets and variants will be checked.",
                    kind="source-review",
                    context={
                        "files": [
                            file_snapshot(workspace.contract_root / "integration.json"),
                            file_snapshot(usage),
                        ],
                        "targets": manifest["integrations"]["transformers"]["targets"],
                        "variants": list(
                            manifest["integrations"]["transformers"]["variants"]
                        ),
                    },
                )
                if before != fingerprint():
                    continue
                state["reviewed_source_sha256"] = workspace.contract_digest()
            elif name == "build":
                code, log = stage(
                    name,
                    ["build", "--rebuild"]
                    if state.get("rebuild_after_repair")
                    else ["build"],
                )
                if not code:
                    state.pop("rebuild_after_repair", None)
            elif name == "auto":
                if (
                    "fixtures"
                    in workspace.manifest["integrations"]["transformers"]["targets"]
                ):
                    code, log = stage(name, ["auto"])
            elif name in ("record", "check", "published-check"):
                if args.baseline_execution == "cloud":
                    cloud_root = folder / "cloud" / model_baselines.inputs(workspace)[0]
                    options = [
                        "baselines",
                        "run" if name == "record" else "verify-model",
                        "--out",
                        str(cloud_root),
                        "--model-workspace",
                        str(root),
                    ]
                    if name == "record":
                        ask(
                            args,
                            "Upload the reviewed model input to a Hub staging branch and measure CPU/GPU baselines?",
                            f"Repository: {workspace.fixture}\nOnly the staging branch changes; model main and fleet pins are published after review.",
                            kind="cloud-baselines",
                            context={
                                "repository": workspace.fixture,
                                "source_sha256": model_baselines.inputs(workspace)[0],
                                "profiles": ["cpu", "cuda"],
                            },
                        )
                        if before != fingerprint():
                            state["next_stage"] = "source"
                            save()
                            continue
                        options += [
                            "--transport",
                            args.baseline_transport,
                            "--fleet",
                            args.fleet,
                            "--dashboard",
                            args.dashboard,
                        ]
                    code, log = stage(name, options)
                else:
                    code, log = stage(
                        name,
                        ["record" if name == "record" else "check", "--all-variants"],
                    )
            elif name == "review":
                code, log = stage(name, ["review", "--approve"])
                if code == 1 and list(workspace.expectations.rglob("*.candidate.json")):
                    review_changed(args, workspace)
                    code = 0
            elif name == "publish-preview":
                code, log = stage(name, ["publish"])
            elif name == "publish":
                ask(
                    args,
                    "Publish the checked MIC model and expectations publicly to the Hub?",
                    f"Model: {workspace.fixture}\nWorkspace: {root}\n"
                    f"Preview: {folder / 'logs/publish-preview.log'}",
                    kind="model-publication",
                    context={
                        "repository": workspace.fixture,
                        "preview": file_snapshot(folder / "logs/publish-preview.log"),
                    },
                )
                if before != fingerprint():
                    state["next_stage"] = "source"
                    save()
                    continue
                code, log = stage(name, ["publish", "--execute"])
            elif name == "export-preview":
                options = ["export", "--fleet", args.fleet]
                if args.contract_name:
                    options += ["--contract-name", args.contract_name]
                code, log = stage(name, options)
            elif name == "export":
                preview = last_json(folder / "logs/export-preview.log")
                ask(
                    args,
                    "Update the shared MIC fleet dataset?",
                    json.dumps(
                        {
                            "repository": fleet.REPOSITORY,
                            "fleet": args.fleet,
                            "base_revision": preview["revision"],
                            "contract": preview["contract"],
                            "pins": preview["contracts"][preview["contract"]],
                        },
                        indent=2,
                    ),
                    kind="fleet-update",
                    context={
                        "repository": fleet.REPOSITORY,
                        "base_revision": preview["revision"],
                        "contract": preview["contract"],
                        "pins": preview["contracts"][preview["contract"]],
                    },
                )
                if before != fingerprint():
                    state["next_stage"] = "source"
                    save()
                    continue
                options = [
                    "export",
                    "--fleet",
                    args.fleet,
                    "--execute",
                    "--expected-revision",
                    preview["revision"],
                ]
                if args.contract_name:
                    options += ["--contract-name", args.contract_name]
                code, log = stage(name, options)
                if not code:
                    state["dataset_revision"] = last_json(log)["revision"]
            elif name == "pin":
                if args.transformers_checkout:
                    code, log = stage(
                        name,
                        [
                            "pin",
                            "--transformers-checkout",
                            args.transformers_checkout,
                            "--revision",
                            state["dataset_revision"],
                        ],
                    )
                    if not code:
                        state["pin_checkout"] = str(
                            Path(args.transformers_checkout).resolve()
                        )
            if code:
                if (
                    contract_repair.eligible(
                        workspace, name, log, state["stages"][name]["command"]
                    )
                    and len(state.get("repairs", [])) < args.max_contract_repairs
                ):
                    print(
                        "tmic: inspecting failure and proposing a bounded contract repair…",
                        file=output,
                        flush=True,
                    )
                    proposal = contract_repair.propose(
                        args,
                        workspace,
                        name,
                        log,
                        folder,
                        state,
                        save,
                        state["stages"][name]["command"],
                    )
                    if proposal is not None and review_repair(proposal):
                        continue
                    if state.get("repairs"):
                        print(
                            "tmic: repair was not applied: "
                            + state["repairs"][-1].get(
                                "error", "proposal became stale"
                            ),
                            file=output,
                            flush=True,
                        )
                if name == "export":
                    # Refresh and review the exact dataset base after a conflict or rejection.
                    state["next_stage"] = "export-preview"
                save()
                failed_fingerprint = fingerprint()
                ask(
                    args,
                    f"{name} failed (exit {code}). Fix the problem and retry?",
                    str(log),
                    kind="retry",
                    context={
                        "failed_stage": name,
                        "exit_code": code,
                        "log": file_snapshot(log),
                    },
                )
                if failed_fingerprint != fingerprint():
                    state["next_stage"] = "source"
                save()
                continue
            index = STAGES.index(name) + 1
            state["next_stage"] = STAGES[index] if index < len(STAGES) else "done"
            state["status"] = "running"
            state.pop("question", None)
            state.pop("details", None)
            save()
        state["status"] = "complete"
        save()
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": "complete",
                    "workspace": str(root),
                    "fleet_repository": fleet.REPOSITORY,
                    "fleet": args.fleet,
                    "dataset_revision": state["dataset_revision"],
                }
            )
        )
        return 0
    except Pending as pending:
        state.update(
            status="needs-input", question=pending.question, details=pending.details
        )
        save()
        print(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": "needs-input",
                    "pending_question": pending.record,
                    **({"answer_error": pending.error} if pending.error else {}),
                    "question": pending.question,
                    "details": pending.details,
                    "workspace": str(root),
                    "resume": resume,
                }
            )
        )
        return NEEDS_INPUT
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        save()
        print(f"\ntmic: interrupted; resume with {resume}", file=sys.stderr)
        if args.mode == "agent":
            print(
                json.dumps(
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "status": "interrupted",
                        "resume": resume,
                    }
                )
            )
        return 130


def resume_options(args):
    options = [
        "--max-contract-repairs",
        str(args.max_contract_repairs),
        "--baseline-transport",
        args.baseline_transport,
        "--baseline-execution",
        args.baseline_execution,
        "--dashboard",
        args.dashboard,
        "--baseline-timeout",
        str(args.baseline_timeout),
        "--mode",
        args.mode,
        "--timeout",
        str(args.timeout),
        "--fleet",
        args.fleet,
        "--namespace",
        args.namespace,
        "--weights",
        args.weights,
        "--provider",
        args.provider,
        "--llm-model",
        args.llm_model,
    ]
    for name in (
        "revision",
        "name",
        "contract",
        "contract_name",
        "transformers_checkout",
    ):
        if value := getattr(args, name):
            options += ["--" + name.replace("_", "-"), value]
    for name in ("regenerate", "llm", "fixtures_only", "readable_output", "no_history"):
        if getattr(args, name):
            options.append("--" + name.replace("_", "-"))
    if args.local_only:
        options.append("--local-only")
    return options
