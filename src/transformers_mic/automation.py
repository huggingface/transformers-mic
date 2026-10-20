"""One-model workflow with resumable stages and explicit human decisions."""

import difflib
import hashlib
import json
from pathlib import Path
import shlex
import sys

from huggingface_hub.utils import validate_repo_id

from . import batch, fleet
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


class Pending(Exception):
    def __init__(self, question, details):
        self.question, self.details = question, details


def ask(args, question, details, *, reason=False):
    print(question, flush=True)
    print(details, flush=True)
    if args.non_interactive or not sys.stdin.isatty():
        raise Pending(question, details)
    try:
        reply = input(
            "Reason for approving this change: " if reason else "Continue? [y/N] "
        ).strip()
    except EOFError:
        reply = ""
    if (reason and not reply) or (not reason and reply.lower() not in ("y", "yes")):
        raise Pending(question, details)
    return reply


def last_json(path):
    for line in reversed(Path(path).read_text().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(f"No result JSON in {path}")


def review_changed(args, workspace):
    """Only a human can approve changed or removed baseline outputs."""
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
        ask(args, f"Approve changed baseline {candidate.name}?", detail)
        reason = ask(
            args, "Why is this baseline change expected?", str(candidate), reason=True
        )
        if (
            candidate.read_text() != after
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
    validate_repo_id(args.model)
    if args.timeout <= 0:
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
    if state["config"] != config:
        raise ValueError(
            "Automation options differ from this workspace; reuse the original options or choose another --out"
        )
    if workspace.state_path.exists() and workspace.state["model"] != args.model:
        raise ValueError("Workspace belongs to a different model; choose another --out")
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

    def stage(name, options, *, create=False):
        print(f"tmic: {name} (log: {folder / 'logs' / (name + '.log')})", flush=True)
        command = options if create else ["--workspace", str(root), *options]
        result = batch.invoke(folder, name, command, args.timeout)
        state["stages"][name] = result
        state["stages"][name]["command"] = command
        save()
        log = folder / result["log"]
        text = log.read_text()
        if result["exit_code"]:
            print("\n".join(text.splitlines()[-60:]), flush=True)
        return result["exit_code"], log

    try:
        if not workspace.state_path.exists():
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
                raise Pending(
                    "Workspace creation needs attention; fix the reported problem and rerun.",
                    str(log),
                )
            if workspace.state["model"] != args.model:
                raise ValueError("Created workspace does not match the requested model")
        elif state.get("fingerprint") != fingerprint():
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
            name = state["next_stage"]
            if args.local_only and STAGES.index(name) >= STAGES.index(
                "publish-preview"
            ):
                state["status"] = "validated"
                save()
                print(
                    json.dumps(
                        {
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
                )
                if before != fingerprint():
                    continue
            elif name == "build":
                code, log = stage(name, ["build"])
            elif name == "auto":
                if (
                    "fixtures"
                    in workspace.manifest["integrations"]["transformers"]["targets"]
                ):
                    code, log = stage(name, ["auto"])
            elif name in ("record", "check", "published-check"):
                code, log = stage(
                    name, ["record" if name == "record" else "check", "--all-variants"]
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
                if name == "export":
                    # Refresh and review the exact dataset base after a conflict or rejection.
                    state["next_stage"] = "export-preview"
                save()
                failed_fingerprint = fingerprint()
                ask(
                    args,
                    f"{name} failed (exit {code}). Fix the problem and retry?",
                    str(log),
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
                    "status": "needs-input",
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
        return 130


def resume_options(args):
    options = [
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
    return options
