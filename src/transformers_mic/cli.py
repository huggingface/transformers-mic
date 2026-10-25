"""Command line interface for measured model integration contracts."""

import argparse
import json
from pathlib import Path
import sys

import jsonschema
from huggingface_hub.errors import HfHubHTTPError

from . import __version__, backend, commands, fleet, planner
from .workspace import Workspace


def llm_options(parser):
    parser.add_argument("--llm-model", default=planner.DEFAULT_MODEL)
    parser.add_argument(
        "--provider", default="auto", help="Hugging Face Inference Provider"
    )


def history_options(parser):
    parser.add_argument("--history-repo", default="huggingface/transformers")
    parser.add_argument(
        "--history-file",
        help="Normalized JSON list exported from relore or another source",
    )
    parser.add_argument("--limit", type=int, default=50)


def weights_options(parser, default=None):
    parser.add_argument(
        "--weights", choices=("auto", "random", "overfit"), default=default
    )
    parser.add_argument(
        "--readable-output",
        action="store_true",
        help="Require an overfit fixture for meaningful decoded output",
    )


def parser():
    root = argparse.ArgumentParser(
        prog="tmic",
        description="Build, measure and review Transformers model integration contracts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Automated modes:
  Human: tmic run org/model --mode human
  Agent: tmic run org/model --mode agent

Agent mode writes one JSON result to stdout; logs go to stderr.
Exit 3 means a pending_question needs an answer. Review its snapshot, write
{"question_id": "<pending ID>", "answer": "approve", "reason": "<explanation>"}
to answer.json, then use the emitted resume command with --answer answer.json.
Approvals are single-use and bound to the reviewed snapshot.
See 'tmic run --help' for the protocol and exit codes.""",
    )
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument(
        "--workspace",
        default=".",
        help="Project directory (default: current directory)",
    )
    sub = root.add_subparsers(dest="command", required=True)
    baselines = sub.add_parser(
        "baselines",
        help="Generate a complete baseline collection on the dashboard's ARC runners",
    )
    baseline_sub = baselines.add_subparsers(dest="baselines_command", required=True)
    for action in ("run", "status", "upload", "review", "publish"):
        command = baseline_sub.add_parser(action)
        command.add_argument(
            "--out", required=True, help="Resumable local collection directory"
        )
        if action in ("upload", "publish"):
            command.add_argument(
                "--execute",
                action="store_true",
                help="Upload/open the previewed Hub changes",
            )
        if action == "upload":
            command.add_argument(
                "--repo",
                required=True,
                help="Hub dataset for the measured candidate collection",
            )
        if action == "review":
            command.add_argument(
                "--approve",
                action="store_true",
                help="Approve new and unchanged measurements",
            )
            command.add_argument(
                "--approve-changed",
                action="store_true",
                help="Explicitly approve displayed changed values",
            )
            command.add_argument(
                "--reason", help="Explanation required for changed values"
            )
        if action in ("run", "status"):
            command.add_argument(
                "--dashboard", default="https://transformers-ci.huggingface.cool"
            )
        if action == "run":
            command.add_argument(
                "--fleet", choices=("all", "control", "prototypes"), default="all"
            )
            command.add_argument(
                "--fleet-revision",
                help="Immutable fleet dataset SHA (default: resolve main once)",
            )
            command.add_argument(
                "--transformers-ref",
                default="main",
                help="Framework main or immutable commit SHA",
            )
            command.add_argument(
                "--fixtures-only",
                action="store_true",
                help="Explicitly exclude real checkpoints",
            )
            command.add_argument(
                "--no-wait",
                action="store_true",
                help="Submit without waiting; repeat to resume",
            )
            command.add_argument(
                "--upload-repo",
                help="Have the workflow upload models and candidate baselines directly to this Hub dataset",
            )
            command.add_argument(
                "--no-download",
                action="store_true",
                help="Return the Hub revision without downloading artifacts (requires --upload-repo)",
            )
    auth = sub.add_parser(
        "auth", help="Sign in to the dashboard with your GitHub account"
    )
    auth_sub = auth.add_subparsers(dest="auth_command", required=True)
    for action in ("login", "status", "logout"):
        command = auth_sub.add_parser(action)
        command.add_argument(
            "--dashboard",
            default="https://transformers-ci.huggingface.cool",
            help="Dashboard HTTPS origin",
        )
        if action == "login":
            command.add_argument(
                "--no-browser",
                action="store_true",
                help="Print the authorization link without opening a browser",
            )
    run = sub.add_parser(
        "run",
        help="Automate one model in human or agent mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Modes:
  --mode human (default): interactive questions in a terminal.
  --mode agent: no stdin reads; stdout is one JSON object (protocol_version 1).
                Stage logs go to stderr. --non-interactive is an alias.

Agent protocol:
  1. tmic run org/model --mode agent --out work/model
  2. Exit 3 returns pending_question with question_id, kind, choices,
     requires_reason, context, and context_sha256. Inspect the referenced
     files/diffs, and ask the operator when their decision is needed.
  3. Write an answer file:
     {"question_id": "<ID>", "answer": "approve", "reason": "Reviewed change"}
     Use 'retry' for retry questions or 'defer' to leave the question pending.
     Baseline changes require a nonempty reason.
  4. Run the emitted resume command with --answer answer.json.
     The answer is consumed once; the next question needs a separate answer.

Workspace changes invalidate approvals. Stale or reused answers return exit 3
with answer_error and the current pending_question. Malformed answer files exit 2. No approval
is transferred to a new snapshot. Human and agent modes share resumable state.
--local-only stops after validation; it is preserved in resume commands.

Exit codes: 0 validated/complete; 2 invalid invocation or state; 3 needs input;
            130 interrupted. Read status in the JSON object, not only the code.
No Git commit or framework PR is created automatically.""",
    )
    run.add_argument("model")
    run.add_argument(
        "--mode",
        choices=("human", "agent"),
        default="human",
        help="Interaction mode (default: human)",
    )
    run.add_argument(
        "--answer",
        type=Path,
        help="One-use JSON answer to a pending question; requires --mode agent",
    )
    run.add_argument("--out", help="Resumable workspace (default: work/mic/ORG--MODEL)")
    run.add_argument("--namespace", default="huggingface-mic")
    run.add_argument("--fleet", choices=tuple(fleet.FILES), default="prototypes")
    run.add_argument("--contract-name")
    run.add_argument("--transformers-checkout")
    run.add_argument("--revision")
    run.add_argument("--name")
    run.add_argument("--contract")
    run.add_argument("--regenerate", action="store_true")
    run.add_argument("--llm", action="store_true")
    run.add_argument("--fixtures-only", action="store_true")
    run.add_argument("--no-history", action="store_true")
    run.add_argument(
        "--local-only",
        action="store_true",
        help="Stop after local validation without uploading",
    )
    run.add_argument(
        "--non-interactive",
        action="store_true",
        help="Alias for --mode agent",
    )
    run.add_argument(
        "--timeout", type=int, default=1800, help="Timeout in seconds per stage"
    )
    weights_options(run, "auto")
    llm_options(run)
    create = sub.add_parser(
        "create", help="Reuse a catalog contract or ask the HF planner for a draft"
    )
    create.add_argument("model")
    create.add_argument("--out", required=True)
    create.add_argument("--contract", help="Choose a catalog usage path")
    create.add_argument(
        "--namespace",
        default="huggingface-mic",
        help="Hub owner for new artifacts (default: huggingface-mic)",
    )
    create.add_argument("--revision")
    create.add_argument(
        "--regenerate",
        action="store_true",
        help="Create new mic- artifacts using a catalog path where available",
    )
    create.add_argument("--name", help="Stable contract slug for generated artifacts")
    create.add_argument(
        "--fixtures-only",
        action="store_true",
        help="Declare this prototype as slim-only; full checkpoint validation remains separate",
    )
    create.add_argument(
        "--llm",
        action="store_true",
        help="Generate a new draft even if a catalog contract exists",
    )
    create.add_argument(
        "--offline",
        action="store_true",
        help="Catalog reuse without Hub or history requests",
    )
    create.add_argument("--no-history", action="store_true")
    create.add_argument("--build", action="store_true")
    create.add_argument("--record", action="store_true")
    create.set_defaults(all_variants=True)
    weights_options(create, "auto")
    history_options(create)
    llm_options(create)
    research = sub.add_parser(
        "research", help="Collect bounded, unverified historical failure candidates"
    )
    research.add_argument("--model-type")
    history_options(research)
    build = sub.add_parser(
        "build", help="Reuse or build a fixture using measured weight-selection gates"
    )
    build.add_argument("--rebuild", action="store_true")
    weights_options(build)
    for name in ("record", "check"):
        run = sub.add_parser(
            name,
            help="Measure candidate expectations"
            if name == "record"
            else "Check reviewed expectations",
        )
        run.add_argument("--variant", default="default")
        run.add_argument("--all-variants", action="store_true")
        run.add_argument("--isolate", action="store_true")
        run.add_argument("--target", action="append")
        if name == "check":
            run.add_argument(
                "--lock",
                type=Path,
                help="Run an exported fleet lock without a MIC workspace",
            )
            run.add_argument(
                "--fleet",
                choices=tuple(fleet.FILES),
                help="Check a fleet from the shared Hub dataset",
            )
            run.add_argument(
                "--fleet-revision",
                help="Dataset revision (defaults to tests/contracts.lock, then main)",
            )
            run.add_argument("--fixtures-only", action="store_true")
            run.add_argument("--changed-files", type=Path)
            run.add_argument("--output", type=Path)
            run.add_argument("--run-all", action="store_true")
            run.add_argument(
                "--contract",
                action="append",
                help="Select contracts in an exported fleet",
            )
            run.add_argument(
                "--no-baseline",
                action="store_true",
                help="Run contracts without comparing against reviewed baselines",
            )
    review = sub.add_parser(
        "review", help="Compare measured candidates with approved expectations"
    )
    review.add_argument("--platform")
    review.add_argument(
        "--approve",
        action="store_true",
        help="Approve new and unchanged candidates; changed values require manual review",
    )
    update = sub.add_parser(
        "update",
        help="Ask the HF planner to add a path while retaining existing outputs and weights",
    )
    update.add_argument("request")
    update.add_argument("--retrain", action="store_true")
    llm_options(update)
    replay = sub.add_parser(
        "replay",
        help="Require a historical failure on the bad revision and a pass on the fix",
    )
    replay.add_argument("fix")
    replay.add_argument("--bad")
    replay.add_argument(
        "--repo",
        required=True,
        help="Local Transformers Git checkout with both revisions",
    )
    replay.add_argument("--variant", default="default")
    replay.add_argument(
        "--match-output", help="Require this output to fail on the bad build"
    )
    publish = sub.add_parser(
        "publish",
        help="Preview publication; --execute uploads reviewed artifacts to the Hub",
    )
    publish.add_argument("--execute", action="store_true")
    publish.add_argument("--private", action="store_true")
    export = sub.add_parser(
        "export",
        help="Preview or publish immutable pins to the shared fleet dataset",
    )
    export.add_argument(
        "--transformers-checkout",
        help="Write tests/contracts.lock after a successful upload",
    )
    export.add_argument("--fleet", choices=tuple(fleet.FILES), default="control")
    export.add_argument(
        "--expected-revision", help="Reject upload if the dataset changed since preview"
    )
    export.add_argument(
        "--execute", action="store_true", help="Upload the previewed fleet update"
    )
    pin = sub.add_parser(
        "pin", help="Pin a dataset revision in a Transformers checkout"
    )
    pin.add_argument("--transformers-checkout", required=True)
    pin.add_argument("--revision", default="main")
    export.add_argument(
        "--contract-name", help="Destination name, e.g. mic-bert alongside BERT"
    )
    report = sub.add_parser(
        "report", help="Render CI results and reproducible tmic commands"
    )
    report.add_argument("results", type=Path)
    report.add_argument("--log", type=Path)
    report.add_argument("--lock", type=Path)
    sub.add_parser(
        "auto",
        help="Check AutoConfig and Auto-model factory mappings for the built slim",
    )
    sub.add_parser(
        "status", help="Show project provenance, weight choice and check receipts"
    )
    batch = sub.add_parser(
        "batch", help="Regenerate a declared prototype fleet and report every stage"
    )
    batch.add_argument("--models-file", type=Path, required=True)
    batch.add_argument("--out", required=True)
    batch.add_argument("--namespace", default="huggingface-mic")
    batch.add_argument("--phase", choices=("prepare", "run", "all"), default="all")
    batch.add_argument(
        "--select", action="append", help="Model names to run (repeatable)"
    )
    batch.add_argument("--resume", action="store_true")
    batch.add_argument(
        "--publish",
        action="store_true",
        help="Upload only after all declared checks pass",
    )
    batch.add_argument("--timeout", type=int, default=1800, help="Seconds per stage")
    batch.add_argument("--no-history", action="store_true")
    batch.add_argument("--history-limit", type=int, default=10)
    llm_options(batch)
    return root


def external_check(args):
    options = [
        "--framework",
        "transformers",
        "--lock",
        str(args.lock.resolve()),
        "--require-all",
    ]
    for target in args.target or []:
        options += ["--target", target]
    if args.isolate:
        options.append("--isolate")
    if args.fixtures_only:
        options.append("--fixtures-only")
    if args.changed_files:
        options += ["--changed-files", str(args.changed_files.resolve())]
    if args.output:
        options += ["--output", str(args.output.resolve())]
    if args.run_all or not args.changed_files:
        options.append("--run-all")
    if args.no_baseline:
        options.append("--no-baseline")
    if not args.all_variants:
        for name in args.contract or []:
            options += ["--contract", name]
        return backend.invoke(
            "run", Path.cwd(), [*options, "--variant", args.variant]
        ).returncode
    from .workspace import read
    from .hub import canonical_repo
    from huggingface_hub import hf_hub_download

    lock = read(args.lock)
    selected = args.contract or list(lock["contracts"])
    if any(name not in lock["contracts"] for name in selected):
        raise ValueError("Unknown contract selected in the exported fleet")
    status = 1 if not selected else 0
    for name in selected:
        pin = lock["contracts"][name]["integration"]
        manifest = read(
            hf_hub_download(
                canonical_repo(pin["repo"]),
                "integration.json",
                revision=pin["revision"],
            )
        )
        for variant in manifest["integrations"]["transformers"].get(
            "variants", {"default": {}}
        ):
            status = max(
                status,
                backend.invoke(
                    "run",
                    Path.cwd(),
                    [*options, "--contract", name, "--variant", variant],
                ).returncode,
            )
    return status


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "baselines":
            from .baselines import run

            return run(args)
        if args.command == "auth":
            from .dashboard_auth import run

            return run(args)
        if args.command == "run":
            from .automation import run

            return run(args)
        if args.command == "batch":
            from .batch import run

            return run(args)
        if args.command == "check":
            if args.lock and (args.fleet or args.fleet_revision):
                raise ValueError(
                    "--lock cannot be combined with --fleet or --fleet-revision"
                )
            if args.lock and args.lock.name == "contracts.lock":
                args.fleet_revision = fleet.read_pin(args.lock)
                args.lock = None
                args.fleet = "control"
            if (
                not args.lock
                and not args.fleet
                and not args.fleet_revision
                and Path("tests/contracts.lock").exists()
                and not (Path(args.workspace) / ".mic/project.json").exists()
            ):
                args.fleet = "control"
            if args.fleet or args.fleet_revision:
                revision = args.fleet_revision
                if not revision and Path("tests/contracts.lock").exists():
                    revision = fleet.read_pin("tests/contracts.lock")
                args.lock, revision = fleet.download(
                    args.fleet or "control", revision or "main"
                )
                print(
                    json.dumps(
                        {
                            "fleet_repository": fleet.REPOSITORY,
                            "fleet_revision": revision,
                        }
                    ),
                    file=sys.stderr,
                )
            if args.lock:
                return external_check(args)
        if args.command == "report":
            options = [str(args.results.resolve()), "--runner", "tmic check"]
            if args.log:
                options += ["--log", str(args.log.resolve())]
            if args.lock:
                options += ["--lock", str(args.lock)]
            return backend.invoke("report", Path.cwd(), options).returncode
        if args.command == "auto":
            project = Workspace(args.workspace)
            return backend.invoke("auto-stage", project.root).returncode
        if args.command == "status":
            print(json.dumps(Workspace(args.workspace).state, indent=2))
            return 0
        if args.command in ("record", "check"):
            return commands.execute(args, record=args.command == "record")
        return getattr(commands, args.command)(args)
    except (ValueError, OSError, jsonschema.ValidationError, HfHubHTTPError) as exc:
        print(f"tmic: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
