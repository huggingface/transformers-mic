"""Command line interface for measured model integration contracts."""

import argparse
import json
from pathlib import Path
import sys

import jsonschema
from huggingface_hub.errors import HfHubHTTPError

from . import __version__, backend, commands, planner
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
        description="Build, measure and review Transformers model integration contracts."
    )
    root.add_argument("--version", action="version", version=__version__)
    root.add_argument(
        "--workspace",
        default=".",
        help="Project directory (default: current directory)",
    )
    sub = root.add_subparsers(dest="command", required=True)
    create = sub.add_parser(
        "create", help="Reuse a catalog contract or ask the HF planner for a draft"
    )
    create.add_argument("model")
    create.add_argument("--out", required=True)
    create.add_argument("--contract", help="Choose a catalog usage path")
    create.add_argument(
        "--namespace", help="Hub owner for new artifacts; defaults to logged-in account"
    )
    create.add_argument("--revision")
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
                "--contract",
                action="append",
                help="Select contracts in an exported fleet",
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
        help="Write immutable pins and expectations into a Transformers checkout",
    )
    export.add_argument("--transformers-checkout", required=True)
    sub.add_parser(
        "status", help="Show project provenance, weight choice and check receipts"
    )
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
    if not args.all_variants:
        for name in args.contract or []:
            options += ["--contract", name]
        return backend.invoke(
            "run", Path.cwd(), [*options, "--variant", args.variant]
        ).returncode
    from .workspace import read
    from huggingface_hub import hf_hub_download

    lock = read(args.lock)
    selected = args.contract or list(lock["contracts"])
    if any(name not in lock["contracts"] for name in selected):
        raise ValueError("Unknown contract selected in the exported fleet")
    status = 1 if not selected else 0
    for name in selected:
        pin = lock["contracts"][name]["integration"]
        manifest = read(
            hf_hub_download(pin["repo"], "integration.json", revision=pin["revision"])
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
        if args.command == "check" and args.lock:
            return external_check(args)
        if args.command == "status":
            print(json.dumps(Workspace(args.workspace).state, indent=2))
            return 0
        if args.command in ("record", "check"):
            return commands.execute(args, record=args.command == "record")
        return getattr(commands, args.command)(args)
    except (ValueError, OSError, jsonschema.ValidationError, HfHubHTTPError) as exc:
        print(f"mic: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
