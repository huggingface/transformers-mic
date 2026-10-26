"""Resumable prototype regeneration; every model gets an explicit stage result."""

import json
import os
import signal
import re
from pathlib import Path
import subprocess
import sys
import threading

from .workspace import Workspace, read, write


def invoke(out, name, arguments, timeout):
    log = out / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("\nCOMMAND " + json.dumps(arguments) + "\n")
        handle.flush()
        stream_progress = "baselines" in arguments
        progress_thread = None
        try:
            child_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if stream_progress and sys.stderr.isatty() and os.getenv("TERM") != "dumb":
                child_env["TMIC_LIVE_PROGRESS"] = "1"
            process = subprocess.Popen(
                [sys.executable, "-m", "transformers_mic", *arguments],
                stdout=handle,
                stderr=subprocess.PIPE if stream_progress else subprocess.STDOUT,
                text=True,
                start_new_session=True,
                env=child_env,
            )

            if stream_progress:

                def relay():
                    for line in process.stderr:
                        handle.write(re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", line))
                        handle.flush()
                        print(line, end="", file=sys.stderr, flush=True)

                progress_thread = threading.Thread(target=relay, daemon=True)
                progress_thread.start()
            process.wait(timeout=timeout)
            if progress_thread:
                progress_thread.join()
            return {"exit_code": process.returncode, "log": str(log.relative_to(out))}
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if progress_thread:
                progress_thread.join()
            if isinstance(error, KeyboardInterrupt):
                raise
            return {
                "exit_code": 124,
                "error": "stage timeout",
                "log": str(log.relative_to(out)),
            }


def run(args):
    spec = read(args.models_file)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report = (
        read(report_path)
        if report_path.exists()
        else {
            "scope": spec.get("scope", "fixtures"),
            "auto": spec.get("auto"),
            "models": {},
        }
    )
    selected = set(args.select or [])
    names = set()
    for item in spec["models"]:
        name = item["name"]
        if name in names or (selected and name not in selected):
            continue
        names.add(name)
        row = report["models"].setdefault(name, {**item, "stages": {}})
        row.update(item)
        workspace = out / "projects" / name
        if (
            args.phase in ("prepare", "all")
            and not Workspace(workspace).state_path.exists()
        ):
            command = [
                "create",
                item["checkpoint"],
                "--out",
                str(workspace),
                "--name",
                name,
                "--namespace",
                args.namespace,
                "--regenerate",
                "--provider",
                args.provider,
                "--llm-model",
                args.llm_model,
            ]
            if spec.get("scope") == "fixtures":
                command.append("--fixtures-only")
            if args.no_history:
                command.append("--no-history")
            command += ["--limit", str(args.history_limit)]
            print(f"{name}: prepare", flush=True)
            row["stages"]["prepare"] = invoke(out, name, command, args.timeout)
            write(report_path, report)
            if row["stages"]["prepare"]["exit_code"]:
                row["status"] = "prepare-failed"
                write(report_path, report)
                continue
        if not Workspace(workspace).state_path.exists():
            row["status"] = (
                row.get("status")
                if row.get("status", "").endswith("failed")
                else "not-prepared"
            )
            write(report_path, report)
            continue
        if args.phase == "prepare":
            row["status"] = "prepared"
            write(report_path, report)
            continue
        stages = [
            ("build", ["build"]),
            ("auto", ["auto"]),
            ("record", ["record", "--all-variants"]),
            ("review", ["review"]),
            ("approve", ["review", "--approve"]),
            ("check", ["check", "--all-variants"]),
        ]
        if args.publish:
            stages += [
                ("publish", ["publish", "--execute"]),
                ("published-check", ["check", "--all-variants"]),
            ]
        row["status"] = "running"
        for stage, command in stages:
            previous = row["stages"].get(stage, {})
            # Checks are always refreshed; failed stages can be retried after a
            # source fix. Recording candidates intentionally requires review again.
            if (
                args.resume
                and previous.get("exit_code") == 0
                and stage in ("build", "publish")
            ):
                continue
            print(f"{name}: {stage}", flush=True)
            result = invoke(
                out, name, ["--workspace", str(workspace), *command], args.timeout
            )
            row["stages"][stage] = result
            write(report_path, report)
            if result["exit_code"]:
                row["status"] = stage + "-failed"
                break
        else:
            row["status"] = "published" if args.publish else "validated"
        write(report_path, report)
    counts = {}
    for row in report["models"].values():
        counts[row.get("status", "unknown")] = (
            counts.get(row.get("status", "unknown"), 0) + 1
        )
    print(json.dumps({"report": str(report_path), "summary": counts}), flush=True)
    return int(
        any(
            row.get("status", "").endswith("failed")
            or row.get("status") == "not-prepared"
            for name, row in report["models"].items()
            if not selected or name in selected
        )
    )
